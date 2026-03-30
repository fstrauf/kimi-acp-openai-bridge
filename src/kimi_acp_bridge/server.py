"""FastAPI server for Kimi ACP Bridge."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kimi_acp_bridge.acp_client import ACPClient
from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    Message,
    ModelInfo,
    ModelList,
    StreamingChoice,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from kimi_acp_bridge.translator import (
    acp_to_openai_chunk,
    create_final_chunk,
    estimate_token_count,
    generate_completion_id,
    generate_tool_call_id,
    openai_to_acp_messages,
    openai_to_acp_tools,
)

logger = structlog.get_logger()

# Available models (Kimi K2.5 is the primary model)
AVAILABLE_MODELS = [
    ModelInfo(
        id="kimi-k2.5",
        created=1677610602,
        owned_by="moonshot-ai",
    ),
]


def create_app(config: BridgeConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = BridgeConfig.load()
    
    # Configure structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), config.log_level.upper())
        ),
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager."""
        logger.info(
            "starting_kimi_acp_bridge",
            host=config.host,
            port=config.port,
            kimi_binary=config.kimi_binary,
        )
        yield
        logger.info("stopping_kimi_acp_bridge")
    
    app = FastAPI(
        title="Kimi ACP Bridge",
        description="OpenAI-compatible API bridge for Kimi ACP",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error",
                    type="internal_error",
                    code="internal_error",
                )
            ).model_dump(),
        )
    
    @app.get("/health")
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        kimi_available = False
        try:
            # Quick check if kimi binary exists
            proc = await asyncio.create_subprocess_exec(
                config.kimi_binary, "--version",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            kimi_available = proc.returncode == 0
        except Exception:
            pass
        
        return HealthResponse(
            status="healthy",
            kimi_available=kimi_available,
            version="0.1.0",
        )
    
    @app.get("/v1/models")
    async def list_models() -> ModelList:
        """List available models."""
        return ModelList(data=AVAILABLE_MODELS)
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> StreamingResponse | JSONResponse:
        """OpenAI-compatible chat completions endpoint."""
        start_time = time.time()
        request_id = generate_completion_id()
        created = int(time.time())
        
        logger.info(
            "chat_completion_request",
            request_id=request_id,
            model=request.model,
            stream=request.stream,
            num_messages=len(request.messages),
            has_tools=request.tools is not None,
        )
        
        # Validate model
        model_ids = [m.id for m in AVAILABLE_MODELS]
        if request.model not in model_ids:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Model '{request.model}' not found. Available: {model_ids}",
                        type="invalid_request_error",
                        param="model",
                        code="model_not_found",
                    )
                ).model_dump(),
            )
        
        # Convert OpenAI messages to ACP format
        preamble, acp_messages = openai_to_acp_messages(request.messages)
        
        # Convert tools if present
        acp_tools = openai_to_acp_tools(request.tools) if request.tools else None
        
        try:
            client = ACPClient(config)
            await client.connect()
            
            session = await client.create_session(
                preamble=preamble,
                tools=acp_tools,
            )
            
            if request.stream:
                # Streaming response
                async def generate_stream() -> AsyncIterator[str]:
                    completion_id = request_id
                    has_tool_calls = False
                    
                    try:
                        async for event in client.prompt(session, acp_messages, stream=True):
                            event_type = event.get("type", "")
                            
                            # Handle tool calls
                            if event_type == "tool_call.start":
                                has_tool_calls = True
                            
                            # Convert ACP event to OpenAI chunk
                            chunk = acp_to_openai_chunk(
                                event, request.model, completion_id, created
                            )
                            
                            if chunk:
                                data = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                                yield data
                            
                            # Check for completion
                            if event_type == "done":
                                break
                            
                            if event_type == "error":
                                error_msg = event.get("error", {}).get("message", "Unknown error")
                                error_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        StreamingChoice(
                                            index=0,
                                            delta=ChoiceDelta(),
                                            finish_reason="stop",
                                        )
                                    ],
                                )
                                yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
                                break
                        
                        # Send final chunk
                        final_chunk = create_final_chunk(
                            request.model, completion_id, created,
                            finish_reason="tool_calls" if has_tool_calls else "stop"
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    finally:
                        await client.close()
                        duration = time.time() - start_time
                        logger.info(
                            "chat_completion_complete",
                            request_id=request_id,
                            duration_ms=round(duration * 1000, 2),
                            streaming=True,
                        )
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                    },
                )
            
            else:
                # Non-streaming response
                full_content = ""
                tool_calls: list[ToolCall] = []
                
                async for event in client.prompt(session, acp_messages, stream=False):
                    if event.get("type") == "complete":
                        full_content = event.get("content", "")
                        raw_tool_calls = event.get("tool_calls", [])
                        
                        for tc in raw_tool_calls:
                            tool_calls.append(
                                ToolCall(
                                    id=tc.get("id", generate_tool_call_id()),
                                    function=ToolCallFunction(
                                        name=tc.get("function", {}).get("name", ""),
                                        arguments=tc.get("function", {}).get("arguments", ""),
                                    ),
                                )
                            )
                        break
                
                await client.close()
                
                # Build response
                prompt_text = "\n".join(m.content or "" for m in request.messages)
                prompt_tokens = estimate_token_count(prompt_text)
                completion_tokens = estimate_token_count(full_content)
                
                message = Message(
                    role="assistant",
                    content=full_content if not tool_calls else None,
                    tool_calls=tool_calls if tool_calls else None,
                )
                
                response = ChatCompletionResponse(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            message=message,
                            finish_reason="tool_calls" if tool_calls else "stop",
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                
                duration = time.time() - start_time
                logger.info(
                    "chat_completion_complete",
                    request_id=request_id,
                    duration_ms=round(duration * 1000, 2),
                    streaming=False,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                
                return JSONResponse(content=response.model_dump(exclude_none=True))
        
        except RuntimeError as e:
            logger.error("kimi_runtime_error", error=str(e))
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=str(e),
                        type="service_unavailable",
                        code="kimi_unavailable",
                    )
                ).model_dump(),
            ) from e
        
        except Exception as e:
            logger.error("chat_completion_error", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Internal error: {str(e)}",
                        type="internal_error",
                        code="internal_error",
                    )
                ).model_dump(),
            ) from e
    
    return app
