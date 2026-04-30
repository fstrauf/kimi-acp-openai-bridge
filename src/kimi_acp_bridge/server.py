"""FastAPI server for Kimi ACP Bridge."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kimi_acp_bridge.acp_client import ACPClient
from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.direct_client import DirectClient
from kimi_acp_bridge.models import (
    BackendCapabilities,
    BridgeError,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    Limits,
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
    compute_prompt_bytes,
    create_final_chunk,
    estimate_token_count,
    generate_completion_id,
    generate_tool_call_id,
    openai_to_acp_messages,
    openai_to_acp_tools,
)

logger = structlog.get_logger()

BRIDGE_VERSION = "0.1.0"

# Available models (Kimi K2.5 is the primary model)
AVAILABLE_MODELS = [
    ModelInfo(
        id="kimi-k2.5",
        created=1677610602,
        owned_by="moonshot-ai",
    ),
]


def _make_request_id() -> str:
    """Generate a stable request id."""
    return f"req_{uuid.uuid4().hex[:24]}"


def resolve_backend(
    request: ChatCompletionRequest,
    header_value: str,
    config: BridgeConfig,
) -> str:
    """Resolve the effective backend using deterministic auto-routing rules.

    Rules (in order):
    1. If the client explicitly asks for ``direct`` or ``acp``, honour it.
    2. If ``auto`` (or config default is ``auto``):
       - Tools requested and tool_choice != "none"  → ``acp``
       - response_format == "json_object"           → ``direct`` (fast)
       - Otherwise                                  → ``direct``
    """
    hv = header_value.lower()
    if hv in ("direct", "acp"):
        return hv

    # auto or anything else → apply rules
    if request.tools and request.tool_choice != "none":
        return "acp"
    if request.response_format and request.response_format.type == "json_object":
        return "direct"
    return "direct"


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

    request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)


    def _error_response(
        status_code: int,
        code: str,
        message: str,
        request_id: str,
        backend: str | None = None,
        phase: str | None = None,
        details: dict | None = None,
        retryable: bool = False,
    ) -> JSONResponse:
        """Build a structured JSON error response."""
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=message,
                    type="invalid_request_error" if status_code < 500 else "service_unavailable",
                    code=code,
                    retryable=retryable,
                    backend=backend,
                    request_id=request_id,
                    phase=phase,
                    details=details,
                )
            ).model_dump(exclude_none=True),
            headers={"x-request-id": request_id},
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
            ).model_dump(exclude_none=True),
        )

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint with capability discovery."""
        request_id = _make_request_id()
        kimi_available = False
        kimi_cli_version = None
        try:
            proc = await asyncio.create_subprocess_exec(
                config.kimi_binary,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=config.health_timeout)
            kimi_available = proc.returncode == 0
            if kimi_available:
                version_text = (
                    stdout.decode("utf-8", errors="replace").strip()
                    or stderr.decode("utf-8", errors="replace").strip()
                )
                match = re.search(r"(\d+\.\d+\.\d+)", version_text)
                if match:
                    kimi_cli_version = match.group(1)
        except Exception:
            pass

        body = HealthResponse(
            status="healthy" if kimi_available else "degraded",
            kimi_available=kimi_available,
            bridge_version=BRIDGE_VERSION,
            kimi_cli_version=kimi_cli_version,
            models=[m.id for m in AVAILABLE_MODELS],
            backends={
                "direct": BackendCapabilities(
                    available=kimi_available,
                    tool_calls=False,
                    json_mode=True,
                    file_io=False,
                ),
                "acp": BackendCapabilities(
                    available=kimi_available,
                    tool_calls=config.enable_tools,
                    json_mode=True,
                    file_io=True,
                ),
            },
            limits=Limits(
                max_prompt_bytes_direct=config.max_prompt_bytes_direct,
                max_prompt_bytes_acp=config.max_prompt_bytes_acp,
                max_concurrent_requests=config.max_concurrent_requests,
            ),
        )
        return JSONResponse(
            content=body.model_dump(exclude_none=True),
            headers={"x-request-id": request_id},
        )

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        request_id = _make_request_id()
        body = ModelList(data=AVAILABLE_MODELS)
        return JSONResponse(
            content=body.model_dump(exclude_none=True),
            headers={"x-request-id": request_id},
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        http_request: Request,
        request: ChatCompletionRequest,
    ) -> StreamingResponse | JSONResponse:
        """OpenAI-compatible chat completions endpoint."""
        # Concurrency guard
        try:
            await asyncio.wait_for(request_semaphore.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            rid = generate_completion_id()
            return _error_response(
                status_code=429,
                code="rate_limit",
                message="Too many concurrent requests. Please retry later.",
                request_id=rid,
                retryable=True,
                details={"max_concurrent_requests": config.max_concurrent_requests},
            )


        async def _do() -> StreamingResponse | JSONResponse:
            start_time = time.time()
            request_id = generate_completion_id()
            created = int(time.time())

            # Resolve backend: header > auto rules > config default
            header_backend = http_request.headers.get("x-kimi-backend", config.kimi_backend)
            effective_backend = resolve_backend(request, header_backend, config)

            logger.info(
                "request_received",
                request_id=request_id,
                method=http_request.method,
                path=str(http_request.url.path),
                content_length=http_request.headers.get("content-length"),
            )
            logger.info(
                "chat_completion_request",
                request_id=request_id,
                model=request.model,
                backend=effective_backend,
                requested_backend=header_backend,
                stream=request.stream,
                num_messages=len(request.messages),
                has_tools=request.tools is not None,
                response_format=request.response_format.type if request.response_format else None,
                request_body_bytes=len(request.model_dump_json(exclude_none=True).encode("utf-8")),
            )

            # Validate model
            model_ids = [m.id for m in AVAILABLE_MODELS]
            if request.model not in model_ids:
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            message=f"Model '{request.model}' not found. Available: {model_ids}",
                            type="invalid_request_error",
                            param="model",
                            code="model_not_found",
                        )
                    ).model_dump(exclude_none=True),
                    headers={"x-request-id": request_id},
                )

            # Handle tool_choice: "none" by stripping tools
            effective_tools = request.tools
            if request.tool_choice == "none":
                effective_tools = None
                logger.debug(
                    "tool_choice_none_set",
                    message="Stripping tools from request due to tool_choice=none",
                )

            # Convert OpenAI messages to ACP format
            translate_start = time.perf_counter()
            preamble, acp_messages = openai_to_acp_messages(
                request.messages,
                response_format=request.response_format,
            )

            # Convert tools if present
            acp_tools = openai_to_acp_tools(effective_tools) if effective_tools else None

            # Prompt size guard
            prompt_bytes = compute_prompt_bytes(preamble, acp_messages)
            prompt_tokens = estimate_token_count(
                "\n\n".join([preamble or ""] + [m.get("content", "") for m in acp_messages])
            )
            max_bytes = (
                config.max_prompt_bytes_direct
                if effective_backend == "direct"
                else config.max_prompt_bytes_acp
            )
            if prompt_bytes > config.max_prompt_bytes_warning:
                logger.warning(
                    "prompt_size_warning",
                    request_id=request_id,
                    backend=effective_backend,
                    prompt_bytes=prompt_bytes,
                    estimated_tokens=prompt_tokens,
                    limit=max_bytes,
                )
            if prompt_bytes > max_bytes:
                logger.error(
                    "prompt_too_large",
                    request_id=request_id,
                    backend=effective_backend,
                    prompt_bytes=prompt_bytes,
                    estimated_tokens=prompt_tokens,
                    limit=max_bytes,
                )
                return JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            message=f"Prompt too large ({prompt_bytes} bytes). Limit: {max_bytes} bytes.",
                            type="invalid_request_error",
                            code="prompt_too_large",
                            details={
                                "prompt_bytes": prompt_bytes,
                                "estimated_tokens": prompt_tokens,
                                "limit": max_bytes,
                                "backend": effective_backend,
                            },
                        )
                    ).model_dump(exclude_none=True),
                    headers={"x-request-id": request_id},
                )

            logger.info(
                "chat_completion_translated",
                request_id=request_id,
                duration_ms=round((time.perf_counter() - translate_start) * 1000, 2),
                preamble_bytes=len((preamble or "").encode("utf-8")),
                acp_messages_count=len(acp_messages),
                acp_messages_json_bytes=len(json.dumps(acp_messages).encode("utf-8")),
                acp_tools_count=len(acp_tools or []),
                prompt_bytes=prompt_bytes,
                estimated_tokens=prompt_tokens,
            )

            if effective_backend == "direct":
                if effective_tools is not None:
                    return JSONResponse(
                        status_code=422,
                        content=ErrorResponse(
                            error=ErrorDetail(
                                message=(
                                    "Kimi direct backend does not support native tool calls. "
                                    "Use tool_choice='none' or backend='acp'."
                                ),
                                type="invalid_request_error",
                                param="tools",
                                code="tools_not_supported",
                            )
                        ).model_dump(exclude_none=True),
                        headers={"x-request-id": request_id},
                    )

                direct_client = DirectClient(config, request_id=request_id)

                try:
                    if request.stream:
                        async def generate_direct_stream() -> AsyncIterator[str]:
                            result = await direct_client.prompt(preamble, acp_messages)
                            content_chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamingChoice(
                                        index=0,
                                        delta=ChoiceDelta(role="assistant", content=result.content),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

                            final_chunk = create_final_chunk(request.model, request_id, created)
                            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                            yield "data: [DONE]\n\n"

                            logger.info(
                                "chat_completion_complete",
                                request_id=request_id,
                                duration_ms=round((time.time() - start_time) * 1000, 2),
                                streaming=True,
                                backend="direct",
                            )

                        return StreamingResponse(
                            generate_direct_stream(),
                            media_type="text/plain",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                                "Content-Type": "text/event-stream",
                                "x-request-id": request_id,
                            },
                        )

                    result = await direct_client.prompt(preamble, acp_messages)

                    prompt_tokens = estimate_token_count(result.prompt_text)
                    completion_tokens = estimate_token_count(result.content)
                    response = ChatCompletionResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            Choice(
                                index=0,
                                message=Message(role="assistant", content=result.content),
                                finish_reason="stop",
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                    )

                    logger.info(
                        "chat_completion_complete",
                        request_id=request_id,
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                        streaming=False,
                        backend="direct",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    logger.info(
                        "response_serialized",
                        request_id=request_id,
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                        backend="direct",
                    )
                    return JSONResponse(
                        content=response.model_dump(exclude_none=True),
                        headers={"x-request-id": request_id},
                    )
                except BridgeError as e:
                    logger.error(
                        "kimi_bridge_error",
                        error=str(e),
                        request_id=request_id,
                        code=e.code,
                        phase=e.phase,
                    )
                    status = 504 if "timeout" in e.code else 503
                    return _error_response(
                        status_code=status,
                        code=e.code,
                        message=str(e),
                        request_id=request_id,
                        backend="direct",
                        phase=e.phase,
                        details=e.details,
                        retryable="timeout" in e.code or e.code in ("backend_step_limit",),
                    )

            try:
                client = ACPClient(config, request_id=request_id)
                connect_start = time.perf_counter()
                await client.connect()
                logger.info(
                    "chat_completion_acp_connected",
                    request_id=request_id,
                    duration_ms=round((time.perf_counter() - connect_start) * 1000, 2),
                )

                session_start = time.perf_counter()
                session = await client.create_session(
                    preamble=preamble,
                    tools=acp_tools,
                )
                logger.info(
                    "chat_completion_session_ready",
                    request_id=request_id,
                    session_id=session.session_id,
                    duration_ms=round((time.perf_counter() - session_start) * 1000, 2),
                )

                if request.stream:
                    # Streaming response
                    async def generate_stream() -> AsyncIterator[str]:
                        completion_id = request_id
                        has_tool_calls = False
                        prompt_start = time.perf_counter()
                        first_event_logged = False

                        try:
                            async for event in client.prompt(
                                session,
                                acp_messages,
                                stream=True,
                                enable_native_tools=effective_tools is not None,
                            ):
                                event_type = event.get("type", "")
                                if not first_event_logged:
                                    first_event_logged = True
                                    logger.info(
                                        "chat_completion_first_event",
                                        request_id=request_id,
                                        event_type=event_type,
                                        duration_ms=round(
                                            (time.perf_counter() - prompt_start) * 1000, 2
                                        ),
                                    )

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
                                request.model,
                                completion_id,
                                created,
                                finish_reason="tool_calls" if has_tool_calls else "stop",
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
                            "x-request-id": request_id,
                        },
                    )

                else:
                    # Non-streaming response
                    full_content = ""
                    tool_calls: list[ToolCall] = []
                    prompt_start = time.perf_counter()
                    first_event_logged = False

                    try:
                        async for event in client.prompt(
                            session,
                            acp_messages,
                            stream=False,
                            enable_native_tools=effective_tools is not None,
                        ):
                            if not first_event_logged and event.get("type"):
                                first_event_logged = True
                                logger.info(
                                    "chat_completion_first_event",
                                    request_id=request_id,
                                    event_type=event.get("type", ""),
                                    duration_ms=round((time.perf_counter() - prompt_start) * 1000, 2),
                                )
                            if event.get("type") == "error":
                                err = event.get("error", {})
                                raise BridgeError(
                                    code="backend_process_failed",
                                    message=err.get("message", "ACP returned an error event"),
                                    phase="acp_completion",
                                    details=err,
                                )

                            if event.get("type") == "complete":
                                full_content = event.get("content", "")
                                raw_tool_calls = event.get("tool_calls", [])

                                for tc in raw_tool_calls:
                                    tool_calls.append(
                                        ToolCall(
                                            id=tc.get("id", generate_tool_call_id()),
                                            function=ToolCallFunction(
                                                name=tc.get("name", ""),
                                                arguments=tc.get("arguments", ""),
                                            ),
                                        )
                                    )
                                break
                    finally:
                        await client.close()

                    if not full_content.strip() and not tool_calls:
                        logger.error(
                            "chat_completion_empty_response",
                            request_id=request_id,
                            duration_ms=round((time.time() - start_time) * 1000, 2),
                        )
                        raise BridgeError(
                            code="backend_empty_response",
                            message="Kimi ACP process exited successfully but returned no content",
                            phase="acp_completion",
                        )

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

                    logger.info(
                        "response_serialized",
                        request_id=request_id,
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                        backend="acp",
                    )
                    return JSONResponse(
                        content=response.model_dump(exclude_none=True),
                        headers={"x-request-id": request_id},
                    )

            except BridgeError as e:
                logger.error(
                    "kimi_bridge_error",
                    error=str(e),
                    request_id=request_id,
                    code=e.code,
                    phase=e.phase,
                )
                status = 504 if "timeout" in e.code else 503
                return _error_response(
                    status_code=status,
                    code=e.code,
                    message=str(e),
                    request_id=request_id,
                    backend=effective_backend,
                    phase=e.phase,
                    details=e.details,
                    retryable="timeout" in e.code or e.code in ("backend_step_limit",),
                )

            except Exception as e:
                logger.error("chat_completion_error", error=str(e), request_id=request_id)
                return _error_response(
                    status_code=500,
                    code="internal_error",
                    message=f"Internal error: {str(e)}",
                    request_id=request_id,
                    backend=effective_backend,
                )

        try:
            return await _do()
        finally:
            request_semaphore.release()
    return app
