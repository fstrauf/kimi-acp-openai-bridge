"""ACP client for communicating with Kimi CLI using official ACP protocol."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.models import BridgeError
from kimi_acp_bridge.utils import build_controlled_env, validate_work_dir

logger = structlog.get_logger()


@dataclass
class ACPSession:
    """Represents an ACP session."""

    session_id: str
    preamble: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


class ACPClient:
    """Client for communicating with Kimi ACP process using JSON-RPC over stdio."""

    def __init__(self, config: BridgeConfig, request_id: str | None = None):
        self.config = config
        self.request_id = request_id
        self.process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._session: ACPSession | None = None
        self._message_id = 0

    async def connect(self) -> None:
        """Spawn kimi acp process and initialize connection."""
        async with self._lock:
            if self.process is not None:
                return

            logger.info(
                "spawning_kimi_process",
                request_id=self.request_id,
                binary=self.config.kimi_binary,
                args=self.config.kimi_args,
            )

            try:
                connect_start = time.perf_counter()
                self.process = await asyncio.create_subprocess_exec(
                    self.config.kimi_binary,
                    *self.config.kimi_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if self.config.log_acp_messages else None,
                    env=build_controlled_env(),
                )
                spawn_ms = round((time.perf_counter() - connect_start) * 1000, 2)
                logger.info(
                    "kimi_process_spawned",
                    request_id=self.request_id,
                    pid=self.process.pid,
                    duration_ms=spawn_ms,
                )

                # Send initialize request per ACP spec
                await self._send_request(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "kimi-acp-bridge",
                            "version": "0.1.0",
                        },
                    },
                )

                # Wait for initialize response
                response = await self._read_response(timeout=self.config.acp_initialize_timeout)
                if response.get("result"):
                    logger.info(
                        "acp_initialized",
                        request_id=self.request_id,
                        duration_ms=round((time.perf_counter() - connect_start) * 1000, 2),
                    )
                else:
                    error = response.get("error", {})
                    raise BridgeError(
                        code="backend_process_failed",
                        message=f"ACP initialization failed: {error}",
                        phase="acp_initialize",
                    )

                # Send initialized notification
                await self._send_notification("initialized", {})

            except FileNotFoundError as e:
                logger.error(
                    "kimi_binary_not_found",
                    request_id=self.request_id,
                    binary=self.config.kimi_binary,
                )
                raise BridgeError(
                    code="kimi_not_found",
                    message=f"Kimi CLI not found at '{self.config.kimi_binary}'. Please ensure Kimi CLI is installed and in PATH.",
                    phase="acp_initialize",
                ) from e
            except asyncio.TimeoutError as e:
                logger.error("acp_initialize_timeout", request_id=self.request_id)
                await self.close()
                raise BridgeError(
                    code="acp_initialize_timeout",
                    message=f"ACP initialize timed out after {self.config.acp_initialize_timeout}s",
                    phase="acp_initialize",
                    details={"timeout_seconds": self.config.acp_initialize_timeout},
                ) from e
            except Exception as e:
                logger.error("failed_to_spawn_kimi", request_id=self.request_id, error=str(e))
                await self.close()
                if isinstance(e, BridgeError):
                    raise
                raise BridgeError(
                    code="backend_process_failed",
                    message=f"Failed to start Kimi ACP process: {e}",
                    phase="acp_initialize",
                ) from e

    async def create_session(
        self,
        preamble: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        cwd: str | None = None,
    ) -> ACPSession:
        """Create a new ACP session using session/new method."""
        if self.process is None:
            await self.connect()

        work_dir = cwd or os.getcwd()
        try:
            work_dir = validate_work_dir(work_dir)
        except ValueError as e:
            raise BridgeError(
                code="invalid_request_error",
                message=str(e),
                phase="acp_session",
            ) from e

        # Use ACP session/new method
        params: dict[str, Any] = {
            "cwd": work_dir,
            "mcpServers": [],  # No MCP servers for now
        }
        # preamble is accepted by the bridge but currently ignored by Kimi ACP.
        # We include it for forward-compatibility and also prepend it to prompts.
        if preamble:
            params["preamble"] = preamble

        session_start = time.perf_counter()
        await self._send_request("session/new", params)

        response = await self._read_response(timeout=self.config.acp_session_timeout)
        result = response.get("result", {})

        if "error" in response:
            raise BridgeError(
                code="backend_process_failed",
                message=f"Failed to create session: {response['error']}",
                phase="acp_session",
            )

        session_id = result.get("sessionId", str(uuid.uuid4()))
        self._session = ACPSession(
            session_id=session_id,
            preamble=preamble,
            tools=tools or [],
        )

        logger.info(
            "session_created",
            request_id=self.request_id,
            session_id=session_id,
            duration_ms=round((time.perf_counter() - session_start) * 1000, 2),
            preamble_bytes=len((preamble or "").encode("utf-8")),
            tools_count=len(tools or []),
        )
        return self._session

    def _build_prompt_text(self, session: ACPSession, messages: list[dict[str, Any]]) -> str:
        """Build a single prompt text from the message history.

        Kimi ACP's session/prompt only accepts the current user turn, and
        session/new ignores preamble. We therefore fold the system message
        and any prior turns into one text block so the model sees the full
        context.
        """
        parts: list[str] = []

        if session.preamble:
            parts.append(f"System: {session.preamble}")

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Already handled as preamble, but include any additional system msgs
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        arguments = fn.get("arguments", "")
                        tc_parts.append(f"<tool_call>{name}({arguments})</tool_call>")
                    parts.append(f"Assistant: {''.join(tc_parts)}")
                else:
                    parts.append(f"Assistant: {content}")
            elif role == "tool_result":
                parts.append(f"Tool result: {content}")

        return "\n\n".join(parts)

    async def prompt(
        self,
        session: ACPSession,
        messages: list[dict[str, Any]],
        stream: bool = True,
        enable_native_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt using session/prompt and yield session/update events.

        Args:
            session: The ACP session
            messages: List of messages (already converted to ACP format)
            stream: Whether to stream responses
            enable_native_tools: If False, native ACP tool_call notifications are
                converted to text deltas instead of OpenAI tool_call events. This
                prevents Kimi from returning empty tool_calls when the client is
                doing prompt-based tool calling.

        Yields:
            ACP events as dictionaries
        """
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("ACP client not connected")

        prompt_text = self._build_prompt_text(session, messages)
        prompt_start = time.perf_counter()

        logger.info(
            "acp_prompt_prepared",
            request_id=self.request_id,
            session_id=session.session_id,
            message_count=len(messages),
            prompt_bytes=len(prompt_text.encode("utf-8")),
            stream=stream,
            enable_native_tools=enable_native_tools,
        )

        # Send session/prompt request - prompt is a list of content items
        await self._send_request(
            "session/prompt",
            {
                "sessionId": session.session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
            },
        )
        logger.info(
            "acp_prompt_sent",
            request_id=self.request_id,
            session_id=session.session_id,
            duration_ms=round((time.perf_counter() - prompt_start) * 1000, 2),
        )

        # Collect all streaming events
        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        buffer = ""
        events_seen = 0
        first_event_logged = False
        first_content_logged = False

        # Read events until done or timeout
        overall_start = time.perf_counter()
        first_readline_done = False
        while True:
            try:
                # Enforce overall ACP total timeout
                elapsed = time.perf_counter() - overall_start
                if elapsed > self.config.acp_total_timeout:
                    raise BridgeError(
                        code="backend_timeout",
                        message=f"ACP total request timed out after {self.config.acp_total_timeout}s",
                        phase="acp_total",
                        details={"timeout_seconds": self.config.acp_total_timeout, "elapsed_seconds": round(elapsed, 2)},
                    )

                # Phase-specific timeouts:
                # 1. First ACP event gets acp_first_event_timeout
                # 2. First content chunk gets acp_first_content_timeout
                # 3. Everything after first content gets idle_timeout
                if not first_readline_done:
                    read_timeout = self.config.acp_first_event_timeout
                elif not first_content_logged:
                    read_timeout = self.config.acp_first_content_timeout
                else:
                    read_timeout = self.config.idle_timeout
                first_readline_done = True

                # Read raw message (could be response or notification)
                line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=read_timeout,
                )

                if not line:
                    logger.warning("acp_stdout_eof")
                    break

                buffer += line.decode("utf-8", errors="replace")

                try:
                    event = json.loads(buffer)
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more data
                    continue

                buffer = ""  # Successfully parsed

                if self.config.log_acp_messages:
                    logger.debug("acp_event", message=event)

                events_seen += 1
                if not first_event_logged:
                    first_event_logged = True
                    update_type = ""
                    if event.get("method") == "session/update":
                        update_type = (
                            event.get("params", {}).get("update", {}).get("sessionUpdate", "")
                        )
                    logger.info(
                        "acp_first_event_received",
                        request_id=self.request_id,
                        session_id=session.session_id,
                        duration_ms=round((time.perf_counter() - prompt_start) * 1000, 2),
                        method=event.get("method", ""),
                        update_type=update_type,
                    )

                # Handle incoming requests from the ACP server (has both id and method)
                if "id" in event and "method" in event:
                    await self._handle_incoming_request(event)
                    continue

                # Handle notifications (no id, has method)
                if "id" not in event and "method" in event:
                    method = event.get("method", "")
                    params = event.get("params", {})

                    if method == "session/update":
                        update = params.get("update", {})
                        update_type = update.get("sessionUpdate", "")

                        if update_type == "agent_message_chunk":
                            chunk = update.get("content", {}).get("text", "")
                            if chunk and not first_content_logged:
                                first_content_logged = True
                                logger.info(
                                    "acp_first_content_chunk",
                                    request_id=self.request_id,
                                    session_id=session.session_id,
                                    duration_ms=round(
                                        (time.perf_counter() - prompt_start) * 1000, 2
                                    ),
                                    chunk_chars=len(chunk),
                                )
                            full_content += chunk
                            if stream:
                                yield {"type": "message.delta", "delta": chunk}

                        elif update_type == "agent_thought_chunk":
                            # Log reasoning but don't include in output
                            pass

                        elif update_type == "available_commands_update":
                            # Ignore available commands update
                            pass

                        elif update_type == "plan":
                            # Ignore plan updates
                            pass

                        elif update_type == "tool_call":
                            tool_call = {
                                "id": update.get("toolCallId", ""),
                                "name": update.get("toolName", ""),
                                "arguments": json.dumps(update.get("arguments", {})),
                            }
                            if enable_native_tools:
                                tool_calls.append(tool_call)
                                if stream:
                                    yield {
                                        "type": "tool_call.start",
                                        "tool_call": tool_call,
                                    }
                            else:
                                # Fold native tool call back into text for prompt-based usage
                                text = (
                                    f"<tool_call>{tool_call['name']}"
                                    f"({tool_call['arguments']})</tool_call>"
                                )
                                full_content += text
                                if stream:
                                    yield {"type": "message.delta", "delta": text}

                        elif update_type == "tool_call_update":
                            status = update.get("status", "")
                            if status in ("completed", "failed") and enable_native_tools and stream:
                                yield {
                                    "type": "tool_result",
                                    "result": update.get("result", {}),
                                }

                        elif update_type == "tool_result":
                            # Legacy handling – kept for compatibility
                            if enable_native_tools and stream:
                                yield {
                                    "type": "tool_result",
                                    "result": update.get("result", {}),
                                }

                        elif update_type == "done":
                            if stream:
                                yield {"type": "done"}
                            break

                # Handle response to our prompt request (has matching id)
                elif event.get("id") == self._message_id:
                    if event.get("error"):
                        raise BridgeError(
                            code="backend_process_failed",
                            message=f"Prompt failed: {event['error']}",
                            phase="acp_prompt",
                        )
                    # Got final result - signal completion
                    if stream:
                        yield {"type": "done"}
                    break

                # Handle other responses
                elif "id" in event:
                    # Response to some other request, skip
                    pass

            except asyncio.TimeoutError:
                if not first_event_logged:
                    phase = "acp_first_event"
                    code = "acp_first_event_timeout"
                elif not first_content_logged:
                    phase = "acp_first_content"
                    code = "acp_first_content_timeout"
                else:
                    phase = "idle"
                    code = "backend_timeout"
                logger.error(
                    "acp_read_timeout",
                    request_id=self.request_id,
                    phase=phase,
                    code=code,
                    elapsed_seconds=round(time.perf_counter() - overall_start, 2),
                )
                yield {"type": "error", "error": {"message": f"ACP {phase} timeout", "code": code}}
                break
            except json.JSONDecodeError as e:
                logger.error("acp_json_error", error=str(e))
                buffer = ""
                continue
            except Exception as e:
                logger.error("acp_read_error", error=str(e))
                yield {"type": "error", "error": {"message": str(e)}}
                break

        logger.info(
            "acp_prompt_finished",
            request_id=self.request_id,
            session_id=session.session_id,
            duration_ms=round((time.perf_counter() - prompt_start) * 1000, 2),
            events_seen=events_seen,
            content_chars=len(full_content),
            completion_bytes=len(full_content.encode("utf-8")),
            tool_calls_count=len(tool_calls),
        )

        # Final yield for non-streaming
        if not stream:
            yield {
                "type": "complete",
                "content": full_content,
                "tool_calls": tool_calls,
            }

    async def _handle_incoming_request(self, event: dict[str, Any]) -> None:
        """Handle JSON-RPC requests sent from the ACP server to us."""
        method = event.get("method", "")
        request_id = event.get("id")

        logger.debug("acp_incoming_request", method=method, request_id=request_id)

        if method == "session/request_permission":
            # Auto-approve all permission requests to prevent deadlocks.
            await self._send_response(
                request_id,
                {
                    "outcome": {
                        "outcome": "selected",
                        "option_id": "approve",
                    }
                },
            )
            logger.debug("auto_approved_permission_request", request_id=request_id)
            return

        if method in (
            "terminal/create",
            "terminal/output",
            "terminal/wait_for_exit",
            "terminal/kill",
            "terminal/release",
            "fs/read_text_file",
            "fs/write_text_file",
        ):
            logger.warning("acp_method_not_supported", method=method)
            await self._send_response(
                request_id,
                error={
                    "code": -32601,
                    "message": f"Method {method} is not supported by the bridge",
                },
            )
            return

        # Unknown method
        logger.warning("acp_unknown_method", method=method)
        await self._send_response(
            request_id,
            error={
                "code": -32601,
                "message": f"Method {method} not found",
            },
        )

    async def _send_response(
        self,
        request_id: Any,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC response."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Not connected")

        response: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
        }
        if error is not None:
            response["error"] = error
        else:
            response["result"] = result or {}

        data = json.dumps(response) + "\n"

        if self.config.log_acp_messages:
            logger.debug("acp_send_response", response=response)

        self.process.stdin.write(data.encode("utf-8"))
        await self.process.stdin.drain()

    async def _stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Stream events from ACP."""
        if self.process is None or self.process.stdout is None:
            return

        buffer = ""

        while True:
            try:
                # Read line by line
                line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.config.session_timeout,
                )

                if not line:
                    # EOF
                    logger.warning("acp_stdout_eof")
                    break

                line_str = line.decode("utf-8", errors="replace")

                if self.config.log_acp_messages:
                    logger.debug("acp_raw_output", line=line_str.strip())

                buffer += line_str

                # Try to parse complete JSON objects
                while buffer:
                    try:
                        # Find the first complete JSON object
                        event = json.loads(buffer)
                        buffer = ""  # Successfully parsed

                        if self.config.log_acp_messages:
                            logger.debug("acp_event", event=event)

                        yield event

                        # Check for completion
                        if event.get("type") == "done":
                            return

                        if event.get("type") == "error":
                            logger.error("acp_error", error=event.get("error"))
                            return

                    except json.JSONDecodeError:
                        # Incomplete JSON, wait for more data
                        break

            except asyncio.TimeoutError:
                logger.error("acp_read_timeout")
                yield {"type": "error", "error": {"message": "Session timeout"}}
                return
            except Exception as e:
                logger.error("acp_read_error", error=str(e))
                yield {"type": "error", "error": {"message": str(e)}}
                return

    async def _send_request(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC request."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Not connected")

        self._message_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._message_id,
            "method": method,
            "params": params,
        }

        data = json.dumps(request) + "\n"

        if self.config.log_acp_messages:
            logger.debug("acp_send", request=request)

        self.process.stdin.write(data.encode("utf-8"))
        await self.process.stdin.drain()

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Not connected")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        data = json.dumps(notification) + "\n"

        if self.config.log_acp_messages:
            logger.debug("acp_send_notification", notification=notification)

        self.process.stdin.write(data.encode("utf-8"))
        await self.process.stdin.drain()

    async def _read_response(self, timeout: float | None = None) -> dict[str, Any]:
        """Read a single JSON-RPC response."""
        if self.process is None or self.process.stdout is None:
            raise BridgeError(
                code="backend_unavailable",
                message="ACP client not connected",
                phase="acp_read",
            )

        buffer = ""
        while True:
            try:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=timeout or self.config.session_timeout,
                )
            except asyncio.TimeoutError as e:
                raise BridgeError(
                    code="backend_timeout",
                    message="Timeout reading ACP response",
                    phase="acp_read",
                    details={"timeout_seconds": timeout or self.config.session_timeout},
                ) from e
            if not line:
                raise BridgeError(
                    code="backend_empty_response",
                    message="EOF while reading ACP response",
                    phase="acp_read",
                )

            buffer += line.decode("utf-8", errors="replace")

            try:
                return json.loads(buffer)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue  # Incomplete, read more

    async def close(self) -> None:
        """Clean up resources."""
        if self.process is not None:
            logger.info("closing_acp_client", request_id=self.request_id, pid=self.process.pid)

            try:
                # Try graceful shutdown
                if self.process.stdin is not None:
                    self.process.stdin.close()
                    await self.process.stdin.wait_closed()
            except Exception as e:
                logger.debug("error_closing_stdin", error=str(e))

            # Terminate process
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("kimi_process_kill_timeout")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.debug("error_terminating_process", error=str(e))

            self.process = None
            self._session = None

    async def __aenter__(self) -> ACPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
