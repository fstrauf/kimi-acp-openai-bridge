"""ACP client for communicating with Kimi CLI using official ACP protocol."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from kimi_acp_bridge.config import BridgeConfig

logger = structlog.get_logger()


@dataclass
class ACPSession:
    """Represents an ACP session."""

    session_id: str
    preamble: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


class ACPClient:
    """Client for communicating with Kimi ACP process using JSON-RPC over stdio."""

    def __init__(self, config: BridgeConfig):
        self.config = config
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
                binary=self.config.kimi_binary,
                args=self.config.kimi_args,
            )

            try:
                self.process = await asyncio.create_subprocess_exec(
                    self.config.kimi_binary,
                    *self.config.kimi_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE if self.config.log_acp_messages else None,
                    env={
                        **os.environ,
                        "KIMI_AUTO_APPROVE": "true" if self.config.auto_approve_tools else "false",
                    },
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
                response = await self._read_response()
                if response.get("result"):
                    logger.info("acp_initialized")
                else:
                    error = response.get("error", {})
                    raise RuntimeError(f"ACP initialization failed: {error}")

                # Send initialized notification
                await self._send_notification("initialized", {})

            except FileNotFoundError as e:
                logger.error("kimi_binary_not_found", binary=self.config.kimi_binary)
                raise RuntimeError(
                    f"Kimi CLI not found at '{self.config.kimi_binary}'. "
                    "Please ensure Kimi CLI is installed and in PATH."
                ) from e
            except Exception as e:
                logger.error("failed_to_spawn_kimi", error=str(e))
                await self.close()
                raise

    async def create_session(
        self,
        preamble: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        cwd: str | None = None,
    ) -> ACPSession:
        """Create a new ACP session using session/new method."""
        if self.process is None:
            await self.connect()

        # Use ACP session/new method
        import os

        params: dict[str, Any] = {
            "cwd": cwd or os.getcwd(),
            "mcpServers": [],  # No MCP servers for now
        }
        if preamble:
            params["preamble"] = preamble

        await self._send_request("session/new", params)

        response = await self._read_response()
        result = response.get("result", {})

        if "error" in response:
            raise RuntimeError(f"Failed to create session: {response['error']}")

        session_id = result.get("sessionId", str(uuid.uuid4()))
        self._session = ACPSession(
            session_id=session_id,
            preamble=preamble,
            tools=tools or [],
        )

        logger.debug("session_created", session_id=session_id)
        return self._session

    async def prompt(
        self,
        session: ACPSession,
        messages: list[dict[str, Any]],
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt using session/prompt and yield session/update events.

        Args:
            session: The ACP session
            messages: List of messages (already converted to ACP format)
            stream: Whether to stream responses

        Yields:
            ACP events as dictionaries
        """
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("ACP client not connected")

        # Build prompt text from messages
        prompt_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break

        # Send session/prompt request - prompt is a list of content items
        await self._send_request(
            "session/prompt",
            {
                "sessionId": session.session_id,
                "prompt": [{"type": "text", "text": prompt_text}],
            },
        )

        # Collect all streaming events
        full_content = ""
        tool_calls: list[dict[str, Any]] = []

        # Read events until done or timeout
        while True:
            try:
                # Read raw message (could be response or notification)
                line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.config.session_timeout,
                )

                if not line:
                    logger.warning("acp_stdout_eof")
                    break

                event = json.loads(line.decode("utf-8", errors="replace"))

                if self.config.log_acp_messages:
                    logger.debug("acp_event", message=event)

                # Handle notifications (no id, has method)
                if "id" not in event and "method" in event:
                    method = event.get("method", "")
                    params = event.get("params", {})

                    if method == "session/update":
                        update = params.get("update", {})
                        update_type = update.get("sessionUpdate", "")

                        if update_type == "agent_message_chunk":
                            chunk = update.get("content", {}).get("text", "")
                            full_content += chunk
                            if stream:
                                yield {"type": "message.delta", "delta": chunk}

                        elif update_type == "agent_thought_chunk":
                            # Log reasoning but don't include in output
                            pass

                        elif update_type == "available_commands_update":
                            # Ignore available commands update
                            pass

                        elif update_type == "tool_call":
                            tool_call = {
                                "id": update.get("toolCallId", ""),
                                "name": update.get("toolName", ""),
                                "arguments": json.dumps(update.get("arguments", {})),
                            }
                            tool_calls.append(tool_call)
                            if stream:
                                yield {
                                    "type": "tool_call.start",
                                    "tool_call": tool_call,
                                }

                        elif update_type == "tool_result":
                            if stream:
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
                        raise RuntimeError(f"Prompt failed: {event['error']}")
                    # Continue reading for notifications

                # Handle other responses
                elif "id" in event:
                    # Response to some other request, skip
                    pass

            except asyncio.TimeoutError:
                logger.error("acp_read_timeout")
                yield {"type": "error", "error": {"message": "Session timeout"}}
                break
            except json.JSONDecodeError as e:
                logger.error("acp_json_error", error=str(e))
                continue
            except Exception as e:
                logger.error("acp_read_error", error=str(e))
                yield {"type": "error", "error": {"message": str(e)}}
                break

        # Final yield for non-streaming
        if not stream:
            yield {
                "type": "complete",
                "content": full_content,
                "tool_calls": tool_calls,
            }

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

    async def _read_response(self) -> dict[str, Any]:
        """Read a single JSON-RPC response."""
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Not connected")

        buffer = ""
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("EOF while reading response")

            buffer += line.decode("utf-8", errors="replace")

            try:
                return json.loads(buffer)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue  # Incomplete, read more

    async def close(self) -> None:
        """Clean up resources."""
        if self.process is not None:
            logger.info("closing_acp_client")

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
