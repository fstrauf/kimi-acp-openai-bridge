"""Direct Kimi CLI client using print mode behind the OpenAI bridge."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import structlog

from kimi_acp_bridge.config import BridgeConfig

logger = structlog.get_logger()


@dataclass
class DirectResult:
    """Result from a direct Kimi print-mode invocation."""

    content: str
    prompt_text: str
    duration_ms: float


class DirectClient:
    """Run Kimi through print mode instead of ACP."""

    def __init__(self, config: BridgeConfig, request_id: str | None = None):
        self.config = config
        self.request_id = request_id

    def _build_prompt_text(self, preamble: str | None, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []

        if preamble:
            parts.append(f"System: {preamble}")

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool_result":
                parts.append(f"Tool result: {content}")
            else:
                parts.append(str(content))

        return "\n\n".join(parts)

    async def prompt(
        self,
        preamble: str | None,
        messages: list[dict[str, Any]],
        cwd: str | None = None,
    ) -> DirectResult:
        """Run a prompt through `kimi --print` and return the final stdout."""
        prompt_text = self._build_prompt_text(preamble, messages)
        session_id = f"bridge-{self.request_id or int(time.time() * 1000)}"
        work_dir = cwd or os.getcwd()
        command = [
            self.config.kimi_binary,
            "--print",
            "--no-thinking",
            "--output-format",
            "text",
            "--final-message-only",
            "--session",
            session_id,
            "--work-dir",
            work_dir,
            "-p",
            prompt_text,
        ]

        start = time.perf_counter()
        logger.info(
            "direct_prompt_prepared",
            request_id=self.request_id,
            prompt_bytes=len(prompt_text.encode("utf-8")),
            cwd=work_dir,
        )
        logger.info(
            "spawning_kimi_direct_process",
            request_id=self.request_id,
            binary=self.config.kimi_binary,
            args=command[1:-1] + ["<prompt>"],
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Kimi CLI not found at '{self.config.kimi_binary}'. "
                "Please ensure Kimi CLI is installed and in PATH."
            ) from e

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.session_timeout,
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Kimi direct process timed out after {self.config.session_timeout} seconds"
            ) from e

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        content = stdout.decode("utf-8", errors="replace").strip()
        error_text = stderr.decode("utf-8", errors="replace").strip()

        logger.info(
            "kimi_direct_process_finished",
            request_id=self.request_id,
            exit_code=proc.returncode,
            duration_ms=duration_ms,
            stdout_chars=len(content),
            stderr_chars=len(error_text),
        )

        if proc.returncode != 0 and not content:
            raise RuntimeError(error_text or f"Kimi direct exited with code {proc.returncode}")

        return DirectResult(
            content=content,
            prompt_text=prompt_text,
            duration_ms=duration_ms,
        )
