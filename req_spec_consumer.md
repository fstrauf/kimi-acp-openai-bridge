# Kimi Bridge Provider Specification

Date: 2026-04-30

## Purpose

This document specifies what PageSeeds needs from the Kimi model provider bridge when used through the Rig-based provider layer.

The goal is to make Kimi feel like a boring, reliable, OpenAI-compatible provider from inside PageSeeds while hiding the differences between Kimi direct CLI mode, Kimi ACP mode, and any future provider implementation. This spec is intended for implementation on the Kimi bridge side of the project.

No PageSeeds application code changes are proposed here.

## Current PageSeeds Architecture

PageSeeds agentic workflow steps route through `engine::agent::run_agent`, which now delegates into the Rig provider layer when a compatible backend is available. The legacy direct CLI path remains available through `agent-wrapper`.

The current relevant paths are:

- `src-tauri/src/engine/workflows/handlers.rs`: builds workflow prompts and chooses `direct` or `acp` preference for Kimi bridge calls.
- `src-tauri/src/engine/agent.rs`: maintains the legacy synchronous `run_agent` interface and delegates to the Rig provider layer.
- `src-tauri/src/rig/provider.rs`: resolves `kimi`, `claude`, `openai`, and `ollama` into concrete backends.
- `src-tauri/src/rig/compat/kimi.rs`: custom Kimi adapter for strict OpenAI-compatible wire format.
- `src-tauri/src/rig/extraction.rs`: typed structured extraction via Rig, with a custom Kimi bridge path.

PageSeeds currently has two distinct Kimi workloads:

1. Fast stateless calls: JSON extraction, analysis, recommendation drafting, and short prose generation.
2. Slower project-aware calls: content writing and file-aware tasks that need ACP-style context and repository access.

The bridge must support both without requiring workflow code to know Kimi internals.

## Current Pain Points

### Kimi Is Not Yet A Fully Ordinary Rig/OpenAI Provider

The bridge exposes an OpenAI-compatible `/v1/chat/completions` endpoint, but PageSeeds still carries a custom compatibility adapter because Kimi rejects some request shapes that Rig's standard OpenAI provider can emit.

The most important example is message content shape. Rig can serialize system message content as OpenAI content-part arrays, while Kimi currently expects plain string content in some paths. PageSeeds works around this by forcing all Kimi request message content to strings.

The ideal bridge should accept normal Rig/OpenAI request shapes and normalize them internally before calling Kimi.

### Prompt Size Can Trigger ACP Stalls Or Timeouts

Recent PageSeeds investigations found failures around large ACP prompts:

- A schema-renderer task sent roughly 75 KB of prompt content and failed through the bridge with a 500/timeout.
- A CTR audit path sent roughly 46 KB of prompt content; ACP accepted the prompt but never emitted first content.

The bridge should not allow large prompts to become opaque hangs. It should enforce request-size limits, report prompt bytes and estimated tokens, and return structured errors such as `413 prompt_too_large` or `504 backend_timeout`.

### Empty ACP Output Must Not Look Successful

When ACP accepts a prompt but produces no assistant content, PageSeeds cannot safely treat that as a valid completion. Empty backend output should be a structured non-200 error unless Kimi explicitly completed with an intentional empty message and a reliable finish reason.

For PageSeeds workflows, the safe default is:

- Empty ACP output after accepted prompt: `503 backend_empty_response`.
- No first content before timeout: `504 backend_timeout`.

### Kimi CLI Step Limits Need Structured Errors

The Kimi CLI can return text like `Max number of steps reached: 100`. PageSeeds should not receive that as normal assistant content when a workflow expects JSON or a patch. The bridge should map it to a structured error such as `backend_step_limit`.

### Fallback Semantics Are Ambiguous

PageSeeds currently has settings for Kimi backend mode: `auto`, `bridge`, and `direct`. Comments and behavior have drifted around whether `auto` means "try bridge, then fallback to direct" or "use bridge if healthy, otherwise fail clearly."

The bridge should expose enough capability information for PageSeeds to make this explicit. The bridge itself should also define clear routing rules for `direct`, `acp`, and `auto`.

## Provider Contract

The bridge should present a strict OpenAI-compatible interface suitable for Rig.

Required endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Optional but recommended:

- Streaming support on `POST /v1/chat/completions` with `stream: true`.

## Request Compatibility Requirements

The bridge must accept standard OpenAI chat completion request shapes, including those emitted by Rig.

It must accept:

- `messages[].content` as plain strings.
- `messages[].content` as OpenAI content-part arrays.
- `system`, `user`, `assistant`, and `tool` roles.
- `tools` as OpenAI function tools.
- `tool_choice` as `"auto"`, `"none"`, `"required"`, or a forced function object.
- `response_format: { "type": "json_object" }`.
- `temperature` and `max_tokens`, even if advisory.
- Model names that map cleanly to Kimi models.

Recommended future support:

- `response_format: { "type": "json_schema", ... }`.
- Multiple choices, even if the bridge initially only supports `n = 1`.

The bridge should normalize all accepted request shapes before calling Kimi. PageSeeds should not need a custom serializer solely for Kimi.

## Response Compatibility Requirements

Successful responses must use OpenAI chat completion shape:

```json
{
  "id": "chatcmpl_...",
  "object": "chat.completion",
  "created": 1777507200,
  "model": "kimi-k2.5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

For tool calls, the bridge must return OpenAI-compatible `tool_calls`:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "submit",
              "arguments": "{\"field\":\"value\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

The response should include:

- `id` for every request.
- `usage` whenever available.
- Estimated usage if exact Kimi token counts are unavailable.
- `finish_reason` with a stable OpenAI-compatible value.
- `x-request-id` response header for log correlation.

## Backend Modes

The bridge should support three backend modes.

### `direct`

Direct mode is the fast, stateless path. It should be equivalent in spirit to the old PageSeeds path:

```bash
kimi --print --output-format text --final-message-only --no-thinking --work-dir <project>
```

Direct mode is useful for:

- Short analysis.
- JSON extraction without native tool calls.
- Emergency fallback and A/B testing.
- Fast tasks that do not need project file access through ACP.

### `acp`

ACP mode is the project-aware path. It is useful for:

- Content writing.
- File-aware workflows.
- Tasks that require Kimi's persistent/session/project context.
- Native tool-call style structured output when direct mode cannot support it.

### `auto`

Auto mode must be deterministic and documented. It may inspect request requirements and bridge capabilities, but it must not silently choose a backend that cannot satisfy the request contract.

Recommended routing rules:

- If file/project context is required, choose `acp`.
- If native tool calls are required and only ACP supports them, choose `acp`.
- If `response_format: json_object` is requested and direct can satisfy it reliably, direct is allowed.
- If the selected backend cannot support the request, return a clear capability error before launching Kimi.
- Do not silently downgrade structured extraction into unstructured prose.

Routing inputs:

- `X-Kimi-Backend: direct | acp | auto`
- Optional request metadata for PageSeeds task type and step name.
- Optional project/work directory metadata when ACP or direct CLI execution needs a project root.

## Capability Discovery

`GET /health` should return bridge and backend capabilities, not only a boolean.

Recommended shape:

```json
{
  "status": "ok",
  "kimi_available": true,
  "bridge_version": "x.y.z",
  "kimi_cli_version": "1.39.0",
  "models": ["kimi-k2.5"],
  "backends": {
    "direct": {
      "available": true,
      "tool_calls": false,
      "json_mode": true,
      "file_io": false
    },
    "acp": {
      "available": true,
      "tool_calls": true,
      "json_mode": true,
      "file_io": true
    }
  },
  "limits": {
    "max_prompt_bytes_direct": 20000,
    "max_prompt_bytes_acp": 20000,
    "max_concurrent_requests": 2
  }
}
```

PageSeeds should be able to use this response to decide whether a workflow can run with Kimi bridge, should fall back to direct mode, or should fail with a clear user-facing setup message.

## Timeout Requirements

The bridge should use phase-specific timeouts. One opaque global timeout makes diagnosis too hard.

Recommended defaults:

| Phase | Default |
| --- | ---: |
| Health request | 2 seconds |
| Process spawn | 10 seconds |
| ACP initialize | 30 seconds |
| ACP session creation | 30 seconds |
| Time to first ACP event | 60 seconds |
| Time to first content/tool call | 90 seconds |
| Direct total request | 120 seconds |
| ACP total request | 300 seconds |
| Idle after first content | 60 seconds |

On timeout, the bridge must:

1. Kill or close the Kimi subprocess/session.
2. Avoid zombie processes.
3. Return a structured non-200 error.
4. Include request id, backend, phase, elapsed time, prompt bytes, and model.

## Prompt Size Requirements

The bridge should enforce explicit prompt size limits per backend.

Recommended initial limits:

| Backend | Warning Threshold | Hard Limit |
| --- | ---: | ---: |
| direct | 15 KB | 20 KB |
| acp | 15 KB | 20 KB |

The exact values can be tuned with live evidence, but the behavior should be fixed:

- If under warning threshold, run normally.
- If over warning threshold, log a warning with `prompt_bytes` and estimated tokens.
- If over hard limit, return `413 prompt_too_large` before launching Kimi.

Large prompt failures should be explicit. PageSeeds can then batch requests deterministically rather than waiting for ACP stalls.

## Error Contract

All errors should use one JSON shape:

```json
{
  "error": {
    "type": "backend_timeout",
    "code": "acp_first_content_timeout",
    "message": "Kimi ACP did not produce content within 90s",
    "retryable": true,
    "backend": "acp",
    "request_id": "req_...",
    "phase": "first_content",
    "details": {
      "prompt_bytes": 46231,
      "elapsed_ms": 90000,
      "model": "kimi-k2.5",
      "kimi_cli_version": "1.39.0"
    }
  }
}
```

Recommended status codes:

| Status | Use |
| --- | --- |
| 400 | Invalid request shape |
| 401/403 | Auth or permission failure if auth is enabled |
| 413 | Prompt too large |
| 422 | Valid OpenAI request, unsupported Kimi capability |
| 429 | Rate limit or concurrency limit |
| 503 | Kimi unavailable, subprocess failed, empty backend response |
| 504 | Timeout |

Avoid generic `500` for known bridge or Kimi states.

Recommended error codes:

- `bridge_unhealthy`
- `kimi_not_found`
- `backend_unavailable`
- `tools_not_supported`
- `json_mode_not_supported`
- `prompt_too_large`
- `acp_initialize_timeout`
- `acp_session_timeout`
- `acp_first_content_timeout`
- `backend_empty_response`
- `backend_step_limit`
- `backend_process_failed`
- `response_parse_error`

## Observability Requirements

Every request should have a request id visible in:

- Response header: `x-request-id`.
- Success response body: `id`.
- Error response body: `error.request_id`.
- All bridge logs related to the request.

Bridge logs should include:

- Request id.
- Backend selected: `direct` or `acp`.
- Requested backend preference.
- Model.
- Sanitized work directory.
- Prompt bytes.
- Estimated prompt tokens.
- Tool count.
- Response format.
- Kimi CLI version.
- Subprocess pid.
- Completion bytes.
- Usage tokens or estimates.
- Exit status and stderr summary on subprocess failure.

Phase timings should include:

- Request received.
- Request translated.
- Kimi process spawned.
- ACP initialized.
- Session created.
- Prompt sent.
- First ACP event.
- First content or first tool call.
- Completion finished.
- Response serialized.

This is the critical data PageSeeds needs to distinguish bridge latency, Kimi CLI latency, model latency, prompt-size stalls, and PageSeeds-side post-processing.

## Streaming Requirements

Streaming is useful for visibility and time-to-first-token diagnostics, but it should not be required for PageSeeds workflows.

Requirements:

- Non-streaming requests must remain fully supported.
- Streaming should emit OpenAI-compatible SSE chunks when `stream: true`.
- Streaming startup failures must return structured errors, not generic `500` responses.
- If ACP never emits first content, streaming will not solve the failure; the bridge still needs first-content timeouts.

## Security Requirements

The bridge should default to localhost-only binding.

If the bridge accepts a work directory or project root:

- Validate the path.
- Restrict file access to the project root.
- Do not allow arbitrary shell execution.
- Pass a controlled environment to Kimi.
- Log environment key names only, never secret values.

If auth is ever added, it should be optional for local development but explicit in health output.

## PageSeeds Workload Support Matrix

| Workload | Preferred Backend | Required Bridge Behavior |
| --- | --- | --- |
| Short prose or recommendation | direct | Fast stateless completion, clear timeout |
| JSON extraction | direct or acp | Valid JSON mode or tool-call extraction |
| Rig typed extraction | acp unless direct emulates tools | OpenAI-compatible tool calls or JSON fallback |
| Content writing | acp | Project-aware execution, file context support |
| CTR/content review analysis | direct or acp, depending on size | Prompt-size limits, structured errors, reliable JSON |
| Emergency fallback | direct | Old CLI-like behavior remains available |

## Acceptance Tests

The bridge should pass these tests before PageSeeds treats it as the preferred Kimi path.

1. Accept Rig/OpenAI array-form system content and normalize it.
2. Accept plain string message content.
3. Return normal assistant content for a tiny non-streaming prompt.
4. Return usage fields or estimates.
5. Return OpenAI-compatible tool calls for a `submit` function.
6. Support JSON mode with `response_format: { "type": "json_object" }`.
7. Route `X-Kimi-Backend: direct` to direct mode.
8. Route `X-Kimi-Backend: acp` to ACP mode.
9. Make `X-Kimi-Backend: auto` deterministic and documented.
10. Reject unsupported tools in direct mode with `422 tools_not_supported`, unless direct mode emulates them reliably.
11. Return `413 prompt_too_large` for prompts over the hard byte limit.
12. Return `504 acp_first_content_timeout` when ACP accepts a prompt but never emits content.
13. Return `503 backend_empty_response` when Kimi exits successfully but provides empty output.
14. Map `Max number of steps reached: 100` to `backend_step_limit`.
15. Kill subprocesses on timeout and prove no zombies remain.
16. Include a stable request id in headers, body, and logs.
17. Report backend capabilities from `/health`.
18. Support non-streaming and streaming without changing response semantics.

## Recommended Rollout

1. Implement capability-rich `/health` first.
2. Normalize all Rig/OpenAI message content shapes in the bridge.
3. Add structured error responses and remove generic `500` for known cases.
4. Add phase-specific timeouts and subprocess cleanup.
5. Add prompt-size limits and early `413` errors.
6. Add request-id phase telemetry.
7. Validate tool-call and JSON-mode behavior against PageSeeds typed extraction.
8. Keep direct CLI-style mode as an explicit fallback and A/B test path.

## Working Recommendation

Keep the Rig/bridge architecture.

The old direct Kimi CLI behavior should remain available as an emergency escape hatch, but the primary investment should be making the bridge a strict, observable OpenAI-compatible provider. The bridge should absorb Kimi-specific request quirks so PageSeeds can increasingly use standard Rig provider behavior without custom Kimi serialization paths.

The highest-value bridge-side work is:

- Accept Rig's native OpenAI message shapes.
- Formalize `direct`, `acp`, and `auto` routing.
- Add capability discovery.
- Return structured non-500 errors.
- Enforce prompt-size and phase timeouts.
- Emit request-id phase telemetry.
- Preserve direct CLI-style mode for fallback and comparison.

Once those are in place, PageSeeds will have a provider boundary that is much easier to reason about, debug, and eventually swap for another model API.