# Kimi ACP ↔ OpenAI Bridge

A local HTTP server that translates between the **OpenAI-compatible API format** (used by Rig, PydanticAI, and other LLM frameworks) and the **Agent Client Protocol (ACP)** used by Kimi Code CLI.

## Features

- ✅ OpenAI-compatible chat completions API (`/v1/chat/completions`)
- ✅ Streaming responses (Server-Sent Events)
- ✅ Tool/function calling (bidirectional)
- ✅ List models endpoint (`/v1/models`)
- ✅ Stateless ephemeral sessions (one HTTP request = one ACP session)
- ✅ Zero external dependencies beyond local Kimi CLI installation
- ✅ Simple single-command startup

## Installation

```bash
# Install from source
pip install git+https://github.com/yourname/kimi-acp-openai-bridge.git

# Or clone and install
git clone https://github.com/yourname/kimi-acp-openai-bridge.git
cd kimi-acp-openai-bridge
pip install -e .
```

## Requirements

- Python 3.10+
- [Kimi Code CLI](https://moonshotai.github.io/kimi-cli/) installed and available in PATH

## Quick Start

```bash
# Start the bridge server
kimi-acp-bridge

# Or with custom port
kimi-acp-bridge --port 9000
```

The server will start on `http://127.0.0.1:8080` by default.

## Usage Examples

### With Rig (Rust)

```rust
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Point to local bridge instead of OpenAI
    let client = openai::Client::from_url(
        "http://localhost:8080/v1",
        "dummy-api-key"  // Bridge ignores this
    );
    
    let agent = client
        .agent("kimi-k2.5")
        .preamble("You are a helpful coding assistant.")
        .build();
    
    let response = agent.prompt("Explain this codebase").await?;
    println!("{}", response);
    
    Ok(())
}
```

### With PydanticAI (Python)

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    "kimi-k2.5",
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # Bridge ignores this
)

agent = Agent(model)

async def main():
    result = await agent.run("Hello, Kimi!")
    print(result.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### With cURL

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIMI_BRIDGE_HOST` | `127.0.0.1` | Server bind address |
| `KIMI_BRIDGE_PORT` | `8080` | Server port |
| `KIMI_BINARY` | `kimi` | Path to kimi CLI binary |
| `KIMI_BRIDGE_SESSION_MODE` | `ephemeral` | Session persistence mode |
| `KIMI_BRIDGE_AUTO_APPROVE` | `true` | Auto-approve tool calls |
| `KIMI_BRIDGE_LOG_LEVEL` | `INFO` | Logging level |
| `KIMI_BRIDGE_LOG_ACP` | `false` | Log all ACP messages |

### Config File

Create `~/.config/kimi-acp-bridge/config.yaml`:

```yaml
server:
  host: 127.0.0.1
  port: 8080

kimi:
  binary: /usr/local/bin/kimi
  args: ["acp"]
  
session:
  mode: ephemeral
  timeout: 300
  
features:
  enable_tools: true
  enable_streaming: true
  auto_approve_tools: true
  
logging:
  level: INFO
  log_acp_messages: false
```

## API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "kimi-k2.5",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "tools": [...]
}
```

### `GET /v1/models`

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "kimi-k2.5",
      "object": "model",
      "created": 1677610602,
      "owned_by": "moonshot-ai"
    }
  ]
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "kimi_available": true,
  "version": "0.1.0"
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP CLIENT (Rig, PydanticAI)                │
│                        OpenAI API Format                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/1.1
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KIMI ACP-OPENAI BRIDGE                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ FastAPI  │ │ OpenAI → │ │ Session  │ │ ACP →    │           │
│  │ Server   │ │ ACP      │ │ Manager  │ │ OpenAI   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/WebSocket to Kimi ACP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         KIMI CODE CLI                            │
│                      (kimi acp)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone the repo
git clone https://github.com/yourname/kimi-acp-openai-bridge.git
cd kimi-acp-openai-bridge

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov

# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy src/kimi_acp_bridge
```

## License

MIT
