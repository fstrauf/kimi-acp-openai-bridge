"""Entry point for Kimi ACP Bridge."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from kimi_acp_bridge.config import BridgeConfig
from kimi_acp_bridge.server import create_app


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="kimi-acp-bridge",
        description="Kimi ACP ↔ OpenAI Bridge - HTTP server for OpenAI API compatibility",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Server bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--kimi-binary",
        type=str,
        help="Path to kimi CLI binary (default: kimi)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    config = BridgeConfig.load(config_path)

    # Override with CLI args
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.kimi_binary:
        config.kimi_binary = args.kimi_binary
    if args.log_level:
        config.log_level = args.log_level

    # Create and run the app
    app = create_app(config)

    print(f"Starting Kimi ACP Bridge on http://{config.host}:{config.port}")
    print(f"Using Kimi binary: {config.kimi_binary}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=config.log_level == "DEBUG",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
