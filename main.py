#!/usr/bin/env python3
"""
LM Arena - Main Entry Point

Command-line interface for starting and managing LM Arena.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn

from lm_arena.config.settings import load_config, get_config
from lm_arena.api.main import app


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LM Arena - Multi-Model AI Agent Framework")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path",
        default=None
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind to",
        default=None
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to bind to",
        default=None
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes",
        default=None
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
        default=None
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.host:
        config.api.host = args.host
    if args.port:
        config.api.port = args.port
    if args.workers:
        config.api.workers = args.workers
    if args.reload:
        config.api.reload = True
    if args.debug:
        config.debug = True
        config.logging.level = "DEBUG"
    if args.log_level:
        config.logging.level = args.log_level.upper()

    # Create necessary directories
    config.create_directories()

    print(f"🚀 Starting LM Arena on {config.api.host}:{config.api.port}")
    print(f"📊 Configuration: {config.environment.value} environment")
    print(f"📝 Logs: {config.logs_dir}")
    print(f"📚 Prompts: {config.prompts_dir}")
    print(f"🤖 Models: {len(get_config().models.default_model.split(','))} configured")
    print()

    # Start the server
    try:
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            workers=config.api.workers,
            reload=config.api.reload,
            log_level=config.logging.level.lower(),
            access_log=config.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down LM Arena...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()