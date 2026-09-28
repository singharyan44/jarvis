#!/usr/bin/env python3
"""
Entry point for running JARVIS as a module: `python -m jarvis`

This maintains backward compatibility with the original `python jarvis.py` usage.
"""

from __future__ import annotations

import asyncio
import httpx

from core import main
from core.config import HONCHO_API_URL, HONCHO_ENABLED
from core.logging import log


async def check_honcho() -> bool:
    """Check if Honcho memory server is reachable."""
    if not HONCHO_ENABLED:
        return False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{HONCHO_API_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


def check_honcho_sync() -> bool:
    """Synchronous Honcho check for startup message."""
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(check_honcho())
    except Exception:
        return False


if __name__ == "__main__":
    # Check Honcho availability before starting
    if HONCHO_ENABLED:
        if check_honcho_sync():
            log.info("Honcho memory server detected at %s", HONCHO_API_URL)
        else:
            log.warning(
                "Honcho memory server not reachable at %s. "
                "Memory features will be limited. "
                "To enable: start PostgreSQL + Redis, configure honcho/config.toml, "
                "then run: cd honcho && uv run fastapi dev src/main.py",
                HONCHO_API_URL,
            )
    main()