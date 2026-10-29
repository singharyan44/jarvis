#!/usr/bin/env python3
"""Thread-safe event bus between the core orchestrator and the UI layer.

Direction of flow: core → UI only. The UI is a presentation layer over what
JARVIS perceived and decided (player knowledge); it never mutates core state
directly. The one control surface it exposes — resolving a pending safety
confirmation — goes through an explicit orchestrator method, never through
shared mutable state.
"""

from __future__ import annotations

import queue
import time


class EventBus:
    """Bounded FIFO of UI events with drop-oldest behavior.

    Producers (core subsystem threads) never block on the UI; if the UI is
    slow, the oldest events are shed, matching a live-monitor contract rather
    than a durability contract.
    """

    def __init__(self, maxsize: int = 512):
        self._q: queue.Queue[dict] = queue.Queue(maxsize=maxsize)

    def publish(self, kind: str, **fields) -> None:
        event = {"ts": time.time(), "kind": kind, **fields}
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(event)
            except queue.Full:
                pass

    def drain(self) -> list[dict]:
        """Return all queued events, emptying the bus."""
        out: list[dict] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out
