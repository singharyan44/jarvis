#!/usr/bin/env python3
"""Unit tests for the core→UI event bus (core.events.EventBus)."""

from core.events import EventBus


def test_publish_drain_order():
    bus = EventBus()
    bus.publish("transcript", text="a")
    bus.publish("speak", text="b")
    events = bus.drain()
    assert [e["kind"] for e in events] == ["transcript", "speak"]
    assert events[0]["text"] == "a"
    assert events[1]["text"] == "b"


def test_drain_empties_bus():
    bus = EventBus()
    bus.publish("info", text="x")
    assert len(bus.drain()) == 1
    assert bus.drain() == []


def test_drop_oldest_when_full():
    bus = EventBus(maxsize=2)
    for i in range(5):
        bus.publish("tick", i=i)
    events = bus.drain()
    assert len(events) == 2
    # Oldest events were shed; only the last two survive.
    assert [e["i"] for e in events] == [3, 4]


def test_events_carry_timestamp_and_fields():
    bus = EventBus()
    bus.publish("state", state="idle")
    ev = bus.drain()[0]
    assert isinstance(ev["ts"], float)
    assert ev["state"] == "idle"
    assert ev["kind"] == "state"
