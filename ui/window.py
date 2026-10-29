#!/usr/bin/env python3
"""JARVIS popup window.

Presentation layer only (UNSEEN architecture): the window renders events the
core publishes — what JARVIS heard, what the safety gate decided, what it
spoke — plus a control surface for pending confirmations that resolves them
through the orchestrator's public method. It never reads or mutates core
state internals.

Layout:
- Header: state badge text
- Canvas: animated orb (breathes when idle, pulses fast when listening,
  orbits when thinking, ripples when speaking, strobes amber while a
  confirmation is pending)
- Streaming output box: timestamped lines (you / jarvis / gate / system)
- Confirmation panel: shows the pending command + safety explanation with
  Run / Cancel buttons
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import scrolledtext

from core.events import EventBus

POLL_MS = 60
ANIM_MS = 40

BG = "#0b0f14"
PANEL_BG = "#11161d"
FG = "#d7e2ec"
DIM = "#6b7a88"

COLOR_STATE = {
    "idle": "#5aa3c8",
    "listening": "#59d38a",
    "thinking": "#ffd166",
    "speaking": "#5aa3c8",
    "confirm": "#ff9d54",
    "error": "#ff6b6b",
}

COLOR_TAG = {
    "you": "#8fd6ff",
    "jarvis": "#ffd166",
    "gate": "#c792ea",
    "ok": "#59d38a",
    "err": "#ff6b6b",
    "sys": "#6b7a88",
}


class JarvisApp:
    def __init__(self):
        self._root = tk.Tk()
        self._root.title("JARVIS")
        self._root.geometry("430x640")
        self._root.configure(bg=BG)
        self._root.minsize(380, 540)

        self._orchestrator = None
        self._events: EventBus | None = None
        self._anim_phase = 0.0
        self._state = "idle"
        self._confirm_shown = False

        self._build()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(ANIM_MS, self._animate)
        self._root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        header = tk.Frame(self._root, bg=BG)
        header.pack(fill="x", padx=14, pady=(12, 2))

        tk.Label(
            header,
            text="JARVIS",
            font=("Consolas", 15, "bold"),
            fg="#e8f1f8",
            bg=BG,
        ).pack(side="left")

        self._state_label = tk.Label(
            header,
            text="idle",
            font=("Consolas", 11, "bold"),
            fg=COLOR_STATE["idle"],
            bg=BG,
        )
        self._state_label.pack(side="right")

        self._canvas = tk.Canvas(
            self._root, width=230, height=230, bg=BG, highlightthickness=0
        )
        self._canvas.pack(pady=(2, 0))

        self._box = scrolledtext.ScrolledText(
            self._root,
            wrap="word",
            bg=PANEL_BG,
            fg=FG,
            font=("Consolas", 10),
            relief="flat",
            padx=10,
            pady=8,
            state="disabled",
        )
        self._box.pack(fill="both", expand=True, padx=14, pady=(6, 6))

        for tag, color in COLOR_TAG.items():
            self._box.tag_configure(tag, foreground=color)
        self._box.tag_configure("date", foreground=DIM, font=("Consolas", 8))

        self._confirm_frame = tk.Frame(self._root, bg="#1c1207", highlightthickness=1, highlightbackground="#7a4a12")
        self._confirm_title = tk.Label(
            self._confirm_frame,
            text="",
            font=("Consolas", 9),
            fg="#ffce8a",
            bg="#1c1207",
            justify="left",
            wraplength=380,
        )
        self._confirm_title.pack(fill="x", padx=10, pady=(8, 2))
        btn_row = tk.Frame(self._confirm_frame, bg="#1c1207")
        btn_row.pack(fill="x", padx=10, pady=(2, 8))
        tk.Button(
            btn_row,
            text="Run it  (Enter)",
            command=self._approve,
            bg="#1f7a45",
            fg="#eafff2",
            activebackground="#27935a",
            activeforeground="#ffffff",
            relief="flat",
            font=("Consolas", 10, "bold"),
            padx=14,
        ).pack(side="left")
        tk.Button(
            btn_row,
            text="Cancel  (Esc)",
            command=self._deny,
            bg="#7a2424",
            fg="#ffecec",
            activebackground="#93302f",
            activeforeground="#ffffff",
            relief="flat",
            font=("Consolas", 10, "bold"),
            padx=14,
        ).pack(side="left", padx=8)

        self._append_line("sys", "UI attached. Toolcalls stream here; say 'hey jarvis' or 'jarvis' to activate.")

        self._root.bind("<Return>", self._on_return)
        self._root.bind("<Escape>", self._on_escape)

    # ------------------------------------------------------------------ #
    # Event dispatch
    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        if self._events is None:
            self._root.after(POLL_MS, self._poll)
            return
        for ev in self._events.drain():
            try:
                self._dispatch(ev)
            except Exception as e:  # UI must never die from a bad event
                self._append_line("err", f"ui dispatch error: {e}")
        self._root.after(POLL_MS, self._poll)

    def _dispatch(self, ev: dict) -> None:
        kind = ev.get("kind", "")
        if kind == "state":
            self._set_state(str(ev.get("state", "idle")))
        elif kind == "transcript":
            self._append_line("you", f"YOU ▸ {ev.get('text', '')}")
        elif kind == "speak":
            self._append_line("jarvis", f"JARVIS ▸ {ev.get('text', '')}")
        elif kind == "intent":
            params = ev.get("parameters") or {}
            self._append_line(
                "sys",
                f"intent ▸ {ev.get('tool', '?')}.{ev.get('action', '?')} "
                f"{params if params else ''} (conf {ev.get('confidence', 0):.2f})",
            )
        elif kind == "decision":
            d = str(ev.get("decision", "")).upper()
            tag = {"DENY": "err", "CONFIRM": "gate", "ALLOW": "ok"}.get(d, "sys")
            self._append_line(
                tag,
                f"gate ▸ {d} (conf {ev.get('confidence', 0):.2f}, risk {ev.get('risk', 0):.2f})",
            )
            if str(ev.get("explanation", "")):
                self._append_line("sys", f"      {ev.get('explanation')}")
        elif kind == "result":
            self._append_line(
                "ok" if ev.get("success") else "err",
                ("done ▸ " if ev.get("success") else "failed ▸ ") + str(ev.get("text", "")),
            )
        elif kind == "confirm_pending":
            self._show_confirm(ev.get("transcript", ""), ev.get("explanation", ""))
        elif kind == "confirm_result":
            self._hide_confirm()
            self._append_line("sys", f"confirm ▸ {ev.get('outcome', '')}")
        elif kind == "info":
            self._append_line("sys", str(ev.get("text", "")))
        elif kind == "error":
            self._append_line("err", f"error ▸ {ev.get('text', '')}")
            self._set_state("idle")

    # ------------------------------------------------------------------ #
    # State + animation
    # ------------------------------------------------------------------ #
    def _set_state(self, state: str) -> None:
        if state not in COLOR_STATE:
            state = "idle"
        self._state = state
        self._state_label.config(text=state, fg=COLOR_STATE[state])

    def _animate(self) -> None:
        canv = self._canvas
        canv.delete("all")
        w = int(canv["width"]) / 2
        h = int(canv["height"]) / 2
        self._anim_phase = (self._anim_phase + (ANIM_MS / 1000.0)) % 1.0
        phase = self._anim_phase
        state = self._state
        color = COLOR_STATE.get(state, COLOR_STATE["idle"])

        if state == "idle":
            base = 34.0
            r = base + 6.0 * (phase * 2 - 1 if phase < 0.5 else (1 - phase) * 2 - 1)
            canv.create_oval(w - r, h - r, w + r, h + r, fill=color, outline="")
        elif state == "listening":
            r = 30.0 + 14.0 * phase
            canv.create_oval(w - r, h - r, w + r, h + r, outline=color, width=3)
            core = 22.0
            canv.create_oval(w - core, h - core, w + core, h + core, fill=color, outline="")
        elif state == "thinking":
            import math

            for i in range(3):
                a = phase * 6.28318530718 + i * 2.0944
                x = w + 52 * math.cos(a)
                y = h + 52 * math.sin(a)
                size = 10 - (i * 3)
                canv.create_oval(x - size, y - size, x + size, y + size, fill=color, outline="")
            canv.create_oval(w - 14, h - 14, w + 14, h + 14, fill=PANEL_BG, outline=color, width=2)
        elif state == "speaking":
            r_outer = 60.0 * (0.35 + 0.65 * phase)
            canv.create_oval(w - r_outer, h - r_outer, w + r_outer, h + r_outer, outline=color, width=2)
            r2 = 30 + 18 * phase
            canv.create_oval(w - r2, h - r2, w + r2, h + r2, outline=color, width=2)
            core = 16.0
            canv.create_oval(w - core, h - core, w + core, h + core, fill=color, outline="")
        else:  # confirm
            blink = 0.5 + 0.5 * ((phase * 3) % 1.0)
            r = 40.0
            canv.create_oval(w - r, h - r, w + r, h + r, outline=color, width=int(2 + 2 * blink))
            canv.create_text(w, h, text="?", fill=color, font=("Consolas", 22, "bold"))

        self._root.after(ANIM_MS, self._animate)

    # ------------------------------------------------------------------ #
    # Output box
    # ------------------------------------------------------------------ #
    def _append_line(self, tag: str, text: str) -> None:
        stamp = _timestamp()
        self._box.configure(state="normal")
        self._box.insert("end", stamp + " ", "date")
        self._box.insert("end", text + "\n", tag)
        self._trim_box()
        self._box.configure(state="disabled")
        self._box.yview_moveto(1.0)

    def _trim_box(self, max_lines: int = 500) -> None:
        try:
            last_line = int(self._box.index("end-1c").split(".")[0])
        except Exception:
            return
        if last_line > max_lines:
            self._box.delete("1.0", f"{last_line - max_lines}.0")

    # ------------------------------------------------------------------ #
    # Confirmation panel + controls
    # ------------------------------------------------------------------ #
    def _show_confirm(self, transcript: str, explanation: str) -> None:
        if not self._confirm_shown:
            self._confirm_frame.pack(fill="x", padx=14, pady=(0, 8))
            self._confirm_shown = True
        self._confirm_title.config(text=f"PENDING ▸ {transcript}\n{explanation}")

    def _hide_confirm(self) -> None:
        if self._confirm_shown:
            self._confirm_frame.pack_forget()
            self._confirm_shown = False

    def _approve(self) -> None:
        self._resolve(True)

    def _deny(self) -> None:
        self._resolve(False)

    def _on_return(self, _event) -> None:
        if self._confirm_shown:
            self._approve()

    def _on_escape(self, _event) -> None:
        if self._confirm_shown:
            self._deny()

    def _resolve(self, approved: bool) -> None:
        orch = self._orchestrator
        if orch is None or not self._confirm_shown:
            return
        self._hide_confirm()
        threading.Thread(
            target=orch.resolve_pending_confirmation_ui,
            args=(approved,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def run(self, events: EventBus, orchestrator) -> None:
        """Enter the Tk mainloop; returns when the window closes."""
        self._events = events
        self._orchestrator = orchestrator
        self._root.mainloop()

    def _on_close(self) -> None:
        if self._orchestrator is not None:
            try:
                self._orchestrator.stop()
            except Exception:
                pass
        self._root.destroy()


def _timestamp() -> str:
    import time

    return time.strftime("%H:%M:%S")


if __name__ == "__main__":  # pragma: no cover
    from core.events import EventBus

    JarvisApp().run(EventBus(), None)
