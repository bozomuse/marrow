"""Private on-machine recording kit for Marrow.

A small local recorder: a window with a text area. Everything you do
inside it -- pointer moves, clicks, typing, pasted or dictated text --
is written to a JSONL recording in the documented Marrow format, on
your own disk, and nowhere else. No network, no upload, no account.

Run it with:  python3 marrow.py --record [--out FILE] [--style X]

The recorder distinguishes how text arrives:
  typed     -- a keypress produced the character (physical keyboard)
  inserted  -- text appeared without a keypress (dictation software,
               on-screen keyboard, paste). Detected by diffing the
               widget contents against the last key-driven state.
  correct   -- BackSpace / selection delete, recorded as corrections.

Pointer moves are throttled to one sample per 25 ms so a long session
stays small; clicks are always recorded. Positions are fractions of
the recorder window (0.0-1.0), so the recording is OS-neutral.

The GUI layer is thin: all event-to-format logic lives in Recorder,
which is testable without a display. If tkinter is missing, --record
prints how to get it (python3-tk on Debian/Ubuntu, stock on macOS
and Windows).
"""

import json
import os
import time

MOVE_THROTTLE_MS = 25


class Recorder(object):
    """GUI-independent event sink: turns raw events into Marrow format."""

    def __init__(self, session_id, style="local", persona="owner"):
        self.header = {"type": "session", "id": session_id,
                       "style": style, "persona": persona}
        self.events = []
        self.t0 = None
        self._last_move_ms = -10 ** 9
        self._typed_text = ""
        self._last_seen_text = ""

    def _now_ms(self):
        if self.t0 is None:
            self.t0 = time.time()
            return 0
        return int(round((time.time() - self.t0) * 1000))

    def _emit(self, ev):
        if self.t0 is None:
            self._now_ms()
        ev = dict(ev)
        ev["t"] = self._now_ms()
        self.events.append(ev)

    def move(self, x, y):
        """Pointer move; x, y are fractions of the window, 0.0-1.0."""
        now = self._now_ms()
        if now - self._last_move_ms < MOVE_THROTTLE_MS:
            return
        self._last_move_ms = now
        self._emit({"type": "move", "x": round(x, 4), "y": round(y, 4)})

    def click(self, x, y, button="left"):
        self._emit({"type": "click", "x": round(x, 4), "y": round(y, 4),
                    "button": button})

    def key(self, char):
        """A keypress that produced a character (typed text)."""
        if not char:
            return
        self._typed_text += char
        self._last_seen_text = self._typed_text
        self._emit({"type": "text", "source": "typed",
                    "kind": "char", "text": char})

    def backspace(self, n=1):
        self._typed_text = self._typed_text[:-n] if n else self._typed_text
        self._last_seen_text = self._typed_text
        self._emit({"type": "correct", "source": "typed",
                    "kind": "backspace", "n": n})

    def sync_text(self, current_text):
        """Call after the widget contents change without a keypress.

        Text that appeared without a key (dictation insert, paste,
        on-screen keyboard) is recorded as an inserted event; text
        that vanished without a backspace is a delete-word correction.
        """
        if current_text == self._last_seen_text:
            return
        if len(current_text) > len(self._last_seen_text):
            added = current_text[len(self._last_seen_text):]
            # Dictation inserts whole words; pasted text arrives in bulk.
            self._emit({"type": "text", "source": "inserted",
                        "kind": "word", "text": added})
        else:
            removed = self._last_seen_text[len(current_text):]
            words = len(removed.split())
            self._emit({"type": "correct", "source": "inserted",
                        "kind": "delete-word", "n": max(words, 1)})
        self._last_seen_text = current_text
        self._typed_text = current_text

    def to_jsonl(self):
        lines = [json.dumps(self.header)]
        lines += [json.dumps(ev) for ev in self.events]
        return "\n".join(lines) + "\n"

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.to_jsonl())
        return path


def run_gui(out_path, style, persona):
    """Launch the tkinter recorder window. Returns the saved path."""
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit(
            "tkinter is not installed in this Python.\n"
            "  Debian/Ubuntu: sudo apt install python3-tk\n"
            "  macOS / Windows: it ships with stock Python.\n"
            "The rest of Marrow (generate / train / test) needs no GUI.")

    import datetime
    session_id = "local-%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rec = Recorder(session_id, style=style, persona=persona)

    root = tk.Tk()
    root.title("Marrow recorder -- private, local only")
    root.geometry("640x420")
    label = tk.Label(
        root,
        text=("Type, dictate, or paste in the box below. Pointer moves and\n"
              "clicks in this window are recorded too. Nothing leaves\n"
              "this machine. Close the window to save."),
        justify="left")
    label.pack(padx=10, pady=8, anchor="w")
    text = tk.Text(root, wrap="word")
    text.pack(fill="both", expand=True, padx=10, pady=6)
    text.focus_set()
    status = tk.Label(root, text="recording... 0 events")
    status.pack(padx=10, pady=4, anchor="w")

    def refresh_status():
        status.config(text="recording... %d events" % len(rec.events))
        root.after(500, refresh_status)

    def on_motion(event):
        w, h = event.widget.winfo_width(), event.widget.winfo_height()
        if w > 0 and h > 0:
            rec.move(event.x / float(w), event.y / float(h))

    def on_click(event):
        w, h = event.widget.winfo_width(), event.widget.winfo_height()
        if w > 0 and h > 0:
            rec.click(event.x / float(w), event.y / float(h))

    def on_key(event):
        if event.keysym == "BackSpace":
            rec.backspace()
        elif event.char:
            rec.key(event.char)
        # other keys (arrows, modifiers) carry no text; ignore

    def on_text_change(event):
        # Fires after any content change; sync_text only records
        # what did not come from a keypress.
        rec.sync_text(text.get("1.0", "end-1c"))

    for widget in (root, text):
        widget.bind("<Motion>", on_motion)
        widget.bind("<Button-1>", on_click)
    text.bind("<Key>", on_key)
    text.bind("<<Modified>>", on_text_change)

    def on_close():
        path = rec.save(out_path)
        print("saved %d events to %s" % (len(rec.events), path))
        print("score it with: python3 marrow.py --train enrol.jsonl "
              "--test %s" % path)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_status()
    root.mainloop()
    return out_path
