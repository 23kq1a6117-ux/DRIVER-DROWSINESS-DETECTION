"""Threaded audio alarm. Uses pyttsx3 if available, otherwise a system beep."""
from __future__ import annotations

import threading


class Alarm:
    def __init__(self, enabled: bool = True, message: str = "Wake up! Please focus on the road."):
        self.enabled = enabled
        self.message = message
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        if not self.enabled or self.running:
            return
        # A fresh Event per run, so an old thread that is still finishing a
        # sentence can never be "revived" by a later start().
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def _run(self, stop: threading.Event) -> None:
        engine = None
        try:
            import pyttsx3

            # The engine must be created inside the thread that uses it.
            engine = pyttsx3.init()
        except Exception as exc:  # no TTS backend available
            print(f"[WARN] Text-to-speech unavailable ({exc}); using beep.")
        while not stop.is_set():
            if engine is not None:
                try:
                    engine.say(self.message)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"[WARN] TTS failed ({exc}); switching to beep.")
                    engine = None
            else:
                self._beep()
            stop.wait(0.3)

    @staticmethod
    def _beep() -> None:
        try:
            import winsound

            winsound.Beep(1000, 300)
        except ImportError:
            print("\a", end="", flush=True)
