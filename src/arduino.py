"""Optional serial output to an Arduino (LED / buzzer / relay prototype).

Sends b'1' when drowsiness starts and b'0' when it ends. See arduino/alert.ino.
"""
from __future__ import annotations

import time
from typing import Optional


class AlertOutput:
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        self._ser = None
        if port:
            try:
                import serial

                self._ser = serial.Serial(port, baudrate, timeout=0.1)
                time.sleep(2)  # Arduino resets when the port opens
                print(f"[INFO] Arduino connected on {port}.")
            except Exception as exc:
                print(f"[WARN] Arduino not available on {port}: {exc}. Software-only mode.")

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def _write(self, data: bytes) -> None:
        if self._ser is not None:
            try:
                self._ser.write(data)
            except Exception as exc:
                print(f"[WARN] Serial write failed: {exc}")

    def on(self) -> None:
        self._write(b"1")

    def off(self) -> None:
        self._write(b"0")

    def close(self) -> None:
        if self._ser is not None:
            self.off()
            self._ser.close()
            self._ser = None
