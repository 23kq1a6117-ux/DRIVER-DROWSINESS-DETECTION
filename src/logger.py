"""CSV event logger."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional


class EventLogger:
    FIELDS = ["timestamp", "event", "reason", "ear", "mar", "pitch_dev"]

    def __init__(self, path: str = "logs/events.csv"):
        self.path = path
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, event: str, reason: str = "", ear: Optional[float] = None,
            mar: Optional[float] = None, pitch_dev: Optional[float] = None) -> None:
        def fmt(v):
            return "" if v is None else f"{v:.3f}"

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                event, reason, fmt(ear), fmt(mar), fmt(pitch_dev),
            ])
