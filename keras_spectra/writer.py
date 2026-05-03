"""JSONL event writer with metadata header for Spectra run logs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class RunWriter:
    """Writes training events to a JSONL file with a metadata header.

    Format:
        # {"spectra_version":1,"project":"...","run":"...","baseline":"...","config":{...},"created_at":...}
        {"epoch":0,"loss":0.69,...}
        {"epoch":1,"loss":0.42,...}
    """

    def __init__(
        self,
        logdir: str | Path,
        project: str,
        run: str,
        baseline: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.logdir = Path(logdir)
        self.project = project
        self.run = run
        self.baseline = baseline
        self.config = config or {}

        self._run_dir = self.logdir / project / run
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._run_dir / "events.jsonl"
        self._written_header = self._events_path.exists() and self._events_path.stat().st_size > 0

    def write_header(self) -> None:
        if self._written_header:
            return
        import time

        header = {
            "spectra_version": 1,
            "project": self.project,
            "run": self.run,
            "baseline": self.baseline,
            "config": self.config,
            "created_at": int(time.time()),
        }
        with open(self._events_path, "a") as f:
            f.write(f"# {json.dumps(header)}\n")
        self._written_header = True

    def write_row(self, row: dict[str, Any]) -> None:
        self.write_header()
        import time

        row = {**row, "_ts": int(time.time())}
        with open(self._events_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    @property
    def events_path(self) -> Path:
        return self._events_path

    @staticmethod
    def read_header(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with open(path) as f:
            first_line = f.readline().strip()
        if first_line.startswith("# "):
            try:
                return json.loads(first_line[2:])
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
