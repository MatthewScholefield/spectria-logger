"""JSONL event writer with metadata header for Spectria run logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunWriter:
    """Writes training events to a JSONL file with a metadata header.

    Format:
        # {"spectria_version":1,"project":"...","run":"...","config":{...},...}
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
        created_at: int | None = None,
    ) -> None:
        self.logdir = Path(logdir)
        self.project = project
        self.run = run
        self.baseline = baseline
        self.config = config or {}
        self._created_at = created_at

        self._run_dir = self.logdir / project / run
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._run_dir / "events.jsonl"
        self._written_header = self._events_path.exists() and self._events_path.stat().st_size > 0

    def write_header(self) -> None:
        if self._written_header:
            return
        import time

        header = {
            "spectria_version": 1,
            "project": self.project,
            "run": self.run,
            "baseline": self.baseline,
            "config": self.config,
            "created_at": self._created_at or int(time.time()),
        }
        with open(self._events_path, "a") as f:
            f.write(f"# {json.dumps(header)}\n")
        self._written_header = True

    def has_rows(self) -> bool:
        """Check if any data rows exist in the events file."""
        if not self._events_path.exists():
            return False
        with open(self._events_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return True
        return False

    def write_row(self, row: dict[str, Any]) -> None:
        self.write_header()
        import time

        row = {**row}
        row.setdefault("_ts", int(time.time()))
        with open(self._events_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Bulk-write multiple data rows."""
        self.write_header()
        import time

        default_ts = int(time.time())
        with open(self._events_path, "a") as f:
            for row in rows:
                row = {**row}
                row.setdefault("_ts", default_ts)
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


def dump_run(
    logdir: str | Path,
    project: str,
    run: str,
    rows: list[dict[str, Any]],
    baseline: str | None = None,
    config: dict[str, Any] | None = None,
    *,
    skip_existing: bool = True,
    created_at: int | None = None,
) -> bool:
    """Write a complete run's data to Spectria JSONL format.

    Returns True if data was written, False if skipped due to existing data.
    """
    writer = RunWriter(logdir, project, run, baseline, config, created_at=created_at)
    if skip_existing and writer.has_rows():
        return False
    writer.write_rows(rows)
    return True
