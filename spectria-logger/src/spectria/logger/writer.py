"""JSONL event writer with metadata header for Spectria run logs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

RunExistsMode = Literal["rename", "overwrite", "append"]


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
        finished_at: int | None = None,
        if_exists: RunExistsMode = "append",
    ) -> None:
        self.logdir = Path(logdir)
        self.project = project
        self.run = run
        self.baseline = baseline
        self.config = config or {}
        self._created_at = created_at
        self._finished_at = finished_at

        self._run_dir = self.logdir / project / run
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._run_dir / "events.jsonl"
        self._written_header = self._events_path.exists() and self._events_path.stat().st_size > 0

        if self._written_header:
            if if_exists == "rename":
                self._rename_run()
            elif if_exists == "overwrite":
                self._events_path.write_text("")
                self._written_header = False

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
        if self._finished_at is not None:
            header["finished_at"] = self._finished_at
        with open(self._events_path, "a") as f:
            f.write(f"# {json.dumps(header)}\n")
        self._written_header = True

        # Write PID file for liveness detection (only for live runs, not dump_run)
        if self._finished_at is None:
            (self._run_dir / "events.pid").write_text(str(os.getpid()))

    def finish(self) -> None:
        """Remove the PID file to signal the run is complete."""
        pid_path = self._run_dir / "events.pid"
        try:
            existing_pid = int(pid_path.read_text().strip())
            if existing_pid == os.getpid():
                pid_path.unlink()
        except (ValueError, OSError):
            pass

    def _rename_run(self) -> None:
        i = 1
        while True:
            new_run = f"{self.run} ({i})"
            new_dir = self.logdir / self.project / new_run
            new_path = new_dir / "events.jsonl"
            if not new_path.exists() or new_path.stat().st_size == 0:
                self.run = new_run
                self._run_dir = new_dir
                self._run_dir.mkdir(parents=True, exist_ok=True)
                self._events_path = new_path
                self._written_header = False
                return
            i += 1

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
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row:
                    continue
                rows.append(row)
        return rows

    @staticmethod
    def read_rows_from_offset(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Read data rows from a JSONL file starting at byte offset.

        Returns (rows, new_offset) where new_offset is the byte position after the last byte read.
        """
        if not path.exists():
            return [], 0
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            f.seek(offset)
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not row:
                    continue
                rows.append(row)
            new_offset = f.tell()
        return rows, new_offset

    @staticmethod
    def is_run_live(events_path: Path) -> bool:
        """Check if a run is still active.

        Prefer the PID file when present, but fall back to a short mtime grace
        period so externally generated live files without a PID marker don't get
        marked complete between writes.
        """
        header = RunWriter.read_header(events_path)
        if header and header.get("finished_at") is not None:
            return False

        pid_path = events_path.parent / "events.pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                pass
            except PermissionError:
                return True
            except (ValueError, OSError):
                pass
            # Stale PID file — clean it up
            try:
                pid_path.unlink()
            except OSError:
                pass

        if not events_path.exists():
            return False

        try:
            return (time.time() - events_path.stat().st_mtime) < 2.0
        except OSError:
            return False


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
    finished_at: int | None = None,
) -> bool:
    """Write a complete run's data to Spectria JSONL format.

    Returns True if data was written, False if skipped due to existing data.
    """
    writer = RunWriter(logdir, project, run, baseline, config, created_at=created_at, finished_at=finished_at)
    if skip_existing and writer.has_rows():
        return False
    writer.write_rows(rows)
    return True
