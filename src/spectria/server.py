"""HTTP/SSE server for serving training logs to Spectria."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .logger.writer import RunWriter


def _allowed_origins() -> list[str]:
    """Exact-match allowed origins."""
    return [
        "https://matthewscholefield.github.io",
    ]


_LOCALHOST_ORIGIN_REGEX = r"http://localhost:\d+"


def create_app(logdir: str | Path):
    """Create a Starlette app serving Spectria training logs."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    logdir = Path(logdir)

    def list_projects(request: Request) -> JSONResponse:
        projects = []
        if logdir.exists():
            for d in sorted(logdir.iterdir()):
                if d.is_dir():
                    run_count = sum(1 for _ in d.iterdir() if _.is_dir())
                    projects.append({"name": d.name, "run_count": run_count})
        return JSONResponse(projects)

    def list_runs(request: Request) -> JSONResponse:
        project_name = request.path_params["name"]
        project_dir = logdir / project_name
        if not project_dir.exists():
            return JSONResponse([], status_code=404)

        runs = []
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            events_path = run_dir / "events.jsonl"
            header = RunWriter.read_header(events_path)

            # Determine status by PID file liveness
            finished_at = None
            status = "completed"
            if RunWriter.is_run_live(events_path):
                status = "running"
            else:
                if header:
                    finished_at = header.get("finished_at") or header.get("created_at")
                if not finished_at and events_path.exists():
                    finished_at = int(events_path.stat().st_mtime)

            runs.append({
                "run_id": run_dir.name,
                "baseline": header.get("baseline") if header else None,
                "status": status,
                "config": header.get("config", {}) if header else {},
                "finished_at": finished_at,
            })

        # Running first, then by most recently finished
        runs.sort(key=lambda r: (r["status"] != "running", -(r["finished_at"] or 0)))
        return JSONResponse(runs)

    def get_run_data(request: Request) -> JSONResponse:
        project_name = request.path_params["name"]
        run_id = request.path_params["run"]
        events_path = logdir / project_name / run_id / "events.jsonl"
        rows = RunWriter.read_rows(events_path)
        status = "running" if RunWriter.is_run_live(events_path) else "completed"
        return JSONResponse({"rows": rows, "status": status})

    async def stream_events(request: Request) -> StreamingResponse:
        project_name = request.path_params["name"]
        run_id = request.path_params["run"]
        events_path = logdir / project_name / run_id / "events.jsonl"

        async def generate():
            # Send status event first
            is_running = RunWriter.is_run_live(events_path)
            yield f"event: status\ndata: {json.dumps({'status': 'running' if is_running else 'completed'})}\n\n"

            # Send all existing rows using offset tracking
            rows, offset = RunWriter.read_rows_from_offset(events_path)
            for row in rows:
                yield f"event: row\ndata: {json.dumps(row, default=str)}\n\n"

            # If already completed, no need to poll
            if not is_running:
                yield "event: complete\ndata: {}\n\n"
                return

            while True:
                await asyncio.sleep(0.5)
                new_rows, offset = RunWriter.read_rows_from_offset(events_path, offset)

                for row in new_rows:
                    yield f"event: row\ndata: {json.dumps(row, default=str)}\n\n"

                # Check if the run has finished (PID file removed)
                if not RunWriter.is_run_live(events_path) and not new_rows:
                    yield "event: complete\ndata: {}\n\n"
                    break

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    routes = [
        Route("/api/projects", endpoint=list_projects),
        Route("/api/projects/{name}/runs", endpoint=list_runs),
        Route("/api/projects/{name}/runs/{run}/data", endpoint=get_run_data),
        Route("/api/projects/{name}/runs/{run}/events", endpoint=stream_events),
    ]

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        def get_mode(request: Request) -> JSONResponse:
            return JSONResponse({"local_data_mode": True})

        async def serve_spa(request: Request) -> FileResponse:
            path = request.path_params.get("path", "")
            file_path = (static_dir / path).resolve()
            if path and file_path.is_file() and str(file_path).startswith(str(static_dir.resolve())):
                return FileResponse(str(file_path))
            return FileResponse(str(static_dir / "index.html"))

        routes.append(Route("/api/mode", endpoint=get_mode))
        routes.append(Route("/{path:path}", endpoint=serve_spa))

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=_allowed_origins(),
                allow_origin_regex=_LOCALHOST_ORIGIN_REGEX,
                allow_methods=["GET"],
            ),
        ],
    )
    return app
