"""CLI entry point for spectria serve."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="spectria",
        description="Serve Spectria training logs over HTTP/SSE",
    )
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the Spectria log server")
    serve_parser.add_argument(
        "--logdir",
        default="./spectria_logs",
        help="Directory containing training logs (default: ./spectria_logs)",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port to listen on (default: 8420)",
    )
    serve_parser.add_argument(
        "--dev",
        action="store_true",
        help="Launch Vite dev server alongside the API server",
    )
    serve_parser.add_argument(
        "--frontend-dir",
        default=None,
        help="Path to the frontend directory (default: auto-detect ../spectria)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        if args.dev:
            _serve_dev(args)
        else:
            _serve(args)
    else:
        parser.print_help()


def _find_frontend_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_dir():
            print(f"Error: frontend directory not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p

    # Auto-detect: look for ../spectria relative to the package
    package_dir = Path(__file__).resolve().parent
    candidate = package_dir.parent.parent.parent / "spectria"
    if candidate.is_dir() and (candidate / "package.json").exists():
        return candidate

    print(
        "Error: could not auto-detect frontend directory. "
        "Use --frontend-dir to specify it.",
        file=sys.stderr,
    )
    sys.exit(1)


def _serve_dev(args):
    import uvicorn

    from .server import create_app

    frontend_dir = _find_frontend_dir(args.frontend_dir)

    if not shutil.which("npm"):
        print("Error: npm not found. Install Node.js to use --dev.", file=sys.stderr)
        sys.exit(1)

    if not (frontend_dir / "package.json").exists():
        print(f"Error: no package.json in {frontend_dir}", file=sys.stderr)
        sys.exit(1)

    app = create_app(logdir=args.logdir)
    frontend_host = "localhost" if args.host == "0.0.0.0" else args.host
    api_url = f"http://{frontend_host}:{args.port}"

    # Start API server in a background thread
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": args.host, "port": args.port, "log_level": "info"},
        daemon=True,
    )
    api_thread.start()

    print(f"Spectria API server at {api_url}")

    # Launch Vite dev server
    vite_env = {
        **os.environ,
        "VITE_LOCAL_DATA_URL": api_url,
    }
    vite_cmd = ["npm", "run", "dev"]
    if args.host not in ("127.0.0.1", "localhost"):
        vite_cmd += ["--", "--host", args.host]
    vite_proc = subprocess.Popen(
        vite_cmd,
        cwd=str(frontend_dir),
        env=vite_env,
    )

    print(f"Vite dev server started (PID {vite_proc.pid})")

    def _shutdown(signum, frame):
        vite_proc.terminate()
        try:
            vite_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vite_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    vite_proc.wait()


def _serve(args):
    import uvicorn

    from .server import create_app

    app = create_app(logdir=args.logdir)
    print(f"Spectria server serving logs from {args.logdir}")
    print(f"Connect at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
