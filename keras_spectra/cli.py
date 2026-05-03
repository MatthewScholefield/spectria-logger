"""CLI entry point for keras-spectra serve."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="keras-spectra",
        description="Serve Spectra training logs over HTTP/SSE",
    )
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Start the Spectra log server")
    serve_parser.add_argument(
        "--logdir",
        default="./spectra_logs",
        help="Directory containing training logs (default: ./spectra_logs)",
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

    args = parser.parse_args()

    if args.command == "serve":
        _serve(args)
    else:
        parser.print_help()


def _serve(args):
    import uvicorn

    from .server import create_app

    app = create_app(logdir=args.logdir)
    print(f"Spectra server serving logs from {args.logdir}")
    print(f"Connect at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
