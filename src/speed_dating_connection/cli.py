"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .modeling import run_benchmark
from .prepare import prepare_dyads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speed-dating-connection")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Collapse directed ratings into dyads")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark", help="Run grouped wave holdouts")
    benchmark.add_argument("dyads", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--predictions", type=Path, required=True)
    benchmark.add_argument("--hero-chart", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_dyads(args.source, args.output)
        return 0
    if args.command == "benchmark":
        run_benchmark(
            args.dyads,
            output_path=args.output,
            predictions_path=args.predictions,
            hero_chart_path=args.hero_chart,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
