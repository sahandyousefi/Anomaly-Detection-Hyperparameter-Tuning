"""Command-line interface, installed as ``fraud-detection``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from fraud_detection import __version__, evaluate, train, tune


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="fraud-detection",
        description="Fraud detection on the PaySim mobile money dataset.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log debug messages."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train", help="Train and evaluate the configured model."
    )
    train.add_arguments(train_parser)
    train_parser.set_defaults(handler=train.run_from_args)

    tune_parser = commands.add_parser(
        "tune", help="Search hyperparameters on the training and validation periods."
    )
    tune.add_arguments(tune_parser)
    tune_parser.set_defaults(handler=tune.run_from_args)

    evaluate_parser = commands.add_parser(
        "evaluate", help="Compare model families on the same chronological folds."
    )
    evaluate.add_arguments(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate.run_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
