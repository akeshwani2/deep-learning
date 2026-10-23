"""Command-line entry points for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parkinsons_voice.data.cohort import freeze_uci_cohort, verify_frozen_cohort

DEFAULT_CONFIG = Path("configs/data/uci489.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parkinsons-voice",
        description="Participant-level Parkinson's-associated voice research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-cohort",
        help="validate UCI 489 and freeze the Mode B cohort and grouped folds",
    )
    freeze.add_argument("--source", required=True, type=Path, help="path to the verified UCI CSV")
    freeze.add_argument(
        "--output",
        default=Path("artifacts/m1-cohort"),
        type=Path,
        help="ignored output directory for row-level artifacts",
    )
    freeze.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="cohort configuration JSON",
    )

    verify = subparsers.add_parser(
        "verify-cohort",
        help="verify frozen manifests, folds, and provenance fingerprints",
    )
    verify.add_argument(
        "--artifact-dir",
        default=Path("artifacts/m1-cohort"),
        type=Path,
        help="directory created by freeze-cohort",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the selected command and emit a non-identifying JSON summary."""

    args = _parser().parse_args(argv)
    if args.command == "freeze-cohort":
        report = freeze_uci_cohort(args.source, args.output, args.config)
    else:
        report = verify_frozen_cohort(args.artifact_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
