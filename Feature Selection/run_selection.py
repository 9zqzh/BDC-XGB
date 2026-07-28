"""Modular entry point for the feature-selection workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = Path(__file__).resolve().parent
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from double_sort import run as run_double_sort  # noqa: E402
from rank_ic import run as run_rank_ic  # noqa: E402
from refitting_ablation import run as run_refitting  # noqa: E402
from rolling_validation import run as run_rolling  # noqa: E402


RESULTS = FEATURE_DIR / "results"


def run_step(step: str) -> None:
    if step in {"rank_ic", "all"}:
        run_rank_ic(RESULTS / "rank_ic")
    if step in {"refitting", "all"}:
        run_refitting(RESULTS / "refitting")
    if step in {"double_sort", "all"}:
        run_double_sort(RESULTS / "double_sort")
    if step in {"rolling", "all"}:
        run_rolling(RESULTS / "rolling")


def main() -> None:
    parser = argparse.ArgumentParser(description="BDC-XGB 因子筛选流程")
    parser.add_argument("--step", choices=["rank_ic", "refitting", "double_sort", "rolling", "all"], required=True)
    args = parser.parse_args()
    run_step(args.step)


if __name__ == "__main__":
    main()
