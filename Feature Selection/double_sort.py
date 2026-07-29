"""5x5 double-sort tests for core factor-family pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, get_windows, prepare_data, select_dates
from feature_groups import build_feature_groups
from rank_ic import calculate_rank_ic


DEFAULT_PAIRS = (
    ("momentum_trend", "volume_liquidity"),
    ("range_breakout", "volume_liquidity"),
    ("momentum_trend", "volatility_risk"),
    ("range_breakout", "momentum_trend"),
)

# When only one factor family survives ablation, switch to intra-group mode:
# pick the top N features by ICIR within that family and test all C(N, 2) pairs.
DEFAULT_INTRA_TOP_N = 6

# Maps group names to their corresponding "no_<group>" experiment names
# used for per-window retention checks.
_RETAINED_GROUP_EXPERIMENTS = {
    "volume_liquidity": "no_volume",
    "range_breakout": "no_range",
    "momentum_trend": "no_momentum",
    "volatility_risk": "no_volatility",
    "other": "no_other",
}


def best_feature(report: pd.DataFrame, group: list[str]) -> str | None:
    report = report[report["feature"].isin(group)].sort_values(["icir", "ic_mean"], ascending=False)
    if not report.empty:
        return str(report.iloc[0]["feature"])
    return group[0] if group else None


def best_features_n(report: pd.DataFrame, group: list[str], n: int = DEFAULT_INTRA_TOP_N) -> list[str]:
    """Return the top *n* features by ICIR within *group* (for intra-group pairs)."""
    filtered = report[report["feature"].isin(group)].sort_values(["icir", "ic_mean"], ascending=False)
    if not filtered.empty:
        return filtered["feature"].head(n).tolist()
    return list(group[:n])


def sort_one(data: pd.DataFrame, left: str, right: str) -> list[dict]:
    columns = list(dict.fromkeys(["日期", left, right, "label"]))
    frame = data[columns].replace([np.inf, -np.inf], np.nan).dropna()

    def quantile_bins(values: pd.Series) -> pd.Series:
        if values.nunique() < 5:
            return pd.Series(np.nan, index=values.index)
        bins = pd.qcut(values.rank(method="first"), 5, labels=False, duplicates="drop")
        return pd.Series(bins.to_numpy(dtype=float) + 1, index=values.index)

    frame["left_bin"] = frame.groupby("日期", group_keys=False)[left].transform(quantile_bins)
    frame["right_bin"] = frame.groupby("日期", group_keys=False)[right].transform(quantile_bins)
    daily_cells = (
        frame.dropna(subset=["left_bin", "right_bin"])
        .groupby(["日期", "left_bin", "right_bin"])["label"]
        .mean()
        .rename("daily_return")
        .reset_index()
    )
    grouped = (
        daily_cells.groupby(["left_bin", "right_bin"])["daily_return"]
        .agg(mean="mean", count="count")
        .reset_index()
    )
    return grouped.to_dict("records")


def _monotonicity(cells: pd.DataFrame, column: str) -> float:
    curve = cells.groupby(column)["mean"].mean()
    if len(curve) < 3 or curve.nunique() < 2:
        return 0.0
    value = curve.index.to_series().corr(curve, method="spearman")
    return float(value) if pd.notna(value) else 0.0


def _daily_top_return(data: pd.DataFrame, feature: str) -> float:
    clean = data[["日期", feature, "label"]].replace([np.inf, -np.inf], np.nan).dropna()
    top = clean.groupby("日期")[feature].rank(pct=True, method="average") >= 0.8
    daily = clean.loc[top].groupby("日期")["label"].mean()
    return float(daily.mean()) if not daily.empty else 0.0


def add_interaction_features(data: pd.DataFrame, specs: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Add cross-sectional percentile-product features selected from inner data."""
    result = data.copy()
    names = []
    for spec in specs:
        left = spec["left_feature"]
        right = spec["right_feature"]
        name = spec["interaction_feature"]
        if left not in result or right not in result:
            continue
        left_rank = result.groupby("日期")[left].rank(pct=True, method="average")
        right_rank = result.groupby("日期")[right].rank(pct=True, method="average")
        result[name] = left_rank * right_rank
        names.append(name)
    return result, names


def load_window_interactions(path: Path, window: int) -> list[dict]:
    if not path.exists():
        return []
    report = pd.read_csv(path)
    if report.empty:
        return []
    required = {"window", "long_short_spread", "joint_monotonicity", "double_lift"}
    if not required.issubset(report.columns):
        return []
    report = report[report["window"] == window]
    report = report[
        (report["long_short_spread"] > 0)
        & (report["joint_monotonicity"] > 0)
        & (report["double_lift"] > 0)
    ]
    return report.to_dict("records")


def _run_inter_group(
    data: pd.DataFrame,
    groups: dict[str, list[str]],
    refitting_results_path: Path,
    rows: list[dict],
) -> None:
    """Original inter-group double-sort logic (preserved)."""
    for idx, (train_start, train_end, _, _, label) in enumerate(get_windows(), start=1):
        outer_train = select_dates(data, train_start, train_end)
        split_date = outer_train["日期"].min() + (outer_train["日期"].max() - outer_train["日期"].min()) * 0.8
        train_data = outer_train[outer_train["日期"] < split_date].copy()
        ic_report = calculate_rank_ic(
            train_data,
            [feature for values in groups.values() for feature in values if feature != "instrument"],
            min_obs=30,
        )
        retained = set(groups)
        if refitting_results_path.exists():
            refit = pd.read_csv(refitting_results_path)
            if "error" in refit.columns:
                refit = refit[refit["error"].isna()]
            refit = refit[refit["window"] == idx]
            if not refit.empty and "baseline" in set(refit["experiment"]):
                baseline = float(refit.loc[refit["experiment"] == "baseline", "final_score"].iloc[0])
                retained = {
                    group for group, experiment in _RETAINED_GROUP_EXPERIMENTS.items()
                    if experiment in set(refit["experiment"])
                    and float(refit.loc[refit["experiment"] == experiment, "final_score"].iloc[0]) < baseline
                }
        for left_group, right_group in DEFAULT_PAIRS:
            if left_group not in retained or right_group not in retained:
                continue
            left = best_feature(ic_report, groups[left_group])
            right = best_feature(ic_report, groups[right_group])
            if not left or not right:
                continue
            cells = sort_one(train_data, left, right)
            if not cells:
                continue
            cell_frame = pd.DataFrame(cells)
            best = cell_frame.loc[cell_frame["mean"].idxmax()]
            worst = cell_frame.loc[cell_frame["mean"].idxmin()]
            left_curve = _monotonicity(cell_frame, "left_bin")
            right_curve = _monotonicity(cell_frame, "right_bin")
            single_left_return = _daily_top_return(train_data, left)
            single_right_return = _daily_top_return(train_data, right)
            best_return = float(best["mean"])
            rows.append({
                "window": idx, "label": label, "left_group": left_group, "right_group": right_group,
                "left_feature": left, "right_feature": right,
                "best_left_bin": int(best["left_bin"]), "best_right_bin": int(best["right_bin"]),
                "best_mean_return": best_return, "worst_mean_return": float(worst["mean"]),
                "long_short_spread": float(best_return - worst["mean"]),
                "left_monotonicity": left_curve,
                "right_monotonicity": right_curve,
                "joint_monotonicity": float((left_curve + right_curve) / 2),
                "single_left_top_return": single_left_return,
                "single_right_top_return": single_right_return,
                "double_lift": float(best_return - max(single_left_return, single_right_return)),
                "interaction_feature": f"interaction__{left}__{right}",
                "cells": len(cells),
                "test_type": "inter_group",
            })


def _run_intra_group(
    data: pd.DataFrame,
    groups: dict[str, list[str]],
    global_retained: list[str],
    rows: list[dict],
) -> None:
    """Intra-group double-sort: when only one family survives ablation, test
    pairs of features *within* that family."""
    target_group = global_retained[0]
    group_features = groups.get(target_group, [])
    for idx, (train_start, train_end, _, _, label) in enumerate(get_windows(), start=1):
        outer_train = select_dates(data, train_start, train_end)
        split_date = outer_train["日期"].min() + (outer_train["日期"].max() - outer_train["日期"].min()) * 0.8
        train_data = outer_train[outer_train["日期"] < split_date].copy()
        ic_report = calculate_rank_ic(
            train_data,
            [feature for values in groups.values() for feature in values if feature != "instrument"],
            min_obs=30,
        )
        top_features = best_features_n(ic_report, group_features, DEFAULT_INTRA_TOP_N)
        if len(top_features) < 2:
            continue
        # Test all C(N, 2) pairs within the group
        for i, left in enumerate(top_features):
            for right in top_features[i + 1:]:
                cells = sort_one(train_data, left, right)
                if not cells:
                    continue
                cell_frame = pd.DataFrame(cells)
                best = cell_frame.loc[cell_frame["mean"].idxmax()]
                worst = cell_frame.loc[cell_frame["mean"].idxmin()]
                left_curve = _monotonicity(cell_frame, "left_bin")
                right_curve = _monotonicity(cell_frame, "right_bin")
                single_left_return = _daily_top_return(train_data, left)
                single_right_return = _daily_top_return(train_data, right)
                best_return = float(best["mean"])
                rows.append({
                    "window": idx, "label": label,
                    "left_group": target_group, "right_group": target_group,
                    "left_feature": left, "right_feature": right,
                    "best_left_bin": int(best["left_bin"]),
                    "best_right_bin": int(best["right_bin"]),
                    "best_mean_return": best_return,
                    "worst_mean_return": float(worst["mean"]),
                    "long_short_spread": float(best_return - worst["mean"]),
                    "left_monotonicity": left_curve,
                    "right_monotonicity": right_curve,
                    "joint_monotonicity": float((left_curve + right_curve) / 2),
                    "single_left_top_return": single_left_return,
                    "single_right_top_return": single_right_return,
                    "double_lift": float(best_return - max(single_left_return, single_right_return)),
                    "interaction_feature": f"interaction__{left}__{right}",
                    "cells": len(cells),
                    "test_type": "intra_group",
                })


def run(output_dir: Path, refitting_dir: Path | None = None) -> pd.DataFrame:
    """Dispatch to inter-group or intra-group double-sort based on how many
    factor families survive the refitting ablation."""
    data, _ = prepare_data()
    groups = build_feature_groups()
    retained_path = (refitting_dir or ROOT / "Feature Selection" / "results" / "refitting") / "retained_feature_groups.json"
    refitting_results_path = retained_path.parent / "refitting_window_results.csv"

    # Determine global retained groups from the refitting step
    global_retained: list[str] = []
    if retained_path.exists():
        global_retained = json.loads(retained_path.read_text(encoding="utf-8")).get("groups", [])

    rows: list[dict] = []
    if len(global_retained) < 2:
        # Only one (or zero) family survived → intra-group mode
        if global_retained:
            _run_intra_group(data, groups, global_retained, rows)
    else:
        # Two or more families survived → original inter-group mode
        _run_inter_group(data, groups, refitting_results_path, rows)

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "double_sort_results.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Feature Selection" / "results" / "double_sort")
    parser.add_argument("--refitting-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.output_dir, args.refitting_dir)


if __name__ == "__main__":
    main()
