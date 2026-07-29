"""Refitting-based grouped ablation and grouped feature importance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ROOT, get_windows, metric_mean, metric_std, prepare_data, save_json, select_dates, train_window
from feature_groups import build_feature_groups, get_feature_columns
from rank_ic import calculate_rank_ic


EXPERIMENTS = (
    "baseline",
    "volume_only", "range_only", "momentum_only",
    "volatility_only", "other_only",
    "no_volume", "no_range", "no_momentum",
    "no_volatility", "no_other",
    "selected_features",
)

_EXPERIMENT_GROUPS = {
    "volume": "volume_liquidity",
    "range": "range_breakout",
    "momentum": "momentum_trend",
    "volatility": "volatility_risk",
    "other": "other",
}

_RETAINED_GROUP_EXPERIMENTS = {
    "volume_liquidity": "no_volume",
    "range_breakout": "no_range",
    "momentum_trend": "no_momentum",
    "volatility_risk": "no_volatility",
    "other": "no_other",
}


def load_selected_features(path: Path) -> list[str]:
    if not path.exists():
        return []
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("features", []))


def experiment_features(name: str, all_features: list[str], groups: dict[str, list[str]], selected: list[str]) -> list[str]:
    if name == "baseline":
        return list(all_features)
    if name == "selected_features":
        return selected or list(all_features)
    if name.endswith("_only"):
        group = _EXPERIMENT_GROUPS.get(name.removesuffix("_only"), name.removesuffix("_only"))
        return [feature for feature in groups.get(group, []) if feature in all_features]
    if name.startswith("no_"):
        group = _EXPERIMENT_GROUPS.get(name.removeprefix("no_"), name.removeprefix("no_"))
        return [feature for feature in all_features if feature not in set(groups.get(group, []))]
    raise ValueError(f"未知实验: {name}")


def run(output_dir: Path, selected_path: Path | None = None) -> pd.DataFrame:
    data, all_features = prepare_data()
    model_features = [feature for feature in all_features if feature != "instrument"]
    groups = build_feature_groups()
    selected = load_selected_features(selected_path) if selected_path else []
    rows = []
    for experiment in EXPERIMENTS:
        for idx, (train_start, train_end, val_start, val_end, label) in enumerate(get_windows(), start=1):
            outer_train = select_dates(data, train_start, train_end)
            # The outer validation period is not used for feature selection.
            split_date = outer_train["日期"].min() + (outer_train["日期"].max() - outer_train["日期"].min()) * 0.8
            inner_train = outer_train[outer_train["日期"] < split_date]
            inner_val = outer_train[outer_train["日期"] >= split_date]
            if inner_train.empty or inner_val.empty:
                continue
            window_selected = selected
            if not window_selected:
                ic_report = calculate_rank_ic(
                    inner_train,
                    [feature for feature in all_features if feature != "instrument"],
                    min_obs=30,
                )
                window_selected = ic_report[
                    (ic_report["ic_mean"] >= 0)
                    & (ic_report["icir"] >= 0)
                    & (ic_report["valid_days"] >= 60)
                ]["feature"].tolist()
                features = experiment_features(experiment, model_features, groups, window_selected)
            if not features:
                continue
            try:
                metrics = train_window(inner_train, inner_val, features, output_dir / experiment / f"window_{idx}")
                rows.append({"experiment": experiment, "window": idx, "label": label, **metrics})
            except Exception as exc:
                rows.append({"experiment": experiment, "window": idx, "label": label, "error": str(exc)})
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "refitting_window_results.csv", index=False, encoding="utf-8-sig")
    valid = result[result.get("error", pd.Series(index=result.index)).isna()].copy() if not result.empty else result
    summary_rows = []
    if not valid.empty:
        for experiment, group in valid.groupby("experiment"):
            summary_rows.append({
                "experiment": experiment,
                "windows": len(group),
                "final_score_mean": metric_mean(group.to_dict("records")),
                "final_score_std": metric_std(group.to_dict("records")),
                "topk_hit_rate_mean": metric_mean(group.to_dict("records"), "topk_hit_rate"),
                "win_rate_mean": metric_mean(group.to_dict("records"), "win_rate"),
                "pred_return_sum_mean": metric_mean(group.to_dict("records"), "pred_return_sum"),
                "positive_window_ratio": float((group["final_score"] > 0).mean()),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and "baseline" in set(summary["experiment"]):
        baseline = float(summary.loc[summary["experiment"] == "baseline", "final_score_mean"].iloc[0])
        summary["group_importance"] = baseline - summary["final_score_mean"]
    summary.to_csv(output_dir / "refitting_summary.csv", index=False, encoding="utf-8-sig")
    retained = [
        group for group, experiment in _RETAINED_GROUP_EXPERIMENTS.items()
        if not summary.empty and experiment in set(summary["experiment"])
        and float(summary.loc[summary["experiment"] == experiment, "group_importance"].iloc[0]) > 0
        and float(summary.loc[summary["experiment"] == experiment, "positive_window_ratio"].iloc[0]) >= 0.5
    ]
    save_json(output_dir / "retained_feature_groups.json", {"groups": retained})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Feature Selection" / "results" / "refitting")
    parser.add_argument("--selected-path", type=Path, default=None)
    args = parser.parse_args()
    run(args.output_dir, args.selected_path)


if __name__ == "__main__":
    main()
