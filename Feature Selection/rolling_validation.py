"""Outer rolling validation for candidate feature sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ROOT, get_windows, metric_mean, metric_std, prepare_data, save_json, select_dates, train_window
from feature_groups import build_feature_groups
from rank_ic import calculate_rank_ic
from double_sort import add_interaction_features, load_window_interactions


def _load_group_features(path: Path, all_features: list[str], groups: dict[str, list[str]], window: int) -> list[str]:
    window_results = path / "refitting_window_results.csv"
    if window_results.exists():
        results = pd.read_csv(window_results)
        results = results[results["window"] == window]
        if "error" in results.columns:
            results = results[results["error"].isna()]
        if not results.empty and "baseline" in set(results["experiment"]):
            baseline = float(results.loc[results["experiment"] == "baseline", "final_score"].iloc[0])
            retained = {
                group for group, experiment in {
                    "volume_liquidity": "no_volume",
                    "range_breakout": "no_range",
                    "momentum_trend": "no_momentum",
                    "volatility_risk": "no_volatility",
                    "other": "no_other",
                }.items()
                if experiment in set(results["experiment"])
                and float(results.loc[results["experiment"] == experiment, "final_score"].iloc[0]) < baseline
            }
            selected = [feature for group in retained for feature in groups.get(group, [])]
            return selected or list(all_features)
    if not path.exists():
        return list(all_features)
    retained = set(json.loads((path / "retained_feature_groups.json").read_text(encoding="utf-8")).get("groups", [])) if (path / "retained_feature_groups.json").exists() else set()
    selected = [feature for group in retained for feature in groups.get(group, [])]
    return selected or list(all_features)


def run(output_dir: Path) -> pd.DataFrame:
    data, all_features = prepare_data()
    model_features = [feature for feature in all_features if feature != "instrument"]
    groups = build_feature_groups()
    refitting_dir = ROOT / "Feature Selection" / "results" / "refitting"

    rows = []
    for idx, (train_start, train_end, val_start, val_end, label) in enumerate(get_windows(), start=1):
        train_data = select_dates(data, train_start, train_end)
        val_data = select_dates(data, val_start, val_end)
        if train_data.empty or val_data.empty:
            rows.append({"candidate": "all", "window": idx, "label": label, "error": "empty_data"})
            continue

        ic_report = calculate_rank_ic(
            train_data,
            [feature for feature in all_features if feature != "instrument"],
            min_obs=30,
        )
        rank_selected = ic_report[
            (ic_report["ic_mean"] >= 0)
            & (ic_report["icir"] >= 0)
            & (ic_report["valid_days"] >= 60)
        ]["feature"].tolist() or list(model_features)
        refit_selected = _load_group_features(refitting_dir, all_features, groups, idx)
        interaction_specs = load_window_interactions(
            ROOT / "Feature Selection" / "results" / "double_sort" / "double_sort_results.csv",
            idx,
        )
        train_candidate, interaction_features = add_interaction_features(train_data, interaction_specs)
        val_candidate, _ = add_interaction_features(val_data, interaction_specs)
        candidates = {
            "baseline": list(model_features),
            "baseline_with_instrument": list(all_features),
            "rank_ic": rank_selected,
            "refitting_groups": [feature for feature in refit_selected if feature != "instrument"],
            "double_sort_interaction": list(model_features) + interaction_features,
        }
        for candidate_name, features in candidates.items():
            try:
                candidate_train = train_candidate if candidate_name == "double_sort_interaction" else train_data
                candidate_val = val_candidate if candidate_name == "double_sort_interaction" else val_data
                metrics = train_window(candidate_train, candidate_val, features, output_dir / candidate_name / f"window_{idx}")
                rows.append({"candidate": candidate_name, "window": idx, "label": label, **metrics})
            except Exception as exc:
                rows.append({"candidate": candidate_name, "window": idx, "label": label, "error": str(exc)})

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "rolling_window_results.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    if not result.empty:
        valid = result[result["error"].isna()] if "error" in result else result
        for candidate, group in valid.groupby("candidate"):
            records = group.to_dict("records")
            summary_rows.append({
                "candidate": candidate,
                "windows": len(group),
                "final_score_mean": metric_mean(records),
                "final_score_std": metric_std(records),
                "positive_window_ratio": float((group["final_score"] > 0).mean()),
                "topk_hit_rate_mean": metric_mean(records, "topk_hit_rate"),
                "pred_return_sum_mean": metric_mean(records, "pred_return_sum"),
            })
    summary = pd.DataFrame(summary_rows)
    recommended = None
    if not summary.empty:
        baseline_rows = summary[summary["candidate"] == "baseline"]
        baseline_std = float(baseline_rows["final_score_std"].iloc[0]) if not baseline_rows.empty else float("inf")
        stability_limit = baseline_std * 1.25 if baseline_std > 0 else 1e-12
        summary["baseline_final_score_std"] = baseline_std
        summary["stability_limit"] = stability_limit
        summary["positive_window_pass"] = summary["positive_window_ratio"] >= 0.5
        summary["stability_pass"] = summary["final_score_std"] <= stability_limit
        selection_pool = summary[summary["candidate"] != "baseline_with_instrument"]
        eligible = selection_pool[
            selection_pool["positive_window_pass"] & selection_pool["stability_pass"]
        ]
        ranking = eligible if not eligible.empty else selection_pool
        if not ranking.empty:
            recommended = ranking.sort_values("final_score_mean", ascending=False).iloc[0].to_dict()
        summary = summary.sort_values("final_score_mean", ascending=False)
    summary.to_csv(output_dir / "rolling_summary.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        best = recommended or summary.iloc[0].to_dict()
        save_json(output_dir / "recommended_candidate.json", best)
        (output_dir / "rolling_summary.md").write_text(
            "# 滚动窗口特征方案比较\n\n````text\n"
            + summary.to_string(index=False)
            + f"\n````\n\n推荐方案：`{best['candidate']}`\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Feature Selection" / "results" / "rolling")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
