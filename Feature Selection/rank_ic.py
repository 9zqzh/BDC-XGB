"""Single-factor cross-sectional Rank IC and ICIR screening."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, prepare_data, save_json
from feature_groups import build_feature_groups, get_feature_columns


def calculate_rank_ic(data: pd.DataFrame, features: list[str], min_obs: int = 30) -> pd.DataFrame:
    rows = []
    for feature in features:
        daily = []
        for _, group in data[["日期", feature, "label"]].groupby("日期", sort=True):
            clean = group.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < min_obs or clean[feature].nunique() < 2 or clean["label"].nunique() < 2:
                continue
            daily.append(clean[feature].rank().corr(clean["label"].rank(), method="pearson"))
        values = np.asarray([x for x in daily if pd.notna(x)], dtype=float)
        mean_ic = float(values.mean()) if len(values) else 0.0
        std_ic = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append({
            "feature": feature,
            "ic_mean": mean_ic,
            "ic_std": std_ic,
            "icir": mean_ic / std_ic if std_ic > 1e-12 else 0.0,
            "positive_ic_ratio": float((values > 0).mean()) if len(values) else 0.0,
            "valid_days": int(len(values)),
            "direction": "positive" if mean_ic >= 0 else "negative",
        })
    return pd.DataFrame(rows).sort_values(["icir", "ic_mean"], ascending=False)


def run(output_dir: Path, min_ic_mean: float = 0.0, min_icir: float = 0.0, min_days: int = 60) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, features = prepare_data()
    groups = build_feature_groups()
    analysis_features = [f for f in features if f != "instrument" and f in {x for values in groups.values() for x in values}]
    report = calculate_rank_ic(data, analysis_features, min_obs=30)
    report.to_csv(output_dir / "rank_ic_report.csv", index=False, encoding="utf-8-sig")
    selected = report[
        (report["ic_mean"] >= min_ic_mean)
        & (report["icir"] >= min_icir)
        & (report["valid_days"] >= min_days)
    ]["feature"].tolist()
    save_json(output_dir / "selected_features.json", {"features": selected, "count": len(selected)})
    summary = ["# Rank IC / ICIR 初筛结果", "", f"保留特征数：{len(selected)}", "", "筛选条件：", f"- IC均值 >= {min_ic_mean}", f"- ICIR >= {min_icir}", f"- 有效交易日 >= {min_days}"]
    (output_dir / "rank_ic_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Feature Selection" / "results" / "rank_ic")
    parser.add_argument("--min-ic-mean", type=float, default=0.0)
    parser.add_argument("--min-icir", type=float, default=0.0)
    parser.add_argument("--min-days", type=int, default=60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir, args.min_ic_mean, args.min_icir, args.min_days)


if __name__ == "__main__":
    main()
