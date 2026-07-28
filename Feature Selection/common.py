"""Shared data, metric, and XGBRanker adapters for feature selection."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "code" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import config, config_extended, xgb_config  # noqa: E402
from train import (  # noqa: E402
    _continuous_labels_to_ranks,
    evaluate_xgb_model,
    flatten_sequences_to_xgb,
    preprocess_data,
    feature_columns_map,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data() -> pd.DataFrame:
    path = ROOT / config["data_path"] / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"训练数据不存在: {path}")
    data = pd.read_csv(path, dtype={"股票代码": str}, low_memory=False)
    data["日期"] = pd.to_datetime(data["日期"])
    return data


def prepare_data(raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Prepare features and labels once, retaining future labels before date filtering."""
    raw = load_raw_data() if raw is None else raw.copy()
    stock_ids = sorted(raw["股票代码"].astype(str).unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}
    raw["股票代码"] = raw["股票代码"].astype(str)
    processed, features = preprocess_data(raw, is_train=True, stockid2idx=stockid2idx)
    processed["日期"] = pd.to_datetime(processed["日期"])
    return processed, list(features)


def get_windows() -> list[tuple[str, str, str, str, str]]:
    return list(config_extended["cross_val_windows"])


def select_dates(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (data["日期"] >= pd.Timestamp(start)) & (data["日期"] <= pd.Timestamp(end))
    return data.loc[mask].copy()


def metric_mean(results: list[dict], key: str = "final_score") -> float:
    values = [float(item[key]) for item in results if key in item and np.isfinite(item[key])]
    return float(np.mean(values)) if values else 0.0


def metric_std(results: list[dict], key: str = "final_score") -> float:
    values = [float(item[key]) for item in results if key in item and np.isfinite(item[key])]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_window(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    features: list[str],
    output_dir: Path,
) -> dict:
    """Train one XGBRanker window using a caller-selected base feature list."""
    if not features:
        raise ValueError("特征列表为空")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = train_data.copy()
    val_data = val_data.copy()
    for frame in (train_data, val_data):
        frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    if train_data.empty or val_data.empty:
        raise ValueError("训练集或验证集在特征清洗后为空")

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    X_train, y_train_cont, qid_train, _, _, _ = flatten_sequences_to_xgb(
        train_data, features, config["sequence_length"], config.get("xgb_flatten_days", 10)
    )
    X_val, y_val_cont, qid_val, _, _, valid_val_dates = flatten_sequences_to_xgb(
        val_data, features, config["sequence_length"], config.get("xgb_flatten_days", 10)
    )
    y_train = _continuous_labels_to_ranks(y_train_cont, qid_train)
    y_val = _continuous_labels_to_ranks(y_val_cont, qid_val)
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("序列展平后没有有效样本")

    params = {
        "max_depth": xgb_config["max_depth"],
        "learning_rate": xgb_config["learning_rate"],
        "n_estimators": xgb_config["n_estimators"],
        "subsample": xgb_config["subsample"],
        "colsample_bytree": xgb_config["colsample_bytree"],
        "reg_alpha": xgb_config["reg_alpha"],
        "reg_lambda": xgb_config["reg_lambda"],
        "min_child_weight": xgb_config["min_child_weight"],
        "objective": xgb_config["objective"],
        "eval_metric": xgb_config["eval_metric"],
        "ndcg_exp_gain": False,
        "verbosity": 0,
        "n_jobs": xgb_config["n_jobs"],
        "tree_method": "hist",
        "random_state": 42,
    }
    model = xgb.XGBRanker(**params)
    model.fit(X_train, y_train, qid=qid_train, eval_set=[(X_val, y_val)], eval_qid=[qid_val], verbose=False)
    metrics = evaluate_xgb_model(
        model, X_val, y_val_cont, qid_val, valid_val_dates, val_data,
        features, scaler, config["sequence_length"],
        k=config_extended.get("eval_top_k", 5),
        min_gap=config_extended.get("min_gap", 0.005),
    )
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    save_json(output_dir / "features.json", {
        "features": features,
        "feature_count": len(features),
        "sequence_length": config["sequence_length"],
        "flatten_days": config.get("xgb_flatten_days", 10),
    })
    return metrics
