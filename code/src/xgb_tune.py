"""
XGBRanker 超参数网格搜索
对 max_depth, learning_rate, subsample, min_child_weight 做快速筛选。
每个组合用 n_estimators=200 + early_stopping_rounds=20 快速评估，
最终输出 Top5 组合，供后续完整训练参考。

用法：uv run python code/src/xgb_tune.py
"""

import os
import sys
import json
import copy
import itertools
import multiprocessing as mp

import pandas as pd
import numpy as np

from config import config, config_extended, xgb_config
from train import (
    set_seed, train_one_window, split_train_val_by_last_month,
)

# 搜索空间（基于前序搜索结果，聚焦 colsample 精调）
PARAM_GRID = {
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6],
    'subsample': [0.5, 0.6, 0.7],
    # 已锁定最优: max_depth=8, lr=0.03, min_child=10
}
QUICK_ESTIMATORS = 200
QUICK_EARLY_STOP = 20


def main():
    set_seed(42)

    # 加载数据（复用 train.py 的划分逻辑）
    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'),
                          dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length'],
        val_months=config_extended.get('val_months', 12)
    )

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    grid = list(itertools.product(
        PARAM_GRID['colsample_bytree'],
        PARAM_GRID['subsample'],
    ))
    print(f"\ncolsample + subsample 网格搜索：{len(grid)} 组合")
    print(f"锁定参数: max_depth(来自config), lr=0.03, min_child_weight=10")
    print(f"快速模式：n_estimators={QUICK_ESTIMATORS}, early_stopping={QUICK_EARLY_STOP}")
    print("=" * 60)

    results = []

    for idx, (cs, ss) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] colsample={cs}, subsample={ss}")

        cfg = copy.deepcopy(config)
        trial_dir = os.path.join(cfg['output_dir'], 'xgb_tune', f'trial_{idx+1:03d}')
        os.makedirs(trial_dir, exist_ok=True)

        # 覆盖超参数
        orig_n_est = xgb_config['n_estimators']
        orig_es = xgb_config['early_stopping_rounds']
        xgb_config['colsample_bytree'] = cs
        xgb_config['subsample'] = ss
        xgb_config['max_depth'] = xgb_config.get('max_depth', 5)
        xgb_config['learning_rate'] = 0.03
        xgb_config['min_child_weight'] = 10
        xgb_config['n_estimators'] = QUICK_ESTIMATORS
        xgb_config['early_stopping_rounds'] = QUICK_EARLY_STOP

        try:
            best_score, metrics = train_one_window(
                train_df, val_df, val_start,
                stockid2idx, num_stocks, cfg, trial_dir
            )
            results.append({
                'colsample': cs, 'subsample': ss,
                'final_score': best_score,
                'spearman': metrics.get('spearman_rho', 0) if metrics else 0,
            })
            print(f"  → final_score={best_score:.6f}")
        except Exception as e:
            print(f"  → 失败: {e}")
            results.append({
                'colsample': cs, 'subsample': ss,
                'final_score': -999,
            })

        # 恢复
        xgb_config['n_estimators'] = orig_n_est
        xgb_config['early_stopping_rounds'] = orig_es

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("  Top 10 参数组合")
    print(f"{'='*60}")
    valid = [r for r in results if r['final_score'] > -900]
    valid.sort(key=lambda x: x['final_score'], reverse=True)

    print(f"{'排名':<5} {'colsample':>12} {'subsample':>10} {'final_score':>14} {'spearman':>10}")
    print("-" * 60)
    for i, r in enumerate(valid[:10]):
        print(f"{i+1:<5} {r['colsample']:>12.2f} {r['subsample']:>10.2f} "
              f"{r['final_score']:>14.6f} {r['spearman']:>10.4f}")

    best = valid[0]
    print(f"\n推荐参数: colsample={best['colsample']}, subsample={best['subsample']}")
    print(f"推荐 final_score: {best['final_score']:.6f}")

    with open(os.path.join(config['output_dir'], 'xgb_tune', 'best_params.json'), 'w') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
