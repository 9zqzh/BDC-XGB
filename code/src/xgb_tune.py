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

# 搜索空间
PARAM_GRID = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8],
    'min_child_weight': [1, 5, 10],
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
        PARAM_GRID['max_depth'],
        PARAM_GRID['learning_rate'],
        PARAM_GRID['subsample'],
        PARAM_GRID['min_child_weight'],
    ))
    print(f"\n超参数网格搜索：{len(grid)} 组合")
    print(f"快速模式：n_estimators={QUICK_ESTIMATORS}, early_stopping={QUICK_EARLY_STOP}")
    print("=" * 60)

    results = []

    for idx, (md, lr, ss, mcw) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] max_depth={md}, lr={lr}, subsample={ss}, min_child={mcw}")

        cfg = copy.deepcopy(config)
        trial_dir = os.path.join(cfg['output_dir'], 'xgb_tune', f'trial_{idx+1:03d}')
        os.makedirs(trial_dir, exist_ok=True)

        # 覆盖超参数
        import xgboost as xgb
        global xgb_config
        orig_n_est = xgb_config['n_estimators']
        orig_es = xgb_config['early_stopping_rounds']
        xgb_config['max_depth'] = md
        xgb_config['learning_rate'] = lr
        xgb_config['subsample'] = ss
        xgb_config['min_child_weight'] = mcw
        xgb_config['n_estimators'] = QUICK_ESTIMATORS
        xgb_config['early_stopping_rounds'] = QUICK_EARLY_STOP

        try:
            best_score, metrics = train_one_window(
                train_df, val_df, val_start,
                stockid2idx, num_stocks, cfg, trial_dir
            )
            results.append({
                'max_depth': md, 'learning_rate': lr,
                'subsample': ss, 'min_child_weight': mcw,
                'final_score': best_score,
                'spearman': metrics.get('spearman_rho', 0) if metrics else 0,
                'win_rate': metrics.get('win_rate', 0) if metrics else 0,
            })
            print(f"  → final_score={best_score:.6f}")
        except Exception as e:
            print(f"  → 失败: {e}")
            results.append({
                'max_depth': md, 'learning_rate': lr,
                'subsample': ss, 'min_child_weight': mcw,
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

    print(f"{'排名':<5} {'max_depth':>10} {'lr':>8} {'subsample':>10} {'min_child':>10} {'final_score':>14} {'spearman':>10}")
    print("-" * 75)
    for i, r in enumerate(valid[:10]):
        print(f"{i+1:<5} {r['max_depth']:>10} {r['learning_rate']:>8.3f} "
              f"{r['subsample']:>10.2f} {r['min_child_weight']:>10} "
              f"{r['final_score']:>14.6f} {r['spearman']:>10.4f}")

    best = valid[0]
    print(f"\n推荐参数: max_depth={best['max_depth']}, lr={best['learning_rate']}, "
          f"subsample={best['subsample']}, min_child_weight={best['min_child_weight']}")
    print(f"推荐 final_score: {best['final_score']:.6f}")

    with open(os.path.join(config['output_dir'], 'xgb_tune', 'best_params.json'), 'w') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
