"""
超参数搜索脚本（Optuna 贝叶斯优化）
搜索空间：top5_weight ∈ [1.0, 5.0], num_layers ∈ [2, 4]
每个 trial 跑 15 个 epoch（含早停），结果保存到 model/60_158+39/optuna_search/

用法：uv run python code/src/optuna_search.py  [--n_trials 20]
"""
import os
import sys
import copy
import argparse
import multiprocessing as mp
import json

import pandas as pd
import torch
import optuna
from tensorboardX import SummaryWriter

from config import config, config_extended, early_stop_config
from train import set_seed, train_one_window, split_train_val_by_last_month


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=20, help='Optuna 搜索 trial 数')
    parser.add_argument('--quick_epochs', type=int, default=15, help='每个 trial 的 epoch 数')
    return parser.parse_args()


class OptunaObjective:
    def __init__(self, train_df, val_df, val_start, stockid2idx, num_stocks, device, search_dir, quick_epochs):
        self.train_df = train_df
        self.val_df = val_df
        self.val_start = val_start
        self.stockid2idx = stockid2idx
        self.num_stocks = num_stocks
        self.device = device
        self.search_dir = search_dir
        self.quick_epochs = quick_epochs
        self.trial_counter = 0

    def __call__(self, trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(config)
        cfg['num_epochs'] = self.quick_epochs
        cfg['top5_weight'] = trial.suggest_float('top5_weight', 1.0, 5.0, step=0.5)
        cfg['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        cfg['learning_rate'] = trial.suggest_categorical('learning_rate', [3e-5, 1e-5, 5e-6])

        self.trial_counter += 1
        trial_dir = os.path.join(self.search_dir, f'trial_{self.trial_counter:03d}')
        cfg['output_dir'] = trial_dir
        os.makedirs(trial_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join(trial_dir, 'log'))

        try:
            best_score, _ = train_one_window(
                self.train_df, self.val_df, self.val_start,
                self.stockid2idx, self.num_stocks,
                cfg, self.device, writer, trial_dir
            )
            trial.set_user_attr('top5_weight', cfg['top5_weight'])
            trial.set_user_attr('num_layers', cfg['num_layers'])
            trial.set_user_attr('learning_rate', cfg['learning_rate'])
        except Exception as e:
            print(f"  Trial {self.trial_counter} FAILED: {e}")
            best_score = -999.0
        finally:
            writer.close()

        return best_score


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    search_dir = os.path.join(config['output_dir'], 'optuna_search')
    os.makedirs(search_dir, exist_ok=True)

    # 数据加载
    full_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'),
                          dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length'],
        val_months=config_extended.get('val_months', 6)
    )
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    print(f"\n{'='*60}")
    print(f"  Optuna 超参数搜索: {args.n_trials} trials × {args.quick_epochs} epochs")
    print(f"  搜索空间: top5_weight=[1.0,5.0], num_layers=[2,4], lr={{3e-5,1e-5,5e-6}}")
    print(f"{'='*60}")

    objective = OptunaObjective(train_df, val_df, val_start, stockid2idx, num_stocks,
                                device, search_dir, args.quick_epochs)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── 结果 ──
    print(f"\n{'='*60}")
    print(f"  Optuna 搜索结果")
    print(f"{'='*60}")
    print(f"  最佳 trial: #{study.best_trial.number}")
    print(f"  最佳 final_score: {study.best_value:.6f}")
    print(f"  最佳参数: {study.best_params}")

    print(f"\n  Top 5 trials:")
    for t in study.trials[:5]:
        if t.value is not None:
            print(f"    #{t.number:3d}  score={t.value:.6f}  params={t.params}")

    # 保存
    result = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials,
    }
    with open(os.path.join(search_dir, 'optuna_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到: {search_dir}/optuna_result.json")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
