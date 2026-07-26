"""
学习率网格搜索脚本
快速测试 {3e-5, 1e-5, 5e-6} 三个学习率，每个跑 15 个 epoch（含早停）。
结果保存到 model/60_158+39/lr_search/ 下，不覆盖主模型。
"""

import os
import sys
import json
import copy
import multiprocessing as mp

import pandas as pd
import torch
from tensorboardX import SummaryWriter

from config import config, config_extended, early_stop_config
from train import set_seed, train_one_window, split_train_val_by_last_month


LR_CANDIDATES = [3e-5, 1e-5, 5e-6]
QUICK_EPOCHS = 15       # 每个候选值只跑 15 轮（含 warm-up 即可看到收敛速度差异）
SEED = 42


def main():
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    base_output_dir = config['output_dir']
    lr_search_dir = os.path.join(base_output_dir, 'lr_search')
    os.makedirs(lr_search_dir, exist_ok=True)

    # 加载数据（复用 main 的数据加载逻辑）
    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'),
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
    print(f"  学习率网格搜索: {LR_CANDIDATES}")
    print(f"  quick_epochs={QUICK_EPOCHS}, 验证月份={config_extended.get('val_months', 6)}")
    print(f"{'='*60}")

    results = {}

    for lr in LR_CANDIDATES:
        print(f"\n--- 测试 lr={lr:.0e} ---")

        # 每个 lr 用独立的 config 副本
        cfg = copy.deepcopy(config)
        cfg['learning_rate'] = lr
        cfg['num_epochs'] = QUICK_EPOCHS
        cfg['output_dir'] = os.path.join(lr_search_dir, f'lr_{lr:.0e}'.replace('.', 'p'))

        writer = SummaryWriter(log_dir=os.path.join(cfg['output_dir'], 'log'))

        try:
            best_score, best_metrics = train_one_window(
                train_df, val_df, val_start,
                stockid2idx, num_stocks,
                cfg, device, writer, cfg['output_dir']
            )
            results[lr] = {
                'best_score': best_score,
                'best_metrics': best_metrics if best_metrics else {},
            }
            print(f"  lr={lr:.0e}: best_final_score={best_score:.6f}")
        except Exception as e:
            print(f"  lr={lr:.0e}: FAILED - {e}")
            results[lr] = {'best_score': None, 'error': str(e)}
        finally:
            writer.close()

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  学习率搜索汇总")
    print(f"{'='*60}")
    print(f"{'lr':<12} {'best_final_score':>18}")
    print("-" * 32)
    for lr in LR_CANDIDATES:
        r = results.get(lr, {})
        fs = r.get('best_score')
        if fs is not None:
            print(f"{lr:<12.0e} {fs:>18.6f}")
        else:
            print(f"{lr:<12.0e} {'FAILED':>18}")

    best_lr = max(LR_CANDIDATES, key=lambda x: results.get(x, {}).get('best_score', -999))
    print(f"\n推荐学习率: {best_lr:.0e} (best_score={results[best_lr]['best_score']:.6f})")

    # 保存结果
    summary = {str(lr): {'best_score': r['best_score']} for lr, r in results.items()}
    summary['recommended_lr'] = best_lr
    with open(os.path.join(lr_search_dir, 'lr_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存到: {lr_search_dir}/lr_summary.json")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
