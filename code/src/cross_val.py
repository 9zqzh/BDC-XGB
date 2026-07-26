"""
滚动窗口交叉验证脚本
对多个时间窗口独立训练+评估，输出跨窗口稳定性汇总报告。

用法：python code/src/cross_val.py [--config_name 60_158+39]
输出：model/{config_name}/cross_val_report.txt
"""

import argparse
import os
import sys
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

# 复用 train.py 中的核心组件
from config import config, config_extended
from train import set_seed, train_one_window


def parse_args():
    parser = argparse.ArgumentParser(description='滚动窗口交叉验证')
    parser.add_argument('--config_name', type=str, default=None,
                        help='配置名称，默认从 config 自动推导')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='汇总报告输出目录')
    return parser.parse_args()


def run_cross_validation(windows, config, device, base_output_dir=None):
    """
    执行滚动窗口交叉验证。

    Args:
        windows: 窗口列表，每项为 (train_start, train_end, val_start, val_end, label)
        config: 配置字典
        device: torch 设备
        base_output_dir: 基准输出目录

    Returns:
        all_results: 每个窗口的结果列表
    """
    data_path = config['data_path']
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    all_results = []

    for idx, (train_start, train_end, val_start, val_end, label) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(windows)}] 窗口: {label}")
        print(f"  训练: {train_start} ~ {train_end}")
        print(f"  验证: {val_start} ~ {val_end}")
        print(f"{'='*60}")

        # ── 切片数据 ──
        train_start_ts = pd.to_datetime(train_start)
        train_end_ts = pd.to_datetime(train_end)
        val_start_ts = pd.to_datetime(val_start)
        val_end_ts = pd.to_datetime(val_end)

        train_mask = (full_df['日期'] >= train_start_ts) & (full_df['日期'] <= train_end_ts)
        val_mask = (full_df['日期'] >= val_start_ts) & (full_df['日期'] <= val_end_ts)

        window_train_df = full_df[train_mask].copy()
        window_val_df = full_df[val_mask].copy()

        if len(window_train_df) == 0 or len(window_val_df) == 0:
            print(f"  ⚠ 窗口 {label} 数据为空，跳过")
            all_results.append({'label': label, 'error': 'empty_data'})
            continue

        # 为验证集补充序列上下文（往前取 sequence_length-1 天）
        seq_len = config['sequence_length']
        val_context_start = val_start_ts - pd.tseries.offsets.BDay(seq_len - 1)
        val_context_mask = (full_df['日期'] >= val_context_start) & (full_df['日期'] <= val_end_ts)
        window_val_full_df = full_df[val_context_mask].copy()

        # 恢复日期字符串格式
        window_train_df['日期'] = window_train_df['日期'].dt.strftime('%Y-%m-%d')
        window_val_full_df['日期'] = window_val_full_df['日期'].dt.strftime('%Y-%m-%d')

        # ── 建立股票映射 ──
        all_stock_ids = pd.concat([window_train_df, window_val_full_df])['股票代码'].unique()
        stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
        num_stocks = len(stockid2idx)

        # ── 窗口输出目录 ──
        window_output_dir = os.path.join(
            base_output_dir or config['output_dir'],
            f'cross_val_{idx+1}_{label}'
        )
        os.makedirs(window_output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(window_output_dir, 'log'))

        try:
            # ── 调用 train_one_window ──
            best_score, best_extended_metrics = train_one_window(
                window_train_df, window_val_full_df, val_start_ts,
                stockid2idx, num_stocks, config, device, writer, window_output_dir
            )

            result = {
                'label': label,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'best_score': best_score,
            }
            if best_extended_metrics:
                result.update(best_extended_metrics)

            all_results.append(result)
            print(f"  ✓ 窗口完成 — final_score: {best_score:.6f}")

        except Exception as e:
            print(f"  ✗ 窗口失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'label': label, 'error': str(e)})

        finally:
            writer.close()

    return all_results


def generate_summary_report(all_results, output_path):
    """生成滚动窗口交叉验证汇总报告。"""
    valid_results = [r for r in all_results if 'error' not in r]

    if len(valid_results) == 0:
        print("没有成功的窗口，无法生成汇总报告")
        return

    # 提取所有指标
    metric_keys = [
        'final_score', 'ratio_pred', 'ratio_random',
        'topk_hit_rate', 'spearman_rho', 'win_rate',
        'final_score_std', 'max_daily_loss', 'valid_days_ratio',
    ]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("               滚动窗口交叉验证汇总报告")
    lines.append("=" * 80)

    # 表头
    header = f"{'指标':<20}"
    for r in valid_results:
        label_short = r['label'].split('_')[-1] if '_' in r['label'] else r['label']
        header += f" {label_short:>10}"
    header += f" {'均值':>10}  {'标准差':>10}"
    lines.append(header)
    lines.append("-" * 80)

    # 各行指标
    stats = {}
    for key in metric_keys:
        values = []
        for r in valid_results:
            val = r.get(key, float('nan'))
            if isinstance(val, (int, float)) and not np.isnan(val):
                values.append(val)

        if len(values) == 0:
            continue

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

        row = f"{key:<20}"
        for v in values:
            row += f" {v:>10.4f}"
        row += f" {mean_val:>10.4f}  {std_val:>10.4f}"
        lines.append(row)

        stats[key] = {'mean': mean_val, 'std': std_val, 'values': values}

    lines.append("-" * 80)

    # 极值信息
    if 'final_score' in stats:
        fs_vals = stats['final_score']['values']
        fs_labels = [r['label'].split('_')[-1] if '_' in r['label'] else r['label'] for r in valid_results]
        min_idx = np.argmin(fs_vals)
        max_idx = np.argmax(fs_vals)
        lines.append(f"最低窗口 final_score: {fs_vals[min_idx]:.4f} ({fs_labels[min_idx]})")
        lines.append(f"最高窗口 final_score: {fs_vals[max_idx]:.4f} ({fs_labels[max_idx]})")
        lines.append(f"final_score 极差: {fs_vals[max_idx] - fs_vals[min_idx]:.4f}")

        # 稳定性判断
        cv = stats['final_score']['std'] / max(abs(stats['final_score']['mean']), 1e-12)
        if cv < 0.2:
            stability = "配置较稳定"
        elif cv < 0.5:
            stability = "配置一般稳定，存在一定窗口差异"
        else:
            stability = "配置不稳定，窗口间差异较大，建议检查超参数"
        lines.append(f"稳定性判断：final_score 标准差 / 均值 = {cv*100:.2f}%，{stability}")

    lines.append("=" * 80)

    report = '\n'.join(lines)
    print(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 配置
    windows = config_extended.get('cross_val_windows', [])
    if not windows:
        print("错误: config_extended 中没有定义 cross_val_windows")
        sys.exit(1)

    # 输出目录
    config_name = args.config_name or f"{config['sequence_length']}_{config['feature_num']}"
    base_output_dir = args.output_dir or f"./model/{config_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"将执行 {len(windows)} 个窗口的交叉验证")
    print(f"输出目录: {base_output_dir}")
    print(f"使用设备: {device}")

    # 执行
    all_results = run_cross_validation(windows, config, device, base_output_dir)

    # 生成报告
    report_path = os.path.join(base_output_dir, 'cross_val_report.txt')
    generate_summary_report(all_results, report_path)
    print(f"\n汇总报告已保存到: {report_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
