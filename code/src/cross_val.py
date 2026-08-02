"""
滚动窗口交叉验证脚本（XGBRanker 版）
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

from config import config, config_extended
from train import set_seed, train_one_window, _probe_xgb_cuda


def parse_args():
    parser = argparse.ArgumentParser(description='滚动窗口交叉验证 (XGBRanker)')
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--cross_vs_nocross', action='store_true', help='cross vs no-cross 4-window')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'gpu', 'auto'], default='cpu',
                        help='训练设备；cuda/gpu 使用 GPU，auto 自动检测（默认: cpu）')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU 编号（默认: 0）')
    return parser.parse_args()


def run_cross_validation(windows, config, base_output_dir=None, runtime=None):
    """对每个窗口调用 XGBRanker 训练+评估。"""
    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    all_results = []

    for idx, (train_start, train_end, val_start, val_end, label) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(windows)}] 窗口: {label}")
        print(f"  训练: {train_start} ~ {train_end}")
        print(f"  验证: {val_start} ~ {val_end}")
        print(f"{'='*60}")

        train_start_ts = pd.to_datetime(train_start)
        train_end_ts = pd.to_datetime(train_end)
        val_start_ts = pd.to_datetime(val_start)
        val_end_ts = pd.to_datetime(val_end)

        train_mask = (full_df['日期'] >= train_start_ts) & (full_df['日期'] <= train_end_ts)
        val_mask = (full_df['日期'] >= val_start_ts) & (full_df['日期'] <= val_end_ts)

        window_train_df = full_df[train_mask].copy()
        window_val_df = full_df[val_mask].copy()

        if len(window_train_df) == 0 or len(window_val_df) == 0:
            print(f"  [跳过] 窗口 {label} 数据为空")
            all_results.append({'label': label, 'error': 'empty_data'})
            continue

        window_train_df['日期'] = window_train_df['日期'].dt.strftime('%Y-%m-%d')
        window_val_df['日期'] = window_val_df['日期'].dt.strftime('%Y-%m-%d')

        all_stock_ids = pd.concat([window_train_df, window_val_df])['股票代码'].unique()
        stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
        num_stocks = len(stockid2idx)

        window_output_dir = os.path.join(
            base_output_dir or config['output_dir'],
            f'cross_val_{idx+1}_{label}'
        )
        os.makedirs(window_output_dir, exist_ok=True)

        try:
            best_score, best_extended_metrics = train_one_window(
                window_train_df, window_val_df, val_start_ts,
                stockid2idx, num_stocks, config, window_output_dir,
                runtime=runtime,
            )

            result = {
                'label': label, 'train_start': train_start, 'train_end': train_end,
                'val_start': val_start, 'val_end': val_end, 'best_score': best_score,
            }
            if best_extended_metrics:
                result.update(best_extended_metrics)
            all_results.append(result)
            print(f"  [完成] final_score: {best_score:.6f}")

        except Exception as e:
            print(f"  [失败] {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'label': label, 'error': str(e)})

    return all_results


def _compute_reliability_weight(valid_days):
    """根据验证天数计算可靠性权重。>=20天=1.0, 10~19天=0.5, <10天=0.3。"""
    if valid_days is None or np.isnan(valid_days):
        return 1.0, '未知'
    valid_days = int(valid_days)
    if valid_days >= 20:
        return 1.0, '正常'
    elif valid_days >= 10:
        return 0.5, '偏低'
    else:
        return 0.3, '不足'


def generate_summary_report(all_results, output_path):
    """生成滚动窗口交叉验证汇总报告（含可靠性加权）。"""
    valid_results = [r for r in all_results if 'error' not in r]
    if len(valid_results) == 0:
        print("没有成功的窗口，无法生成汇总报告")
        return

    # 计算每个窗口的可靠性权重
    weights = []
    weight_labels = []
    for r in valid_results:
        vd = r.get('valid_days', r.get('total_days', None))
        w, label = _compute_reliability_weight(vd)
        weights.append(w)
        weight_labels.append(label)
    total_weight = sum(weights) if weights else len(valid_results)

    metric_keys = [
        'final_score', 'topk_hit_rate', 'spearman_rho', 'win_rate',
        'final_score_std', 'valid_days_ratio',
    ]

    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("               滚动窗口交叉验证汇总报告 (XGBRanker)  [加权版]")
    lines.append("=" * 95)

    # 表头
    header = f"{'指标':<20}"
    for r in valid_results:
        label_short = r['label'].split('_')[-1] if '_' in r['label'] else r['label']
        header += f" {label_short:>10}"
    header += f" {'加权均值':>10}  {'标准差':>10}"
    lines.append(header)

    # 验证天数和权重行
    days_row = f"{'验证天数':<20}"
    for r in valid_results:
        vd = r.get('valid_days', r.get('total_days', '?'))
        days_row += f" {str(vd):>10}"
    days_row += f" {'—':>10}  {'—':>10}"
    lines.append(days_row)

    weight_row = f"{'可靠性权重':<20}"
    for w, wl in zip(weights, weight_labels):
        weight_row += f" {w:>8.1f}({wl})"
    weight_row += f" {'—':>10}  {'—':>10}"
    lines.append(weight_row)
    lines.append("-" * 95)

    stats = {}
    for key in metric_keys:
        values = []
        w_values = []
        for i, r in enumerate(valid_results):
            val = r.get(key, float('nan'))
            if isinstance(val, (int, float)) and not np.isnan(val):
                values.append(val)
                w_values.append(val * weights[i])
        if len(values) == 0:
            continue
        weighted_mean = sum(w_values) / total_weight if total_weight > 0 else np.mean(values)
        weighted_var_sum = sum(w * (v - weighted_mean) ** 2 for v, w in zip(values, weights))
        weighted_std = np.sqrt(weighted_var_sum / total_weight) if total_weight > 0 else np.std(values, ddof=1)

        row = f"{key:<20}"
        for v in values:
            row += f" {v:>10.4f}"
        row += f" {weighted_mean:>10.4f}  {weighted_std:>10.4f}"
        lines.append(row)
        stats[key] = {'mean': weighted_mean, 'std': weighted_std, 'values': values, 'weights': weights}

    lines.append("-" * 95)

    if 'final_score' in stats:
        fs_vals = stats['final_score']['values']
        fs_labels = [r['label'].split('_')[-1] if '_' in r['label'] else r['label'] for r in valid_results]
        min_idx = np.argmin(fs_vals)
        max_idx = np.argmax(fs_vals)
        lines.append(f"最低窗口 final_score: {fs_vals[min_idx]:.4f} ({fs_labels[min_idx]})")
        lines.append(f"最高窗口 final_score: {fs_vals[max_idx]:.4f} ({fs_labels[max_idx]})")
        lines.append(f"final_score 极差: {fs_vals[max_idx] - fs_vals[min_idx]:.4f}")
        cv = stats['final_score']['std'] / max(abs(stats['final_score']['mean']), 1e-12)
        if cv < 0.2:
            stability = "配置较稳定"
        elif cv < 0.5:
            stability = "配置一般稳定，存在一定窗口差异"
        else:
            stability = "配置不稳定，窗口间差异较大，建议检查超参数"
        lines.append(f"稳定性判断（加权）：final_score 标准差 / 均值 = {cv*100:.2f}%，{stability}")

    # 低可靠性窗口标注
    low_reliability_windows = [
        r['label'] for r, wl in zip(valid_results, weight_labels) if wl != '正常'
    ]
    if low_reliability_windows:
        lines.append(f"\n⚠ 以下窗口验证天数不足，指标仅供参考：")
        for r, wl in zip(valid_results, weight_labels):
            if wl != '正常':
                vd = r.get('valid_days', r.get('total_days', '?'))
                lines.append(f"   · {r['label']}: 验证{vd}天，权重{weights[valid_results.index(r)]:.1f}")

    lines.append("=" * 95)
    report = '\n'.join(lines)
    print(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report
def compare_cross_vs_nocross(runtime=None):
    """
    P2交叉特征四窗口 A/B 对比：含交叉 vs 不含交叉
    """
    CROSS_FEATURES = [
        'cross_MA60_SUMN60', 'cross_MA60_ROC60', 'cross_MA60_ROC30',
        'cross_SUMN60_ROC60', 'cross_SUMN30_ROC60',
        'cross_MA60_MA30', 'cross_ROC30_MA30',
        'cross_vol_price_div', 'cross_liq_adj_ret',
    ]
    windows = config_extended.get('cross_val_windows', [])
    if not windows:
        print("错误: config_extended 中没有定义 cross_val_windows")
        return

    print("=" * 60)
    print("  四窗口交叉验证 A/B 对比")
    print("=" * 60)

    # A组: 含交叉
    base_features = [f for f in config.get('selected_features', []) if f not in CROSS_FEATURES]
    config['selected_features'] = base_features + CROSS_FEATURES
    print(f"\n[A组] 含交叉特征（{len(config['selected_features'])} 因子）...")
    results_a = run_cross_validation(windows, config, config['output_dir'] + '/cross_val_A', runtime=runtime)
    # B组: 无交叉
    config['selected_features'] = base_features
    print(f"\n[B组] 纯IC124基线（{len(base_features)} 因子）...")
    results_b = run_cross_validation(windows, config, config['output_dir'] + '/cross_val_B', runtime=runtime)

    # 对比
    valid_a = [r for r in results_a if 'error' not in r]
    valid_b = [r for r in results_b if 'error' not in r]
    if not valid_a or not valid_b:
        print("窗口数据不足，无法对比")
        return

    print(f"\n{'='*80}")
    print(f"  四窗口交叉 A(含交叉) vs B(无交叉) 对比")
    print(f"{'='*80}")
    print(f"  {'窗口':<14s} {'A fs':>10s} {'B fs':>10s} {'差值':>10s} {'A wr':>8s} {'B wr':>8s}")
    print(f"  {'-'*60}")
    for ra, rb in zip(valid_a, valid_b):
        label = ra['label'].split('_')[-1] if '_' in ra['label'] else ra['label']
        afs = ra.get('final_score', 0)
        bfs = rb.get('final_score', 0)
        awr = ra.get('win_rate', 0)
        bwr = rb.get('win_rate', 0)
        print(f"  {label:<14s} {afs:>10.4f} {bfs:>10.4f} {(bfs-afs):>+10.4f} {awr:>8.4f} {bwr:>8.4f}")
    print(f"{'='*80}")



def main():
    args = parse_args()
    set_seed(args.seed)

    # ── 解析设备 ──
    if args.device in ('cuda', 'gpu'):
        device = _probe_xgb_cuda(f'cuda:{args.gpu_id}')
        print(f'运行模式: GPU ({device})')
    elif args.device == 'auto':
        try:
            device = _probe_xgb_cuda(f'cuda:{args.gpu_id}')
            print(f'运行模式: GPU ({device})')
        except RuntimeError:
            device = 'cpu'
            print('运行模式: CPU（GPU 不可用，自动回退）')
    else:
        device = 'cpu'
        print('运行模式: CPU')
    runtime = {'device': device, 'gpu_id': args.gpu_id if device.startswith('cuda') else None,
               'n_jobs': 8 if device.startswith('cuda') else None,
               'max_bin': 128 if device.startswith('cuda') else None,
               'feature_workers': 6}

    windows = config_extended.get('cross_val_windows', [])
    if not windows:
        print("错误: config_extended 中没有定义 cross_val_windows")
        sys.exit(1)

    config_name = args.config_name or f"{config['sequence_length']}_{config['feature_num']}"
    base_output_dir = args.output_dir or f"./model/{config_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    if args.cross_vs_nocross:
        compare_cross_vs_nocross(runtime=runtime)
        return

    print(f"将执行 {len(windows)} 个窗口的交叉验证 (XGBRanker)")
    print(f"输出目录: {base_output_dir}")

    all_results = run_cross_validation(windows, config, base_output_dir, runtime=runtime)
    report_path = os.path.join(base_output_dir, 'cross_val_report.txt')
    generate_summary_report(all_results, report_path)
    print(f"\n汇总报告已保存到: {report_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
