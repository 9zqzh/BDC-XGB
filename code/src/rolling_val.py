"""
50次滚动窗口验证脚本

每次用最近260个交易日训练 XGBRanker → 向前滚动5个交易日验证 → 共滚动50次。
使用置信度驱动后处理替代等权 TopK，评估模型在不同时间段的稳定性和尾部风险。

用法：
    python code/src/rolling_val.py                     # 基础评估（50次滚动）
    python code/src/rolling_val.py --tune               # 后处理参数网格搜索
    python code/src/rolling_val.py --compare_feats      # 特征筛选A/B对比

评估指标：
    - 正收益周比例：50次中 final_score > 0 的比例
    - 平均 final_score：50次周收益率的均值
    - 最长连续亏损：连续 final_score < 0 的最大次数
    - 最大单周亏损/收益：尾部风险度量
"""

import os
import sys
import json
import gc
import copy
import argparse
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from config import config, config_extended, xgb_config
from postprocess import confidence_aware_postprocess
from train import (
    set_seed,
    preprocess_data, preprocess_val_data,
    _merge_fundamentals,
    flatten_sequences_to_xgb,
    _continuous_labels_to_ranks,
    feature_columns_map,
)

# ── 滚动参数 ──
TRAIN_DAYS = 260       # 每次训练窗口（交易日）
VAL_DAYS = 5           # 每次验证窗口（交易日）
N_ROLLS = 50           # 总滚动次数
STEP_DAYS = 5          # 向前滚动步长（交易日）


def _spearman_numpy(a, b):
    """纯 NumPy Spearman 秩相关。"""
    n = len(a)
    if n < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    cov = (ra * rb).sum()
    std = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    return float(cov / std) if std > 1e-12 else 0.0


def _evaluate_day_with_postprocess(day_preds, day_labels, day_stocks, k=5, min_gap=0.005, postproc_params=None):
    """
    对单日预测应用置信度驱动后处理，计算评估指标。

    与等权 TopK 不同：后处理可能选取少于 k 只股票，权重也不均等。
    因此 final_score 公式中的 random_return_sum 需根据实际选取数量调整。

    Args:
        postproc_params: dict, 传递给 confidence_aware_postprocess 的 params 参数

    Returns:
        dict: 当日指标（含 final_score, topk_hit, spearman, win 等）
    """
    n = len(day_preds)
    if n < k:
        return None

    # 真实 TopK 的最大收益和随机收益（用于 final_score 分母）
    true_topk_idx = np.argsort(day_labels)[::-1][:k]
    max_return_sum = float(day_labels[true_topk_idx].sum())
    random_return_sum = float(k * day_labels.mean())
    gap = max_return_sum - random_return_sum
    if abs(gap) < min_gap:
        return None

    # ── 置信度驱动后处理 ──
    selected_stocks, weights, conf_info = confidence_aware_postprocess(
        day_preds, day_stocks, top_k=k, params=postproc_params
    )

    if len(selected_stocks) == 0:
        return None

    # 找到选中股票在数组中的索引
    stock_to_idx = {s: i for i, s in enumerate(day_stocks)}
    selected_idx = np.array([stock_to_idx[s] for s in selected_stocks])

    # 加权收益
    selected_labels = day_labels[selected_idx]
    pred_return_sum = float(np.dot(weights, selected_labels))

    # TopK 命中率（选中的股票中有多少在真实 TopK 中）
    true_topk_set = set(true_topk_idx)
    pred_set = set(selected_idx)
    hit_count = len(true_topk_set & pred_set)
    topk_hit = hit_count

    # Spearman
    spearman = _spearman_numpy(day_preds, day_labels)

    # Win rate（选中股票均收益 vs 全市场均收益）
    market_mean = float(day_labels.mean())
    model_mean = float(selected_labels.mean())
    win = 1.0 if model_mean > market_mean else 0.0

    return {
        'final_score': (pred_return_sum - random_return_sum) / (gap + 1e-12),
        'topk_hit': topk_hit,
        'spearman': spearman,
        'win': win,
        'pred_ret': model_mean,
        'mkt_ret': market_mean,
        'n_selected': len(selected_stocks),
        'confidence_gap': conf_info['confidence_gap'],
        'n_stocks': n,
    }


def run_single_roll(train_df, val_df, stockid2idx, roll_idx, config_override=None):
    """
    执行单次滚动训练 + 置信度驱动后处理评估。

    Args:
        train_df: 训练集 DataFrame
        val_df: 验证集 DataFrame
        stockid2idx: 股票代码→整数映射
        roll_idx: 滚动编号（用于日志）
        config_override: 可选的后处理参数覆盖 dict

    Returns:
        dict: 含 final_score, win_rate, spearman, n_selected_avg 等汇总指标
    """
    import xgboost as xgb

    cfg = copy.deepcopy(config)
    seq_len = cfg['sequence_length']
    feat_name = cfg['feature_num']
    flatten_days = cfg.get('xgb_flatten_days', 10)
    features_list = list(feature_columns_map[feat_name])

    # ── 1. 特征工程 ──
    train_data, _ = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # 检查特征工程后数据是否为空（验证集在数据末尾可能因缺少未来标签而被全部丢弃）
    if len(train_data) == 0:
        return {'final_score': 0.0, 'error': 'empty_train_data'}
    if len(val_data) == 0:
        return {'final_score': 0.0, 'error': 'empty_val_data'}

    # ── 2. 标准化 ──
    scaler = StandardScaler()
    for col_set in [train_data, val_data]:
        col_set[features_list] = col_set[features_list].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features_list)
    val_data = val_data.dropna(subset=features_list)
    if len(train_data) == 0 or len(val_data) == 0:
        return {'final_score': 0.0, 'error': 'empty_after_dropna'}
    train_data[features_list] = scaler.fit_transform(train_data[features_list])
    val_data[features_list] = scaler.transform(val_data[features_list])

    # ── 3. 合并基本面 ──
    fp = os.path.join(cfg['data_path'], 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(cfg['data_path'], 'hs300_fundamentals.csv')
    train_data, fund_cols = _merge_fundamentals(train_data, fp)
    if fund_cols:
        val_data, _ = _merge_fundamentals(val_data, fp)
        features_list = features_list + fund_cols

    # ── 4. 特征筛选 ──
    if cfg.get('selected_features'):
        valid_features = [f for f in cfg['selected_features'] if f in features_list]
        features_list = valid_features

    # ── 5. 展平特征 ──
    X_train, y_train_cont, qid_train, _, _, _ = flatten_sequences_to_xgb(
        train_data, features_list, seq_len
    )
    X_val, y_val_cont, qid_val, _, _, valid_val_dates = flatten_sequences_to_xgb(
        val_data, features_list, seq_len
    )

    if len(X_train) == 0 or len(X_val) == 0:
        return {'final_score': 0.0, 'error': 'empty_data'}

    # ── 6. 标签转换 ──
    y_train = _continuous_labels_to_ranks(y_train_cont, qid_train)
    y_val = _continuous_labels_to_ranks(y_val_cont, qid_val)

    # ── 7. 训练 XGBRanker ──
    xgb_params = {
        'max_depth': xgb_config['max_depth'],
        'learning_rate': xgb_config['learning_rate'],
        'n_estimators': xgb_config['n_estimators'],
        'subsample': xgb_config['subsample'],
        'colsample_bytree': xgb_config['colsample_bytree'],
        'reg_alpha': xgb_config['reg_alpha'],
        'reg_lambda': xgb_config['reg_lambda'],
        'min_child_weight': xgb_config['min_child_weight'],
        'objective': xgb_config['objective'],
        'eval_metric': xgb_config['eval_metric'],
        'ndcg_exp_gain': False,
        'verbosity': 0,
        'n_jobs': xgb_config['n_jobs'],
        'tree_method': 'hist',
        'random_state': 42 + roll_idx,
    }

    model = xgb.XGBRanker(**xgb_params)
    model.fit(
        X_train, y_train,
        qid=qid_train,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=False,
    )

    # ── 8. 预测 + 逐日后处理评估 ──
    preds = model.predict(X_val)

    unique_qids = sorted(set(qid_val))
    daily_results = []

    for qi in unique_qids:
        mask = qid_val == qi
        day_preds = preds[mask].astype(np.float64)
        day_labels = y_val_cont[mask].astype(np.float64)

        # 获取股票代码：通过 instrument 列
        val_sub = val_data[val_data['instrument'].isin(
            np.unique(qid_val[mask])  # 这里需要通过索引找到对应的股票
        )]

        # 由于展平后失去了股票代码列，我们需要另一种方式获取
        # 简化处理：用整数编码作为股票标识
        n_day = len(day_preds)
        day_stocks = [f"s{i}" for i in range(n_day)]

        result = _evaluate_day_with_postprocess(
            day_preds, day_labels, day_stocks, k=5, min_gap=0.005,
            postproc_params=config_override
        )
        if result is not None:
            daily_results.append(result)

    # ── 9. 汇总 ──
    if not daily_results:
        return {'final_score': 0.0, 'error': 'no_valid_days'}

    n_valid = len(daily_results)
    fs_values = [r['final_score'] for r in daily_results]

    del X_train, X_val, y_train, y_val, model
    gc.collect()

    return {
        'final_score': np.mean(fs_values),
        'final_score_std': np.std(fs_values, ddof=1) if n_valid > 1 else 0.0,
        'win_rate': np.mean([r['win'] for r in daily_results]),
        'spearman': np.mean([r['spearman'] for r in daily_results]),
        'topk_hit_rate': np.mean([r['topk_hit'] for r in daily_results]) / 5,
        'n_selected_avg': np.mean([r['n_selected'] for r in daily_results]),
        'confidence_gap_avg': np.mean([r['confidence_gap'] for r in daily_results]),
        'valid_days': n_valid,
        'min_fs': min(fs_values),
        'max_fs': max(fs_values),
    }


def run_rolling_validation(config_override=None):
    """
    执行完整50次滚动验证，汇总结果。

    性能注意：50次独立训练，每次260天数据，预期耗时较长（取决于硬件）。
    建议在后台运行。
    """
    set_seed(42)

    full_df = pd.read_csv(
        os.path.join(config['data_path'], 'train.csv'),
        dtype={'股票代码': str}, low_memory=False
    )
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    all_dates = sorted(full_df['日期'].unique())
    total_dates = len(all_dates)

    print(f"数据范围: {all_dates[0].date()} ~ {all_dates[-1].date()}, 共 {total_dates} 个交易日")
    print(f"滚动参数: 训练{TRAIN_DAYS}天, 验证{VAL_DAYS}天, 步长{STEP_DAYS}天, 共{N_ROLLS}次")

    # 预计算滚动窗口边界
    max_start = total_dates - TRAIN_DAYS - VAL_DAYS - N_ROLLS * STEP_DAYS
    if max_start < 0:
        actual_rolls = (total_dates - TRAIN_DAYS - VAL_DAYS) // STEP_DAYS
        print(f"[警告] 数据不足以完成{N_ROLLS}次滚动，调整为{actual_rolls}次")
        n_rolls = max(1, actual_rolls)
    else:
        n_rolls = N_ROLLS

    results = []

    for roll in range(n_rolls):
        # 计算窗口：从最新日期向前推
        train_end_idx = total_dates - VAL_DAYS - roll * STEP_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0:
            print(f"[跳过] 滚动{roll+1}: 训练起始不足")
            continue

        train_dates = all_dates[train_start_idx:train_end_idx]
        val_dates = all_dates[train_end_idx:train_end_idx + VAL_DAYS]

        train_df = full_df[full_df['日期'].isin(train_dates)].copy()
        val_df = full_df[full_df['日期'].isin(val_dates)].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
        val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

        all_sids = sorted(set(train_df['股票代码'].unique()) | set(val_df['股票代码'].unique()))
        stockid2idx = {s: i for i, s in enumerate(all_sids)}

        print(f"\n[滚动 {roll+1}/{n_rolls}] "
              f"训练: {train_dates[0].date()}~{train_dates[-1].date()} "
              f"({len(train_dates)}天), "
              f"验证: {val_dates[0].date()}~{val_dates[-1].date()} "
              f"({len(val_dates)}天)")

        result = run_single_roll(
            train_df, val_df, stockid2idx, roll,
            config_override=config_override
        )

        if 'error' not in result:
            print(f"  → final_score={result['final_score']:.6f}, "
                  f"win_rate={result['win_rate']:.4f}, "
                  f"spearman={result['spearman']:.4f}, "
                  f"conf_gap={result['confidence_gap_avg']:.3f}, "
                  f"selected_avg={result['n_selected_avg']:.1f}")
            results.append(result)
        else:
            print(f"  → 失败: {result['error']}")
            results.append({'final_score': 0.0, 'error': result['error']})

        del train_df, val_df
        gc.collect()

    # ── 汇总 ──
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("\n错误: 没有成功的滚动窗口")
        return None, results

    fs_values = [r['final_score'] for r in valid_results]
    win_weeks = sum(1 for fs in fs_values if fs > 0)

    # 最长连续亏损
    max_consec_loss = 0
    current_loss = 0
    for fs in fs_values:
        if fs < 0:
            current_loss += 1
            max_consec_loss = max(max_consec_loss, current_loss)
        else:
            current_loss = 0

    summary = {
        'n_rolls': len(valid_results),
        'positive_ratio': win_weeks / max(len(valid_results), 1),
        'mean_final_score': np.mean(fs_values),
        'std_final_score': np.std(fs_values, ddof=1) if len(fs_values) > 1 else 0.0,
        'max_consecutive_loss': max_consec_loss,
        'max_single_loss': min(fs_values),
        'max_single_gain': max(fs_values),
        'mean_win_rate': np.mean([r.get('win_rate', 0) for r in valid_results]),
        'mean_spearman': np.mean([r.get('spearman', 0) for r in valid_results]),
        'mean_selected': np.mean([r.get('n_selected_avg', 0) for r in valid_results]),
    }

    print(f"\n{'='*60}")
    print(f"  50次滚动验证汇总")
    print(f"{'='*60}")
    print(f"  成功滚动次数:    {summary['n_rolls']}")
    print(f"  正收益周比例:    {summary['positive_ratio']:.2%}")
    print(f"  平均 final_score: {summary['mean_final_score']:.4f}")
    print(f"  fs 标准差:        {summary['std_final_score']:.4f}")
    print(f"  最长连续亏损:     {summary['max_consecutive_loss']} 周")
    print(f"  最大单周亏损:     {summary['max_single_loss']:.4f}")
    print(f"  最大单周收益:     {summary['max_single_gain']:.4f}")
    print(f"  平均 win_rate:    {summary['mean_win_rate']:.4f}")
    print(f"  平均 spearman:    {summary['mean_spearman']:.4f}")
    print(f"  平均选股数:       {summary['mean_selected']:.1f}")
    print(f"{'='*60}")

    return summary, results


def tune_postprocess_params():
    """
    对置信度驱动后处理的超参数做网格搜索。

    搜索空间：confidence_gap 阈值 × temperature × z_threshold
    每个组合跑全部50次滚动验证，选综合最优。

    警告：计算量极大（每组合≈50次训练），建议在 GPU/多核机器上运行。
    """
    # 定义参数搜索空间
    from itertools import product

    gap_thresholds = [0.5, 1.0, 1.5, 2.0]      # 高/中/低置信度分界
    temperatures = [0.5, 0.7, 1.0, 1.5]          # softmax 温度
    z_thresholds = [0.5, 1.0, 1.5]                # z-score 入选阈值

    param_combos = list(product(gap_thresholds, temperatures, z_thresholds))
    print(f"后处理参数搜索: {len(param_combos)} 组合")
    print(f"  每个组合跑50次滚动验证")

    best_score = -999
    best_params = None
    best_summary = None

    for idx, (gap, temp, z_thresh) in enumerate(param_combos):
        print(f"\n[{idx+1}/{len(param_combos)}] gap={gap}, temp={temp}, z_thresh={z_thresh}")

        # 构建后处理参数
        postproc_params = {
            'gap_thresholds': [gap, gap * 0.5, gap * 0.25],
            'temperatures': [temp, temp * 0.7, temp * 0.3, temp * 0.1],
            'z_thresholds': [z_thresh, z_thresh * 2.0, z_thresh * 3.0, z_thresh * 4.0],
            'n_selects': [5, 4, 2, 1],
        }

        config_override = postproc_params

        summary, _ = run_rolling_validation(config_override=config_override)

        if summary is None:
            continue

        # 综合评分：正收益比例 × 2 + 平均fs × 10 - 最长连亏
        composite = (
            summary['positive_ratio'] * 2.0
            + summary['mean_final_score'] * 10.0
            - summary['max_consecutive_loss'] * 0.1
        )

        print(f"  → composite={composite:.3f} (pos={summary['positive_ratio']:.2%}, "
              f"fs={summary['mean_final_score']:.4f}, "
              f"consec_loss={summary['max_consecutive_loss']})")

        if composite > best_score:
            best_score = composite
            best_params = {'gap': gap, 'temperature': temp, 'z_threshold': z_thresh}
            best_summary = summary

    print(f"\n{'='*60}")
    print(f"  最佳后处理参数")
    print(f"{'='*60}")
    print(f"  参数: {best_params}")
    print(f"  综合评分: {best_score:.3f}")
    if best_summary:
        print(f"  正收益比例: {best_summary['positive_ratio']:.2%}")
        print(f"  平均 fs: {best_summary['mean_final_score']:.4f}")
        print(f"  最长连亏: {best_summary['max_consecutive_loss']} 周")
    print(f"{'='*60}")

    return best_params, best_summary


def compare_features():
    """
    特征筛选 A/B 对比：全量特征 vs IC筛选后特征。
    分别跑50次滚动验证，对比正收益比例和稳定性。
    """
    print("=" * 60)
    print("  特征筛选 A/B 对比")
    print("=" * 60)

    # ── A组：全量特征 ──
    print("\n[A组] 全量特征（不筛选）...")
    original_selected = config.get('selected_features')
    config['selected_features'] = None
    summary_a, results_a = run_rolling_validation()

    # ── B组：IC筛选后特征 ──
    config['selected_features'] = original_selected
    if config.get('selected_features') is None:
        print("\n[B组] 跳过: 请先运行 factor_ic.py 并将显著因子填入 config.py 的 selected_features")
        return

    print(f"\n[B组] IC筛选特征（{len(config['selected_features'])} 个因子）...")
    summary_b, results_b = run_rolling_validation()

    # ── 对比 ──
    if summary_a and summary_b:
        print(f"\n{'='*60}")
        print(f"  A/B 对比结果")
        print(f"{'='*60}")
        print(f"  {'指标':<22s} {'A(全量)':>12s} {'B(筛选)':>12s} {'变化':>10s}")
        print(f"  {'-'*56}")

        for key, fmt in [
            ('positive_ratio', '.2%'), ('mean_final_score', '.4f'),
            ('std_final_score', '.4f'), ('max_consecutive_loss', 'd'),
            ('mean_win_rate', '.4f'), ('mean_spearman', '.4f'),
        ]:
            va = summary_a.get(key, 0)
            vb = summary_b.get(key, 0) if summary_b else 0
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                change = vb - va
                if fmt == '.2%':
                    print(f"  {key:<22s} {va:>12.2%} {vb:>12.2%} {change:>+10.2%}")
                elif fmt == 'd':
                    print(f"  {key:<22s} {int(va):>12d} {int(vb):>12d} {int(change):>+10d}")
                else:
                    print(f"  {key:<22s} {va:>12.4f} {vb:>12.4f} {change:>+10.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser(description='50次滚动窗口验证 (XGBRanker)')
    parser.add_argument('--tune', action='store_true', help='后处理参数网格搜索')
    parser.add_argument('--compare_feats', action='store_true', help='特征筛选A/B对比')
    parser.add_argument('--n_rolls', type=int, default=N_ROLLS, help=f'滚动次数 (默认{N_ROLLS})')
    args = parser.parse_args()

    if args.n_rolls != N_ROLLS:
        # 允许命令行覆盖滚动次数
        import builtins
        N_ROLLS = args.n_rolls

    if args.tune:
        tune_postprocess_params()
    elif args.compare_feats:
        compare_features()
    else:
        run_rolling_validation()
