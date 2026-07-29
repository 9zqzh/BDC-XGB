"""
因子IC分析脚本
计算每个特征因子对未来5日超额收益的截面 Rank IC，
筛选出 IC 均值显著不为零（|t-stat| > 2）的因子。

标签计算：直接复用 train.py 的 _preprocess_common（内含 _build_label_and_clean），
确保与训练流程的标签逻辑完全一致。

用法：
    python code/src/factor_ic.py

输出：
    model/60_158+39/factor_ic_analysis.csv  — 全部因子的IC统计量
    控制台打印显著因子列表 — 手动填入 config.py 的 selected_features
"""

import os
import sys
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from config import config, config_extended
from train import (
    _preprocess_common, _merge_fundamentals,
    feature_engineer_func_map, feature_columns_map,
    set_seed,
)


def compute_factor_ic_analysis(output_path=None):
    """
    对 feature_columns_map 中所有特征因子逐一计算 Rank IC。

    流程：
    1. 加载 train.csv，取训练集部分（不含最近12个月验证集）
    2. 特征工程（复用 _preprocess_common）
    3. 对每个因子，逐日计算因子值与 label（超额收益）的 Spearman 秩相关
    4. 输出每个因子的：IC均值、IC标准差、t-stat、ICIR

    Returns:
        ic_df: DataFrame, 各因子的 IC 统计量
        significant_factors: list, t-stat 绝对值 > 2 的显著因子名
    """
    set_seed(42)

    # ── 1. 加载数据 ──
    data_path = config['data_path']
    full_df = pd.read_csv(
        os.path.join(data_path, 'train.csv'),
        dtype={'股票代码': str}, low_memory=False
    )
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    # ── 2. 只用训练集部分（不含最近12个月验证集），避免使用未来数据 ──
    last_date = full_df['日期'].max()
    train_end = last_date - pd.DateOffset(months=config_extended.get('val_months', 12))
    train_df = full_df[full_df['日期'] <= train_end].copy()
    del full_df
    print(f"IC分析数据范围: {train_df['日期'].min().date()} ~ {train_df['日期'].max().date()}")
    print(f"验证集排除范围: {train_end.date()} ~ {last_date.date()}")

    # ── 3. 股票映射 ──
    all_sids = sorted(train_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_sids)}
    print(f"训练集股票数: {len(all_sids)}")

    # ── 4. 特征工程（复用 train.py 的 _preprocess_common，内部调用 _build_label_and_clean） ──
    features_list = list(feature_columns_map[config['feature_num']])

    train_data, _ = _preprocess_common(
        train_df, stockid2idx, desc="IC分析特征工程", drop_small_open=True
    )

    # ── 5. 合并基本面因子 ──
    fp = os.path.join(data_path, 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(data_path, 'hs300_fundamentals.csv')
    train_data, fund_cols = _merge_fundamentals(train_data, fp)
    if fund_cols:
        features_list = features_list + fund_cols
        print(f"已合并基本面因子: {len(fund_cols)} 个")

    # ── 6. 数据清洗 ──
    train_data = train_data.replace([np.inf, -np.inf], np.nan)
    # label 列由 _build_label_and_clean 生成，即超额收益
    if 'label' not in train_data.columns:
        raise RuntimeError("特征工程后缺少 'label' 列，请检查 _preprocess_common")
    train_data = train_data.dropna(subset=['label'])
    train_data['日期'] = pd.to_datetime(train_data['日期'])

    # ── 7. 对每个因子计算逐日 Rank IC ──
    print(f"\n计算 {len(features_list)} 个因子的逐日 Rank IC ...")
    ic_results = []
    unique_dates = sorted(train_data['日期'].unique())
    print(f"有效交易日数: {len(unique_dates)}")

    for feat_idx, feat in enumerate(features_list):
        if feat not in train_data.columns:
            ic_results.append({
                'factor': feat, 'ic_mean': np.nan, 'ic_std': np.nan,
                't_stat': np.nan, 'icir': np.nan, 'n_days': 0,
            })
            continue

        daily_ics = []
        for date in unique_dates:
            day_data = train_data[train_data['日期'] == date].dropna(subset=[feat, 'label'])
            if len(day_data) < 30:  # 至少30只股票才有统计意义
                continue
            # Spearman 秩相关
            ic = day_data[[feat, 'label']].corr(method='spearman').iloc[0, 1]
            if not np.isnan(ic):
                daily_ics.append(ic)

        if len(daily_ics) >= 20:
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics, ddof=1)
            t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ics))) if ic_std > 0 else 0.0
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
        else:
            ic_mean = np.nan
            ic_std = np.nan
            t_stat = np.nan
            icir = np.nan

        ic_results.append({
            'factor': feat,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            't_stat': t_stat,
            'icir': icir,
            'n_days': len(daily_ics),
        })

        if (feat_idx + 1) % 50 == 0:
            print(f"  进度: {feat_idx + 1}/{len(features_list)}")

    # ── 8. 汇总筛选 ──
    ic_df = pd.DataFrame(ic_results)
    ic_df['significant'] = ic_df['t_stat'].abs() > 2
    significant_factors = ic_df[ic_df['significant']]['factor'].tolist()

    print(f"\n{'='*70}")
    print(f"  因子IC分析结果")
    print(f"{'='*70}")
    print(f"  总因子数: {len(ic_df)}")
    print(f"  显著因子数 (|t|>2): {len(significant_factors)}")
    print(f"  IC均值范围: [{ic_df['ic_mean'].min():+.4f}, {ic_df['ic_mean'].max():+.4f}]")
    print(f"  IC最强Top10（按|IC|降序）:")
    print(f"  {'因子':<25s} {'IC均值':>8s} {'t-stat':>8s} {'ICIR':>8s} {'显著':>6s}")
    print(f"  {'-'*55}")
    top10 = ic_df.dropna(subset=['ic_mean']).copy()
    top10 = top10.iloc[top10['ic_mean'].abs().argsort()[::-1]].head(10)
    for _, row in top10.iterrows():
        print(f"  {row['factor']:<25s} {row['ic_mean']:>+8.4f} {row['t_stat']:>+8.2f} "
              f"{row['icir']:>+8.3f} {str(row['significant']):>6s}")

    # ── 9. 打印显著因子列表（方便复制到 config.py） ──
    if significant_factors:
        print(f"\n{'='*70}")
        print(f"  显著因子列表 (共{len(significant_factors)}个)")
        print(f"  请将以下列表填入 config.py 的 selected_features：")
        print(f"{'='*70}")
        # 格式化为 Python list
        formatted = "selected_features = [\n"
        for i, f in enumerate(significant_factors):
            formatted += f"    '{f}'"
            if i < len(significant_factors) - 1:
                formatted += ","
            formatted += "\n"
        formatted += "]"
        print(formatted)

    # ── 10. 保存 ──
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ic_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存到: {output_path}")

    return ic_df, significant_factors


if __name__ == '__main__':
    mp.freeze_support()
    output_path = os.path.join(config['output_dir'], 'factor_ic_analysis.csv')
    ic_df, sig_factors = compute_factor_ic_analysis(output_path=output_path)
