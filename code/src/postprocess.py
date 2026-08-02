"""
TopK 选股 + softmax 赋权后处理模块

策略：每个交易日直接取 XGBRanker 预测分数最高的前 K 只股票，
通过 softmax 归一化分配持仓权重——高分多配、低分少配。

相比旧版 z-score 三层防御策略的优势：
- 每天稳定选出 K 只股票，不会因 z-threshold 过滤导致空仓
- 权重由模型相对排序直接决定，无需额外置信度分层逻辑
- 参数简洁：仅 top_k（选股数）和 temperature（softmax 温度）
"""

import numpy as np
import pandas as pd


def confidence_aware_postprocess(scores, stock_codes, top_k=5, params=None):
    """
    TopK 选股 + softmax 赋权后处理。

    直接取预测分数最高的前 top_k 只股票，用 softmax 归一化分配权重。
    保持与旧版相同的函数签名和返回格式，确保所有调用方兼容。

    Args:
        scores: np.array, 原始预测分数
        stock_codes: list, 对应的股票代码
        top_k: int, 选取的股票数量（默认 5）
        params: dict, 可选参数覆盖：
            - temperature: float, softmax 温度系数（默认 1.0）
              >1.0 → 权重更均匀；<1.0 → 头部集中

    Returns:
        selected_stocks: list, 入选股票代码（长度 = min(top_k, n)）
        weights: list, 对应权重（总和 = 1.0）
        conf_info: dict, 元信息 {
            'confidence_gap': float,  Top1 vs Top5 分数差距（保持向后兼容）
            'n_selected':   int,    实际选出股票数
            'top_k':        int,    配置的 top_k
            'temperature':  float,  softmax 温度
            'method':       str,    'topk_softmax'
        }
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return [], [], {
            'confidence_gap': 0.0, 'n_selected': 0,
            'top_k': top_k, 'temperature': 1.0, 'method': 'topk_softmax'
        }

    if params is None:
        params = {}
    temperature = float(params.get('temperature', 1.0))

    # ── Step 1: TopK 选股 ──
    k = min(top_k, n)
    topk_idx = np.argsort(scores)[::-1][:k]
    selected_scores = scores[topk_idx]
    selected_stocks = [stock_codes[i] for i in topk_idx]

    # ── Step 2: softmax 赋权（数值稳定版：减最大值防溢出） ──
    safe_temp = max(temperature, 1e-8)
    shifted = selected_scores - selected_scores.max()
    exp_scores = np.exp(shifted / safe_temp)
    weights = (exp_scores / exp_scores.sum()).tolist()

    # ── Step 3: 置信度 gap（保持向后兼容，供已有分析脚本参考） ──
    sorted_scores = np.sort(scores)[::-1]
    if len(sorted_scores) >= 5:
        confidence_gap = float(sorted_scores[0] - sorted_scores[4])
    elif len(sorted_scores) >= 2:
        confidence_gap = float(sorted_scores[0] - sorted_scores[-1])
    else:
        confidence_gap = 0.0

    conf_info = {
        'confidence_gap': confidence_gap,
        'n_selected': len(selected_stocks),
        'top_k': top_k,
        'temperature': temperature,
        'method': 'topk_softmax',
    }

    return selected_stocks, weights, conf_info


def compute_market_state_for_postprocess(df, latest_date=None):
    """
    计算后处理阶段的市场状态辅助指标。
    仅作为置信度判断的参考，非主要驱动。

    Args:
        df: DataFrame，需含 '日期'、'涨跌幅' 列
        latest_date: 最新日期（用于计算窗口终点），默认使用 df 中最大日期

    Returns:
        dict with: volatility_5d, trend_60d, ad_ratio
    """
    if '涨跌幅' not in df.columns or '日期' not in df.columns:
        return {'volatility_5d': 0.01, 'trend_60d': 0.0, 'ad_ratio': 1.0}

    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    market_daily = df.groupby('日期')['涨跌幅'].mean().sort_index()

    if latest_date is None:
        latest_date = df['日期'].max()

    # 5日波动率
    vol_5d = float(
        market_daily.rolling(5, min_periods=3).std().iloc[-1]
    ) if len(market_daily) >= 5 else 0.01
    if np.isnan(vol_5d):
        vol_5d = 0.01

    # 60日趋势
    trend_60d_series = market_daily.rolling(60, min_periods=10).sum()
    trend_60d = float(trend_60d_series.iloc[-1]) if len(trend_60d_series) > 0 else 0.0
    if np.isnan(trend_60d):
        trend_60d = 0.0

    # 近5日涨跌比
    recent = df[df['日期'] >= pd.to_datetime(latest_date) - pd.DateOffset(days=30)]
    if len(recent) > 0:
        daily_ret = recent.groupby('日期')['涨跌幅'].mean()
        up_days = int((daily_ret > 0).sum())
        down_days = int((daily_ret < 0).sum())
        ad_ratio = up_days / max(down_days, 1)
    else:
        ad_ratio = 1.0

    return {
        'volatility_5d': vol_5d,
        'trend_60d': trend_60d,
        'ad_ratio': float(ad_ratio),
    }



