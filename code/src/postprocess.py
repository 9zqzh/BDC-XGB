"""
置信度驱动的预测后处理模块

核心洞察（来自窗口4月度分析）：验证期连续12个月横盘，但模型表现月度间剧烈波动
（win_rate 17%↔95%）。说明模型自身的"自信程度"比市场状态更有区分度。

置信度定义：同一天所有股票的预测 z-score 分布中，Top1 与 Top5 的 z-score 差距。
- 高置信度（gap > 2.0）：模型明确区分出鹤立鸡群的股票 → 满仓5只
- 中置信度（1.0 < gap < 2.0）：模型有一些偏好但不够自信 → 中等仓位
- 低置信度（0.5 < gap < 1.0）：模型不太确定 → 少选
- 极低置信度（gap < 0.5）：模型在猜 → 几乎空仓（最坏情况预案）
"""

import numpy as np
import pandas as pd


def confidence_aware_postprocess(scores, stock_codes, top_k=5, params=None):
    """
    基于模型预测置信度的后处理：z-score 标准化 + 置信度阈值 + softmax 权重。

    置信度定义：Top1 与 Top5 的 z-score 差距。
    差距大 = 模型明确知道哪些股票更好 = 高置信度。
    差距小 = 所有股票分数接近 = 模型在"猜" = 低置信度。

    Args:
        scores: np.array, 原始预测分数
        stock_codes: list, 对应的股票代码
        top_k: int, 最多选取的股票数量
        params: dict, 可选参数覆盖，支持以下键：
            - gap_thresholds: [high, mid, low] 置信度分界 (默认 [2.0, 1.0, 0.5])
            - n_selects: [high, mid, low, min] 各档选取数 (默认 [5, 4, 2, 1])
            - z_thresholds: [high, mid, low, min] z-score 入选阈值 (默认 [0.5, 1.0, 1.5, 2.0])
            - temperatures: [high, mid, low, min] softmax 温度 (默认 [1.0, 0.7, 0.3, 0.1])

    Returns:
        selected_stocks: list, 入选股票代码
        weights: list, 对应权重（总和≤1）
        conf_info: dict, 置信度元信息（方便复盘）
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return [], [], {'confidence_gap': 0.0, 'n_selected': 0, 'z_threshold_used': 0.0}

    # ── Step 1: z-score 标准化 ──
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    if std_score < 1e-8:
        z_scores = np.zeros_like(scores)
    else:
        z_scores = (scores - mean_score) / std_score

    # ── Step 2: 按 z-score 降序排列 ──
    sorted_idx = np.argsort(z_scores)[::-1]
    sorted_z = z_scores[sorted_idx]
    sorted_stocks = [stock_codes[i] for i in sorted_idx]

    # ── Step 3: 计算置信度 gap（Top1 vs Top5） ──
    if len(sorted_z) >= 5:
        confidence_gap = float(sorted_z[0] - sorted_z[4])
    elif len(sorted_z) >= 2:
        confidence_gap = float(sorted_z[0] - sorted_z[-1])
    else:
        confidence_gap = 0.0

    # ── Step 4: 根据置信度决定选取策略 ──
    if params is None:
        params = {}
    # 默认参数已通过 rolling_val.py --tune 两阶段调优（gap=0.15, temp=0.5, z=0.5）
    gap_thresholds = params.get('gap_thresholds', [0.15, 0.075, 0.0375])
    n_selects = params.get('n_selects', [5, 4, 2, 1])
    z_thresholds = params.get('z_thresholds', [0.5, 1.0, 1.5, 2.0])
    temperatures = params.get('temperatures', [0.5, 0.35, 0.15, 0.05])

    if confidence_gap > gap_thresholds[0]:
        # 高置信度：模型有明确判断
        n_select = min(n_selects[0], n)
        z_threshold = z_thresholds[0]
        temperature = temperatures[0]
    elif confidence_gap > gap_thresholds[1]:
        # 中置信度：模型有一定把握
        n_select = min(n_selects[1], n)
        z_threshold = z_thresholds[1]
        temperature = temperatures[1]
    elif confidence_gap > gap_thresholds[2]:
        # 低置信度：模型不太确定
        n_select = min(n_selects[2], n)
        z_threshold = z_thresholds[2]
        temperature = temperatures[2]
    else:
        # 极低置信度：模型在猜，几乎不选（最坏情况预案）
        n_select = min(n_selects[3], n)
        z_threshold = z_thresholds[3]
        temperature = temperatures[3]

    # ── Step 5: 按 z_threshold 筛选 ──
    qualified_mask = sorted_z >= z_threshold
    qualified_idx = np.where(qualified_mask)[0]

    if len(qualified_idx) == 0:
        # 没有股票过阈值 → 至少选一只（即使低置信度）
        selected_idx = np.array([0])
    else:
        selected_idx = qualified_idx[:n_select]

    selected_z = sorted_z[selected_idx]
    selected_stocks = [sorted_stocks[i] for i in selected_idx]

    # ── Step 6: softmax 权重分配 ──
    safe_temp = max(temperature, 1e-8)
    exp_z = np.exp(selected_z / safe_temp)
    weights = (exp_z / exp_z.sum()).tolist()

    conf_info = {
        'confidence_gap': confidence_gap,
        'n_selected': len(selected_stocks),
        'z_threshold_used': z_threshold,
        'temperature': temperature,
        'top1_z': float(sorted_z[0]) if len(sorted_z) > 0 else 0.0,
        'top5_z': float(sorted_z[4]) if len(sorted_z) >= 5 else 0.0,
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


def apply_market_overlay(n_select, temperature, confidence_gap, market_state):
    """
    市场状态作为辅助参考，微调置信度驱动的参数。
    仅在高波动 + 低置信度 的组合下触发防御性降仓。

    Args:
        n_select: 当前计划选取数量
        temperature: 当前 softmax 温度
        confidence_gap: 置信度 gap
        market_state: compute_market_state_for_postprocess 的返回值

    Returns:
        (n_select, temperature): 调整后的参数
    """
    vol = market_state.get('volatility_5d', 0.01)

    if vol > 0.025 and confidence_gap < 1.0:
        # 高波动 + 模型不确定 → 额外降仓
        n_select = max(1, n_select - 1)
        temperature = temperature * 0.5

    return n_select, temperature
