"""
多维评估模块
提供扩展的排序评估指标和报告格式化功能。
不修改 train.py 中的训练循环，仅作为额外的评估层。

指标说明：
- topk_hit_rate  : 预测 Top k 中落入真实 Top k 的股票数 / k，跨日取均值
- spearman_rho   : 每日所有有效股票的预测分与真实 label 的 Spearman 秩相关，跨日取均值
- win_rate       : 每日模型 Top k 均收益 > 当日全部股票均收益的天数占比
- final_score_std: 每日 final_score 的样本标准差
- max_daily_loss : 所有天中模型 Top k 收益最低的值
- valid_days_ratio: 过滤低信息量日后保留的天数占比
"""

import numpy as np
import torch
from typing import Optional


def _spearman_rho_pytorch(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    使用纯 PyTorch 计算 Spearman 秩相关系数，不依赖 scipy。

    Args:
        pred: 预测分数，shape (n,)
        true: 真实标签，shape (n,)

    Returns:
        秩相关系数，范围 [-1, 1]
    """
    n = len(pred)
    if n < 2:
        return 0.0

    # 对 pred 和 true 分别求秩（argsort 两次等价于 rank，处理并列值用 average）
    # 使用两次 argsort 获得稳定的 rank（最小值为 0）
    pred_rank = torch.argsort(torch.argsort(pred)).float()
    true_rank = torch.argsort(torch.argsort(true)).float()

    pred_mean = pred_rank.mean()
    true_mean = true_rank.mean()

    # Pearson correlation on ranks
    pred_centered = pred_rank - pred_mean
    true_centered = true_rank - true_mean

    cov = (pred_centered * true_centered).sum()
    std = pred_centered.norm() * true_centered.norm()

    if std < 1e-12:
        return 0.0
    return (cov / std).item()


def calculate_extended_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    masks: torch.Tensor,
    k: int = 5,
    min_gap: float = 0.005,
) -> dict:
    """
    相对于原有 calculate_ranking_metrics 的扩展版本。
    在原指标基础上，增加 topk_hit_rate、spearman_rho、win_rate 等。

    Args:
        y_pred: 模型预测分数，shape (batch_size, max_stocks)
        y_true: 真实涨跌幅，shape (batch_size, max_stocks)
        masks: 有效位置 mask，shape (batch_size, max_stocks)
        k: Top-k 值
        min_gap: final_score 分母过滤阈值

    Returns:
        扩展指标字典
    """
    batch_size = y_pred.size(0)

    # 累加器
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    topk_hit_list = []          # 每天预测 Top k 命中真实 Top k 的数量
    spearman_list = []          # 每天的 Spearman 秩相关
    win_list = []               # 每天是否跑赢全市场均值（1=赢, 0=输）

    num_total_days = 0
    num_valid_days = 0

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero(as_tuple=True)[0]

        if valid_indices.numel() < k:
            continue

        num_total_days += 1

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        # ---- 第一层过滤：跳过低信息量交易日 ----
        _, true_topk_local = torch.topk(valid_true, min(k, len(valid_true)))
        true_top_returns = valid_true[true_topk_local]
        max_return_sum = true_top_returns.sum().item()
        random_return_sum = k * valid_true.mean().item()
        gap = max_return_sum - random_return_sum

        if abs(gap) < min_gap:
            continue
        # ---- 过滤结束 ----

        num_valid_days += 1

        # Predicted Top k
        _, pred_indices = torch.topk(valid_pred, min(k, len(valid_pred)))
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        # ---- 原有指标 ----
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)

        denominator = max_return_sum - random_return_sum
        fs = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        final_score_list.append(fs)

        # ---- 新增指标：Top k 命中率 ----
        true_k_set = set(true_topk_local.cpu().numpy().tolist())
        pred_k_set = set(pred_indices.cpu().numpy().tolist())
        hit_count = len(true_k_set & pred_k_set)
        topk_hit_list.append(hit_count)

        # ---- 新增指标：Spearman 秩相关 ----
        sp = _spearman_rho_pytorch(valid_pred, valid_true)
        spearman_list.append(sp)

        # ---- 新增指标：胜率 ----
        market_mean = valid_true.mean().item()
        model_mean = pred_top_returns.mean().item()
        win_list.append(1.0 if model_mean > market_mean else 0.0)

    # 汇总
    n = num_valid_days
    n_total = max(num_total_days, 1)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if n > 0 else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if n > 0 else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if n > 0 else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if n > 0 else 0.0,
        'ratio_random': np.mean(ratio_random_list) if n > 0 else 0.0,
        'final_score': np.mean(final_score_list) if n > 0 else 0.0,
        # 新增指标
        'topk_hit_rate': np.mean(topk_hit_list) / k if n > 0 else 0.0,
        'topk_hit_count': np.mean(topk_hit_list) if n > 0 else 0.0,
        'spearman_rho': np.mean(spearman_list) if n > 0 else 0.0,
        'win_rate': np.mean(win_list) if n > 0 else 0.0,
        'final_score_std': np.std(final_score_list, ddof=1) if n > 1 else 0.0,
        'max_daily_loss': np.min(pred_return_sum_list) if n > 0 else 0.0,
        'valid_days_ratio': n / n_total,
        'valid_days': n,
        'total_days': num_total_days,
    }

    return metrics


def _pad_to_width(text: str, width: int, align: str = '<') -> str:
    """辅助：按指定宽度填充中文文本（考虑中文字符占2个宽度）。"""
    text_width = sum(2 if ord(c) > 127 else 1 for c in text)
    padding = max(0, width - text_width)
    if align == '<':
        return text + ' ' * padding
    elif align == '>':
        return ' ' * padding + text
    else:
        left = padding // 2
        return ' ' * left + text + ' ' * (padding - left)


def format_eval_report(metrics: dict) -> str:
    """
    将扩展指标格式化为可读的评估报告字符串。

    Args:
        metrics: calculate_extended_metrics 返回的指标字典

    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("")
    lines.append("=" * 56)
    lines.append("           扩展评估报告 (Extended Evaluation Report)")
    lines.append("=" * 56)

    field_width = 20
    for key, label in [
        ('final_score', 'Final Score'),
        ('ratio_pred', 'Ratio vs Max'),
        ('topk_hit_rate', 'TopK Hit Rate'),
        ('spearman_rho', 'Spearman Rho'),
        ('win_rate', 'Win Rate'),
        ('final_score_std', 'FS Std Dev'),
        ('max_daily_loss', 'Max Daily Loss'),
        ('valid_days_ratio', 'Valid Days Ratio'),
        ('valid_days', 'Valid Days'),
        ('total_days', 'Total Days'),
    ]:
        value = metrics.get(key, None)
        if value is not None:
            label_padded = _pad_to_width(label + ':', field_width)
            if isinstance(value, float):
                lines.append(f"  {label_padded} {value:>10.6f}")
            else:
                lines.append(f"  {label_padded} {str(value):>10}")

    lines.append("=" * 56)
    return '\n'.join(lines)
