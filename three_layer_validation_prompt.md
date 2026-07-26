# BDC 三层验证体系改造提示词

## 核心约束（必须遵守）

1. **不改变 `train.py` 中的训练循环**：模型的 forward、反向传播、优化器更新、学习率调度的逻辑保持原样不动。
2. **不改变项目目录结构**：不新增目录，只在 `code/src/` 下新增一个 `evaluation.py` 文件、一个 `cross_val.py` 脚本，以及在 `train.py` 中对 `calculate_ranking_metrics` 做最小化增强（不改接口签名，只增加统计量）。
3. **不改变 `config.py` 中已有的任何参数**：只能新增配置项，已有配置项的键和默认值不动。

---

## 第一层：修复 `calculate_ranking_metrics` 的分母不稳定问题

### 问题

当前 `final_score = (pred_return_sum - random_return_sum) / (max_return_sum - random_return_sum)`。在窄幅震荡日（所有股票收益都差不多），分母接近 0，导致 `final_score` 估值极大且不可靠，污染日均值。

### 修改方案

在 `train.py` 的 `calculate_ranking_metrics` 函数中，新增一个过滤逻辑：

```python
def calculate_ranking_metrics(y_pred, y_true, masks, k=5, min_gap=0.005):
    # ... 原有代码不变 ...
    
    for i in range(batch_size):
        # ... 原有代码不变 ...
        
        # ---- 新增：跳过低信息量交易日 ----
        max_return_sum = true_top_returns.sum().item()
        random_return_sum = k * valid_true.mean().item()
        gap = max_return_sum - random_return_sum
        
        # 如果理论最优和随机期望的差距小于阈值，这一天不计入统计
        if abs(gap) < min_gap:
            continue
        # ---- 新增结束 ----
        
        # 下面原有代码完全不动
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        # ...
```

同时，在该批次循环结束后，增加一个被过滤天数的计数器，方便观察有多少比例的低信息量日被排除：

```python
# 在 metrics 字典中新增：
metrics['valid_days_ratio'] = len(pred_return_sum_list) / num_total_days  # 保留比例
```

`min_gap` 的默认值 `0.005` 代表最优与随机之间的差距必须超过 0.5%（50 个基点），这是一个合理的初始值，可由用户通过 config 调整。

---

## 第二层：构建多维度评估体系

### 问题

当前训练过程中只记录 `final_score`、`ratio_pred` 等原始指标，缺少以下维度的信息：
- 排序一致性（模型排序与真实排序的整体吻合度）
- Top-k 命中率（预测 Top 5 中有多少进入了真实 Top 5）
- 胜率（模型 Top 5 跑赢全市场均值的频率）

### 修改方案

**在 `code/src/` 下新增 `evaluation.py` 文件**，内容包含：

1. **`calculate_extended_metrics(y_pred, y_true, masks, k=5, min_gap=0.005)`**
   - 调用原有的 `calculate_ranking_metrics` 获得基础指标
   - 在此基础上新增以下指标：

   | 新增指标 | 含义 | 计算方式 |
   |---------|------|----------|
   | `topk_hit_rate` | Top 5 命中率 | 预测 Top 5 中落入真实 Top 5 的股票数 / 5，跨日取均值 |
   | `spearman_rho` | 排序相关系数 | 对每天所有有效股票的预测分和真实 label 计算 Spearman 秩相关，跨日取均值 |
   | `win_rate` | 胜率 | 每日模型 Top 5 均收益 > 当日全部股票均收益的天数占比 |
   | `final_score_std` | Final Score 标准差 | 每日 final_score 的样本标准差 |
   | `max_daily_loss` | 最差单日表现 | 所有天中模型 Top 5 收益最低的值 |
   | `valid_days_ratio` | 有效交易日比例 | 过滤低信息量日后保留的天数占比 |

2. **`format_eval_report(metrics_dict)`**
   - 将上述指标格式化为可读的字符串报告，方便在训练结束时打印或写入日志文件。

**在 `train.py` 中的改动**（不影响训练循环）：

- 在 `train.py` 顶部增加 `from evaluation import calculate_extended_metrics, format_eval_report`
- 在 `evaluate_ranking_model` 函数中，每完成一个 epoch 的评估后，调用 `calculate_extended_metrics` 获取扩展指标
- 将扩展指标也写入 TensorBoard（`writer.add_scalar(f'eval/{k}', v, epoch)`），与原有逻辑保持一致
- 同时保持原有 `calculate_ranking_metrics` 在训练循环中的调用不变，确保训练过程不受影响

**在训练结束时的输出增强**：

```python
# 训练结束后，在 main() 的末尾（保存完 best_model.pth 和 final_score.txt 之后）
eval_report = format_eval_report(extended_metrics)
print(eval_report)
with open(os.path.join(output_dir, 'eval_report.txt'), 'w') as f:
    f.write(eval_report)
```

### Spearman 秩相关系数的实现注意

`spearman_rho` 的计算不依赖 `scipy`，直接用 PyTorch 实现，避免增加依赖：

```python
def _spearman_rho_pytorch(pred, true):
    """计算 Spearman 秩相关系数，使用 PyTorch 实现"""
    # 对 pred 和 true 分别求秩（argsort 两次得到 rank）
    pred_rank = torch.argsort(torch.argsort(pred)).float()
    true_rank = torch.argsort(torch.argsort(true)).float()
    n = len(pred)
    # Pearson correlation on ranks
    pred_mean = pred_rank.mean()
    true_mean = true_rank.mean()
    cov = ((pred_rank - pred_mean) * (true_rank - true_mean)).sum()
    std = (pred_rank - pred_mean).norm() * (true_rank - true_mean).norm()
    return (cov / (std + 1e-12)).item()
```

---

## 第三层：时间序列滚动窗口验证脚本

### 问题

当前只用最后一个月做单次验证，无法得知模型在不同市场阶段（趋势市 vs 震荡市、牛市 vs 熊市）是否稳定。你无法回答"这个超参数配置是真的好，还是恰好在这一个月好"。

### 修改方案

**在 `code/src/` 下新增 `cross_val.py` 文件**，这是一个**完全独立的脚本**，不修改 `train.py` 的任何逻辑。它的行为是：

1. 读取来自命令行参数或脚本内硬编码的窗口定义
2. 对每个窗口，调用 `train.py` 的核心函数（通过 `import train` 方式复用）完成一次独立的"训练 + 评估"
3. 汇总所有窗口的评估结果，输出一个汇总表

核心实现思路：

```python
"""
滚动窗口交叉验证脚本
用法：python code/src/cross_val.py
输出：model/{config_name}/cross_val_report.txt
"""
import pandas as pd
import numpy as np
import torch
import multiprocessing as mp
from train import (
    main as train_single,  # 或拆出一个 train_single_window 函数
    set_seed, preprocess_data, preprocess_val_data,
    RankingDataset, collate_fn, train_ranking_model, evaluate_ranking_model,
)
from train import config as train_config
from evaluation import calculate_extended_metrics
import json

# 窗口定义
# 每个窗口：(train_start, train_end, val_start, val_end, label)
WINDOWS = [
    ("2024-01-02", "2025-08-29", "2025-09-01", "2025-10-31", "窗口1_早期"),
    ("2024-05-06", "2025-12-31", "2026-01-02", "2026-02-28", "窗口2_中期"),
    ("2024-09-02", "2026-02-19", "2026-02-20", "2026-03-06", "窗口3_近期"),
]

def run_single_window(train_start, train_end, val_start, val_end, label, config):
    """
    对单个窗口执行完整的训练+评估流程。
    复用 train.py 中的函数，但数据范围由窗口定义控制。
    """
    # 1. 加载并切片数据
    # 2. 复用 train.py 的特征工程和数据集构建
    # 3. 复用 train.py 的模型初始化和训练循环
    # 4. 复用 train.py 的评估函数（现在会调用 calculate_extended_metrics）
    # 5. 返回该窗口的全部指标
    pass
```

### 关键设计

**不要复制粘贴 `train.py` 的代码**。`cross_val.py` 应该通过 `import` 复用 `train.py` 中已有的函数。如果 `train.py` 当前的 `main()` 函数耦合太紧（从加载数据到训练全在一个函数里），只需要对 `train.py` 做一个最低限度的重构——把 `main()` 中的核心逻辑提取为一个接受参数的函数。改动方式：

```python
# train.py 中，在现有 main() 之前增加一个参数化版本：
def train_one_window(train_df, val_df, val_start, stockid2idx, num_stocks, config):
    """
    参数化的单窗口训练函数。
    所有逻辑从原 main() 中提取，不改变原有行为。
    """
    # 内容与原 main() 中从第 2 步到训练结束的逻辑完全一致
    ...

def main():
    """保持原有的 main 函数签名和行为完全不变"""
    # ... 原有代码不动 ...
    # 内部调用 train_one_window
    best_score = train_one_window(train_df, val_df, val_start, ...)
    return best_score
```

这样 `cross_val.py` 可以直接调用 `train_one_window`，而原有的 `sh train.sh` 运行方式完全不受影响。

### 汇总输出

`cross_val.py` 运行完毕后，打印并保存如下格式的汇总表：

```
═══════════════════════════════════════════════════════════════
               滚动窗口交叉验证汇总报告
═══════════════════════════════════════════════════════════════
指标            窗口1_早期  窗口2_中期  窗口3_近期  均值   标准差
───────────────────────────────────────────────────────────────
final_score      0.4521     0.3892     0.5103     0.4505  0.0494
ratio_pred       0.6721     0.5893     0.7156     0.6590  0.0522
topk_hit_rate    2.12       1.89       2.34       2.12    0.18
spearman_rho     0.0834     0.0712     0.0918     0.0821  0.0084
win_rate         0.6123     0.5678     0.6341     0.6047  0.0276
valid_days_ratio 0.8912     0.9034     0.8765     0.8904  0.0110
───────────────────────────────────────────────────────────────
最低窗口 final_score: 0.3892 (窗口2_中期)
final_score 极差: 0.1211 (最低窗口与最高窗口的差距)
═══════════════════════════════════════════════════════════════
稳定性判断：final_score 标准差 / 均值 = 10.97%，配置较稳定
═══════════════════════════════════════════════════════════════
```

### 配置扩展

在 `config.py` 中新增以下可选项（不修改任何已有键）：

```python
# 验证增强配置（新增项，不影响已有逻辑）
config_extended = {
    'min_gap': 0.005,              # 第一层：filnal_score 分母过滤阈值
    'eval_top_k': 5,               # 第二层：Top-k 命中率的 k
    'cross_val_windows': [         # 第三层：滚动窗口定义
        ("2024-01-02", "2025-08-29", "2025-09-01", "2025-10-31"),
        ("2024-05-06", "2025-12-31", "2026-01-02", "2026-02-28"),
        ("2024-09-02", "2026-02-19", "2026-02-20", "2026-03-06"),
    ],
}
```

---

## 文件变更清单

| 文件 | 操作 | 改动量 | 说明 |
|------|------|--------|------|
| `code/src/evaluation.py` | **新增** | ~150 行 | 多维评估指标函数 + 报告格式化 |
| `code/src/cross_val.py` | **新增** | ~200 行 | 滚动窗口交叉验证脚本 |
| `code/src/config.py` | 追加 | ~10 行 | 新增 `config_extended` 字典 |
| `code/src/train.py` | 最小修改 | ~30 行改动 | `calculate_ranking_metrics` 增加过滤逻辑；`evaluate_ranking_model` 增加扩展指标计算；提取 `train_one_window`；训练结束打印扩展报告 |
| `code/src/predict.py` | 不改 | 0 行 | 推理逻辑不受影响 |
| `code/src/model.py` | 不改 | 0 行 | 模型定义不受影响 |
| `code/src/utils.py` | 不改 | 0 行 | 特征工程不受影响 |

---

## 验收标准

改完后按以下顺序验证：

1. **`sh train.sh` 仍能正常运行**，训练逻辑不变，输出 model 目录结构与原来完全一致。
2. **`final_score.txt` 的值应该与改动前基本一致**（因为只加了过滤，过滤掉的天数少所以均值变化小），但额外出现了 `eval_report.txt`。
3. **TensorBoard 日志中出现了新指标**：`eval/topk_hit_rate`、`eval/spearman_rho`、`eval/win_rate`、`eval/final_score_std`、`eval/valid_days_ratio`。
4. **`python code/src/cross_val.py` 可独立运行**，遍历所有窗口，输出汇总表到 `model/{config_name}/cross_val_report.txt`。
5. **`sh test.sh` 仍能正常运行**，推理产出的 `output/result.csv` 格式不变。
