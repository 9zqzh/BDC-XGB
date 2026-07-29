# BDC(XGB) 项目优化操作指南

> **文档性质**：面向 AI 的项目修改操作指南
> **生成日期**：2026-07-29
> **数据依据**：窗口4月度切分分析（2025-07 ~ 2026-06，237个交易日）
> **目标赛事窗口**：2026-08-03 至 2026-08-07（5个交易日）
> **当前市场状态**：持续性横盘震荡（验证期12个月市场月度累计收益最高+0.53%，最低-1.58%，无明确方向）
> **核心模型**：XGBRanker（objective=rank:ndcg, eval_metric=ndcg@5）

---

## 目录

1. [项目现状诊断](#1-项目现状诊断)
2. [窗口4月度分析：关键发现](#2-窗口4月度分析关键发现)
3. [核心教训：市场状态特征的无效性与修正](#3-核心教训市场状态特征的无效性与修正)
4. [各调优方向最终判定（修正版）](#4-各调优方向最终判定修正版)
5. [P0 最高优先：因子IC分析——筛选有效特征](#5-p0-最高优先因子ic分析筛选有效特征)
6. [P0 最高优先：置信度驱动的后处理改造](#6-p0-最高优先置信度驱动的后处理改造)
7. [P1 高优先：方向分类器辅助特征](#7-p1-高优先方向分类器辅助特征)
8. [P1 高优先：50次滚动验证脚本](#8-p1-高优先50次滚动验证脚本)
9. [P2 中优先：精选特征交叉](#9-p2-中优先精选特征交叉)
10. [P3 低优先：超参数重搜 + BUG修复](#10-p3-低优先超参数重搜--bug修复)
11. [P4 未来方向：多模型融合](#11-p4-未来方向多模型融合)
12. [验证框架设计](#12-验证框架设计)
13. [开发约束与注意事项](#13-开发约束与注意事项)

---

## 1. 项目现状诊断

### 1.1 当前模型配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 模型 | XGBRanker | `objective='rank:ndcg'` |
| 特征方案 | 158+39 | 158个Alpha因子 + 39个技术指标 |
| 特征维度 | 10天×221维 + 7市场 = 2,217维 | `xgb_flatten_days=10` |
| max_depth | 5 | 树最大深度 |
| learning_rate | 0.03 | 学习率 |
| n_estimators | 300 | 树数量 |
| subsample | 0.6 | 行采样 |
| colsample_bytree | 0.3 | 列采样 |
| reg_alpha | 1.0 | L1正则化 |
| reg_lambda | 5.0 | L2正则化 |
| min_child_weight | 10 | 最小叶子权重 |
| 验证方案 | 末尾12个月 | `val_months=12` |
| 标签 | 未来5日超额收益 | `(open_t5 - open_t1) / open_t1 - daily_mean` |

### 1.2 四窗口交叉验证指标

| 指标 | 全量训练 | 窗口1(早期回调) | 窗口2(弱复苏) | 窗口3(924行情) | 窗口4(近期市场) | 加权均值 |
|------|---------|----------------|--------------|---------------|----------------|---------|
| final_score | **0.0068** | **-0.0035** | 0.0511 | 0.0192 | 0.0369 | 0.0259 |
| topk_hit_rate | 0.0101 | 0.0151 | 0.0188 | 0.0038 | 0.0146 | 0.0131 |
| spearman_rho | 0.0557 | 0.1213 | 0.0518 | 0.0325 | 0.0455 | 0.0628 |
| win_rate | 0.5196 | 0.6038 | 0.6875 | 0.5192 | 0.5281 | 0.5846 |
| final_score_std | 0.1279 | 0.2570 | 0.1762 | 0.1392 | 0.1602 | 0.1832 |

**跨窗口稳定性**：final_score 变异系数 = **78.77%**（>50%，判定为不稳定）

### 1.3 窗口4月度切分数据（2025-07 ~ 2026-06，237个交易日）

| 月份 | 天数 | fs_mean | win_rate | topk_hit | spearman | pred_ret | mkt_cum | 评价 |
|------|------|---------|----------|----------|----------|----------|---------|------|
| 2025-07 | 23 | 0.0063 | 0.5217 | 0.0174 | -0.0520 | 0.0013 | +0.0053 | 偏差 |
| 2025-08 | 21 | **-0.0575** | 0.1905 | 0.0000 | -0.1673 | -0.0151 | -0.0008 | **严重亏损** |
| 2025-09 | 22 | **-0.0765** | 0.2273 | 0.0000 | -0.0714 | -0.0146 | +0.0046 | **严重亏损** |
| 2025-10 | 17 | **0.0902** | 0.6471 | 0.0235 | 0.1734 | 0.0117 | -0.0001 | **良好** |
| 2025-11 | 20 | **0.1685** | 0.8500 | 0.0600 | 0.1532 | 0.0167 | -0.0003 | **良好** |
| 2025-12 | 23 | **-0.0823** | 0.1739 | 0.0087 | -0.1338 | -0.0095 | +0.0042 | **严重亏损** |
| 2026-01 | 20 | 0.0028 | 0.4500 | 0.0100 | 0.0029 | -0.0016 | +0.0021 | 偏差 |
| 2026-02 | 14 | **-0.0506** | 0.2857 | 0.0000 | 0.0118 | -0.0076 | -0.0048 | **严重亏损** |
| 2026-03 | 22 | **0.1399** | 0.9545 | 0.0091 | 0.1813 | 0.0199 | -0.0010 | **良好** |
| 2026-04 | 21 | 0.0060 | 0.5238 | 0.0000 | -0.0482 | -0.0008 | -0.0000 | 偏差 |
| 2026-05 | 18 | 0.0567 | 0.7222 | 0.0111 | 0.1194 | 0.0104 | -0.0000 | **良好** |
| 2026-06 | 16 | 0.0489 | 0.5625 | 0.0125 | 0.0072 | 0.0123 | -0.0158 | 一般 |
| **合计** | **237** | **0.0195** | **0.5063** | **0.0127** | **0.0089** | **0.0016** | — | — |

**关键统计**：
- 正收益月份：8/12（67%）
- 亏损月份：2025-08, 2025-09, 2025-12, 2026-02
- 最差月份：2025-12（fs=-0.0823, wr=17.4%）
- 最佳月份：2025-11（fs=0.1685, wr=85.0%）
- 月度 final_score 标准差：0.0826
- **月度间变异系数：423.5%**

### 1.4 五个核心问题

1. **排序能力趋近于零**：Spearman 整体 0.0089。这不是"比较弱"，是统计上等于零。好的月份（2025-10/11, 2026-03）Spearman 也只有 0.12-0.18，差月份（2025-08, 2025-12）是负数——**模型选的是反向排序**。
2. **TopK命中率几乎为零**：全年 0.0127，即 237 天中平均每天命中 0.06 只（5只中）。好月份也只有 0.06，差月份直接是 0——一次都没命中过。
3. **模型是单因子/单风格过拟合**：相邻月份间 win_rate 从 17% 跳到 95%，final_score 从 -0.08 跳到 +0.17，但市场状态完全一样（都是横盘）。模型学的不是泛化的选股信号，而是某些在特定月份有效的噪声模式。
4. **市场状态特征完全无效**：7维市场特征全部低于Top20重要性阈值——不是因为特征质量差，而是数据结构决定了它们不可分裂（同一天所有股票共享相同值，组内方差为零）。
5. **特征中80%可能是噪声**：221个特征因子（158 Alpha + 39 技术指标 + 基本面 + 行业）中很多高度共线（如 MA5/MA10/MA20，ROC5/ROC10/ROC20）。高维噪声淹没了少数有效信号。

### 1.5 已知BUG

- **`config.py` 中 max_depth=5，但 `xgb_tune.py` 锁定 max_depth=8** — 调参脚本搜索出的最优参数组合（colsample=0.4, subsample=0.5）是基于 max_depth=8 的结果，实际训练却用 max_depth=5。在不同深度下，最优 colsample 和 subsample 完全不同。
- **`optuna_search.py` 完全不兼容当前 XGBRanker** — 该脚本是为旧版 StockTransformer（PyTorch）编写的，调用 `train_one_window` 的方式与当前 XGBRanker 签名不匹配。

---

## 2. 窗口4月度分析：关键发现

### 2.1 整个验证期市场都是横盘——"牛熊适配"的假设破产

验证期12个月中，月度累计市场收益（mkt_cum）最高 +0.53%，最低 -1.58%。**全部是横盘，没有一个月的市场有明确方向。**

这意味着两个方面：

**第一，此前设计的"市场-regime驱动的自适应后处理"方案在该验证期内完全无法验证**——因为市场从未发生过显著的regime切换。模型在2025-11（fs=0.17, wr=85%）和2025-12（fs=-0.08, wr=17%）之间的巨大波动，发生在完全相同的横盘市场中，与市场方向无关。

**第二，比赛窗口（8月3日-7日）大概率也是横盘延续**，历史数据显示模型在横盘期表现极不稳定，这是最大风险。

### 2.2 模型有弱方向感但无排序能力

关键证据：好月份（2025-10/11, 2026-03）的 win_rate 高达 65-95%，但 Spearman 只有 0.12-0.18。

**这揭示了一个重要的非对称性**：模型能大致判断"哪些股票会涨"（方向判断），但无法区分"涨得多和涨得少"（排序能力）。TopK命中率始终趋近于零也印证了这一点——模型选的Top5从来不是真实Top5，但有时候这5只确实涨了。

这为"方向分类器辅助特征"提供了清晰的定位：专门强化方向判断能力，让 XGBRanker 在方向确定的基础上做排序。

### 2.3 月度波动的根因是特征噪声过拟合

好月份和差月份之间市场状态完全相同（都是横盘），但 win_rate 从 17% 跳到 95%。说明模型过拟合到了某些只在特定月份有效的噪声模式。

过拟合的数学原因：2,210 维特征 × 约260天训练数据。当大部分特征是噪声时，XGBoost 有足够多的候选分裂点来"记忆"训练集的随机模式——这些模式换到下个月就失效。**降维（因子IC分析→只保留有效特征）是解决这个问题的关键。**

---

## 3. 核心教训：市场状态特征的无效性与修正

### 3.1 问题本质

市场状态特征（如"今天是牛市还是熊市"、"全市场波动率"）在 XGBRanker 中**不可能被选中做分裂**。原因不是特征质量差，而是**数据结构决定的**：

> 同一个交易日的所有股票共享完全相同的市场状态值 → 该特征在组内（同一个 qid）的方差为零 → XGBoost 树分裂需要特征在组内有方差来区分不同样本 → **该特征不可分裂**。

这与特征工程质量无关。无论怎么编码它们，只要仍然是"同一天所有股票共享的全局值"，就永远不会被 XGBoost 选中做分裂。

**2026-07-29 实测确认**：当前全量训练的 Top20 特征重要性中，7维市场状态特征全部未上榜。

```
Top20 最低阈值: 0.002240 (T-1天_MAX60)
市场_当日          < 0.0022
市场_5日波动        < 0.0022
市场_5日均          < 0.0022
市场_10日波动       < 0.0022
市场_20日波动       < 0.0022
市场_60日趋势       < 0.0022
市场_regime         < 0.0022
```

### 3.2 正确用法

市场状态特征的唯一正确使用位置是**后处理阶段**，而非模型输入：

| 使用方式 | 位置 | 示例 |
|---------|------|------|
| 入选阈值调整 | 预测后处理 | 高波动日提高 z-score 阈值 |
| 仓位控制 | 预测后处理 | 熊市降低持仓数量 |
| 模型输出校准 | 预测后处理 | 根据 regime 调整预测分数 |
| 标签调整 | 训练标签 | 市场下跌日降低目标收益基准 |
| 样本加权 | 训练样本 | 震荡市样本降权 |

**核心原则**：市场状态信息应当影响"怎么用模型的预测结果"，而不是"模型预测什么"。

### 3.3 月度数据揭示的修正：从"市场驱动"到"置信度驱动"

月度数据揭示了一个关键事实：**验证期全部是横盘，模型表现的月度波动与市场方向无关，而是来源于模型自身对不同月份数据模式的过拟合程度不同。**

因此，后处理的核心驱动因素应该从"市场状态"改为**"模型预测置信度"**：

```
修正前（不可靠）：
  市场是高波动吗？ → 调整阈值
  市场是下跌趋势吗？ → 调整仓位

修正后（更可靠）：
  模型对今天的预测有多自信？ → Top1 vs Top5 的z-score差距大吗？
  → 差距大（高置信度）：满仓5只
  → 差距小（低置信度）：少选或不选
```

市场状态特征仍然有用——但作为置信度判断的**辅助参考**而非**主要驱动**。例如：高波动日 + 低置信度 → 空仓；高波动日 + 高置信度 → 仍可持有但降低仓位。

### 3.4 操作建议

- **从 XGBRanker 输入中移除7维市场特征**（`flatten_sequences_to_xgb` 中的 `+7`）
- 将市场状态保存在后处理阶段使用的全局字典中
- 特征维度从 2,217 降至 2,210，训练速度略有提升
- **主推"置信度驱动后处理"，市场状态作为辅助参考**

---

## 4. 各调优方向最终判定（修正版）

### 方向1：市场环境特征 + 多窗口稳健性约束

**判定**：方向正确但使用方式需大幅修正。

- 市场特征必须从模型输入移除，改为后处理使用 ✓
- **关键修正**：后处理应以"模型置信度"为主要驱动，市场状态为辅助参考（月度数据证明市场始终横盘，regime无区分度）
- 多窗口稳健性约束（特征因子和模型参数在多个交叉窗口上表现稳定）保留 ✓

### 方向2：target_precision_gate + 非满仓

**判定**：分离为两个独立子方向。

- **target_precision_gate（标签过滤）**：不推荐硬过滤。震荡市中同时满足"收益为正 AND 排名前25% AND 短中期稳定 AND 回撤≤3%"的样本不足5%，会导致训练数据严重不足。建议改用 sample_weight 做软加权。
- **非满仓机制**：可行。赛制允许权重和≤1（剩余视为现金），应在后处理阶段根据**模型置信度**（非市场状态）动态调整。

### 方向3：特征交叉

**判定**：方向正确，但需要前置步骤。

- **新前置条件**：必须先做因子IC分析，筛选出真正有效的特征（当前221维中80%可能是噪声）。在此基础上对高IC因子做精选交叉特征，而非盲目给全部特征加交叉。
- 只构造3-5个最有经济学含义的交叉特征
- 推荐：量价背离度、夏普比率因子、流动性调整收益

### 方向4：多模型融合 + 辅助任务

**判定**：分离评估。

- **辅助任务（方向分类器 → 额外特征）**：优先级从P2提升到**P1**。月度数据揭示模型有弱方向感（好月份 win_rate 65-95%）但无排序能力（Spearman ≈ 0），方向分类器可以针对性地强化方向判断。
- **OOF stacking**：前置条件未满足（基模型 Spearman < 0.10），保留到P4。

### 方向5：预测后处理与分数校准

**判定**：**当前最高优先级方向，但方案需从"市场驱动"改为"置信度驱动"**。

- z-score 标准化 + 置信度驱动的动态阈值 + softmax 权重 ✓
- 按模型自信程度分配权重而非等权 ✓
- 市场状态仅作为置信度判断的辅助参考（非主要驱动）

### 方向6：50次滚动验证

**判定**：推荐作为所有改动的验证工具。

- 用于验证因子IC分析后特征筛选的有效性
- 用于调优"置信度驱动"后处理的超参数（confidence_gap阈值、temperature等）
- 四窗口验证保留做压力测试

### 方向7：超参数调优

**判定**：不是当前阶段重点，但需修复BUG。

- 先对齐 max_depth 配置（config.py=5 vs xgb_tune.py=8）
- 重写 optuna_search.py 适配 XGBRanker
- **在降维+新特征稳定后做**，否则调参是在噪声特征上优化

---

## 5. P0 最高优先：因子IC分析——筛选有效特征

### 5.1 为什么这是最高优先级

当前 158+39+基本面+行业 = 221 个基础特征，展平后 2,210 维。月度数据揭示模型在相邻月份间表现天差地别（win_rate 17%↔95%），这是典型的高维噪声过拟合。

从量化金融的角度，单个 Alpha 因子的 Rank IC（因子值与未来收益的截面秩相关）通常在 0.02-0.05 之间。221 个因子中，真正IC显著的（|IC| > 0.02 且 t-stat > 2）可能不超过 30-50 个。其余 170+ 个因子是噪声——它们不仅不贡献信号，还会在 XGBoost 的高维分裂空间中淹没有效信号。

**目标**：从 221 个特征中筛选出 IC 显著的 30-50 个，特征维度从 2,210 降至 300-500，显著减少过拟合。

### 5.2 实现方案

**新建文件**：`code/src/factor_ic.py`

```python
"""
因子IC分析脚本
计算每个特征因子对未来5日超额收益的截面 Rank IC，
筛选出 IC 均值显著不为零（|t-stat| > 2）的因子。
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import config, feature_columns_map
from train import _preprocess_common, _merge_fundamentals

def compute_factor_ic_analysis(output_path=None):
    """
    对 feature_columns_map 中所有特征因子逐一计算 Rank IC。
    
    流程：
    1. 加载 train.csv，做特征工程
    2. 构建未来5日超额收益标签
    3. 对每个因子，逐日计算因子值与超额收益的 Spearman 秩相关
    4. 输出每个因子的：IC均值、IC标准差、t-stat、ICIR
    
    Returns:
        DataFrame: 各因子的 IC 统计量
    """
    # 加载数据
    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'), 
                          dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    
    # 只用训练集部分（不含最近12个月验证集），避免使用未来数据
    last_date = full_df['日期'].max()
    train_end = last_date - pd.DateOffset(months=12)
    train_df = full_df[full_df['日期'] <= train_end].copy()
    print(f"IC分析数据范围: {train_df['日期'].min().date()} ~ {train_df['日期'].max().date()}")
    
    # 标签构建：未来5日超额收益
    train_df = train_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    train_df['open_t1'] = train_df.groupby('股票代码')['开盘'].shift(-1)
    train_df['open_t5'] = train_df.groupby('股票代码')['开盘'].shift(-5)
    train_df = train_df[train_df['open_t1'] > 1e-4]
    train_df['future_ret'] = (train_df['open_t5'] - train_df['open_t1']) / (train_df['open_t1'] + 1e-12)
    train_df['daily_mean'] = train_df.groupby('日期')['future_ret'].transform('mean')
    train_df['excess_ret'] = train_df['future_ret'] - train_df['daily_mean']
    train_df = train_df.dropna(subset=['excess_ret'])
    
    # 特征工程
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    features_list = list(feature_columns_map[config['feature_num']])
    
    all_sids = sorted(train_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_sids)}
    
    train_data, _ = _preprocess_common(train_df, stockid2idx, desc="特征工程", drop_small_open=True)
    
    # 合并基本面，扩展特征列表
    fp = os.path.join(data_path, 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(data_path, 'hs300_fundamentals.csv')
    train_data, fund_cols = _merge_fundamentals(train_data, fp)
    if fund_cols:
        features_list = features_list + fund_cols
    
    # 对每个因子计算逐日 Rank IC
    print(f"\n计算 {len(features_list)} 个因子的逐日 Rank IC ...")
    ic_results = []
    unique_dates = sorted(train_data['日期'].unique())
    
    for feat in features_list:
        if feat not in train_data.columns:
            ic_results.append({'factor': feat, 'ic_mean': np.nan, 'ic_std': np.nan, 
                              't_stat': np.nan, 'icir': np.nan, 'n_days': 0})
            continue
        
        daily_ics = []
        for date in unique_dates:
            day_data = train_data[train_data['日期'] == date].dropna(subset=[feat, 'excess_ret'])
            if len(day_data) < 30:
                continue
            ic = day_data[[feat, 'excess_ret']].corr(method='spearman').iloc[0, 1]
            if not np.isnan(ic):
                daily_ics.append(ic)
        
        if len(daily_ics) >= 20:
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics, ddof=1)
            t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ics)))
            icir = ic_mean / ic_std if ic_std > 0 else 0
        else:
            ic_mean = np.nan; ic_std = np.nan; t_stat = np.nan; icir = np.nan
        
        ic_results.append({
            'factor': feat,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            't_stat': t_stat,
            'icir': icir,
            'n_days': len(daily_ics),
        })
    
    ic_df = pd.DataFrame(ic_results)
    
    # 筛选：|t_stat| > 2（IC均值显著不为零）
    ic_df['significant'] = ic_df['t_stat'].abs() > 2
    significant_factors = ic_df[ic_df['significant']]['factor'].tolist()
    
    print(f"\n=== 因子IC分析结果 ===")
    print(f"总因子数: {len(ic_df)}")
    print(f"显著因子数 (|t|>2): {len(significant_factors)}")
    print(f"IC最强Top10:")
    for _, row in ic_df.sort_values('ic_mean', key=abs, ascending=False).head(10).iterrows():
        print(f"  {row['factor']:<25s} IC={row['ic_mean']:+.4f}  t={row['t_stat']:+.2f}  "
              f"ICIR={row['icir']:+.3f}  sig={row['significant']}")
    
    if output_path:
        ic_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存到: {output_path}")
    
    return ic_df, significant_factors


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    ic_df, sig_factors = compute_factor_ic_analysis(
        output_path=os.path.join(config['output_dir'], 'factor_ic_analysis.csv')
    )
```

### 5.3 特征筛选后集成到训练流程

在 `config.py` 中新增配置项：

```python
# 在 config.py 中新增
selected_features = None  # 初始为 None，表示使用全部特征
# 运行 factor_ic.py 后，将显著因子列表填入：
# selected_features = ['KMID', 'KLEN', 'ROC5', 'ROC10', ...]  # 30-50个因子
```

在 `train.py` 的 `_preprocess_common` 完成后、`flatten_sequences_to_xgb` 调用前，新增特征筛选步骤：

```python
# 在 train.py 的 train_one_window 函数中，features_list 构建之后
if config.get('selected_features'):
    # 只保留IC显著的因子
    valid_features = [f for f in config['selected_features'] 
                     if f in features_list]
    features_list = valid_features
    print(f"特征筛选: {len(features_list)} 个显著因子 (从 {len(feature_columns_map[config['feature_num']])} 缩减)")
```

### 5.4 验证方式

在因子IC分析完成后，用50次滚动验证对比：
- 全量221特征 → final_score, win_rate, 正收益比例
- 筛选后30-50特征 → 同上

预期：筛选后 final_score 均值持平或略降，但月度间波动（fs_std）显著减小，说明过拟合得到缓解。

---

## 6. P0 最高优先：置信度驱动的后处理改造

### 6.1 设计核心：从"市场驱动"改为"置信度驱动"

**核心洞察（来自月度数据）**：市场连续12个月横盘，但模型表现月度间剧烈波动（win_rate 17%↔95%）。这说明模型自身的"自信程度"比市场状态更有区分度。

**置信度的量化方式**：同一天所有股票的预测z-score分布中，Top1 和 Top5 之间的 z-score 差距：

- **高置信度**（gap > 2.0）：模型明确区分出了鹤立鸡群的股票 → 满仓5只
- **中置信度**（1.0 < gap < 2.0）：模型有一些偏好但不够自信 → 中等仓位
- **低置信度**（gap < 1.0）：模型对所有股票判断接近 → 少选或空仓

### 6.2 实现方案

**涉及文件**：`code/src/predict.py`

**新增函数**：

```python
def confidence_aware_postprocess(scores, stock_codes, top_k=5):
    """
    基于模型预测置信度的后处理：z-score标准化 + 置信度阈值 + softmax权重。
    
    置信度定义：Top1 与 Top5 的 z-score 差距。
    差距大 = 模型明确知道哪些股票更好 = 高置信度。
    差距小 = 所有股票分数接近 = 模型在"猜" = 低置信度。
    
    Args:
        scores: np.array, 原始预测分数
        stock_codes: list, 对应的股票代码
        top_k: int, 最多选取的股票数量
    
    Returns:
        selected_stocks: list, 入选股票代码
        weights: list, 对应权重（总和≤1）
    """
    # Step 1: z-score 标准化
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)
    if std_score < 1e-8:
        z_scores = np.zeros_like(scores)
    else:
        z_scores = (scores - mean_score) / std_score
    
    # Step 2: 按 z-score 降序排列
    sorted_idx = np.argsort(z_scores)[::-1]
    sorted_z = z_scores[sorted_idx]
    
    # Step 3: 计算置信度 gap
    if len(sorted_z) >= 5:
        confidence_gap = sorted_z[0] - sorted_z[4]
    elif len(sorted_z) >= 2:
        confidence_gap = sorted_z[0] - sorted_z[-1]
    else:
        confidence_gap = 0.0
    
    # Step 4: 根据置信度决定选取策略
    if confidence_gap > 2.0:
        # 高置信度：模型有明确判断
        n_select = 5
        z_threshold = 0.5       # 低阈值，容纳更多候选
        temperature = 1.0       # 中等集中度
    elif confidence_gap > 1.0:
        # 中置信度：模型有一定把握
        n_select = 4
        z_threshold = 1.0
        temperature = 0.7
    elif confidence_gap > 0.5:
        # 低置信度：模型不太确定
        n_select = 2
        z_threshold = 1.5
        temperature = 0.3       # 高度集中权重
    else:
        # 极低置信度：模型在猜，几乎不选
        n_select = 1
        z_threshold = 2.0
        temperature = 0.1
    
    # Step 5: 按 z_threshold 筛选
    qualified = sorted_z[sorted_z >= z_threshold]
    qualified_idx = sorted_idx[:len(qualified)]
    
    if len(qualified_idx) == 0:
        qualified_idx = sorted_idx[:1]
    
    selected_idx = qualified_idx[:n_select]
    selected_z = z_scores[selected_idx]
    
    # Step 6: softmax 权重分配
    exp_z = np.exp(selected_z / max(temperature, 1e-8))
    weights = exp_z / exp_z.sum()
    
    # 注意：权重总和 = 1.0（不满仓由 n_select 控制，非权重缩放）
    
    selected_stocks = [stock_codes[i] for i in selected_idx]
    
    # 记录置信度信息（方便复盘）
    return selected_stocks, weights.tolist(), {
        'confidence_gap': confidence_gap,
        'n_selected': len(selected_stocks),
        'z_threshold_used': z_threshold,
    }
```

**在主函数中调用**（替换现有 Top5 等权选取）：

```python
# 修改前：
# top5 = ranked_stock_ids[:5]
# output_df = pd.DataFrame({'stock_id': top5, 'weight': [0.2] * len(top5)})

# 修改后：
selected_stocks, weights, conf_info = confidence_aware_postprocess(
    scores, stock_codes, top_k=5
)
print(f"置信度信息: gap={conf_info['confidence_gap']:.3f}, "
      f"选取{conf_info['n_selected']}只, z阈值={conf_info['z_threshold_used']:.1f}")

output_df = pd.DataFrame({
    'stock_id': selected_stocks, 
    'weight': weights
})
```

### 6.3 市场状态作为辅助参考（可选叠加）

在置信度判断的基础上，可选叠加市场状态调整：

```python
def compute_market_state_for_postprocess(df, latest_date):
    """
    计算后处理阶段的市场状态辅助指标。
    仅作为置信度判断的参考，非主要驱动。
    
    Returns:
        dict with: volatility_5d, trend_60d, ad_ratio
    """
    market_daily = df.groupby('日期')['涨跌幅'].mean().sort_index()
    
    vol_5d = market_daily.rolling(5, min_periods=3).std().iloc[-1] if len(market_daily) >= 5 else 0.01
    trend_60d = market_daily.rolling(60, min_periods=10).sum().iloc[-1] if len(market_daily) >= 60 else 0.0
    
    # 涨跌比
    recent = df[df['日期'] >= latest_date - pd.DateOffset(days=5)]
    up_days = (recent.groupby('日期')['涨跌幅'].mean() > 0).sum()
    down_days = (recent.groupby('日期')['涨跌幅'].mean() < 0).sum()
    ad_ratio = up_days / max(down_days, 1)
    
    return {
        'volatility_5d': float(vol_5d) if not np.isnan(vol_5d) else 0.01,
        'trend_60d': float(trend_60d) if not np.isnan(trend_60d) else 0.0,
        'ad_ratio': float(ad_ratio),
    }
```

叠加逻辑（在 `confidence_aware_postprocess` 返回前）：

```python
# 可选叠加：市场状态微调
def apply_market_overlay(n_select, temperature, market_state):
    """
    市场状态作为辅助参考，微调置信度驱动的参数。
    仅在高波动+低置信度的组合下触发防御性降仓。
    """
    vol = market_state.get('volatility_5d', 0.01)
    
    if vol > 0.025 and confidence_gap < 1.0:
        # 高波动 + 模型不确定 → 额外降仓
        n_select = max(1, n_select - 1)
        temperature = temperature * 0.5
    
    return n_select, temperature
```

### 6.4 验证方案

用窗口4模型，逐日计算 confidence_gap，将237个交易日按 gap 分成高/中/低三组，看三组的 final_score 是否有显著差异：

```
假设：
  高置信度组（gap > 2.0）：final_score 应该显著 > 0
  低置信度组（gap < 1.0）：final_score 应该接近或小于 0
```

在 `analyze_window4_monthly.py` 的输出基础上，新增置信度维度的分析列。如果三组 final_score 差异显著，说明置信度驱动策略有效。

---

## 7. P1 高优先：方向分类器辅助特征

### 7.1 为什么从P2提升到P1

月度数据揭示：好月份 win_rate 高达 65-95%，但 Spearman ≈ 0.12-0.18。这说明：

- **模型有弱方向判断能力**：能大致判断哪些股票会涨
- **但无排序能力**：无法区分涨得多和涨得少

方向分类器可以针对性地强化"方向判断"这一模型已经具备的弱能力，将更强的方向信号作为特征输入 XGBRanker，让 Ranker 专注于它目前缺失的"排序"任务。

### 7.2 实现方案

**新建文件**：`code/src/direction_classifier.py`

```python
"""
方向分类器：预测个股未来5日超额收益是否为正（二分类）。
用 OOF 方式生成特征，避免数据泄露。
"""
import numpy as np, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def generate_direction_features(X, y_cont, qid, n_splits=5):
    """
    使用时序交叉验证生成 OOF 方向概率特征。
    
    Args:
        X: np.array, 特征矩阵 (N, D)
        y_cont: np.array, 连续收益标签 (N,)
        qid: np.array, 组ID (N,)——按交易日
        n_splits: 交叉验证折数
    
    Returns:
        direction_proba: np.array, OOF预测的上涨概率 (N,)
    """
    y_binary = (y_cont > 0).astype(int)
    direction_proba = np.zeros(len(X))
    
    unique_qids = sorted(set(qid))
    fold_size = len(unique_qids) // n_splits
    
    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_splits - 1 else len(unique_qids)
        
        val_qids = set(unique_qids[val_start:val_end])
        train_mask = np.array([q not in val_qids for q in qid])
        val_mask = np.array([q in val_qids for q in qid])
        
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc',
            num_leaves=31, learning_rate=0.05, n_estimators=100,
            subsample=0.6, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbose=-1,
        )
        
        model.fit(X[train_mask], y_binary[train_mask])
        direction_proba[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    
    # 训练集部分补充——用全部训练数据训练一个最终模型做预测
    final_model = lgb.LGBMClassifier(
        objective='binary', metric='auc',
        num_leaves=31, learning_rate=0.05, n_estimators=100,
        subsample=0.6, colsample_bytree=0.3,
        reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, verbose=-1,
    )
    final_model.fit(X, y_binary)
    
    return direction_proba, final_model
```

**集成到 train.py**（在 `train_one_window` 函数中，展平特征后、训练 XGBRanker 前）：

```python
# 方向分类器 OOF 特征
from direction_classifier import generate_direction_features
direction_proba_train, dir_model = generate_direction_features(X_train, y_train_cont, qid_train)
direction_proba_val = dir_model.predict_proba(X_val)[:, 1]

# 作为额外特征追加
X_train = np.column_stack([X_train, direction_proba_train.reshape(-1, 1)])
X_val = np.column_stack([X_val, direction_proba_val.reshape(-1, 1)])
features_list = features_list + ['direction_proba']
```

**集成到 predict.py**：加载方向分类器模型，在推理时生成 direction_proba 特征。

### 7.3 注意事项

- OOF（Out-of-Fold）生成训练集特征是必须的，否则会导致数据泄露（XGBRanker 在训练时看到了验证集的标签信息）
- 方向分类器的 IC（作为特征时）预计在 0.03-0.06 之间，高于大部分原始特征
- 该方案也适用于未来扩展：训练波动率预测器，将其输出同样作为额外特征

---

## 8. P1 高优先：50次滚动验证脚本

### 8.1 设计目标

作为 P0 和后续所有改动的评估工具。每次改动后用50次滚动验证量化效果对比。

**新建文件**：`code/src/rolling_val.py`

**滚动参数**：
- 每次训练窗口：260个交易日（约1年）
- 每次验证窗口：5个交易日（1周）
- 向前滚动步长：5个交易日
- 总滚动次数：50次

### 8.2 核心实现

```python
"""
50次滚动窗口验证脚本
每次用最近260个交易日训练 → 向前滚动5个交易日验证 → 共滚动50次
评估指标：正收益窗口比例、最长连续亏损窗口、周收益率波动幅度

用法：
    python code/src/rolling_val.py                    # 基础评估
    python code/src/rolling_val.py --tune              # 后处理参数搜索
    python code/src/rolling_val.py --compare_feats     # 特征筛选A/B对比
"""
import os, sys, json, copy, gc, warnings, argparse, multiprocessing as mp
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from config import config, config_extended, xgb_config
from train import (
    set_seed, train_one_window, preprocess_data, 
    flatten_sequences_to_xgb, feature_columns_map,
    _continuous_labels_to_ranks, evaluate_xgb_model,
)

# 滚动参数
TRAIN_DAYS = 260
VAL_DAYS = 5
N_ROLLS = 50
STEP_DAYS = 5

# ===== 后处理函数（从 predict.py 导入或在此定义） =====
def confidence_aware_postprocess(scores, stock_codes, top_k=5):
    """同 Section 6.2 的实现"""
    # ... （完整实现见 6.2 节）
    pass


def run_single_roll(train_df, val_df, roll_idx, config_override=None):
    """
    执行单次滚动训练+评估。
    
    Returns:
        dict: 含 final_score, win_rate, spearman, n_stocks_selected 等
    """
    # ... 实现训练+预测+后处理+评估 ...
    pass


def run_rolling_validation():
    """
    执行完整50次滚动验证，汇总结果。
    """
    full_df = pd.read_csv(os.path.join(config['data_path'], 'train.csv'),
                          dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    all_dates = sorted(full_df['日期'].unique())
    
    results = []
    for roll in range(N_ROLLS):
        train_end_idx = len(all_dates) - VAL_DAYS - roll * STEP_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS
        
        train_dates = all_dates[train_start_idx:train_end_idx]
        val_dates = all_dates[train_end_idx:train_end_idx + VAL_DAYS]
        
        train_df = full_df[full_df['日期'].isin(train_dates)].copy()
        val_df = full_df[full_df['日期'].isin(val_dates)].copy()
        
        result = run_single_roll(train_df, val_df, roll)
        results.append(result)
        
        del train_df, val_df; gc.collect()
    
    # 汇总指标
    fs_values = [r['final_score'] for r in results]
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
        'positive_ratio': win_weeks / N_ROLLS,
        'mean_final_score': np.mean(fs_values),
        'std_final_score': np.std(fs_values, ddof=1),
        'max_consecutive_loss': max_consec_loss,
        'max_single_loss': min(fs_values),
        'max_single_gain': max(fs_values),
    }
    
    print(f"\n{'='*60}")
    print(f"  50次滚动验证汇总")
    print(f"{'='*60}")
    print(f"  正收益周比例:    {summary['positive_ratio']:.2%}")
    print(f"  平均 final_score: {summary['mean_final_score']:.4f}")
    print(f"  fs 标准差:        {summary['std_final_score']:.4f}")
    print(f"  最长连续亏损:     {summary['max_consecutive_loss']} 周")
    print(f"  最大单周亏损:     {summary['max_single_loss']:.4f}")
    print(f"  最大单周收益:     {summary['max_single_gain']:.4f}")
    print(f"{'='*60}")
    
    return summary, results


def tune_postprocess_params():
    """
    对后处理超参数做网格搜索。
    参数空间：confidence_gap阈值 ∈ {0.5, 1.0, 1.5, 2.0}, 
             temperature ∈ {0.5, 0.7, 1.0}
    每个组合跑50次滚动验证，选综合最优。
    """
    # ... 实现参数搜索 ...
    pass


if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='后处理参数搜索')
    parser.add_argument('--compare_feats', action='store_true', help='特征筛选A/B对比')
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.tune:
        tune_postprocess_params()
    elif args.compare_feats:
        # 分别用全量特征和筛选后特征跑50次滚动，做对比
        pass
    else:
        run_rolling_validation()
```

**评估指标**：

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| 正收益窗口比例 | 50次中 final_score > 0 的比例 | 模型选股的稳定性 |
| 最长连续亏损窗口 | 连续 final_score < 0 的最大次数 | 最坏情况风险 |
| 周收益率波动幅度 | 50次周收益率的标准差 | 收益一致性 |
| 平均 final_score | 50次 final_score 的均值 | 综合排序能力 |
| 最大单周亏损 | 50次中最差的 final_score | 尾部风险 |

### 8.3 用于后处理参数调优

```python
# 搜索空间
confidence_gap_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
temperature_values = [0.3, 0.5, 0.7, 1.0, 1.5]
z_thresholds = [0.5, 1.0, 1.5, 2.0]

# 评价标准：最终以"正收益比例 × 2 + 平均fs × 10 - 最大连亏"作为综合得分
```

---

## 9. P2 中优先：精选特征交叉

### 9.1 前置条件

必须在 P0（因子IC分析）完成后执行。只在 IC 显著的因子上构造交叉特征，避免在噪声特征上浪费维度。

### 9.2 实现方案

**涉及文件**：`code/src/utils.py`

**新增函数**：`engineer_cross_features(df, high_ic_factors)`

```python
def engineer_cross_features(df, high_ic_factors=None):
    """
    构造精选交叉特征（3-5个）。
    
    Args:
        df: 特征 DataFrame
        high_ic_factors: 因子IC分析筛选出的显著因子列表（用于判断哪些基础计算可用）
    
    交叉特征列表：
    1. 量价背离度：价格趋势方向 vs 成交量趋势方向
    2. 夏普比率因子：10日动量 / 20日波动率
    3. 流动性调整收益：5日收益 / (1 + 换手率变化率)
    4. (可选) 趋势质量：趋势斜率 × 趋势拟合优度
    5. (可选) 波动率调整RSI
    """
    df = df.copy()
    
    # 1. 量价背离度
    price_trend_5d = df['收盘'].pct_change(5).apply(np.sign)
    volume_trend_5d = df['成交量'].pct_change(5).apply(np.sign)
    df['cross_vol_price_div'] = price_trend_5d * volume_trend_5d
    
    # 2. 夏普比率因子
    momentum_10d = df['收盘'].pct_change(10)
    vol_20d = df['收盘'].pct_change().rolling(20).std()
    df['cross_sharpe'] = momentum_10d / (vol_20d + 1e-12)
    
    # 3. 流动性调整收益
    ret_5d = df['收盘'].pct_change(5)
    turnover_change = df['换手率'].pct_change(5).fillna(0)
    df['cross_liq_adj_ret'] = ret_5d / (1 + abs(turnover_change))
    
    # 4. 趋势质量（依赖 BETA20 和 RSQR20 已计算）
    if 'BETA20' in df.columns and 'RSQR20' in df.columns:
        df['cross_trend_quality'] = df['BETA20'] * df['RSQR20']
    
    # 5. 波动率调整RSI
    if 'RSI' in df.columns and 'volatility_10' in df.columns:
        df['cross_vol_adj_rsi'] = (df['RSI'] - 50) / (df['volatility_10'] + 1e-12)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, ['cross_vol_price_div', 'cross_sharpe', 'cross_liq_adj_ret',
                'cross_trend_quality', 'cross_vol_adj_rsi']
```

**集成方式**：在 `engineer_features_158plus39()` 的最后调用 `engineer_cross_features()`，将交叉特征列名追加到 `feature_columns_map['158+39']`。

---

## 10. P3 低优先：超参数重搜 + BUG修复

### 10.1 修复 max_depth 配置不一致

**涉及文件**：`code/src/xgb_tune.py`

```python
# 第75行，删除硬编码：
# xgb_config['max_depth'] = 8  # ← 删除这行

# 改为在 main() 开头从 xgb_config 读取锁定参数：
locked_params = {
    'max_depth': xgb_config['max_depth'],  # 从 config.py 读取
}

# 在 PARAM_GRID 中增加 max_depth 搜索：
PARAM_GRID = {
    'max_depth': [4, 5, 6, 7],           # 新增
    'colsample_bytree': [0.3, 0.4, 0.5],
    'subsample': [0.5, 0.6, 0.7],
}
```

### 10.2 重写 optuna_search.py（适配 XGBRanker）

**涉及文件**：`code/src/optuna_search.py`（需要完全重写）

新搜索空间：

```python
def objective(trial):
    xgb_config['max_depth'] = trial.suggest_int('max_depth', 4, 8)
    xgb_config['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    xgb_config['subsample'] = trial.suggest_float('subsample', 0.4, 0.8)
    xgb_config['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.2, 0.6)
    xgb_config['reg_alpha'] = trial.suggest_float('reg_alpha', 0.1, 10.0, log=True)
    xgb_config['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 20.0, log=True)
    xgb_config['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 50)
    xgb_config['n_estimators'] = trial.suggest_int('n_estimators', 100, 500, step=50)
    
    # 调用 train_one_window（快速模式：n_estimators=200 + early_stopping=20）
    return best_score

# TPESampler 贝叶斯优化，25 trials
```

**搜索策略**：
- 先用 Optuna TPESampler 做贝叶斯粗搜（25 trials × 200 trees each）
- 取 Top3 trial 的参数区域
- 在该区域做小规模网格精调（n_estimators=300，完整训练）

**重要**：超参数调优在 P0-P2 完成后执行。在噪声特征上做超参数搜索是低效的。

---

## 11. P4 未来方向：多模型融合

> **前置条件**：单模型 Spearman > 0.10 且不同基模型之间预测相关性 < 0.7。不满足时投入产出比低。

### 11.1 OOF Stacking 方案（当条件满足时）

```
基模型层 (Level 0):
├── XGBRanker v1 (max_depth=5, objective=rank:ndcg)
├── XGBRanker v2 (max_depth=7, objective=rank:pairwise)
├── LightGBM Ranker (objective=lambdarank)
└── CatBoost Ranker (objective=YetiRank)

融合层 (Level 1):
└── XGBRegressor (输入=基模型OOF预测, 目标=真实超额收益)
    或简单的加权平均（权重由近期表现决定）
```

### 11.2 辅助任务扩展（当条件满足时）

```
主任务: XGBRanker (rank:ndcg, 对所有股票排序)
├── 辅助任务1: 方向分类器 → 输出概率作为主模型特征 ✓ (P1实现)
├── 辅助任务2: 波动率回归 → 输出预期波动作为主模型特征
└── 辅助任务3: Top1专项模型 → 辅助最终选股决策
```

---

## 12. 验证框架设计

### 12.1 三层验证体系

```
层次              用途                    评估内容                    使用频率
──────────────────────────────────────────────────────────────────────────
L1: 50次滚动验证   后处理参数调优             正收益比、连亏、波动幅度    每次改参数后
                   + 特征筛选A/B对比
L2: 四窗口交叉验证  模型在极端行情下的稳健性    final_score极差、稳定性    重大改动后
L3: 窗口4按月切分   震荡市各阶段适应性诊断     月度fs分布、置信度分析      出结果前诊断
```

### 12.2 关键决策阈值

| 指标 | 绿灯（可提交） | 黄灯（需改进） | 红灯（不可用） |
|------|--------------|--------------|--------------|
| 50次滚动正收益比例 | > 60% | 50-60% | < 50% |
| 50次滚动平均fs | > 0.03 | 0.01-0.03 | < 0.01 |
| 四窗口 fs 变异系数 | < 30% | 30-50% | > 50% |
| 窗口4月度 fs 为正月份 | > 8/12 | 6-8/12 | < 6/12 |
| 最长连续亏损窗口 | ≤ 3周 | 4-5周 | ≥ 6周 |

### 12.3 测试流程（赛前）

```bash
# 1. P0: 因子IC分析 → 生成 factor_ic_analysis.csv
python code/src/factor_ic.py

# 2. P0: 将显著因子列表填入 config.py 的 selected_features
# (手动编辑 config.py)

# 3. P1: 用50次滚动验证对比全量特征 vs 筛选特征
python code/src/rolling_val.py --compare_feats

# 4. P0: 用50次滚动验证调优后处理参数
python code/src/rolling_val.py --tune

# 5. P1: 集成方向分类器后，跑全量训练
python code/src/train.py

# 6. 四窗口交叉验证
python code/src/cross_val.py

# 7. 窗口4月度诊断
python analyze_window4_monthly.py

# 8. 用最佳参数生成最终提交
python code/src/predict.py

# 9. 自评得分
python test/score_self.py
```

---

## 13. 开发约束与注意事项

### 13.1 赛制约束

- 最终输出：不超过5只不同股票代码，每行一个代码及权重
- 权重累加和**不超过1**，不到1的部分视为持有现金
- 等权不是必需的，可以根据置信度差异化配权
- 允许输出少于5只股票

### 13.2 技术约束

- 特征工程依赖 TA-Lib，需确保系统已安装
- 训练使用多进程（spawn模式），Windows下**所有脚本必须加 `if __name__ == '__main__':` 保护**，否则子进程递归启动导致死循环
- memmap临时`.dat`文件不会自动清理，调参后需手动清理
- XGBoost 版本升级后，旧版 pickle 序列化的模型需要 re-save（用 `model.save_model()` 而非 pickle）

### 13.3 风险点

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| 后处理参数过拟合 | 高 | 用50次滚动验证而非单次验证调参 |
| 因子IC分析在12个月前训练集上做，结论可能不适用于当前 | 中 | 同时在最近12个月验证集上复算IC，对比确认 |
| 测试窗口5天随机性太大 | 高 | 通过50次滚动验证量化不确定性；置信度低时空仓 |
| 方向分类器数据泄露 | 中 | 用OOF方式生成训练集特征，严格时序划分 |
| 月度fs剧烈波动（423.5%变异系数）意味着即使改进后也可能在8月初踩到差月 | **极高** | 最坏情况预案：如果8月3日confidence_gap < 0.5，空仓 |

### 13.4 明确不建议做的事情

- ❌ 不要在 XGBRanker 输入中保留市场特征（数据结构决定了它们不可分裂）
- ❌ 不要增加更多原始特征（在完成IC分析筛选前）
- ❌ 不要硬过滤训练样本（会严重减少训练数据）
- ❌ 不要在基模型 Spearman < 0.10 时做 stacking
- ❌ 不要用 random split 做验证集划分（必须时序划分）
- ❌ 不要用当前版本的 `optuna_search.py`
- ❌ 不要在模块顶层执行多进程代码而不加 `if __name__ == '__main__':` 保护

### 13.5 最坏情况预案

月度数据显示即使模型在好月份（2025-11, fs=0.17）表现优异，紧邻的差月份（2025-12, fs=-0.08）可能完全失效。8月3日-7日恰好是月/季初，历史上这个时间窗口的模型表现不确定性极高。

**预案**：如果 `confidence_aware_postprocess` 在 8月3日 返回 confidence_gap < 0.5，执行保守策略——只选1-2只股票或空仓（权重全部为0，输出空文件或极小仓位），宁可放弃这周的收益也不冒险踩到负收益。

---

## 附录A：文件改动清单（修正版）

| 文件 | 改动类型 | 优先级 | 说明 |
|------|---------|--------|------|
| `code/src/factor_ic.py` | **新建** | **P0** | 因子IC分析，筛选有效特征 |
| `code/src/predict.py` | 重写后处理逻辑 | **P0** | confidence_aware_postprocess（置信度驱动）替代原有的市场驱动方案 |
| `code/src/config.py` | 修改 + 新增 | **P0** | 新增 selected_features 配置项 |
| `code/src/rolling_val.py` | **新建** | **P1** | 50次滚动验证脚本（含后处理参数调优） |
| `code/src/direction_classifier.py` | **新建** | **P1** | 辅助方向分类器（OOF方式防止泄露） |
| `code/src/train.py` | 修改 | **P0** | 集成特征筛选 + 方向分类器特征；移除市场特征输入 |
| `code/src/xgb_tune.py` | 修改 | P3 | 修复 max_depth=8 硬编码 |
| `code/src/utils.py` | 新增函数 | P2 | engineer_cross_features()（需等IC分析完成） |
| `code/src/optuna_search.py` | 重写 | P3 | 适配 XGBRanker（需等P0-P1完成） |
| `analyze_window4_monthly.py` | 已生成并修复 | P3 | 窗口4月度诊断脚本 |

## 附录B：特征维度演进

```
当前: 10天×221维(158+39+基本面+行业) + 7市场 = 2,217维
P0后: 10天×30~50维(IC显著因子) = 300~500维 (移除市场特征 + 大幅降维)
P1后: P0 + 1维(方向概率) = 301~501维
P2后: P1 + 3~5维(交叉特征) = 304~506维
```

## 附录C：项目文件结构速查

```
BDC(XGB)/
├── code/src/
│   ├── config.py              # 主配置文件
│   ├── train.py               # XGBRanker 训练主脚本
│   ├── predict.py             # 推理 + 置信度驱动后处理
│   ├── cross_val.py           # 四窗口交叉验证
│   ├── factor_ic.py           # [NEW P0] 因子IC分析
│   ├── rolling_val.py         # [NEW P1] 50次滚动验证
│   ├── direction_classifier.py # [NEW P1] 方向分类器辅助特征
│   ├── xgb_tune.py            # 超参数网格搜索(待修)
│   ├── optuna_search.py       # Optuna搜索(待重写)
│   ├── utils.py               # 特征工程
│   ├── evaluation.py          # 评估指标
│   └── model.py               # 旧版StockTransformer(未使用)
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── hs300_stock_list.csv
│   ├── hs300_fundamentals.csv
│   └── history_factors_nan.csv
├── model/60_158+39/
│   ├── best_model.pkl/json
│   ├── scaler.pkl
│   ├── factor_ic_analysis.csv  # [NEW] IC分析结果
│   ├── cross_val_1~4/
│   ├── cross_val_4_窗口4_近期市场/
│   │   └── monthly_breakdown.csv  # [NEW] 月度切分结果
│   └── xgb_tune/
├── output/result.csv          # 最终提交
├── test/
│   ├── score_self.py
│   └── test_windows.py
├── analyze_window4_monthly.py  # 月度诊断脚本
├── OPTIMIZATION_GUIDE.md       # 本指南
└── pyproject.toml
```
