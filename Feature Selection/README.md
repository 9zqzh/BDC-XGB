# BDC-XGB 特征筛选运行说明

所有命令默认在项目根目录执行：

```powershell
cd "D:\藏经阁\大数据挑战赛\THU_project\BDC-XGB"
uv sync
```

不执行 GroupMRMR。运行顺序：

```text
Rank IC / ICIR -> Refitting -> 双因子 25 组 -> 滚动窗口 final_score
```

## 运行配置（RTX 5060 级别 GPU）

特征筛选中的 XGBRanker 默认使用 CUDA：`tree_method=hist`、`device=cuda`，默认使用4个 CPU 线程配合 GPU。RTX 5060 8GB 显存可先使用默认配置；流程按窗口和实验顺序执行，不会同时启动多个 XGBoost 模型。

首次运行前检查 GPU：

```powershell
uv run python -c "import xgboost as xgb; print(xgb.__version__); print(xgb.build_info())"
nvidia-smi
```

运行特征筛选：

```powershell
$env:BDC_XGB_DEVICE = "cuda"
$env:BDC_XGB_N_JOBS = "4"
uv run python "Feature Selection/run_selection.py" --step all
```

如果对方没有可用 CUDA，必须切换到 CPU：

```powershell
$env:BDC_XGB_DEVICE = "cpu"
$env:BDC_XGB_N_JOBS = "-1"
uv run python "Feature Selection/run_selection.py" --step all
```

`rank_ic` 和 `double_sort` 主要由 Pandas 执行，GPU 加速有限；GPU 主要用于 `refitting` 和 `rolling` 中反复训练 XGBRanker。显存不足时，优先将 `BDC_XGB_N_JOBS` 调为 `2`，并减少同时运行的其他 GPU 程序。

## 1. Rank IC / ICIR

```powershell
uv run python "Feature Selection/run_selection.py" --step rank_ic
```

输出到 `Feature Selection/results/rank_ic/`：`rank_ic_report.csv`、`selected_features.json`、`rank_ic_summary.md`。
`instrument` 不参与 Rank IC 筛选。

## 2. Refitting 分组消融

```powershell
uv run python "Feature Selection/run_selection.py" --step refitting
```

测试 `baseline`、`volume_only`、`range_only`、`momentum_only`、`no_volume`、`no_range`、`no_momentum` 和 `selected_features`。

输出到 `Feature Selection/results/refitting/`：`refitting_window_results.csv`、`refitting_summary.csv`、`retained_feature_groups.json`。

分组重要性为：

```text
baseline_mean_final_score - 消融方案_mean_final_score
```

## 3. 双因子 25 组 / 协同效应

```powershell
uv run python "Feature Selection/run_selection.py" --step double_sort
```

默认测试动量趋势、量能/流动性、区间位置/突破和波动风险之间的核心组合。

只有同时满足以下条件的组合才进入滚动验证：

```text
long_short_spread > 0
joint_monotonicity > 0
double_lift > 0
```

25 组收益先按交易日计算组合平均收益，再跨交易日等权平均。
结果保存到 `Feature Selection/results/double_sort/double_sort_results.csv`。

## 4. 滚动窗口最终验证

```powershell
uv run python "Feature Selection/run_selection.py" --step rolling
```

比较 `baseline`、`baseline_with_instrument`、`rank_ic`、`refitting_groups` 和 `double_sort_interaction`。
其中 `baseline_with_instrument` 仅作对照，不参与最终推荐。

推荐规则：

```text
positive_window_ratio >= 0.5
final_score_std <= baseline_std * 1.25
在满足条件的方案中，final_score_mean 最高
```

结果保存到 `Feature Selection/results/rolling/`，重点查看 `rolling_summary.csv` 和 `recommended_candidate.json`。

## 5. 一键执行

```powershell
uv run python "Feature Selection/run_selection.py" --step all
```

该命令依次执行 `rank_ic`、`refitting`、`double_sort` 和 `rolling`。完整流程会多次重训 XGBRanker，运行时间较长。

## 6. 使用推荐模型预测

滚动验证后查看 `Feature Selection/results/rolling/recommended_candidate.json`，再将对应模型目录传给预测脚本：

```powershell
uv run python code/src/predict.py --model-dir "Feature Selection/results/rolling/rank_ic/window_4"
```

模型目录应包含 `model.pkl`、`scaler.pkl` 和 `features.json`。预测脚本会按 `features.json` 构造输入，并自动重建双因子交互特征。结果输出到 `output/result.csv`。

## 7. 重点指标

- `final_score_mean`：滚动窗口平均选股表现；
- `final_score_std`：时间稳定性；
- `positive_window_ratio`：正收益窗口比例；
- `topk_hit_rate_mean`：TopK 命中率；
- `pred_return_sum_mean`：模型 Top5 组合平均收益。
