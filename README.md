# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案。当前训练、预测、调参和交叉验证链路均使用 `XGBRanker`，不使用 `StockTransformer`。

- 输入：每只股票过去一段时间的量价与技术特征；
- 特征：默认使用 `158+39` 特征，并取最近 `xgb_flatten_days=10` 天展平成 XGBoost 输入；
- 模型：`xgboost.XGBRanker`；
- 目标：按交易日分组，对同一天候选股票做排序；
- 输出：排序分最高的前5只股票，默认等权重 `0.2`。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（默认 `data/train.csv`）；
2. 做特征工程（`39` 或 `158+39` 特征，当前默认 `158+39`）；
3. 构建标签：`open_t1` 到 `open_t5` 的未来收益，并减去当日平均收益，得到超额收益；
4. 按交易日组织排序样本，每个交易日是一个 XGBoost ranking group；
5. 将最近若干天特征展平成单行向量，当前默认 `10 × 197 + 3` 维；
6. 训练 `XGBRanker`，用验证集 `final_score` 评估效果；
7. 保存 `best_model.json`、`best_model.pkl` 和 `scaler.pkl`；
8. 推理时加载 `best_model.pkl`，对最新交易日沪深300股票打分，输出 Top5。

---

## 2. 代码结构说明

### [config.py](code/src/config.py)
统一管理训练与推理参数，包括：
- `sequence_length`：历史上下文长度，默认60；
- `feature_num`：特征方案，默认 `158+39`；
- `xgb_flatten_days`：XGBRanker 实际展平的最近天数，默认10；
- `xgb_config`：XGBRanker 超参数，如 `max_depth`、`learning_rate`、`n_estimators`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`、`min_child_weight`；
- `objective='rank:ndcg'`；
- `eval_metric='ndcg@5'`；
- `config_extended`：验证月数、TopK、滚动交叉验证窗口等。

### [train.py](code/src/train.py)
训练主脚本，关键内容：
- `_preprocess_common()`：按股票分组并行特征工程、股票ID映射、标签构建；
- `_build_label_and_clean()`：构建未来5日超额收益标签；
- `split_train_val_by_last_month()`：默认用最后12个月做验证集；
- `flatten_sequences_to_xgb()`：将时序特征展平成 XGBRanker 输入；
- `_continuous_labels_to_ranks()`：将连续超额收益转成组内整数排名；
- `train_one_window()`：训练单个窗口的 `XGBRanker`，并保存模型与评估报告；
- `evaluate_xgb_model()`：按日计算 Top5 表现和 `final_score`。

训练产物：
- `best_model.json`：XGBoost 原生模型文件；
- `best_model.pkl`：joblib 保存的模型，供 `predict.py` 加载；
- `scaler.pkl`：标准化器；
- `config.json`：训练时配置快照；
- `final_score.txt`：验证集分数；
- `eval_report.txt`：扩展评估报告。

注意：`config.py` 中保留了 `early_stopping_rounds` 配置，但当前 `train.py` 的 `model.fit()` 未传入早停参数，因此实际训练按 `n_estimators` 固定轮数运行。

### [predict.py](code/src/predict.py)
推理主脚本，流程：
1. 加载 `best_model.pkl` 和 `scaler.pkl`；
2. 读取 `data/train.csv`，取最新交易日；
3. 只保留沪深300成分股；
4. 执行与训练一致的特征工程和标准化；
5. 展平最近 `xgb_flatten_days` 天特征；
6. 用 `XGBRanker` 打分并取 Top5；
7. 输出 `output/result.csv`。

### [xgb_tune.py](code/src/xgb_tune.py)
XGBRanker 超参数网格搜索脚本。

当前搜索参数：
- `max_depth`
- `learning_rate`
- `subsample`
- `min_child_weight`

每组参数都会调用一次完整训练流程，并按验证集 `final_score` 排序输出推荐组合。

注意：调参过程会为每个 trial 生成模型文件和 memmap 临时 `.dat` 文件；当前代码中 `.dat` 文件使用 `delete=False`，不会自动清理。大规模网格搜索可能占用较多磁盘空间。

### [cross_val.py](code/src/cross_val.py)
滚动窗口交叉验证脚本。

作用：
- 在多个市场阶段分别训练和验证 `XGBRanker`；
- 汇总 `final_score`、`topk_hit_rate`、`spearman_rho`、`win_rate` 等指标；
- 用于判断参数和特征方案的稳定性。

### [utils.py](code/src/utils.py)
包含特征工程与数据处理逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个 Alpha 类特征；
- `engineer_features_158plus39()`：合并 `158+39` 特征。

说明：特征工程依赖 `TA-Lib`。

### [model.py](code/src/model.py)
该文件保留了旧版 `StockTransformer` 定义，但当前训练、预测、调参、交叉验证脚本均未引用它。当前项目实际使用 `XGBRanker`。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- `output/result.csv`

输出列：
- `stock_id`
- `weight`

---

## 4. 运行方法（推荐使用 uv）

1) 安装依赖

```bash
uv sync
```

2) 激活虚拟环境

Linux/macOS：

```bash
source .venv/bin/activate
```

Windows：

```powershell
.\.venv\Scripts\activate
```

3) 训练模型

```bash
sh train.sh
```

Windows 可直接运行：

```powershell
python code/src/train.py
```

4) 生成预测结果

```bash
sh test.sh
```

Windows 可直接运行：

```powershell
python code/src/predict.py
```

5) 可选：超参数搜索

```bash
uv run python code/src/xgb_tune.py
```

6) 可选：滚动窗口交叉验证

```bash
python code/src/cross_val.py
```

---

## 5. 常见问题

1) `TA-Lib` 安装失败

本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装 Python 包。

2) 多进程相关问题

`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS 下请通过脚本入口运行，不建议在交互式环境里直接调用多进程逻辑。

3) 是否使用 GPU

当前主模型是 `XGBRanker`，配置中使用 `tree_method='hist'`，不是 PyTorch GPU 训练链路。

4) 超参搜索磁盘占用

`xgb_tune.py` 会多次调用完整训练流程，并在各 trial 目录下保存中间文件。当前 memmap 临时 `.dat` 文件不会自动删除，长时间调参前建议确认磁盘空间，或手动清理 `model/*/xgb_tune/` 下不需要的 trial 文件。
