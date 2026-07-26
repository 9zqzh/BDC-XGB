# BDC 2026 项目从零运行命令指南

## 前置要求

- Python 3.10 ~ 3.12
- 操作系统：Linux / macOS / Windows 均可

---

## 第 1 步：安装 uv 包管理器

```bash
pip install uv
```

如果已有 conda 环境，也可以直接 `pip install uv`。

---

## 第 2 步：安装 TA-Lib C 库（必须，否则特征工程会报错）

**Linux：**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make -j1
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

**macOS：**
```bash
brew install ta-lib
```

**Windows：**
去 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 下载对应 Python 版本的 `.whl` 文件，然后用 pip 安装。或者使用 conda：
```bash
conda install -c conda-forge ta-lib
```

---

## 第 3 步：安装项目 Python 依赖

在项目根目录下执行：

```bash
cd "D:\9z's ProjectS\BDC(XGB)"
uv sync
```

安装成功后终端会显示 `Resolved XX packages`。

---

## 第 4 步：激活虚拟环境

**Linux / macOS：**
```bash
source .venv/bin/activate
```

**Windows：**
```bash
.\.venv\Scripts\activate
```

激活后终端提示符前面会出现 `(.venv)` 前缀。

---

## 第 5 步：下载股票数据

先确认 `get_stock_data.py` 中的日期范围是否满足需求（默认在文件约第 223-224 行）：

```python
start_date = "2024-01-01"
end_date = "2026-03-15"
```

确认无误后执行：

```bash
python get_stock_data.py
```

数据将保存为 `data/stock_data.csv`。如果中途出现网络错误（Baostock 不太稳定），关掉代理多试几次。

---

## 第 6 步：划分训练集和测试集

先确认 `data/split_train_test.py` 中的日期范围。训练集需要覆盖你想训练的时段，测试集通常取最后 5 个交易日：

```bash
python data/split_train_test.py \
    --train-start 2024-01-02 \
    --train-end 2026-03-06 \
    --test-start 2026-03-09 \
    --test-end 2026-03-13
```

执行后会在 `data/` 目录下生成 `train.csv` 和 `test.csv`。

---

## 第 7 步：检查配置文件

打开 `code/src/config.py`，确认关键参数是否符合你的需求：

```python
sequence_length = 60        # 使用 60 个交易日的历史序列
feature_num = '158+39'      # 特征方案：39 个技术指标 + 158 个 Alpha 特征
batch_size = 4              # 每批处理的天数
num_epochs = 50             # 最大训练轮数
learning_rate = 1e-5        # 初始学习率
```

如果只是想快速跑通，可以临时把 `num_epochs` 改小（比如 10），加速验证流程。

---

## 第 8 步：训练模型

```bash
sh train.sh
```

Windows 下直接运行：

```bash
python code/src/train.py
```

训练产物保存在 `model/60_158+39/` 目录下：
- `best_model.pth` — 最佳模型权重
- `scaler.pkl` — 特征标准化器
- `config.json` — 训练配置快照
- `final_score.txt` — 最佳验证得分记录
- `log/` — TensorBoard 日志

如需查看训练曲线，另开终端：

```bash
tensorboard --logdir model/60_158+39/log
```

---

## 第 9 步：生成预测结果

```bash
sh test.sh
```

Windows 下直接运行：

```bash
python code/src/predict.py
```

预测结果输出到 `output/result.csv`，格式为：

```csv
stock_id,weight
000001,0.2
600519,0.2
...
```

---

## 第 10 步：本地自评得分

```bash
python test/score_self.py
```

得分结果保存在 `temp/tmp.csv`。这个分数是你自己参考用的加权收益率，不代表赛事方最终评分。

---

## 第 11 步（可选）：滚动窗口交叉验证

如果你已经实现了 `cross_val.py`：

```bash
python code/src/cross_val.py
```

它会在 `model/60_158+39/cross_val_report.txt` 中输出每个窗口的 final_score 均值和标准差，帮你判断模型在不同市场阶段的稳定性。

---

## 完整一键流程（所有步骤都确认过后）

```bash
# 1. 进入项目并激活环境
cd "D:\9z's ProjectS\BDC(XGB)"
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 2. 下载数据（如果还没有）
python get_stock_data.py

# 3. 划分数据集
python data/split_train_test.py

# 4. 训练
sh train.sh                 # Windows: python code/src/train.py

# 5. 预测
sh test.sh                  # Windows: python code/src/predict.py

# 6. 本地评分
python test/score_self.py
```

---

## 常见问题速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'talib'` | TA-Lib C 库未安装 | 回到第 2 步 |
| `请安装TA-Lib库` | 同上 | 同上 |
| `uv sync` 报网络错误 | PyPI 或 PyTorch 源不可达 | 关闭代理或换源重试 |
| `get_stock_data.py` 卡住不动 | Baostock API 不稳定 | Ctrl+C 重跑，脚本会自动跳过已有股票 |
| `CUDA out of memory` | batch_size 太大或 GPU 显存不够 | 减小 `batch_size` 或改用 `device='cpu'` |
| 预测结果全是同一只股票 | 模型没收敛，所有股票得分相同 | 检查 config 参数、增加 epochs、降低 lr |
| `FileNotFoundError: train.csv` | 还没划分数据集 | 回到第 6 步 |

---

## A/B 实验流程（对比两种标签方案）

如果你想验证"原标签 vs 新标签"的效果差异：

```bash
# ---- 方案 A：原标签 ----
python code/src/train.py              # 训练
python code/src/predict.py            # 预测
python test/score_self.py             # 记下分数 A
cp model/60_158+39/best_model.pth model/60_158+39/best_model_A.pth
cp model/60_158+39/scaler.pkl model/60_158+39/scaler_A.pkl
cp output/result.csv output/result_A.csv

# ---- 方案 B：改标签后 ----
# （手动修改 train.py 中 _build_label_and_clean 的标签定义）
python code/src/train.py              # 重新训练
python code/src/predict.py            # 重新预测
python test/score_self.py             # 记下分数 B
cp model/60_158+39/best_model.pth model/60_158+39/best_model_B.pth
cp model/60_158+39/scaler.pkl model/60_158+39/scaler_B.pkl
cp output/result.csv output/result_B.csv

# ---- 比较 ----
echo "方案 A 分数:"
cat temp/tmp.csv   # 如果你跑 A 之后保存了
echo "方案 B 分数:"
python test/score_self.py
```
