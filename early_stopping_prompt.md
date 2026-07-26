# 噪声感知自适应早停机制 — 修改提示词

## 核心约束（必须遵守）

1. **不改变训练循环的核心逻辑**：模型的 forward、反向传播、优化器更新、学习率调度保持原样不动。
2. **不改变项目目录结构**：只需在 `code/src/` 下新增 `early_stopping.py`，在 `config.py` 末尾新增配置项，在 `train.py` 的 epoch 循环中插入少量调用。
3. **不触碰 `model.py`、`utils.py`、`predict.py`**。
4. **`sh train.sh` 的行为完全不变**——唯一的区别是训练可能提前终止（如果触发早停），不会跑满 50 个无谓的 epoch。

---

## 背景分析：为什么标准早停不适合本项目

### 问题 1：监控信号天然高噪声

`final_score` 是每日计算后取均值，而金融数据中有些天"选谁都差不多"（窄幅震荡），有些天"选对就是暴赚"。这种日间方差使得 final_score 在相邻 epoch 之间天然波动 5~10%，用原始值做 patience 判断会导致大量误触发——一个好 epoch 后面跟一个差的，不是因为过拟合，只是因为那天的股票分布碰巧不利。

### 问题 2：LinearLR 调度器让后期学习极慢

lr 从 1e-5 线性衰减到 2e-6，第 40 个 epoch 的 lr 只有初始的 36%。在这么小的 lr 下，模型每次改进的幅度极小（final_score 可能只变化 0.001），标准 patience=5 根本等不到就被触发。

### 问题 3：排序任务没有"损失平台期"的清晰信号

分类任务的 val_loss 通常会先降后升（经典 U 形），但对于 ListNet + Pairwise 的排序损失，训练集和验证集的损失经常是同向运动的——训练集损失还在降，验证集 final_score 可能已经横盘了。只盯一个信号会漏掉重要的互补信息。

---

## 早停机制总览：四层防护

```
                    原始 final_score (含噪声)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                 ▼
    [层 1: EMA 平滑]   [层 2: Warmup]   [层 4: Floor 保护]
    滤除日间随机波动      前15 epoch 不触发    未学会时不终止
          │
          ▼
    平滑后的 final_score
          │
          ▼
    [层 3: 自适应 Patience]
    lr 低 → patience 自动延长
    lr 高 → patience 保持正常
          │
          ▼
     判断：是否触发早停？
```

### 层 1：EMA 信号平滑（防噪声误触发）

用指数移动平均平滑 final_score 序列，而不是直接用原始值做判断。

```
smoothed_score[t] = α × raw_score[t] + (1-α) × smoothed_score[t-1]
其中 α = 0.3
```

平滑系数 0.3 意味着当前原始值占 30% 权重、历史平滑值占 70%。第 30 个 epoch 的 0.42 → 第 31 个 epoch 的 0.38 这种随机波动会被滤掉，只有当下降趋势持续多个 epoch 时，平滑值才会反映出来。

### 层 2：Warm-up 保护期（防过早终止）

前 `warmup_epochs` 个 epoch **完全关闭早停逻辑**——不记录最佳分数、不更新计数器。你的模型有 197 维特征通过 StockTransformer 学习排序，前期 epoch 波动剧烈是正常的，需要足够时间建立基础排序能力。

默认配置：`warmup_epochs = 15`（总共 50 个 epoch 的 30%）

### 层 3：自适应 Patience（防后期误杀）

patience 不是固定值，而是根据当前学习率动态缩放：

```
effective_patience = base_patience × max(1.0, 1.0 / (current_lr / initial_lr + 0.1))
```

举例说明效果：

| 当前 lr | 与初始 lr 比值 | 缩放因子 | 实际 patience (base=15) |
|---------|---------------|---------|------------------------|
| 1e-5 | 1.0 | 1/1.1 ≈ 0.91 → clamp 到 1.0 | 15 |
| 6e-6 | 0.6 | 1/0.7 ≈ 1.43 | 21 |
| 4e-6 | 0.4 | 1/0.5 = 2.0 | 30 |
| 2e-6 | 0.2 | 1/0.3 ≈ 3.33 | 50 |

当 lr 衰减到初始的 20% 时（第 40+ epoch），effective_patience 自动提升到 50 个 epoch，这意味着后期模型每次改进虽小，但早停器会给予极其充足的等待时间。

### 层 4：Floor 保护（防无信号市误停）

增加一个额外条件：如果当前 best_score 低于 `min_acceptable_score`（即模型从未超过这个底线），即便 patience 耗尽也**绝不触发早停**。

在你的场景中，`min_acceptable_score` 建议设为 `0.0`。模型起码要比随机选股好才值得保存——如果训练中一直没跨过 0.0 这条线，让它继续跑。

---

## 详细实现方案

### 文件 1：`code/src/early_stopping.py`（新增文件）

```python
"""
噪声感知自适应早停器 (Noise-Aware Adaptive EarlyStopping)

针对金融排序学习任务设计的四层防护早停机制：
  层1 — EMA 平滑，滤除 final_score 的日间随机波动
  层2 — Warm-up 保护期，前 N 个 epoch 不触发早停
  层3 — 自适应 patience，lr 越低 patience 越长
  层4 — Floor 保护，模型未学会任何东西时不终止
"""


class NoiseAwareEarlyStopping:
    """
    参数说明：
        base_patience: int      基础耐心（无改进时容忍的 epoch 数），默认 15
        min_delta: float        视为"改进"的最小阈值（绝对变化量），默认 1e-4
        warmup_epochs: int      前 N 个 epoch 完全不触发早停，默认 15
        smoothing_alpha: float  EMA 平滑系数，0~1 之间，越小越平滑，默认 0.3
        min_acceptable_score: float  底线分数，best_score 低于此值时绝不早停，默认 0.0
        verbose: bool           是否打印早停相关日志，默认 True
    """

    def __init__(
        self,
        base_patience=15,
        min_delta=1e-4,
        warmup_epochs=15,
        smoothing_alpha=0.3,
        min_acceptable_score=0.0,
        verbose=True,
    ):
        self.base_patience = base_patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.smoothing_alpha = smoothing_alpha
        self.min_acceptable_score = min_acceptable_score
        self.verbose = verbose

        # 内部状态
        self.best_score = None          # 历史最佳平滑分数
        self.best_raw_score = None      # 历史最佳原始分数（用于日志）
        self.best_epoch = -1            # 最佳分数所在的 epoch（1-based）
        self.counter = 0                # 无改进的连续 epoch 计数
        self.early_stop = False         # 是否触发早停
        self.smoothed_score = None      # 当前 EMA 平滑分数
        self.initial_lr = None          # 初始学习率（用于自适应 patience）

    def _compute_effective_patience(self, current_lr):
        """
        层3：根据当前 lr 与初始 lr 的比值计算自适应 patience。
        当 lr 衰减得很低时，自动延长 patience。
        """
        if self.initial_lr is None or current_lr is None:
            return self.base_patience
        ratio = current_lr / (self.initial_lr + 1e-12)
        scale = 1.0 / max(ratio + 0.1, 0.05)  # 上界防止除以零
        return max(self.base_patience, int(round(self.base_patience * scale)))

    def step(self, score, epoch, current_lr=None):
        """
        每个 epoch 结束后调用一次。

        参数：
            score: float     当前 epoch 的原始 final_score（未平滑）
            epoch: int       当前 epoch 编号（0-based）
            current_lr: float 当前学习率，用于自适应 patience（可选）

        返回：
            bool  True 表示触发早停，False 表示继续训练
        """
        # 记录初始 lr
        if self.initial_lr is None and current_lr is not None:
            self.initial_lr = current_lr

        # 层1：EMA 平滑
        if self.smoothed_score is None:
            self.smoothed_score = score
        else:
            self.smoothed_score = (
                self.smoothing_alpha * score
                + (1.0 - self.smoothing_alpha) * self.smoothed_score
            )

        epoch_1based = epoch + 1

        # 层2：Warm-up 期间只记录最佳分数，不触发早停
        if epoch_1based <= self.warmup_epochs:
            if self.best_score is None or self.smoothed_score > self.best_score + self.min_delta:
                self.best_score = self.smoothed_score
                self.best_raw_score = score
                self.best_epoch = epoch_1based
            if self.verbose:
                print(
                    f"[早停] Epoch {epoch_1based:3d} | "
                    f"raw_final_score={score:.6f} | "
                    f"smoothed_score={self.smoothed_score:.6f} | "
                    f"best_smoothed={self.best_score:.6f} | "
                    f"状态: warm-up (前{self.warmup_epochs}轮不触发)"
                )
            return False

        # 判断是否有改进
        improved = self.smoothed_score > self.best_score + self.min_delta

        if improved:
            self.best_score = self.smoothed_score
            self.best_raw_score = score
            self.best_epoch = epoch_1based
            self.counter = 0
        else:
            self.counter += 1

        # 层3：计算自适应 patience
        effective_patience = self._compute_effective_patience(current_lr)

        # 层4：Floor 保护 — 如果至今最佳分数低于底线，强制不触发早停
        floor_blocked = self.best_score < self.min_acceptable_score

        # 判断是否触发
        if self.counter >= effective_patience and not floor_blocked:
            self.early_stop = True

        if self.verbose:
            floor_msg = " [FLOOR保护:未达标] " if floor_blocked else ""
            trigger_msg = " *** 触发早停 ***" if self.early_stop else ""
            lr_msg = f"lr={current_lr:.2e}" if current_lr is not None else ""
            effective_pat_msg = (
                f"eff_patience={effective_patience}"
                if not improved and not floor_blocked
                else f"patience={self.base_patience}"
            )
            print(
                f"[早停] Epoch {epoch_1based:3d} | "
                f"raw={score:.6f} | "
                f"smooth={self.smoothed_score:.6f} | "
                f"best={self.best_score:.6f} (epoch {self.best_epoch}) | "
                f"counter={self.counter}/{effective_patience} | "
                f"{lr_msg} | "
                f"{effective_pat_msg}"
                f"{floor_msg}"
                f"{trigger_msg}"
            )

        return self.early_stop

    def summary(self):
        """返回早停过程的摘要字符串。"""
        return (
            f"早停摘要: best_smoothed_score={self.best_score:.6f} "
            f"(raw={self.best_raw_score:.6f}) "
            f"at epoch {self.best_epoch} | "
            f"early_stop={'是' if self.early_stop else '否'} | "
            f"smoothed_at_stop={self.smoothed_score:.6f}"
        )
```

### 文件 2：`code/src/config.py`（追加内容）

在 `config.py` 文件末尾（在已有的 `config_extended` 下面）追加以下内容：

```python
# ============ 早停配置（新增项，不影响已有逻辑） ============
early_stop_config = {
    'enabled': True,                # 是否启用早停，False 则退化为原行为
    'base_patience': 15,            # 基础耐心：无改进时容忍的 epoch 数
    'min_delta': 1e-4,              # 视为"有改进"的最小绝对增量
    'warmup_epochs': 15,            # 前 N 轮不触发早停（给模型学习基本排序的时间）
    'smoothing_alpha': 0.3,         # EMA 平滑系数，越小越平滑（0.3 = 70% 历史 + 30% 当前）
    'min_acceptable_score': 0.0,    # 最佳分数低于此值时强制不早停（模型至少要优于随机）
}
```

### 文件 3：`code/src/train.py`（修改内容）

**3a. 文件顶部新增 import**

在第 3~4 行附近（与现有 import 放在一起）：

```python
from early_stopping import NoiseAwareEarlyStopping
```

**3b. 在 `train_one_window` 函数的 epoch 循环中插入早停逻辑**

找到 `train_one_window` 函数中当前的 epoch 循环部分（大约第 660 行开始）：

```python
    # 8. 排序模型训练
    best_score = -float('inf')
    best_epoch = -1
    best_extended_metrics = {}

    for epoch in range(config['num_epochs']):
```

在 `best_extended_metrics = {}` 之后、`for epoch` 之前，新增早停器初始化：

```python
    # ---- 新增：初始化早停器 ----
    esc = config.get('early_stop_config', {})
    early_stopper = None
    if esc.get('enabled', True):
        early_stopper = NoiseAwareEarlyStopping(
            base_patience=esc.get('base_patience', 15),
            min_delta=esc.get('min_delta', 1e-4),
            warmup_epochs=esc.get('warmup_epochs', 15),
            smoothing_alpha=esc.get('smoothing_alpha', 0.3),
            min_acceptable_score=esc.get('min_acceptable_score', 0.0),
            verbose=True,
        )
    # ---- 新增结束 ----
```

在 epoch 循环内部，找到 `scheduler.step()` 那一行（大约第 682 行），在其**之后**插入早停判断：

```python
        scheduler.step()
        if writer:
            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)

        # ---- 新增：早停判断 ----
        if early_stopper is not None:
            current_lr = scheduler.get_last_lr()[0]
            should_stop = early_stopper.step(
                score=current_final_score,
                epoch=epoch,
                current_lr=current_lr,
            )
            if should_stop:
                print(f"\n[早停] 在第 {epoch+1} 个 epoch 触发早停，停止训练。")
                break
        # ---- 新增结束 ----
```

**3c. 在 epoch 循环结束后新增早停摘要**

找到 `train_one_window` 函数中打印训练完成的那一行（大约第 694 行）：

```python
    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
```

在其**之前**插入早停摘要：

```python
    # ---- 新增：打印早停摘要 ----
    if early_stopper is not None:
        print(f"\n{early_stopper.summary()}")
    # ---- 新增结束 ----
```

**注意**：以上插入位置在 `train_one_window` 函数中的 `for epoch in range(config['num_epochs']):` 循环内。`main()` 函数（第 702 行）无需任何修改——它通过调用 `train_one_window` 间接获得早停能力。

---

## 配置调优指南

如果需要对早停参数进行调优，建议按以下优先级调整：

| 参数 | 默认值 | 何时调大 | 何时调小 |
|------|--------|---------|---------|
| `base_patience` | 15 | 模型在 15 轮内经常有微小改进却来不及显现 | 训练非常快，想早点结束无效 epoch |
| `smoothing_alpha` | 0.3 | 日间波动特别大（final_score 方差 >0.1） | 信号本身很平滑，希望早停更灵敏 |
| `warmup_epochs` | 15 | 模型前期波动剧烈或数据集更大 | 模型收敛快或 num_epochs 本身就小 |
| `min_delta` | 1e-4 | 希望只对明显改进反应 | final_score 的量级本身就小 |
| `min_acceptable_score` | 0.0 | 想提高"及格线"门槛 | 任务极难，0.0 就已是合理目标 |

推荐验证方法：在同一组超参数下，分别用 `enabled=True` 和 `enabled=False` 各跑一次。对比两者的 best_score 和触发的 epoch 位置。如果早停在 20 epoch 就停了但 disabled 版本在第 38 epoch 还有显著提升，说明 patience 或 smoothing_alpha 需要调大。

---

## 文件变更清单

| 文件 | 操作 | 行数变化 | 说明 |
|------|------|---------|------|
| `code/src/early_stopping.py` | **新增** | ~110 行 | `NoiseAwareEarlyStopping` 类 |
| `code/src/config.py` | 末尾追加 | +10 行 | `early_stop_config` 字典 |
| `code/src/train.py` | 内部插入 | +22 行，0 处删除 | import + 初始化 + step 调用 + 摘要 |
| `code/src/model.py` | 不动 | 0 | — |
| `code/src/utils.py` | 不动 | 0 | — |
| `code/src/predict.py` | 不动 | 0 | — |

---

## 验收标准

1. `sh train.sh` 仍能正常运行，产出 model 目录结构与原来一致。
2. 控制台输出中，从第 1 epoch 开始可以看到 `[早停]` 前缀的日志行，包含 raw、smooth、best、counter 等关键值。
3. 前 15 个 epoch 的日志中会显示 `状态: warm-up (前15轮不触发)`。
4. 第 16 epoch 之后，如连续多轮无改进且 best_score 已超过 0.0，日志末尾出现 `*** 触发早停 ***`，训练终止。
5. 无论是否触发早停，训练结束时都会打印 `早停摘要: best_smoothed_score=... (raw=...) at epoch ...`。
