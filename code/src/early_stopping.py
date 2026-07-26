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
