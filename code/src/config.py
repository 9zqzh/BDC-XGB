# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}

# ============ 验证增强配置（新增项，不影响已有逻辑） ============
config_extended = {
    'min_gap': 0.005,              # 第一层：final_score 分母过滤阈值，最优与随机差距须>0.5%
    'eval_top_k': 5,               # 第二层：Top-k 命中率的 k 值
    'val_months': 6,                # 验证集取末尾几个月（默认6个月，确保≥20个有效交易日）
    'cross_val_windows': [         # 第三层：滚动窗口定义 (train_start, train_end, val_start, val_end, label)
        # 窗口1：早期震荡→疫情牛市，验证2022上半年（A股回调期）
        ('2018-01-02', '2021-12-31', '2022-01-04', '2022-06-30', '窗口1_早期回调'),
        # 窗口2：疫情复苏→盘整，验证2023下半年（弱复苏阶段）
        ('2020-01-02', '2023-06-30', '2023-07-03', '2024-01-05', '窗口2_弱复苏'),
        # 窗口3：中期震荡→924行情，验证2024下半年（政策驱动暴涨）
        ('2021-01-04', '2024-08-30', '2024-09-02', '2024-12-31', '窗口3_924行情'),
        # 窗口4：近年数据→当前，验证2025下半年~2026上半年（最新市场）
        ('2022-01-04', '2025-06-30', '2025-07-01', '2026-06-30', '窗口4_近期市场'),
    ],
}

# ============ 早停配置（新增项，不影响已有逻辑） ============
early_stop_config = {
    'enabled': True,                # 是否启用早停，False 则退化为原行为
    'base_patience': 15,            # 基础耐心：无改进时容忍的 epoch 数
    'min_delta': 1e-4,              # 视为"有改进"的最小绝对增量
    'warmup_epochs': 15,            # 前 N 轮不触发早停（给模型学习基本排序的时间）
    'smoothing_alpha': 0.3,         # EMA 平滑系数，越小越平滑（0.3 = 70% 历史 + 30% 当前）
    'min_acceptable_score': 0.0,    # 最佳分数低于此值时强制不早停（模型至少要优于随机）
}