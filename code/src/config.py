# 配置参数
sequence_length = 60
feature_num = '158+39'

# ============ 数据 + 通用配置 ============
config = {
    'sequence_length': sequence_length,   # 特征展平窗口：取最后 N 天做特征
    'feature_num': feature_num,           # 特征方案：39 或 158+39
    'xgb_flatten_days': 10,               # XGBRanker 展平天数（60→10，信噪比↑6×）
    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}

# ============ XGBRanker 超参数 ============
xgb_config = {
    'max_depth': 6,              # 树最大深度
    'learning_rate': 0.05,       # 学习率
    'n_estimators': 500,         # 最大树数量
    'subsample': 0.8,            # 行采样比例
    'colsample_bytree': 0.8,     # 列采样比例
    'reg_alpha': 0.1,            # L1 正则化
    'reg_lambda': 1.0,           # L2 正则化
    'min_child_weight': 5,       # 最小叶子权重
    'objective': 'rank:ndcg',    # 直接优化 NDCG，对 TopK 更敏感
    'eval_metric': 'ndcg@5',     # 评估指标
    'ndcg_exp_gain': False,      # 禁用指数增益（标签>31时必需）
    'early_stopping_rounds': 30, # XGBoost 自带早停
    'verbosity': 1,              # 训练日志级别
    'n_jobs': -1,                # 并行线程数
}

# ============ 验证增强配置 ============
config_extended = {
    'min_gap': 0.005,
    'eval_top_k': 5,
    'val_months': 12,
    'cross_val_windows': [
        ('2018-01-02', '2021-12-31', '2022-01-04', '2022-06-30', '窗口1_早期回调'),
        ('2020-01-02', '2023-06-30', '2023-07-03', '2024-01-05', '窗口2_弱复苏'),
        ('2021-01-04', '2024-08-30', '2024-09-02', '2024-12-31', '窗口3_924行情'),
        ('2022-01-04', '2025-06-30', '2025-07-01', '2026-06-30', '窗口4_近期市场'),
    ],
}
