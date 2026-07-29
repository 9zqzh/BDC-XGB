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
    # 因子IC筛选后的特征列表（None=使用全部特征，运行 factor_ic.py 后手动填入显著因子）
    'selected_features': None,
}

# ============ XGBRanker 超参数 ============
xgb_config = {
    'max_depth': 5,              # 树最大深度（保持不变）
    'learning_rate': 0.03,       # 学习率
    'n_estimators': 300,         # 最大树数量（↓500→300，减少无效训练）
    'subsample': 0.6,            # 行采样比例（保持不变）
    'colsample_bytree': 0.3,     # 列采样比例（↑0.2→0.3，给模型足够特征空间）
    'reg_alpha': 1.0,            # L1 正则化（↓2.0→1.0，避免过度稀疏化）
    'reg_lambda': 5.0,           # L2 正则化（↓10.0→5.0，保留有效信号）
    'min_child_weight': 10,      # 最小叶子权重
    'objective': 'rank:ndcg',    # 直接优化NDCG
    # 'objective': 'rank:pairwise', # 备选：噪声环境可用
    'eval_metric': 'ndcg@5',     # 评估指标
    'ndcg_exp_gain': False,      # 禁用指数增益（标签>31时必需）
    'early_stopping_rounds': 15, # 早停轮数
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
        ('2021-01-04', '2024-08-30', '2024-09-02', '2025-02-28', '窗口3_924行情'),
        ('2022-01-04', '2025-06-30', '2025-07-01', '2026-06-30', '窗口4_近期市场'),
    ],
}
