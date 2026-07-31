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
    # 因子：IC124最优基准 + 全21行业 + P2交叉（9/10 IC通过）
    'selected_features': [
        '开盘', '收盘', '最高', '最低', '成交额', '振幅', '涨跌额', '换手率',
        'KMID', 'KLEN', 'KUP', 'KLOW', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
        'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
        'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
        'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
        'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
        'RESI10', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
        'QTLU20', 'QTLU30', 'QTLU60',
        'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
        'RANK5', 'RANK30',
        'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN20', 'IMIN30', 'IMIN60',
        'IMXD5', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30',
        'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP20', 'CNTP30', 'CNTD20', 'CNTD30',
        'SUMP20', 'SUMP30',
        'SUMN20', 'SUMN30', 'SUMN60',
        'SUMD20', 'SUMD30', 'SUMD60',
        'VMA60',
        'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30',
        'VSUMP30', 'VSUMP60', 'VSUMN30', 'VSUMN60', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
        'obv', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
        'volatility_10', 'volatility_20', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
        # 基本面 + 全21行业
        'PE_TTM', 'PB', 'ROE_approx', '总市值_对数',
        '行业_交通运输', '行业_传媒', '行业_公用事业', '行业_农林牧渔',
        '行业_化工', '行业_医药生物', '行业_国防军工', '行业_地产建筑',
        '行业_家用电器', '行业_有色钢铁', '行业_机械设备', '行业_汽车',
        '行业_消费零售', '行业_电力新能源', '行业_电子', '行业_综合',
        '行业_能源', '行业_计算机', '行业_通信', '行业_金融', '行业_食品饮料',
    ],
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
    # 'objective': 'rank:ndcg',    # 直接优化NDCG
    'objective': 'rank:pairwise', # 噪声环境优先：只要求两两相对方向正确
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
