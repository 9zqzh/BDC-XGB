"""
XGBRanker 排序学习训练脚本
标签：超额收益（已由 _build_label_and_clean 计算）
特征：将 60 天序列展平为单行特征向量（60 × 197 = 11,820 维）
分组：每个交易日为一个 group（qid），group 内股票按超额收益排序
"""

import os
import json
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import config, config_extended, xgb_config
from utils import engineer_features_39, engineer_features_158plus39
from evaluation import calculate_extended_metrics, format_eval_report


# ============================================================
#  特征列映射 & 特征工程（复用 utils.py，不修改特征逻辑）
# ============================================================

feature_columns_map = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
#  标签构建：超额收益（不改计算逻辑）
# ============================================================

def _build_label_and_clean(processed, drop_small_open=True):
    """构建超额收益标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)

    # 转换为超额收益：减去当日等权指数收益
    processed['_daily_mean'] = processed.groupby('日期')['label'].transform('mean')
    processed['label'] = processed['label'] - processed['_daily_mean']
    processed.drop(columns=['_daily_mean'], inplace=True)

    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


# ============================================================
#  数据预处理（复用 utils.py 特征工程）
# ============================================================

def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# ============================================================
#  特征展平：60 天序列 → 单行向量（XGBRanker 输入）
# ============================================================

def flatten_sequences_to_xgb(data, features, sequence_length, flatten_days=None):
    """
    将时序 DataFrame 转换为 XGBRanker 需要的扁平特征矩阵。

    - 历史窗口 = sequence_length (60天，确保上下文)
    - 展平窗口 = flatten_days (默认10天，控制特征维度)
    - 附加市场状态特征（全局均值、波动率、趋势）
    """
    import tempfile

    if flatten_days is None:
        flatten_days = config.get('xgb_flatten_days', 10)
    flatten_days = min(flatten_days, sequence_length)

    data = data.copy()
    data['日期'] = pd.to_datetime(data['日期'])
    data = data.sort_values(['instrument', '日期']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    date_list = sorted(data['日期'].unique())
    valid_dates = date_list[sequence_length - 1:]
    date2qid = {d: i for i, d in enumerate(valid_dates)}

    n_feat = len(features)
    feat_dim = flatten_days * n_feat + 3  # +3 市场状态特征

    # ── 预计算市场状态特征 ──
    print("正在计算市场状态特征...")
    market_daily = data.groupby('日期')['label'].mean().sort_index()
    market_returns = market_daily.values
    # 20日波动率 + 60日累计趋势
    market_vol = pd.Series(market_returns, index=market_daily.index).rolling(20, min_periods=5).std().values
    market_trend = pd.Series(market_returns, index=market_daily.index).rolling(60, min_periods=10).sum().values
    date2market = {}
    for i, d in enumerate(market_daily.index):
        date2market[d] = (
            float(market_returns[i]),
            float(market_vol[i]) if not np.isnan(market_vol[i]) else 0.0,
            float(market_trend[i]) if not np.isnan(market_trend[i]) else 0.0,
        )

    # ── 第一遍：统计总样本数 ──
    print("正在统计样本数...")
    total_samples = 0
    for _, group in data.groupby('instrument', sort=False):
        n = len(group)
        group_dates = group['日期'].values
        for i in range(sequence_length - 1, n):
            if group_dates[i] in date2qid:
                total_samples += 1
    print(f"总样本数: {total_samples:,}")

    # ── 预分配 memmap（写到项目 output 目录而非系统临时目录） ──
    mmap_dir = config.get('output_dir', './model')
    os.makedirs(mmap_dir, exist_ok=True)
    tmpfile = tempfile.NamedTemporaryFile(suffix='.dat', delete=False, dir=mmap_dir)
    X = np.memmap(tmpfile.name, dtype=np.float32, mode='w+', shape=(total_samples, feat_dim))
    y = np.empty(total_samples, dtype=np.float32)
    qid = np.empty(total_samples, dtype=np.int32)

    # ── 预建每只股票的索引（日期 → 行号），用于快速查找 ──
    print("正在构建索引...")
    stock_groups = {}
    for stock, group in data.groupby('instrument', sort=False):
        group = group.set_index('日期').sort_index()
        stock_groups[stock] = {
            'feat': group[features].values.astype(np.float32),
            'label': group['label'].values.astype(np.float32),
            'dates': group.index.values,
        }

    # ── 第二遍：按日期顺序遍历，天然保证 qid 有序 ──
    print(f"正在展平时序特征（最后{flatten_days}天 × {n_feat}维 + 3市场 = {feat_dim:,}维，memmap 模式）...")
    write_pos = 0
    for q, d in enumerate(tqdm(valid_dates, desc="展平特征")):
        mkt_feat = date2market.get(d, (0.0, 0.0, 0.0))
        for stock, grp in stock_groups.items():
            idx = np.where(grp['dates'] == d)[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            if i < sequence_length - 1:
                continue
            # 只取最后 flatten_days 天展平
            seq = grp['feat'][max(0, i - flatten_days + 1): i + 1]
            # 补齐：若不足 flatten_days 天，前面补零
            if len(seq) < flatten_days:
                pad = np.zeros((flatten_days - len(seq), n_feat), dtype=np.float32)
                seq = np.vstack([pad, seq])
            flat = seq.flatten()
            # 追加市场状态特征
            row = np.concatenate([flat, np.array(mkt_feat, dtype=np.float32)])
            X[write_pos] = row
            y[write_pos] = grp['label'][i]
            qid[write_pos] = q
            write_pos += 1

    X_final = X[:write_pos]        # memmap 切片，不占额外 RAM
    y = y[:write_pos]
    qid = qid[:write_pos]

    print(f"展平完成：{write_pos:,} 个样本，{feat_dim:,} 维特征，{len(valid_dates)} 个交易组")
    return X_final, y, qid, None, None, valid_dates


# ============================================================
#  验证集划分
# ============================================================

def split_train_val_by_last_month(df, sequence_length, val_months=12):
    """按末尾 N 个月做验证集划分。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_start].copy()

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ============================================================
#  标签转换：连续超额收益 → 整数排名（XGBRanker 要求）
# ============================================================

def _continuous_labels_to_ranks(y, qid):
    """
    XGBRanker rank:pairwise 要求标签为整数。
    将每组 (qid) 内的连续标签按降序排名，最高收益→最高整数排名。

    例：y=[0.05, -0.02, 0.03], qid=[0,0,0] → rank=[2, 0, 1]
    同时返回原始连续标签，供后续 evaluate_xgb_model 计算实际收益指标。
    """
    y_rank = np.zeros_like(y, dtype=np.int32)
    for q in np.unique(qid):
        mask = qid == q
        group_y = y[mask]
        # argsort 两次得到 dense rank（0-based）
        order = np.argsort(group_y)                     # 升序：最低收益排最前
        rank = np.empty_like(order)
        rank[order] = np.arange(len(group_y))           # 0 = 最低，n-1 = 最高
        y_rank[mask] = rank
    return y_rank


# ============================================================
#  评估：复用 evaluation.py（不修改指标计算）
# ============================================================

def evaluate_xgb_model(model, X_val, y_val, qid_val, valid_dates, val_df, features,
                       scaler, sequence_length, k=5, min_gap=0.005):
    """
    在验证集上使用自定义指标评估 XGBRanker。
    步骤：按日期分组 → 对每组内的股票打分 → 计算 extended_metrics
    """
    import torch
    preds = model.predict(X_val)

    # 按 qid 分组，构建每日的 pred/true/mask
    daily_metrics = {
        'pred_return_sum': [], 'max_return_sum': [], 'random_return_sum': [],
        'ratio_pred': [], 'ratio_random': [], 'final_score': [],
        'topk_hit': [], 'spearman': [], 'win': [],
    }

    unique_qids = sorted(set(qid_val))
    num_total = 0
    num_valid = 0

    for q in unique_qids:
        mask = qid_val == q
        day_preds = torch.tensor(preds[mask], dtype=torch.float32)
        day_labels = torch.tensor(y_val[mask], dtype=torch.float32)
        n = len(day_preds)
        if n < k:
            continue
        num_total += 1

        # 按预测排序取 top k
        _, topk_idx = torch.topk(day_preds, k)
        topk_returns = day_labels[topk_idx]
        pred_sum = topk_returns.sum().item()

        _, true_topk_idx = torch.topk(day_labels, k)
        max_sum = day_labels[true_topk_idx].sum().item()
        random_sum = k * day_labels.mean().item()

        gap = max_sum - random_sum
        if abs(gap) < min_gap:
            continue

        num_valid += 1
        daily_metrics['pred_return_sum'].append(pred_sum)
        daily_metrics['max_return_sum'].append(max_sum)
        daily_metrics['random_return_sum'].append(random_sum)

        fs = (pred_sum - random_sum) / (gap + 1e-12) if abs(gap) > 1e-6 else 0.0
        daily_metrics['final_score'].append(fs)

        # TopK 命中
        true_set = set(true_topk_idx.numpy())
        pred_set = set(topk_idx.numpy())
        daily_metrics['topk_hit'].append(len(true_set & pred_set))

        # Spearman
        from evaluation import _spearman_rho_pytorch
        daily_metrics['spearman'].append(_spearman_rho_pytorch(day_preds, day_labels))

        # Win rate
        daily_metrics['win'].append(1.0 if topk_returns.mean().item() > day_labels.mean().item() else 0.0)

    n = num_valid
    metrics = {
        'final_score': np.mean(daily_metrics['final_score']) if n > 0 else 0.0,
        'topk_hit_rate': (np.mean(daily_metrics['topk_hit']) / k) if n > 0 else 0.0,
        'topk_hit_count': np.mean(daily_metrics['topk_hit']) if n > 0 else 0.0,
        'spearman_rho': np.mean(daily_metrics['spearman']) if n > 0 else 0.0,
        'win_rate': np.mean(daily_metrics['win']) if n > 0 else 0.0,
        'final_score_std': np.std(daily_metrics['final_score'], ddof=1) if n > 1 else 0.0,
        'pred_return_sum': np.mean(daily_metrics['pred_return_sum']) if n > 0 else 0.0,
        'valid_days_ratio': n / max(num_total, 1),
        'valid_days': n,
        'total_days': num_total,
    }
    return metrics


# ============================================================
#  单窗口训练函数（XGBRanker）
# ============================================================

def train_one_window(train_df, val_df, val_start, stockid2idx, num_stocks, config, output_dir):
    """
    XGBRanker 单窗口训练 + 评估。

    Args:
        train_df: 训练集 DataFrame
        val_df:   验证集 DataFrame
        val_start: 验证集起始日期
        stockid2idx: 股票代码映射
        num_stocks: 总股票数
        config: 配置字典
        output_dir: 输出目录

    Returns:
        best_score, extended_metrics
    """
    sequence_length = config['sequence_length']
    features_list = feature_columns_map[config['feature_num']]
    flatten_days = config.get('xgb_flatten_days', 10)

    # ── 特征工程 ──
    train_data, _ = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # ── 标准化 ──
    scaler = StandardScaler()
    for col_set in [train_data, val_data]:
        col_set[features_list] = col_set[features_list].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features_list)
    val_data = val_data.dropna(subset=features_list)
    train_data[features_list] = scaler.fit_transform(train_data[features_list])
    val_data[features_list] = scaler.transform(val_data[features_list])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # ── 展平特征 ──
    X_train, y_train_cont, qid_train, _, _, valid_train_dates = flatten_sequences_to_xgb(
        train_data, features_list, sequence_length
    )
    X_val, y_val_cont, qid_val, val_sample_dates, val_sample_stocks, valid_val_dates = flatten_sequences_to_xgb(
        val_data, features_list, sequence_length
    )

    # ── 标签转换：连续超额收益 → 整数排名（XGBRanker rank:pairwise 要求） ──
    y_train = _continuous_labels_to_ranks(y_train_cont, qid_train)
    y_val = _continuous_labels_to_ranks(y_val_cont, qid_val)

    # ── XGBRanker 按组的样本数 ──
    train_groups = [np.sum(qid_train == q) for q in sorted(set(qid_train))]
    val_groups = [np.sum(qid_val == q) for q in sorted(set(qid_val))]

    print(f"\nXGBRanker 训练配置:")
    print(f"  训练样本: {len(X_train):,} 行，{X_train.shape[1]} 维特征")
    print(f"  训练组数: {len(train_groups)} 天")
    print(f"  验证样本: {len(X_val):,} 行")
    print(f"  验证组数: {len(val_groups)} 天")

    # ── 构建 XGBRanker ──
    xgb_params = {
        'max_depth': xgb_config['max_depth'],
        'learning_rate': xgb_config['learning_rate'],
        'n_estimators': xgb_config['n_estimators'],
        'subsample': xgb_config['subsample'],
        'colsample_bytree': xgb_config['colsample_bytree'],
        'reg_alpha': xgb_config['reg_alpha'],
        'reg_lambda': xgb_config['reg_lambda'],
        'min_child_weight': xgb_config['min_child_weight'],
        'objective': xgb_config['objective'],
        'eval_metric': xgb_config['eval_metric'],
        'ndcg_exp_gain': False,                         # 禁用指数增益（标签>31时必需）
        'verbosity': xgb_config['verbosity'],
        'n_jobs': xgb_config['n_jobs'],
        'tree_method': 'hist',
        'random_state': 42,
    }

    model = xgb.XGBRanker(**xgb_params)

    print("\n开始训练 XGBRanker ...")
    model.fit(
        X_train, y_train,
        qid=qid_train,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=20,
    )

    # ── 输出特征重要性 ──
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-20:][::-1]
    n_feat_per_day = len(features_list)
    print(f"\n特征重要性 Top20 (共{len(importance)}维, 每{n_feat_per_day}维=1天特征, 最后3维=市场状态):")
    for rank, idx in enumerate(top_idx):
        if idx >= len(importance) - 3:
            label = ["市场均值", "市场波动率", "市场趋势"][idx - (len(importance) - 3)]
        else:
            day = idx // n_feat_per_day + 1
            f_idx = idx % n_feat_per_day
            label = f"T-{flatten_days - day + 1}天_{features_list[f_idx][:8]}"
        print(f"  {rank+1:2d}. {label}: {importance[idx]:.6f}")

    # ── 评估（使用原始连续收益标签，非整数排位） ──
    min_gap_val = config_extended.get('min_gap', 0.005)
    k_val = config_extended.get('eval_top_k', 5)
    extended_metrics = evaluate_xgb_model(
        model, X_val, y_val_cont, qid_val, valid_val_dates,
        val_data, features_list, scaler, sequence_length,
        k=k_val, min_gap=min_gap_val
    )

    best_score = extended_metrics.get('final_score', 0.0)

    # ── 保存模型 ──
    model_path = os.path.join(output_dir, 'best_model.json')
    model.save_model(model_path)
    # 同时保存 pkl（兼容 cross_val.py）
    joblib.dump(model, os.path.join(output_dir, 'best_model.pkl'))

    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best final_score: {best_score:.6f}\n")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({**config, **xgb_config}, f, indent=4, ensure_ascii=False)

    print(f"\n模型已保存到: {model_path}")
    print(f"验证集最终得分 (final_score): {best_score:.6f}")

    eval_report = format_eval_report(extended_metrics)
    print(eval_report)
    with open(os.path.join(output_dir, 'eval_report.txt'), 'w', encoding='utf-8') as f:
        f.write(eval_report)

    return best_score, extended_metrics


# ============================================================
#  主程序
# ============================================================

def main():
    set_seed(42)
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_path = config['data_path']
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str}, low_memory=False)

    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config['sequence_length'],
        val_months=config_extended.get('val_months', 12)
    )

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    print(f"全量数据范围: {full_df['日期'].min()} 到 {full_df['日期'].max()}")
    print(f"训练集范围: {train_df['日期'].min()} 到 {train_df['日期'].max()}")
    print(f"验证集范围: {val_df['日期'].min()} 到 {val_df['日期'].max()}")

    best_score, best_extended_metrics = train_one_window(
        train_df, val_df, val_start, stockid2idx, num_stocks, config, output_dir
    )

    print(f"\n{'#'*50}")
    print(f"  训练完成！最佳 final_score: {best_score:.6f}")
    print(f"{'#'*50}")
    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
