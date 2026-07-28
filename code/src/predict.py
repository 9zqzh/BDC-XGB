"""
XGBRanker 推理预测脚本
加载训练好的 XGBRanker 模型，只对沪深300成分股进行排序预测，输出 Top5。
"""

import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import config
from utils import engineer_features_39, engineer_features_158plus39


def main():
    data_file = os.path.join(config['data_path'], 'train.csv')
    model_path_pkl = os.path.join(config['output_dir'], 'best_model.pkl')
    scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
    output_path = os.path.join('./output/', 'result.csv')
    hs300_path = os.path.join(config['data_path'], 'hs300_stock_list.csv')

    # 加载模型
    if not os.path.exists(model_path_pkl):
        raise FileNotFoundError(f'未找到模型文件: {model_path_pkl}')
    model = joblib.load(model_path_pkl)
    print(f"已加载模型 (pkl): {model_path_pkl}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到 Scaler 文件: {scaler_path}')

    # 加载沪深300成分股列表
    hs300_set = set()
    if os.path.exists(hs300_path):
        hs300_df = pd.read_csv(hs300_path, dtype={'code': str})
        hs300_set = set(hs300_df['code'].str.zfill(6))
        print(f"沪深300成分股: {len(hs300_set)} 只")

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    # 只保留沪深300成分股
    all_stocks = sorted(raw_df['股票代码'].unique())
    if hs300_set:
        stock_ids = sorted([s for s in all_stocks if s in hs300_set])
        print(f"全量股票: {len(all_stocks)} 只 -> 沪深300过滤后: {len(stock_ids)} 只")
    else:
        stock_ids = all_stocks

    # 过滤数据
    raw_df = raw_df[raw_df['股票代码'].isin(stock_ids)].copy()

    # 建立股票ID映射（与训练时一致）
    stockid2idx = {sid: i for i, sid in enumerate(stock_ids)}

    # 特征工程
    from train import feature_columns_map, _merge_fundamentals
    features = feature_columns_map[config['feature_num']]
    feature_engineer = engineer_features_158plus39 if config['feature_num'] == '158+39' else engineer_features_39

    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in raw_df.groupby('股票代码', sort=False)]

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(scaler_path)
    processed[features] = scaler.transform(processed[features])

    # ── 合并基本面因子（与训练时一致） ──
    fundamental_path = os.path.join(config['data_path'], 'history_factors_nan.csv')
    if not os.path.exists(fundamental_path):
        fundamental_path = os.path.join(config['data_path'], 'hs300_fundamentals.csv')
    processed, fund_cols = _merge_fundamentals(processed, fundamental_path)
    if fund_cols:
        features = features + fund_cols

    # 展平特征
    sequence_length = config['sequence_length']
    flatten_days = config.get('xgb_flatten_days', 10)
    n_feat = len(features)

    # 市场状态特征（从 processed 数据实时计算，与训练保持一致）
    market_daily = processed.groupby('日期')['涨跌幅'].mean().sort_index() if '涨跌幅' in processed.columns else pd.Series(dtype=float)

    rows, stock_codes = [], []
    for stock_id in stock_ids:
        stock_history = processed[
            (processed['股票代码'] == stock_id)
        ].sort_values('日期').tail(sequence_length)
        if len(stock_history) == sequence_length:
            feat = stock_history[features].values.astype(np.float32)
            # 只取最后 flatten_days 天
            feat = feat[-flatten_days:]
            # 补齐
            if len(feat) < flatten_days:
                pad = np.zeros((flatten_days - len(feat), n_feat), dtype=np.float32)
                feat = np.vstack([pad, feat])
            flat = feat.flatten()
            # 市场状态（与训练时一致的7维：6市场+1 regime，推理时填0）
            latest_date_stock = pd.to_datetime(stock_history['日期'].values[-1])
            mkt_ret = float(market_daily.get(latest_date_stock, 0.0)) if len(market_daily) > 0 else 0.0
            mkt_feat = np.array([mkt_ret, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            rows.append(np.concatenate([flat, mkt_feat]))
            stock_codes.append(stock_id)

    if len(rows) == 0:
        raise ValueError('没有可用于预测的股票序列')

    X_infer = np.array(rows, dtype=np.float32)
    scores = model.predict(X_infer)
    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [stock_codes[i] for i in order]

    if len(ranked_stock_ids) < 5:
        raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')
    top5 = ranked_stock_ids[:5]
    output_df = pd.DataFrame({'stock_id': top5, 'weight': [0.2] * len(top5)})
    output_df.to_csv(output_path, index=False)

    print(f'预测日期: {latest_date.date()}')
    print(f'参与排序股票数(沪深300): {len(ranked_stock_ids)}')
    print(f'Top5 股票: {top5}')
    print(f'结果已写入: {output_path}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
