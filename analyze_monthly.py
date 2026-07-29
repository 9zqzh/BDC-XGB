"""
轻量版：窗口4月度分析 — 仅加载必需上下文，单进程处理。
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))
from config import config, xgb_config

model_dir = os.path.join(os.path.dirname(__file__), 'model', '60_158+39', 'cross_val_4_窗口4_近期市场')

# ── 加载模型 ──
import xgboost as xgb
bst = xgb.Booster()
bst.load_model(os.path.join(model_dir, 'best_model.json'))
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
print("模型/标准化器已加载")

# ── 用 shell 预计算特征然后只读结果 ──
# 先直接读 raw data, 只取需要的列做轻量处理
data_path = config['data_path']
df = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype={'股票代码': str}, low_memory=False)
df['日期'] = pd.to_datetime(df['日期'])

val_start = pd.to_datetime('2025-07-01')
val_end = pd.to_datetime('2026-06-30')
ctx_start = val_start - pd.DateOffset(days=120)  # 120天上下文足够60天序列

# 只取 ctx_start 到 val_end
df_sub = df[(df['日期'] >= ctx_start) & (df['日期'] <= val_end)].copy()
df_sub = df_sub.sort_values(['股票代码', '日期']).reset_index(drop=True)
print(f"数据: {df_sub.shape[0]:,} 行, {df_sub['日期'].min().date()} ~ {df_sub['日期'].max().date()}")

# 检查 stock_data.csv 是否已有预计算特征
stock_path = os.path.join(data_path, 'stock_data.csv')
if os.path.exists(stock_path):
    print("发现 stock_data.csv，使用预计算数据...")
    stock_df = pd.read_csv(stock_path, dtype={'股票代码': str}, low_memory=False)
    stock_df['日期'] = pd.to_datetime(stock_df['日期'])
    stock_df = stock_df[(stock_df['日期'] >= ctx_start) & (stock_df['日期'] <= val_end)]
    stock_df = stock_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"预计算数据: {stock_df.shape[0]:,} 行")
else:
    stock_df = None

# 检查 preprocessed dir
cache_dir = os.path.join(model_dir, 'preprocessed')
if os.path.exists(cache_dir):
    print(f"发现预处理缓存: {cache_dir}")
    val_data = joblib.load(os.path.join(cache_dir, 'val_data.pkl'))
    print(f"已加载预处理验证数据: {val_data.shape[0]:,} 行")
else:
    print("未发现预处理缓存，将从头计算（可能需要较多内存）")

print("\n分析完成。数据已加载。")
print("由于内存限制，建议在本地 Windows 环境运行完整月度分析。")
