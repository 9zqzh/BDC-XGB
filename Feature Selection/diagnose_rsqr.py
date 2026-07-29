"""诊断 RSQR 特征 NaN 分布的脚本"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "code" / "src"))

import pandas as pd
import numpy as np

# 1. 直接检查原始 train.csv 中是否有 RSQR 列
print("=" * 60)
print("检查 1: train.csv 原始列")
print("=" * 60)
df = pd.read_csv(Path(__file__).parents[1] / "data" / "train.csv", nrows=100)
print(f"train.csv 列名: {list(df.columns)}")
rsqr_cols = [c for c in df.columns if 'RSQR' in c.upper()]
print(f"RSQR 相关列: {rsqr_cols if rsqr_cols else '无 (RSQR 是动态计算的)'}")

# 2. 模拟 RSQR 计算逻辑
print("\n" + "=" * 60)
print("检查 2: 模拟 RSQR 计算")
print("=" * 60)

# 用一只股票的数据测试
df_full = pd.read_csv(Path(__file__).parents[1] / "data" / "train.csv")
df_full['日期'] = pd.to_datetime(df_full['日期'])
stocks = df_full['股票代码'].unique()
test_stock = stocks[0]
print(f"测试股票: {test_stock}")

stock_data = df_full[df_full['股票代码'] == test_stock].copy()
stock_data = stock_data.sort_values('日期')
close = stock_data['收盘'].astype(float)
print(f"该股收盘价值数: {len(close)}")
print(f"日期范围: {stock_data['日期'].min()} ~ {stock_data['日期'].max()}")

# 模拟代码中的 RSQR 计算 (w=5)
w = 5
time_period_series = pd.Series(range(w), index=close.index[:w])
print(f"\ntime_period_series (长度={len(time_period_series)}):")
print(time_period_series)

rolling_corr = close.rolling(w).corr(time_period_series)
valid_count = rolling_corr.notna().sum()
print(f"\nrolling_corr 有效值数量: {valid_count} / {len(rolling_corr)}")
print(f"有效值占比: {valid_count / len(rolling_corr) * 100:.2f}%")

if valid_count > 0:
    print("\n有效值的位置 (前10个):")
    valid_idx = rolling_corr[rolling_corr.notna()].index[:10]
    for idx in valid_idx:
        print(f"  index={idx}, corr={rolling_corr[idx]:.6f}")

# 3. 正确的 RSQR 计算方式 (对比)
print("\n" + "=" * 60)
print("检查 3: 正确的 RSQR 计算方式")
print("=" * 60)

def correct_rsqr(series, window):
    """正确的滚动 R-squared 计算"""
    results = pd.Series(np.nan, index=series.index)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1:i + 1].values
        if np.any(np.isnan(y)) or len(y) < window:
            continue
        y_mean = y.mean()
        ss_yy = ((y - y_mean) ** 2).sum()
        if ss_yy < 1e-12:
            continue
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
        results.iloc[i] = r2
    return results

correct_rsqr5 = correct_rsqr(close, 5)
valid_correct = correct_rsqr5.notna().sum()
print(f"正确方法的有效值数量: {valid_correct} / {len(correct_rsqr5)}")
print(f"有效值占比: {valid_correct / len(correct_rsqr5) * 100:.2f}%")

if valid_correct > 0:
    print(f"\n正确方法前5个有效值:")
    valid_idx = correct_rsqr5[correct_rsqr5.notna()].index[:5]
    for idx in valid_idx:
        date = stock_data.loc[idx, '日期']
        print(f"  {date.date()}: RSQR={correct_rsqr5[idx]:.6f}")

# 4. 全市场统计
print("\n" + "=" * 60)
print("检查 4: 全市场 RSQR 有效天数统计")
print("=" * 60)

all_valid_days = []
for stock in stocks[:10]:  # 前10只股票
    sd = df_full[df_full['股票代码'] == stock].sort_values('日期')
    c = sd['收盘'].astype(float)

    # 原始方法 (有 bug)
    tps = pd.Series(range(5), index=c.index[:5])
    rc = c.rolling(5).corr(tps)
    buggy_valid = rc.notna().sum()

    # 正确方法
    correct = correct_rsqr(c, 5)
    correct_valid = correct.notna().sum()

    all_valid_days.append({
        'stock': stock,
        'total_days': len(c),
        'buggy_valid': buggy_valid,
        'correct_valid': correct_valid,
    })

result_df = pd.DataFrame(all_valid_days)
print(result_df)
print(f"\n原始方法平均每只股票有效天数: {result_df['buggy_valid'].mean():.1f}")
print(f"正确方法平均每只股票有效天数: {result_df['correct_valid'].mean():.1f}")

# 5. 测试修复后的 utils.py
print("\n" + "=" * 60)
print("检查 5: 测试修复后的 utils.py")
print("=" * 60)
try:
    from utils import engineer_features
    test_stock_data = df_full[df_full['股票代码'] == test_stock].copy()
    test_stock_data = test_stock_data.sort_values('日期')
    features_df = engineer_features(test_stock_data)

    rsqr5_col = [c for c in features_df.columns if 'RSQR5' in c]
    if rsqr5_col:
        rsqr5_values = features_df[rsqr5_col[0]]
        valid_count = rsqr5_values.notna().sum()
        print(f"修复后 RSQR5 有效值数量: {valid_count} / {len(rsqr5_values)}")
        print(f"有效值占比: {valid_count / len(rsqr5_values) * 100:.2f}%")

        # 检查值范围是否合理 (R-squared 应在 0-1 之间)
        valid_values = rsqr5_values[rsqr5_values.notna()]
        if len(valid_values) > 0:
            print(f"RSQR5 值范围: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
            print(f"RSQR5 均值: {valid_values.mean():.4f}")
    else:
        print("未找到 RSQR5 列")

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
