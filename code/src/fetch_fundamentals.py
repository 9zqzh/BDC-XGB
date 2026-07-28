"""
获取沪深300成分股基本面数据 (tushare 版)。
使用 tushare.pro 接口，无需访问东方财富。

前置条件：
  1. 注册 https://tushare.pro 获取 API token
  2. 设置环境变量 TUSHARE_TOKEN=<你的token>
     或直接修改变量 TUSHARE_TOKEN

产出：data/hs300_fundamentals.csv
字段：股票代码, PE_TTM, PB, ROE_approx, 总市值_对数, 行业
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tushare as ts

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")


def fetch_hs300_fundamentals():
    if not TUSHARE_TOKEN:
        print("未设置 TUSHARE_TOKEN 环境变量")
        print("请注册 https://tushare.pro 获取 token，然后:")
        print('  $env:TUSHARE_TOKEN="你的token"')
        return

    pro = ts.pro_api(TUSHARE_TOKEN)

    # ── 1. 从本地文件读取沪深300成分股 ──
    hs300_path = os.path.join(DATA_DIR, 'hs300_stock_list.csv')
    if not os.path.exists(hs300_path):
        raise FileNotFoundError(f"缺少 {hs300_path}，请确保该文件存在")
    hs300_list = pd.read_csv(hs300_path, dtype={'code': str})
    codes = [c.zfill(6) for c in hs300_list['code'].tolist()]
    print(f"沪深300成分股(本地): {len(codes)} 只")

    # ── 2. 获取 PE/PB/市值（跳过 trade_cal，直接用近期日期） ──
    print("正在获取 PE/PB/市值数据...")
    try:
        # 尝试多个可能交易日，从近到远
        for test_date in ['20260710', '20260709', '20260708', '20260707',
                          '20260703', '20260630', '20250710']:
            try:
                df_all = pro.daily_basic(ts_code='', trade_date=test_date,
                                         fields='ts_code,pe_ttm,pb,total_mv,circ_mv')
                if len(df_all) > 100:
                    latest_trade_date = test_date
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("无法获取任何交易日数据")

        df_all['code'] = df_all['ts_code'].str[:6].str.zfill(6)
        print(f"  全A股数据: {len(df_all)} 只, 日期: {latest_trade_date}")
    except Exception as e:
        raise RuntimeError(f"获取 daily_basic 失败: {e}") from e

    # ── 3. 筛选 HS300 ──
    df_hs300 = df_all[df_all['code'].isin(codes)].copy()
    print(f"  HS300 匹配: {len(df_hs300)} 只")

    # ── 4. 构建输出 ──
    result = pd.DataFrame()
    result['股票代码'] = df_hs300['code']

    # PE_TTM (tushare 直接提供 pe_ttm)
    pe = pd.to_numeric(df_hs300['pe_ttm'], errors='coerce')
    pe[pe <= 0] = np.nan
    result['PE_TTM'] = pe

    # PB
    pb = pd.to_numeric(df_hs300['pb'], errors='coerce')
    pb[pb <= 0] = np.nan
    result['PB'] = pb

    # 总市值（万元→元），再取对数
    total_mv = pd.to_numeric(df_hs300['total_mv'], errors='coerce') * 1e4
    result['总市值_对数'] = np.where(total_mv > 0, np.log(total_mv), np.nan)

    # ROE_approx = PB / PE_TTM
    result['ROE_approx'] = (
        result['PB'] / result['PE_TTM']
    ).replace([np.inf, -np.inf], np.nan).clip(-1, 1)

    # 行业 (tushare 每日数据不含行业，统一填未知)
    result['行业'] = '未知'

    # ── 5. 数据质量 ──
    n = len(result)
    print(f"\n{'='*50}")
    print(f"  数据质量校验")
    print(f"{'='*50}")
    print(f"  HS300匹配: {n} 只")
    print(f"  PE_TTM 有效: {(pe > 0).sum()} ({(pe > 0).mean():.1%})" if n > 0 else "  PE_TTM 有效: 0")
    print(f"  PB 有效:     {(pb > 0).sum()} ({(pb > 0).mean():.1%})" if n > 0 else "  PB 有效: 0")
    print(f"  总市值有效:   {(total_mv > 0).sum()} ({(total_mv > 0).mean():.1%})" if n > 0 else "  总市值有效: 0")
    print(f"  数据日期: {latest_trade_date}")
    print(f"{'='*50}")

    # ── 6. 保存 ──
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, 'hs300_fundamentals.csv')
    result.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n保存完成: {output_path}")
    print(f"字段: {list(result.columns)}")


if __name__ == '__main__':
    fetch_hs300_fundamentals()
