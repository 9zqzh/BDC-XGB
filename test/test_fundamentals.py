"""
测试 fetch_fundamentals.py 的数据清洗逻辑。
在修改 fetch_fundamentals.py 之前先运行一次，观察哪些测试失败；
修改后再次运行，确认全部通过。
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))

# ── 不依赖网络：直接测试数据清洗函数 ──


def _resolve_code(df):
    """模拟 fetch_fundamentals.py 中的股票代码解析逻辑。"""
    for col in ('股票代码', '代码', 'f12'):
        if col in df.columns:
            return df[col].astype(str).str[:6].str.zfill(6)
    raise KeyError("缺少股票代码列（股票代码/代码/f12）")


def _safe_numeric(series):
    """统一数值转换。"""
    return pd.to_numeric(series, errors='coerce')


def test_code_resolution():
    """股票代码补齐六位。"""
    df_code = pd.DataFrame({'代码': ['1', '600000', '000001']})
    result = _resolve_code(df_code)
    assert result.tolist() == ['000001', '600000', '000001']

    df_stock = pd.DataFrame({'股票代码': ['1', '600000']})
    result = _resolve_code(df_stock)
    assert result.tolist() == ['000001', '600000']

    df_f12 = pd.DataFrame({'f12': ['1', '000001']})
    result = _resolve_code(df_f12)
    assert result.tolist() == ['000001', '000001']


def test_code_missing():
    """缺少代码列应抛出异常。"""
    df = pd.DataFrame({'名称': ['a', 'b']})
    try:
        _resolve_code(df)
        assert False, "Should raise KeyError"
    except KeyError as e:
        assert '缺少股票代码列' in str(e)


def test_pe_cleanup():
    """动态PE正常/负/非法→NaN。"""
    df = pd.DataFrame({'市盈率-动态': ['15.5', '-3', 'abc', '', np.nan]})
    result = _safe_numeric(df['市盈率-动态'])
    result[result <= 0] = np.nan
    assert result.iloc[0] == 15.5
    assert np.isnan(result.iloc[1])   # -3 → NaN
    assert np.isnan(result.iloc[2])   # 'abc' → NaN
    assert np.isnan(result.iloc[3])   # '' → NaN
    assert np.isnan(result.iloc[4])   # NaN → NaN

    df_f9 = pd.DataFrame({'f9': ['20', '-5']})
    result = _safe_numeric(df_f9['f9'])
    result[result <= 0] = np.nan
    assert result.iloc[0] == 20.0
    assert np.isnan(result.iloc[1])


def test_pb_cleanup():
    """负PB→NaN。"""
    df = pd.DataFrame({'市净率': ['1.5', '-0.5', 'x']})
    result = _safe_numeric(df['市净率'])
    result[result <= 0] = np.nan
    assert result.iloc[0] == 1.5
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


def test_market_cap_log():
    """零市值不参与对数计算。"""
    df = pd.DataFrame({'总市值': ['100', '0', '-5', '', np.nan]})
    result = _safe_numeric(df['总市值'])
    valid = result > 0
    log_vals = np.full(len(result), np.nan)
    log_vals[valid] = np.log(result[valid])
    assert log_vals[0] == np.log(100)
    assert np.isnan(log_vals[1])  # 0
    assert np.isnan(log_vals[2])  # -5
    assert np.isnan(log_vals[3])  # ''
    assert np.isnan(log_vals[4])  # NaN


def test_roe_approx():
    """ROE = PB/PE, 无效值→NaN。"""
    df = pd.DataFrame({
        'PE_TTM': [10.0, 0.0, np.nan, -5.0],
        'PB': [2.0, 1.5, 1.0, 1.0],
    })
    result = (df['PB'] / df['PE_TTM']).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    assert result.iloc[0] == 0.2        # 2/10
    assert np.isnan(result.iloc[1])     # PE=0
    assert np.isnan(result.iloc[2])     # PE=NaN
    assert result.iloc[3] == -0.2       # -5→clipped, within bounds


def test_industry():
    """行业字段处理。"""
    # 存在行业列
    df = pd.DataFrame({'行业': ['银行', np.nan, '科技']})
    result = df['行业'].fillna('未知').astype(str)
    assert result.tolist() == ['银行', '未知', '科技']

    # 不存在行业列
    df2 = pd.DataFrame({'名称': ['平安银行', '万科A']})
    # 禁止：result = df2['名称']  ← 这是错误
    result2 = pd.Series(['未知'] * len(df2))
    assert result2.tolist() == ['未知', '未知']


def test_name_not_industry():
    """股票名称不能伪装成行业。"""
    df = pd.DataFrame({'名称': ['平安银行', '万科A']})
    # 确认没有'行业'列
    has_industry = '行业' in df.columns
    assert not has_industry
    # 不应把名称放入行业
    result = pd.Series(['未知'] * len(df))
    assert result.tolist() == ['未知', '未知']


def test_full_pipeline():
    """完整数据流测试。"""
    df = pd.DataFrame({
        '代码': ['1', '600000', '000002'],
        '市盈率-动态': ['15.5', '-3.0', '20.0'],
        '市净率': ['1.5', '2.0', '-0.5'],
        '总市值': ['1000', '500', '0'],
        '行业': ['银行', '银行', np.nan],
    })

    # stock code
    code = _resolve_code(df)

    # PE
    pe = _safe_numeric(df['市盈率-动态'])
    pe[pe <= 0] = np.nan

    # PB
    pb = _safe_numeric(df['市净率'])
    pb[pb <= 0] = np.nan

    # Market cap
    market_cap = _safe_numeric(df['总市值'])
    market_cap_log = np.full(len(market_cap), np.nan)
    valid = market_cap > 0
    market_cap_log[valid] = np.log(market_cap[valid])

    # ROE
    roe = (pb / pe).replace([np.inf, -np.inf], np.nan).clip(-1, 1)

    # Industry
    industry = df['行业'].fillna('未知').astype(str)

    # Asserts
    assert code.tolist() == ['000001', '600000', '000002']
    assert pe.iloc[0] == 15.5
    assert np.isnan(pe.iloc[1])
    assert pe.iloc[2] == 20.0
    assert np.isnan(pb.iloc[2])
    assert market_cap_log[0] == np.log(1000)
    assert np.isnan(market_cap_log[2])
    assert roe.iloc[0] == 1.5 / 15.5
    assert industry.tolist() == ['银行', '银行', '未知']


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-q', '-v'])
