"""
窗口4 月度切分分析脚本
===== 在 Windows 本地运行（需先激活 .venv）=====
用法: python analyze_window4_monthly.py

基于 cross_val_4 已训练好的 XGBRanker 模型，
对验证集 (2025-07 ~ 2026-06) 按月份计算 final_score / win_rate / spearman / TopK命中率，
诊断模型在震荡市不同阶段是否存在系统性偏差。
"""
import os, sys, json, gc, warnings, multiprocessing as mp
import numpy as np, pandas as pd, joblib, torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))


def spearman_rho_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """纯 numpy 计算 Spearman，无 torch 依赖。"""
    n = len(a)
    if n < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    cov = (ra * rb).sum()
    std = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum())
    return float(cov / std) if std > 1e-12 else 0.0


def main():
    # ── Windows spawn 模式下必须在 __main__ guard 内做所有 import ──
    from config import config, xgb_config
    from train import (
        _preprocess_common, _merge_fundamentals, flatten_sequences_to_xgb,
        set_seed, feature_engineer_func_map, feature_columns_map,
    )

    set_seed(42)

    MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'model', '60_158+39', 'cross_val_4_窗口4_近期市场')
    VAL_START  = pd.to_datetime('2025-07-01')
    VAL_END    = pd.to_datetime('2026-06-30')
    CTX_START  = VAL_START - pd.DateOffset(days=180)
    MIN_GAP    = 0.005
    K_TOP      = 5

    # ── 加载模型和标准化器 ──
    print("加载模型...")
    model  = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    # ── 加载数据 ──
    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"全量数据  : {len(full_df):,} 行, {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    # ── 切片上下文 ──
    context_df = full_df[(full_df['日期'] >= CTX_START) & (full_df['日期'] <= VAL_END)].copy()
    context_df['日期'] = context_df['日期'].dt.strftime('%Y-%m-%d')
    del full_df; gc.collect()
    print(f"上下文数据: {len(context_df):,} 行")

    # ── 股票映射 ──
    all_sids = sorted(context_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_sids)}
    print(f"股票数    : {len(all_sids)}")

    seq_len     = config['sequence_length']
    feat_name   = config['feature_num']
    features_list = list(feature_columns_map[feat_name])

    # ── 特征工程 ──
    print("特征工程中... (多进程, 可能需要几分钟)")
    val_data, _ = _preprocess_common(context_df, stockid2idx, desc="特征工程", drop_small_open=True)
    gc.collect()
    print(f"  特征表: {val_data.shape[0]:,} 行 × {val_data.shape[1]} 列")

    val_data[features_list] = val_data[features_list].replace([np.inf, -np.inf], np.nan)
    val_data = val_data.dropna(subset=features_list)
    val_data[features_list] = scaler.transform(val_data[features_list])

    # ── 基本面合并 ──
    fp = os.path.join(data_path, 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(data_path, 'hs300_fundamentals.csv')
    val_data, fund_cols = _merge_fundamentals(val_data, fp)
    if fund_cols:
        features_list = features_list + fund_cols
    gc.collect()

    # ── 展平 ──
    print("展平特征...")
    X_val, y_val_cont, qid_val, _, _, valid_dates = flatten_sequences_to_xgb(
        val_data, features_list, seq_len
    )
    del val_data; gc.collect()
    print(f"  展平后: {len(X_val):,} 样本, {X_val.shape[1]} 维, {len(np.unique(qid_val))} 组")

    # ── 过滤到验证窗口 ──
    valid_dates_dt = pd.to_datetime(valid_dates)
    in_window = (valid_dates_dt >= VAL_START) & (valid_dates_dt <= VAL_END)
    qid_set = set(np.where(in_window)[0])
    mask = np.array([q in qid_set for q in qid_val])
    X_val, y_val_cont, qid_val = X_val[mask], y_val_cont[mask], qid_val[mask]
    print(f"  过滤后(验证窗口): {len(X_val):,} 样本")

    # 重新映射 qid 为连续整数
    uq = sorted(set(qid_val))
    qid_map = {o: n for n, o in enumerate(uq)}
    qid_val = np.array([qid_map[q] for q in qid_val])
    new_valid = [valid_dates[i] for i in range(len(valid_dates)) if i in qid_set]
    print(f"  有效交易日: {len(new_valid)}")
    gc.collect()

    # ── 预测 + 逐日评估 ──
    print("预测中...")
    preds = model.predict(X_val)
    print(f"  预测完成, {len(preds):,} 条")

    daily = []
    unique_qids = sorted(set(qid_val))
    for qi in unique_qids:
        msk = qid_val == qi
        dp = preds[msk].astype(np.float64)
        dl = y_val_cont[msk].astype(np.float64)
        n_st = len(dp)
        if n_st < K_TOP:
            continue

        tki = np.argsort(dp)[::-1][:K_TOP]
        tkr = dl[tki]
        ps = tkr.sum()

        tti = np.argsort(dl)[::-1][:K_TOP]
        ms_val = dl[tti].sum()
        rs = K_TOP * dl.mean()
        gap = ms_val - rs
        if abs(gap) < MIN_GAP:
            continue

        fs = (ps - rs) / (gap + 1e-12) if abs(gap) > 1e-6 else 0.0
        hc = len(set(tti) & set(tki))
        sp = spearman_rho_numpy(dp, dl)
        w = 1.0 if tkr.mean() > dl.mean() else 0.0

        ds = new_valid[qi] if qi < len(new_valid) else str(qi)
        daily.append({
            'date': pd.to_datetime(ds),
            'final_score': fs, 'topk_hit': hc,
            'spearman': sp, 'win': w,
            'pred_ret': float(tkr.mean()),
            'mkt_ret':  float(dl.mean()),
            'n_stocks': n_st,
        })

    if not daily:
        print("错误: 没有任何有效评估日! 请检查: 1) MIN_GAP 是否过大 2) 数据是否有问题")
        return

    dd = pd.DataFrame(daily)
    dd['ym'] = dd['date'].dt.to_period('M')

    # ── 月度明细 ──
    print("\n" + "=" * 120)
    print("                    窗口4 各月验证指标明细（2025-07 ~ 2026-06）")
    print("=" * 120)
    hdr = f"{'月份':<8} {'天数':>5} {'fs_mean':>10} {'win_rate':>9} {'topk_hit':>9} {'spearman':>9} {'pred_ret':>9} {'mkt_ret':>9} {'fs_std':>9} {'mkt_cum':>10}  {'评价'}"
    print(hdr)
    print("-" * 120)

    summary = []
    for m, g in dd.groupby('ym', sort=True):
        n = len(g)
        fm = g['final_score'].mean()
        fs_std = g['final_score'].std(ddof=1) if n > 1 else 0.0
        wr = g['win'].mean()
        th = g['topk_hit'].mean() / K_TOP
        sp = g['spearman'].mean()
        pr = g['pred_ret'].mean()
        mr = g['mkt_ret'].mean()
        mc = g['mkt_ret'].sum()
        if fm < 0:
            tag = "🔴 严重亏损"
        elif fm < 0.02:
            tag = "🟠 偏差"
        elif fm < 0.05:
            tag = "🟡 一般"
        else:
            tag = "🟢 良好"
        print(f"{str(m):<8} {n:>5} {fm:>10.6f} {wr:>9.4f} {th:>9.4f} {sp:>9.4f} {pr:>9.6f} {mr:>9.6f} {fs_std:>9.6f} {mc:>10.4f}  {tag}")
        summary.append({
            'month': str(m), 'days': n, 'final_score': fm, 'win_rate': wr,
            'topk_hit_rate': th, 'spearman': sp,
            'pred_ret': pr, 'market_ret': mr, 'fs_std': fs_std, 'market_cumret': mc,
        })

    print("-" * 120)
    total_fs = dd['final_score'].mean()
    print(f"{'合计':<8} {len(dd):>5} {total_fs:>10.6f} "
          f"{dd['win'].mean():>9.4f} {dd['topk_hit'].mean()/K_TOP:>9.4f} "
          f"{dd['spearman'].mean():>9.4f} {dd['pred_ret'].mean():>9.6f} "
          f"{dd['mkt_ret'].mean():>9.6f} {dd['final_score'].std():>9.6f}")
    print("=" * 120)

    # ── 季度 ──
    dd['q'] = dd['date'].dt.to_period('Q')
    print("\n" + "=" * 90)
    print("                    窗口4 各季度汇总")
    print("=" * 90)
    print(f"{'季度':<8} {'天数':>5} {'fs_mean':>10} {'win_rate':>9} {'spearman':>9} {'topk_hit':>9} {'mkt_cum':>11}")
    print("-" * 90)
    for q, g in dd.groupby('q', sort=True):
        print(f"{str(q):<8} {len(g):>5} {g['final_score'].mean():>10.6f} "
              f"{g['win'].mean():>9.4f} {g['spearman'].mean():>9.4f} "
              f"{g['topk_hit'].mean()/K_TOP:>9.4f} {g['mkt_ret'].sum():>11.6f}")
    print("=" * 90)

    # ── 市场状态 vs 模型表现 ──
    print("\n" + "=" * 80)
    print("  月度市场状态 vs 模型表现")
    print("=" * 80)
    for s in summary:
        if s['market_cumret'] > 0.03:
            mkt = "📈 强上涨"
        elif s['market_cumret'] > 0.01:
            mkt = "↗ 微涨"
        elif s['market_cumret'] < -0.03:
            mkt = "📉 强下跌"
        elif s['market_cumret'] < -0.01:
            mkt = "↘ 微跌"
        else:
            mkt = "↔ 横盘"
        if s['final_score'] > 0.05:
            perf = "✓强"
        elif s['final_score'] > 0.02:
            perf = "△中"
        else:
            perf = "✗弱"
        print(f"  {s['month']}: 市场{mkt}(累计{s['market_cumret']:+.4f}) | 模型:{perf}(fs={s['final_score']:.4f}, wr={s['win_rate']:.2%})")

    # ── 保存 ──
    out = os.path.join(MODEL_DIR, 'monthly_breakdown.csv')
    pd.DataFrame(summary).to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n已保存月度明细到: {out}")

    # ── 关键诊断 ──
    print("\n" + "=" * 80)
    print("  关键诊断")
    print("=" * 80)
    neg_months = [s for s in summary if s['final_score'] < 0]
    pos_months = [s for s in summary if s['final_score'] >= 0]
    print(f"  正收益月份: {len(pos_months)}/{len(summary)} ({len(pos_months)/len(summary)*100:.0f}%)")
    if neg_months:
        print(f"  亏损月份: {', '.join(s['month'] for s in neg_months)}")
    print(f"  最差月份: {min(summary, key=lambda x: x['final_score'])['month']} (fs={min(s['final_score'] for s in summary):.4f})")
    print(f"  最佳月份: {max(summary, key=lambda x: x['final_score'])['month']} (fs={max(s['final_score'] for s in summary):.4f})")
    print(f"  月度 final_score 标准差: {np.std([s['final_score'] for s in summary], ddof=1):.4f}")
    print(f"  月度间变异系数: {np.std([s['final_score'] for s in summary], ddof=1)/max(abs(total_fs), 1e-12)*100:.1f}%")
    print("=" * 80)


if __name__ == '__main__':
    mp.freeze_support()   # Windows 打包兼容
    main()
