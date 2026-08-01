"""
窗口4 月度切分分析脚本
用法: python analyze_window4_monthly.py [--cross_vs_nocross]

--cross_vs_nocross: 自动对比含/不含P2交叉特征的月度表现
"""
import os, sys, json, gc, warnings, argparse, multiprocessing as mp
import numpy as np, pandas as pd, joblib, torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))


def spearman_rho_numpy(a: np.ndarray, b: np.ndarray) -> float:
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

CROSS_FEATURES = [
    'cross_MA60_SUMN60', 'cross_MA60_ROC60', 'cross_MA60_ROC30',
    'cross_SUMN60_ROC60', 'cross_SUMN30_ROC60',
    'cross_MA60_MA30', 'cross_ROC30_MA30',
    'cross_vol_price_div', 'cross_liq_adj_ret',
]


def run_monthly_analysis(features_override=None, model_subdir=''):
    """执行一次完整的月度分析，返回 summary 列表。
    features_override: 若不为None，临时覆盖 config['selected_features']"""
    from config import config, xgb_config
    from train import (
        _preprocess_common, _merge_fundamentals, flatten_sequences_to_xgb,
        set_seed, feature_engineer_func_map, feature_columns_map,
    )
    set_seed(42)

    # 临时覆盖特征
    saved = config.get('selected_features')
    if features_override is not None:
        config['selected_features'] = features_override

    base = os.path.join(os.path.dirname(__file__), 'model', '60_158+39')
    MODEL_DIR = os.path.join(base, model_subdir, 'cross_val_4_窗口4_近期市场') if model_subdir else os.path.join(base, 'cross_val_4_窗口4_近期市场')
    VAL_START = pd.to_datetime('2025-07-01')
    VAL_END   = pd.to_datetime('2026-06-30')
    CTX_START = VAL_START - pd.DateOffset(days=180)
    MIN_GAP   = 0.005
    K_TOP     = 5

    print("加载模型...")
    model  = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    data_path = config['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    print(f"全量数据: {len(full_df):,}行, {full_df['日期'].min().date()}~{full_df['日期'].max().date()}")

    context_df = full_df[(full_df['日期'] >= CTX_START) & (full_df['日期'] <= VAL_END)].copy()
    context_df['日期'] = context_df['日期'].dt.strftime('%Y-%m-%d')
    del full_df; gc.collect()
    print(f"上下文: {len(context_df):,}行")

    all_sids = sorted(context_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_sids)}

    seq_len = config['sequence_length']
    feat_name = config['feature_num']
    features_list = list(feature_columns_map[feat_name])

    print("特征工程中...")
    val_data, _ = _preprocess_common(context_df, stockid2idx, desc="特征工程", drop_small_open=True)
    gc.collect()

    val_data[features_list] = val_data[features_list].replace([np.inf, -np.inf], np.nan)
    val_data = val_data.dropna(subset=features_list)
    val_data[features_list] = scaler.transform(val_data[features_list])

    fp = os.path.join(data_path, 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(data_path, 'hs300_fundamentals.csv')
    val_data, fund_cols = _merge_fundamentals(val_data, fp)
    if fund_cols:
        features_list = features_list + fund_cols

    if config.get('selected_features'):
        features_list = [f for f in config['selected_features'] if f in features_list]
        print(f"特征筛选: {len(features_list)}个因子")
    gc.collect()

    print("展平特征...")
    X_val, y_val_cont, qid_val, _, _, valid_dates = flatten_sequences_to_xgb(val_data, features_list, seq_len)
    del val_data; gc.collect()

    valid_dates_dt = pd.to_datetime(valid_dates)
    in_window = (valid_dates_dt >= VAL_START) & (valid_dates_dt <= VAL_END)
    qid_set = set(np.where(in_window)[0])
    mask = np.array([q in qid_set for q in qid_val])
    X_val, y_val_cont, qid_val = X_val[mask], y_val_cont[mask], qid_val[mask]

    uq = sorted(set(qid_val))
    qid_map = {o: n for n, o in enumerate(uq)}
    qid_val = np.array([qid_map[q] for q in qid_val])
    new_valid = [valid_dates[i] for i in range(len(valid_dates)) if i in qid_set]
    print(f"有效交易日: {len(new_valid)}")
    gc.collect()

    print("预测中...")
    preds = model.predict(X_val)

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
        daily.append({'date': pd.to_datetime(ds), 'final_score': fs, 'topk_hit': hc,
                       'spearman': sp, 'win': w, 'pred_ret': float(tkr.mean()),
                       'mkt_ret': float(dl.mean()), 'n_stocks': n_st})

    # 恢复配置
    config['selected_features'] = saved

    if not daily:
        print("错误: 无有效评估日")
        return None

    dd = pd.DataFrame(daily)
    dd['ym'] = dd['date'].dt.to_period('M')
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
        summary.append({'month': str(m), 'days': n, 'final_score': fm, 'win_rate': wr,
                         'topk_hit_rate': th, 'spearman': sp,
                         'pred_ret': pr, 'market_ret': mr, 'fs_std': fs_std, 'market_cumret': mc})
    return summary


def print_monthly_table(summary, title):
    if not summary:
        return
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    hdr = f"{'月份':<8} {'天数':>5} {'fs_mean':>10} {'win_rate':>9} {'topk_hit':>9} {'spearman':>9}"
    print(hdr)
    print("-" * 120)
    for s in summary:
        print(f"{s['month']:<8} {s['days']:>5} {s['final_score']:>10.6f} {s['win_rate']:>9.4f} "
              f"{s['topk_hit_rate']:>9.4f} {s['spearman']:>9.4f}")
    fs_all = [s['final_score'] for s in summary]
    print("-" * 120)
    print(f"{'合计':<8} {sum(s['days'] for s in summary):>5} {np.mean(fs_all):>10.6f} "
          f"{np.mean([s['win_rate'] for s in summary]):>9.4f} "
          f"{np.mean([s['topk_hit_rate'] for s in summary]):>9.4f} "
          f"{np.mean([s['spearman'] for s in summary]):>9.4f}")
    print(f"{'='*120}")


def main():
    from config import config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_vs_nocross', action='store_true', help='cross vs no-cross对比')
    args = parser.parse_args()

    if not args.cross_vs_nocross:
        summary = run_monthly_analysis()
        if summary:
            print_monthly_table(summary, "窗口4 各月验证指标明细")
        return

    # A组: 含交叉
    base = [f for f in config.get('selected_features', []) if f not in CROSS_FEATURES]
    print("=" * 60)
    print("  月度分析 A/B 对比")
    print("=" * 60)
    print("\n[A组] 含交叉特征...")
    summary_a = run_monthly_analysis(features_override=base + CROSS_FEATURES, model_subdir='cross_val_A')

    print("\n[B组] 纯IC124基线...")
    summary_b = run_monthly_analysis(features_override=base, model_subdir='cross_val_B')

    if not summary_a or not summary_b:
        print("数据不足，无法对比")
        return

    print(f"\n{'='*90}")
    print(f"  A(含交叉) vs B(无交叉) 月度对比")
    print(f"{'='*90}")
    print(f"  {'月份':<8s} {'A fs':>10s} {'B fs':>10s} {'差值':>10s} {'A wr':>8s} {'B wr':>8s}")
    print(f"  {'-'*60}")
    for sa, sb in zip(summary_a, summary_b):
        afs, bfs = sa['final_score'], sb['final_score']
        awr, bwr = sa['win_rate'], sb['win_rate']
        print(f"  {sa['month']:<8s} {afs:>10.4f} {bfs:>10.4f} {(bfs-afs):>+10.4f} {awr:>8.4f} {bwr:>8.4f}")
    print(f"{'='*90}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
