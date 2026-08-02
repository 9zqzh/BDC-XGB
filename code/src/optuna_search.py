"""
Optuna 贝叶斯超参数搜索（XGBRanker 版）
搜索 8 个关键超参数，25 trials，每 trial 用 n_estimators=200+early_stopping=20 快速评估。

用法：uv run python code/src/optuna_search.py [--n_trials 25]
输出：model/60_158+39/optuna_search/optuna_result.json
"""
import os, sys, copy, json, argparse, gc, warnings, multiprocessing as mp
import numpy as np, pandas as pd
import optuna

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from config import config, config_extended, xgb_config
from train import (
    set_seed, train_one_window,
    preprocess_data, preprocess_val_data,
    _merge_fundamentals, flatten_sequences_to_xgb,
    _continuous_labels_to_ranks,
    feature_columns_map,
    _probe_xgb_cuda,
)
from sklearn.preprocessing import StandardScaler

DEVICE = 'cpu'  # XGBoost 设备，由 CLI --device 覆盖


def quick_train_and_eval(trial_params):
    """快速训练+评估，返回 final_score。不保存模型文件。"""
    import xgboost as xgb

    cfg = copy.deepcopy(config)
    seq_len = cfg['sequence_length']
    feat_name = cfg['feature_num']
    features_list = list(feature_columns_map[feat_name])

    # 加载数据
    data_path = cfg['data_path']
    full_df = pd.read_csv(os.path.join(data_path, 'train.csv'),
                          dtype={'股票代码': str}, low_memory=False)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    from train import split_train_val_by_last_month
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, seq_len, val_months=config_extended.get('val_months', 12)
    )

    all_sids = sorted(full_df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_sids)}

    # 特征工程
    train_data, _ = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    scaler = StandardScaler()
    for col_set in [train_data, val_data]:
        col_set[features_list] = col_set[features_list].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features_list)
    val_data = val_data.dropna(subset=features_list)
    if len(train_data) == 0 or len(val_data) == 0:
        return -999.0
    train_data[features_list] = scaler.fit_transform(train_data[features_list])
    val_data[features_list] = scaler.transform(val_data[features_list])

    fp = os.path.join(data_path, 'history_factors_nan.csv')
    if not os.path.exists(fp):
        fp = os.path.join(data_path, 'hs300_fundamentals.csv')
    train_data, fund_cols = _merge_fundamentals(train_data, fp)
    if fund_cols:
        val_data, _ = _merge_fundamentals(val_data, fp)
        features_list = features_list + fund_cols

    if cfg.get('selected_features'):
        features_list = [f for f in cfg['selected_features'] if f in features_list]

    X_train, y_train_cont, qid_train, _, _, _ = flatten_sequences_to_xgb(train_data, features_list, seq_len)
    X_val, y_val_cont, qid_val, _, _, valid_val_dates = flatten_sequences_to_xgb(val_data, features_list, seq_len)

    if len(X_train) == 0 or len(X_val) == 0:
        return -999.0

    y_train = _continuous_labels_to_ranks(y_train_cont, qid_train)
    y_val = _continuous_labels_to_ranks(y_val_cont, qid_val)

    xgb_params = {
        'max_depth': trial_params['max_depth'],
        'learning_rate': trial_params['learning_rate'],
        'n_estimators': 200,
        'subsample': trial_params['subsample'],
        'colsample_bytree': trial_params['colsample_bytree'],
        'reg_alpha': trial_params['reg_alpha'],
        'reg_lambda': trial_params['reg_lambda'],
        'min_child_weight': trial_params['min_child_weight'],
        'objective': xgb_config['objective'],
        'eval_metric': xgb_config['eval_metric'],
        'ndcg_exp_gain': False,
        'verbosity': 0,
        'n_jobs': xgb_config['n_jobs'],
        'tree_method': 'hist',
        'random_state': 42,
    }
    if DEVICE.startswith('cuda'):
        xgb_params['device'] = DEVICE

    model = xgb.XGBRanker(**xgb_params)
    model.fit(X_train, y_train, qid=qid_train,
              eval_set=[(X_val, y_val)], eval_qid=[qid_val],
              verbose=False)

    preds = model.predict(X_val)
    daily_fs = []
    K_TOP, MIN_GAP = 5, 0.005
    for q in sorted(set(qid_val)):
        mask = qid_val == q
        dp, dl = preds[mask].astype(np.float64), y_val_cont[mask].astype(np.float64)
        n_st = len(dp)
        if n_st < K_TOP: continue
        tti = np.argsort(dl)[::-1][:K_TOP]
        ms, rs = dl[tti].sum(), K_TOP * dl.mean()
        gap = ms - rs
        if abs(gap) < MIN_GAP: continue
        tki = np.argsort(dp)[::-1][:K_TOP]
        ps = dl[tki].sum()
        daily_fs.append((ps - rs) / (gap + 1e-12))

    del X_train, X_val, model; gc.collect()
    return np.mean(daily_fs) if daily_fs else -999.0


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
    }
    return quick_train_and_eval(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=25)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'gpu', 'auto'], default='cpu',
                        help='训练设备；cuda/gpu 使用 GPU，auto 自动检测（默认: cpu）')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU 编号（默认: 0）')
    args = parser.parse_args()

    # ── 解析设备 ──
    global DEVICE
    if args.device in ('cuda', 'gpu'):
        DEVICE = _probe_xgb_cuda(f'cuda:{args.gpu_id}')
        print(f'运行模式: GPU ({DEVICE})')
    elif args.device == 'auto':
        try:
            DEVICE = _probe_xgb_cuda(f'cuda:{args.gpu_id}')
            print(f'运行模式: GPU ({DEVICE})')
        except RuntimeError:
            DEVICE = 'cpu'
            print('运行模式: CPU（GPU 不可用，自动回退）')
    else:
        print('运行模式: CPU')

    set_seed(42)

    search_dir = os.path.join(config['output_dir'], 'optuna_search')
    os.makedirs(search_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Optuna 贝叶斯搜索 (XGBRanker): {args.n_trials} trials")
    print(f"  搜索空间: max_depth[4-7], lr[0.01-0.1], subsample[0.4-0.8]")
    print(f"           colsample[0.2-0.5], alpha[0.1-10], lambda[0.1-20], min_child[1-30]")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\n{'='*60}")
    print(f"  最佳结果")
    print(f"{'='*60}")
    print(f"  Trial #{study.best_trial.number}: final_score={study.best_value:.6f}")
    print(f"  最佳参数: {json.dumps(study.best_params, indent=4)}")

    result = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials,
    }
    with open(os.path.join(search_dir, 'optuna_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到: {search_dir}/optuna_result.json")


if __name__ == '__main__':
    mp.freeze_support()
    main()
