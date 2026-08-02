"""
方向分类器 (P1): LightGBM 二分类预测个股未来5日是否上涨。
用 OOF (Out-of-Fold) 时序交叉验证生成训练集特征，避免数据泄露。
输出方向概率作为 XGBRanker 的辅助特征。
"""
import numpy as np
import lightgbm as lgb


def generate_direction_features(X, y_cont, qid, n_splits=5, random_state=42):
    """
    使用时序交叉验证生成 OOF 方向概率特征。

    Args:
        X: np.array, 特征矩阵 (N, D)
        y_cont: np.array, 连续超额收益标签 (N,)
        qid: np.array, 组ID (N,)——按交易日排序
        n_splits: 交叉验证折数（按时序自然切分，不需 shuffle）
        random_state: 随机种子

    Returns:
        direction_proba: np.array, OOF 预测的上涨概率 (N,)
        final_model: LGBMClassifier, 用全部数据训练的最终模型（供推理用）
    """
    y_binary = (y_cont > 0).astype(int)
    n = len(y_binary)
    direction_proba = np.zeros(n, dtype=np.float32)

    unique_qids = sorted(set(qid))
    n_qids = len(unique_qids)
    if n_qids < n_splits * 2:
        # 组数不够做交叉验证，直接用全部数据训练+预测
        n_splits = 1

    if n_splits == 1:
        # 不做 OOF，直接用全部数据训练（数据量太少时的降级方案）
        model = _make_classifier(random_state)
        model.fit(X, y_binary)
        direction_proba = model.predict_proba(X)[:, 1].astype(np.float32)
        return direction_proba, model

    fold_size = n_qids // n_splits

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_splits - 1 else n_qids

        val_qids = set(unique_qids[val_start:val_end])
        train_mask = np.array([q not in val_qids for q in qid])
        val_mask = np.array([q in val_qids for q in qid])

        if train_mask.sum() < 100 or val_mask.sum() < 10:
            continue

        model = _make_classifier(random_state + fold)
        model.fit(X[train_mask], y_binary[train_mask])
        direction_proba[val_mask] = model.predict_proba(X[val_mask])[:, 1]

    # 已预测部分如果仍有0（fold被跳过），用全局模型填充
    unfilled = direction_proba == 0
    if unfilled.any():
        final_model = _make_classifier(random_state)
        final_model.fit(X, y_binary)
        direction_proba[unfilled] = final_model.predict_proba(X[unfilled])[:, 1]
    else:
        final_model = _make_classifier(random_state)
        final_model.fit(X, y_binary)

    return direction_proba.astype(np.float32), final_model


def _make_classifier(random_state=42):
    """构建 LightGBM 二分类器。参数偏保守，避免过拟合。"""
    return lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        num_leaves=15,            # 小树，防过拟合
        learning_rate=0.05,
        n_estimators=80,          # 少量树
        subsample=0.6,
        colsample_bytree=0.3,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_samples=50,     # 叶子最少样本数，防过拟合
        random_state=random_state,
        verbose=-1,
        force_col_wise=True,      # 多特征场景优化
    )
