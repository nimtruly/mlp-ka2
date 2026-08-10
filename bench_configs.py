"""
bench_configs.py — Exhaustively bench different HistGBM configs to find
                   one that reliably gives OOF F1 > 0.66.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import sys

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# ── Shared imputation ──────────────────────────────────────────────────────────
for col in ['credit_score', 'acc_balance', 'prod_count']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)
train['country'] = train['country'].fillna(train['country'].mode()[0])
test['country']  = test['country'].fillna(train['country'].mode()[0])

# ── OOF Target Encodings ───────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y   = train['exit_status'].copy()
g   = y.mean()

train['last_name_te']   = np.nan
train['customer_id_te'] = np.nan

for ti, vi in skf.split(train, y):
    tr = train.iloc[ti]
    ln = tr.groupby('last_name')['exit_status'].agg(['count', 'mean'])
    ln_te = (ln['count'] * ln['mean'] + 20 * g) / (ln['count'] + 20)
    train.loc[vi, 'last_name_te'] = (
        train.iloc[vi]['last_name'].map(ln_te).fillna(g)
    )
    cid = tr.groupby('customer_id')['exit_status'].agg(['count', 'mean'])
    cid_te = (cid['count'] * cid['mean'] + 5 * g) / (cid['count'] + 5)
    train.loc[vi, 'customer_id_te'] = (
        train.iloc[vi]['customer_id'].map(cid_te).fillna(g)
    )

ln_f  = train.groupby('last_name')['exit_status'].agg(['count', 'mean'])
test['last_name_te'] = test['last_name'].map(
    (ln_f['count'] * ln_f['mean'] + 20 * g) / (ln_f['count'] + 20)
).fillna(g)

cid_f = train.groupby('customer_id')['exit_status'].agg(['count', 'mean'])
test['customer_id_te'] = test['customer_id'].map(
    (cid_f['count'] * cid_f['mean'] + 5 * g) / (cid_f['count'] + 5)
).fillna(g)

# ── Feature matrix ────────────────────────────────────────────────────────────
num_cols  = ['credit_score', 'age', 'tenure', 'acc_balance', 'prod_count',
             'has_card', 'is_active', 'estimated_salary']
feat_cols = num_cols + ['last_name_te', 'customer_id_te']
cat_cols  = ['country', 'gender']

train_ohe = pd.get_dummies(train[cat_cols], drop_first=True)
test_ohe  = pd.get_dummies(test[cat_cols],  drop_first=True)

X  = pd.concat([train[feat_cols].reset_index(drop=True),
                train_ohe.reset_index(drop=True)], axis=1)
Xt = pd.concat([test[feat_cols].reset_index(drop=True),
                test_ohe.reindex(columns=train_ohe.columns, fill_value=0)
                         .reset_index(drop=True)], axis=1)

print(f"Feature matrix: {X.shape}", flush=True)

# ── Config bench ──────────────────────────────────────────────────────────────
configs = [
    # (label, max_iter, lr, max_leaf_nodes, min_samples_leaf, l2_reg)
    ("baseline (300, lr=0.03, d=31, msl=20, l2=0.5)",  300, 0.03, 31, 20, 0.5),
    ("config-A  (500, lr=0.02, d=63, msl=10, l2=0.1)",  500, 0.02, 63, 10, 0.1),
    ("config-B  (400, lr=0.02, d=31, msl=20, l2=0.1)",  400, 0.02, 31, 20, 0.1),
    ("config-C  (300, lr=0.02, d=63, msl=20, l2=0.1)",  300, 0.02, 63, 20, 0.1),
    ("config-D  (500, lr=0.03, d=31, msl=10, l2=0.1)",  500, 0.03, 31, 10, 0.1),
    ("config-E  (500, lr=0.02, d=63, msl=20, l2=0.1)",  500, 0.02, 63, 20, 0.1),
]

results = []
skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for label, max_iter, lr, mln, msl, l2 in configs:
    oof = np.zeros(len(X))
    for fold, (ti, vi) in enumerate(skf2.split(X, y)):
        m = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=lr,
            max_leaf_nodes=mln, min_samples_leaf=msl,
            l2_regularization=l2, random_state=42 + fold
        )
        m.fit(X.iloc[ti], y.iloc[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 161):
        s = f1_score(y, (oof >= t).astype(int))
        if s > best_f1:
            best_f1, best_thresh = s, t

    for pct in [0.21, 0.22, 0.23, 0.24, 0.25]:
        n    = int(pct * len(oof))
        topn = sorted(oof, reverse=True)[n]
        s    = f1_score(y, (oof >= topn).astype(int))
        if s > best_f1:
            best_f1, best_thresh = s, topn

    results.append((label, best_f1, best_thresh,
                    (oof >= best_thresh).mean() * 100,
                    max_iter, lr, mln, msl, l2))
    print(f"\n{label}")
    print(f"  Best OOF F1:  {best_f1:.4f}  threshold={best_thresh:.4f}  pos_rate={(oof>=best_thresh).mean()*100:.2f}%")
    print(f"  >0.66:        {'YES' if best_f1 > 0.66 else 'NO'}", flush=True)

print("\n\n=== SUMMARY (sorted by OOF F1) ===")
for r in sorted(results, key=lambda x: x[1], reverse=True):
    label, f1, thresh, rate, *_ = r
    print(f"  F1={f1:.4f}  th={thresh:.4f}  rate={rate:.1f}%  {label}")
