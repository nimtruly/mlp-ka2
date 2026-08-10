"""
bench_configs2.py — Test with 5-fold CV (larger training sets per fold)
                    and also explore additional features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

for col in ['credit_score', 'acc_balance', 'prod_count']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)
train['country'] = train['country'].fillna(train['country'].mode()[0])
test['country']  = test['country'].fillna(train['country'].mode()[0])

y = train['exit_status'].copy()
g = y.mean()

def run(n_folds, feat_label, extra_feats=None, hgb_kwargs=None):
    if hgb_kwargs is None:
        hgb_kwargs = dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63,
                          min_samples_leaf=20, l2_regularization=0.1)

    tr = train.copy()
    te = test.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    tr['last_name_te']   = np.nan
    tr['customer_id_te'] = np.nan

    for ti, vi in skf.split(tr, y):
        fold_tr = tr.iloc[ti]
        ln  = fold_tr.groupby('last_name')['exit_status'].agg(['count','mean'])
        cid = fold_tr.groupby('customer_id')['exit_status'].agg(['count','mean'])
        ln_te  = (ln['count']*ln['mean']   + 20*g) / (ln['count']  + 20)
        cid_te = (cid['count']*cid['mean'] +  5*g) / (cid['count'] +  5)
        tr.loc[vi, 'last_name_te']   = tr.iloc[vi]['last_name'].map(ln_te).fillna(g)
        tr.loc[vi, 'customer_id_te'] = tr.iloc[vi]['customer_id'].map(cid_te).fillna(g)

    ln_f  = tr.groupby('last_name')['exit_status'].agg(['count','mean'])
    cid_f = tr.groupby('customer_id')['exit_status'].agg(['count','mean'])
    te['last_name_te']   = te['last_name'].map(
        (ln_f['count']*ln_f['mean']+20*g)/(ln_f['count']+20)).fillna(g)
    te['customer_id_te'] = te['customer_id'].map(
        (cid_f['count']*cid_f['mean']+5*g)/(cid_f['count']+5)).fillna(g)

    num_cols  = ['credit_score','age','tenure','acc_balance','prod_count',
                 'has_card','is_active','estimated_salary']
    feat_cols = num_cols + ['last_name_te','customer_id_te']
    if extra_feats:
        feat_cols += extra_feats
    cat_cols  = ['country','gender']

    train_ohe = pd.get_dummies(tr[cat_cols], drop_first=True)
    test_ohe  = pd.get_dummies(te[cat_cols],  drop_first=True)

    X  = pd.concat([tr[feat_cols].reset_index(drop=True),
                    train_ohe.reset_index(drop=True)], axis=1)
    Xt = pd.concat([te[feat_cols].reset_index(drop=True),
                    test_ohe.reindex(columns=train_ohe.columns, fill_value=0)
                             .reset_index(drop=True)], axis=1)

    oof  = np.zeros(len(X))
    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(skf2.split(X, y)):
        m = HistGradientBoostingClassifier(random_state=42+fold, **hgb_kwargs)
        m.fit(X.iloc[ti], y.iloc[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 161):
        s = f1_score(y, (oof >= t).astype(int))
        if s > best_f1: best_f1, best_thresh = s, t
    for pct in [0.21, 0.22, 0.23, 0.24, 0.25]:
        n    = int(pct * len(oof))
        topn = sorted(oof, reverse=True)[n]
        s    = f1_score(y, (oof >= topn).astype(int))
        if s > best_f1: best_f1, best_thresh = s, topn

    label = f"{n_folds}-fold, {feat_label}"
    print(f"\n{label}")
    print(f"  Best OOF F1:  {best_f1:.4f}  thresh={best_thresh:.4f}  pos={(oof>=best_thresh).mean()*100:.2f}%")
    print(f"  >0.66: {'YES' if best_f1>0.66 else 'NO'}", flush=True)
    return best_f1, best_thresh

# Build simple extra features
train['zero_balance'] = (train['acc_balance'] == 0).astype(int)
train['age_x_active'] = train['age'] * train['is_active']
test['zero_balance']  = (test['acc_balance'] == 0).astype(int)
test['age_x_active']  = test['age'] * test['is_active']

best_hgb = dict(max_iter=500, learning_rate=0.02, max_leaf_nodes=63,
                min_samples_leaf=20, l2_regularization=0.1)

print("=== Exploring folds and features ===", flush=True)

# A: 5-fold, base features
run(5, "base (ln+cid TE)", hgb_kwargs=best_hgb)

# B: 10-fold, base features (known baseline)
run(10, "base (ln+cid TE)", hgb_kwargs=best_hgb)

# C: 10-fold, base + zero_balance + age_x_active
run(10, "base + zero_balance + age_x_active",
    extra_feats=['zero_balance','age_x_active'], hgb_kwargs=best_hgb)

# D: 20-fold, base features (even larger training slices)
run(20, "base (ln+cid TE)", hgb_kwargs=best_hgb)
