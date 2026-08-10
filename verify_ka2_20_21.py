"""verify_ka2_20_21.py — verify OOF F1 for both notebook configs"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# ── Shared imputation ──────────────────────────────────────────────────────────
for col in ['credit_score', 'acc_balance', 'prod_count']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)
train['country'] = train['country'].fillna(train['country'].mode()[0])
test['country']  = test['country'].fillna(train['country'].mode()[0])

# ── OOF target encodings ───────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y   = train['exit_status'].copy()
g   = y.mean()

train['last_name_te']   = np.nan
train['customer_id_te'] = np.nan

for ti, vi in skf.split(train, y):
    tr = train.iloc[ti]

    ln = tr.groupby('last_name')['exit_status'].agg(['count','mean'])
    ln_te = (ln['count']*ln['mean'] + 20*g) / (ln['count'] + 20)
    train.loc[vi, 'last_name_te'] = train.iloc[vi]['last_name'].map(ln_te).fillna(g)

    cid = tr.groupby('customer_id')['exit_status'].agg(['count','mean'])
    cid_te = (cid['count']*cid['mean'] + 5*g) / (cid['count'] + 5)
    train.loc[vi, 'customer_id_te'] = train.iloc[vi]['customer_id'].map(cid_te).fillna(g)

ln_f = train.groupby('last_name')['exit_status'].agg(['count','mean'])
test['last_name_te'] = test['last_name'].map(
    (ln_f['count']*ln_f['mean'] + 20*g) / (ln_f['count'] + 20)).fillna(g)

cid_f = train.groupby('customer_id')['exit_status'].agg(['count','mean'])
test['customer_id_te'] = test['customer_id'].map(
    (cid_f['count']*cid_f['mean'] + 5*g) / (cid_f['count'] + 5)).fillna(g)

# ── OHE (drop_first=True — the verified working config) ───────────────────────
num_cols = ['credit_score','age','tenure','acc_balance','prod_count',
            'has_card','is_active','estimated_salary']
cat_cols = ['country','gender']
train_ohe = pd.get_dummies(train[cat_cols], drop_first=True)
test_ohe  = pd.get_dummies(test[cat_cols],  drop_first=True)

def run_config(label, feature_cols):
    X  = pd.concat([train[feature_cols].reset_index(drop=True),
                    train_ohe.reset_index(drop=True)], axis=1)
    Xt = pd.concat([test[feature_cols].reset_index(drop=True),
                    test_ohe.reindex(columns=train_ohe.columns, fill_value=0)
                        .reset_index(drop=True)], axis=1)

    oof  = np.zeros(len(X))
    tprob = np.zeros(len(Xt))
    skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (ti, vi) in enumerate(skf2.split(X, y)):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.5, random_state=42+fold)
        m.fit(X.iloc[ti], y.iloc[ti])
        oof[vi]  = m.predict_proba(X.iloc[vi])[:, 1]
        tprob   += m.predict_proba(Xt)[:, 1] / 10

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 161):
        s = f1_score(y, (oof >= t).astype(int))
        if s > best_f1: best_f1, best_thresh = s, t
    n_pos = int(0.22 * len(oof))
    topn  = sorted(oof, reverse=True)[n_pos]
    s22   = f1_score(y, (oof >= topn).astype(int))
    if s22 > best_f1: best_f1, best_thresh = s22, topn

    print(f"\n=== {label} ===")
    print(f"  Features:          {feature_cols}")
    print(f"  OOF F1 @ 0.5:     {f1_score(y, (oof>=0.5).astype(int)):.4f}")
    print(f"  Best threshold:    {best_thresh:.4f}")
    print(f"  Best OOF F1:       {best_f1:.4f}")
    print(f"  Positive rate:     {(oof>=best_thresh).mean()*100:.2f}%")
    print(f"  Target > 0.66:     {'YES ✓' if best_f1 > 0.66 else 'NO ✗'}")
    return best_f1

f1_20 = run_config("ka2_20  (last_name TE only)",
                   num_cols + ['last_name_te'])
f1_21 = run_config("ka2_21  (last_name + customer_id TE)",
                   num_cols + ['last_name_te', 'customer_id_te'])

print(f"\nSummary:")
print(f"  ka2_20 OOF F1: {f1_20:.4f}  {'✓' if f1_20 > 0.66 else '✗'}")
print(f"  ka2_21 OOF F1: {f1_21:.4f}  {'✓' if f1_21 > 0.66 else '✗'}")
