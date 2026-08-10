"""
final_bench.py — Final targeted experiments to push OOF F1 above 0.66.
Tests: different customer_id smoothing (0,1,2,5,10) and n_folds.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
import sys

sys.stdout.reconfigure(encoding='utf-8')

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

num_cols  = ['credit_score','age','tenure','acc_balance','prod_count',
             'has_card','is_active','estimated_salary']
cat_cols  = ['country','gender']


def run_experiment(cid_smoothing, n_folds, label):
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
        cid_te = (cid['count']*cid['mean'] + cid_smoothing*g) / (cid['count'] + cid_smoothing)
        tr.loc[vi, 'last_name_te']   = tr.iloc[vi]['last_name'].map(ln_te).fillna(g)
        tr.loc[vi, 'customer_id_te'] = tr.iloc[vi]['customer_id'].map(cid_te).fillna(g)

    ln_f  = tr.groupby('last_name')['exit_status'].agg(['count','mean'])
    cid_f = tr.groupby('customer_id')['exit_status'].agg(['count','mean'])
    te['last_name_te']   = te['last_name'].map(
        (ln_f['count']*ln_f['mean']+20*g)/(ln_f['count']+20)).fillna(g)
    te['customer_id_te'] = te['customer_id'].map(
        (cid_f['count']*cid_f['mean']+cid_smoothing*g)/(cid_f['count']+cid_smoothing)).fillna(g)

    feat_cols = num_cols + ['last_name_te','customer_id_te']
    train_ohe = pd.get_dummies(tr[cat_cols], drop_first=True)
    test_ohe  = pd.get_dummies(te[cat_cols],  drop_first=True)

    X  = pd.concat([tr[feat_cols].reset_index(drop=True),
                    train_ohe.reset_index(drop=True)], axis=1)
    Xt = pd.concat([te[feat_cols].reset_index(drop=True),
                    test_ohe.reindex(columns=train_ohe.columns, fill_value=0)
                             .reset_index(drop=True)], axis=1)

    oof = np.zeros(len(X))
    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(skf2.split(X, y)):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=0.5,
            random_state=42 + fold)
        m.fit(X.iloc[ti], y.iloc[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.linspace(0.10, 0.90, 161):
        s = f1_score(y, (oof >= t).astype(int))
        if s > best_f1: best_f1, best_thresh = s, t
    for pct in [0.20, 0.21, 0.22, 0.23, 0.24]:
        n    = int(pct * len(oof))
        topn = sorted(oof, reverse=True)[n]
        s    = f1_score(y, (oof >= topn).astype(int))
        if s > best_f1: best_f1, best_thresh = s, topn

    above = 'YES' if best_f1 > 0.66 else 'NO'
    print(f"  {label:55s} OOF F1={best_f1:.4f}  pos={( oof>=best_thresh).mean()*100:.1f}%  >0.66:{above}")
    sys.stdout.flush()
    return best_f1, best_thresh


print("=== Testing customer_id smoothing values (10-fold) ===")
results = []
for sm in [0.001, 1, 2, 5, 10, 20]:
    f1, thresh = run_experiment(sm, 10, f"cid_smooth={sm:6.3f}, 10-fold")
    results.append((f1, thresh, sm, 10))

print("\n=== Testing n_folds with best smoothing ===")
for nf in [5, 15, 20, 30]:
    f1, thresh = run_experiment(5, nf, f"cid_smooth=5, {nf}-fold")
    results.append((f1, thresh, 5, nf))

print("\n=== BEST RESULTS ===")
for r in sorted(results, key=lambda x: x[0], reverse=True)[:5]:
    f1, thresh, sm, nf = r
    print(f"  F1={f1:.4f}  thresh={thresh:.4f}  cid_smooth={sm}  folds={nf}")
