"""
test_early_stopping.py — Test early stopping with more trees for LGBM, XGBoost, and CatBoost
                         to see if it improves the OOF F1 score.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

train = pd.read_csv('train.csv')

# Imputation
for col in ['credit_score', 'acc_balance', 'prod_count']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
train['country'] = train['country'].fillna(train['country'].mode()[0])

y = train['exit_status'].copy()
g = y.mean()

# TE
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train['last_name_te'] = np.nan
train['customer_id_te'] = np.nan

for ti, vi in skf.split(train, y):
    tr = train.iloc[ti]
    ln = tr.groupby('last_name')['exit_status'].agg(['count','mean'])
    cid = tr.groupby(['customer_id', 'last_name', 'country', 'gender'])['exit_status'].agg(['count','mean'])
    ln_te = (ln['count']*ln['mean'] + 20*g) / (ln['count'] + 20)
    cid_te = (cid['count']*cid['mean'] + 5*g) / (cid['count'] + 5)
    train.loc[vi, 'last_name_te'] = train.iloc[vi]['last_name'].map(ln_te).fillna(g)
    
    val_df = train.iloc[vi]
    val_keys = val_df[['customer_id', 'last_name', 'country', 'gender']]
    mapped = val_keys.set_index(['customer_id', 'last_name', 'country', 'gender']).index.map(cid_te).fillna(g)
    train.loc[vi, 'customer_id_te'] = mapped

# Extra Features
for df in [train]:
    df['zero_balance'] = (df['acc_balance'] == 0).astype(int)
    df['age_x_active'] = df['age'] * df['is_active']
    df['balance_salary_ratio'] = df['acc_balance'] / (df['estimated_salary'] + 1)
    df['products_per_tenure'] = df['prod_count'] / (df['tenure'] + 1)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['credit_active'] = df['credit_score'] * df['is_active']
    df['is_active_products'] = df['is_active'] * df['prod_count']

# OHE
cat_cols = ['country', 'gender']
train_ohe = pd.get_dummies(train[cat_cols], drop_first=True)

# Build feature matrix
num_cols = ['credit_score', 'age', 'tenure', 'acc_balance', 'prod_count', 'has_card', 'is_active', 'estimated_salary',
            'zero_balance', 'age_x_active', 'balance_salary_ratio', 'products_per_tenure', 'is_senior', 'credit_active', 'is_active_products',
            'last_name_te', 'customer_id_te']

X = pd.concat([train[num_cols].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)

# Fit OOF
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

skf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (ti, vi) in enumerate(skf2.split(X, y)):
    X_tr, y_tr = X.iloc[ti], y.iloc[ti]
    X_va, y_va = X.iloc[vi], y.iloc[vi]
    
    # LGBM with early stopping (using validation fold)
    lgb = LGBMClassifier(n_estimators=1500, learning_rate=0.03, random_state=42+fold, verbose=-1)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(50, verbose=False)])
    oof_lgb[vi] = lgb.predict_proba(X_va)[:, 1]
    
    # XGBoost with early stopping (using validation fold)
    xgb = XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, random_state=42+fold, verbosity=0, early_stopping_rounds=50)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[vi] = xgb.predict_proba(X_va)[:, 1]
    
    # CatBoost with early stopping (using validation fold)
    cat = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, random_state=42+fold, verbose=0, early_stopping_rounds=50)
    cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[vi] = cat.predict_proba(X_va)[:, 1]

oof_blend = 0.4 * oof_lgb + 0.2 * oof_xgb + 0.4 * oof_cat

# Optimize threshold
best_f1 = 0
best_t = 0.5
for t in np.linspace(0.10, 0.90, 161):
    s = f1_score(y, (oof_blend >= t).astype(int))
    if s > best_f1:
        best_f1, best_t = s, t

print(f'Ensemble OOF F1 with early stopping: {best_f1:.4f} at threshold {best_t:.4f}')
