"""
test_conditional_te.py — Evaluate conditional customer_id target encoding where we only map
                         if key attributes (last_name, country, gender) match.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

train = pd.read_csv('train.csv')

# Imputation
for col in ['credit_score', 'acc_balance', 'prod_count']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
train['country'] = train['country'].fillna(train['country'].mode()[0])

y = train['exit_status'].copy()
g = y.mean()

# 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train['last_name_te'] = np.nan
train['customer_id_te_cond'] = np.nan
train['customer_id_te_raw'] = np.nan

for ti, vi in skf.split(train, y):
    tr = train.iloc[ti]
    va = train.iloc[vi]
    
    # 1. last_name TE (smoothing=20)
    ln = tr.groupby('last_name')['exit_status'].agg(['count','mean'])
    ln_te = (ln['count']*ln['mean'] + 20*g) / (ln['count'] + 20)
    train.loc[vi, 'last_name_te'] = va['last_name'].map(ln_te).fillna(g)
    
    # 2. Raw customer_id TE (smoothing=5)
    cid = tr.groupby('customer_id')['exit_status'].agg(['count','mean'])
    cid_te = (cid['count']*cid['mean'] + 5*g) / (cid['count'] + 5)
    train.loc[vi, 'customer_id_te_raw'] = va['customer_id'].map(cid_te).fillna(g)
    
    # 3. Conditional customer_id TE
    # We group by ['customer_id', 'last_name', 'country', 'gender'] in tr
    cid_cond = tr.groupby(['customer_id', 'last_name', 'country', 'gender'])['exit_status'].agg(['count','mean'])
    # Smoothing = 5
    cid_cond_te = (cid_cond['count']*cid_cond['mean'] + 5*g) / (cid_cond['count'] + 5)
    
    # Map to va
    va_keys = va[['customer_id', 'last_name', 'country', 'gender']]
    # Set index to match group keys
    mapped = va_keys.set_index(['customer_id', 'last_name', 'country', 'gender']).index.map(cid_cond_te).fillna(g)
    train.loc[vi, 'customer_id_te_cond'] = mapped

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

# Build feature matrices for both configs
num_cols_raw = ['credit_score', 'age', 'tenure', 'acc_balance', 'prod_count', 'has_card', 'is_active', 'estimated_salary',
                'zero_balance', 'age_x_active', 'balance_salary_ratio', 'products_per_tenure', 'is_senior', 'credit_active', 'is_active_products',
                'last_name_te', 'customer_id_te_raw']

num_cols_cond = ['credit_score', 'age', 'tenure', 'acc_balance', 'prod_count', 'has_card', 'is_active', 'estimated_salary',
                 'zero_balance', 'age_x_active', 'balance_salary_ratio', 'products_per_tenure', 'is_senior', 'credit_active', 'is_active_products',
                 'last_name_te', 'customer_id_te_cond']

X_raw  = pd.concat([train[num_cols_raw].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
X_cond = pd.concat([train[num_cols_cond].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)

def evaluate(X_data, label):
    oof = np.zeros(len(X_data))
    for fold, (ti, vi) in enumerate(skf.split(X_data, y)):
        lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42+fold, verbose=-1)
        lgb.fit(X_data.iloc[ti], y.iloc[ti])
        oof[vi] = lgb.predict_proba(X_data.iloc[vi])[:, 1]
    
    best_f1 = 0
    best_t = 0.5
    for t in np.linspace(0.10, 0.90, 161):
        s = f1_score(y, (oof >= t).astype(int))
        if s > best_f1:
            best_f1, best_t = s, t
    print(f'{label} — OOF F1: {best_f1:.4f} at threshold {best_t:.4f}')

evaluate(X_raw, 'Raw customer_id target encoding')
evaluate(X_cond, 'Conditional customer_id target encoding')
