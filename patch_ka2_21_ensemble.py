"""
patch_ka2_21_ensemble.py — Patch ka2_21.ipynb to implement a high-performance ensemble of
                            LightGBM, XGBoost, and CatBoost models.
                            Uses CONDITIONAL target encoding for customer_id to avoid leakage and test set noise.
"""
import nbformat

with open('ka2_21.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ── Cell 60: Markdown ─────────────────────────────────────────────────────────
nb.cells[60] = nbformat.v4.new_markdown_cell(
    "## Improved Final Model (ka2_21 — Ensemble model for F1 > 0.70)\n\n"
    "To achieve a test F1-score above 0.70, we implement a powerful, optimized ensemble model combining "
    "**LightGBM**, **XGBoost**, and **CatBoost** along with advanced feature engineering.\n\n"
    "### Techniques:\n"
    "1. **Ensemble Blending** — We train three state-of-the-art gradient boosting classifiers (LGBM, XGBoost, CatBoost) "
    "using 30-fold cross-validation and blend their predictions (`0.4 * LGBM + 0.2 * XGBoost + 0.4 * CatBoost`).\n"
    "2. **Advanced Feature Engineering** — We add zero balance flag, interaction terms (age × active member), credit score activity, "
    "and ratio columns (balance-to-salary ratio, products-per-tenure).\n"
    "3. **Conditional Out-of-Fold Target Encoding** — High-impact target encoding on `last_name` (smoothing=20) and `customer_id` "
    "which is conditional on matching name, country, and gender (smoothing=5) to avoid noise from synthetic duplicate IDs.\n"
    "4. **Precision-Recall Threshold Optimization** — Instead of a fixed 0.5 threshold, we scan the precision-recall space "
    "on the OOF blend probabilities to find the threshold that maximizes the F1 score.\n"
)

# ── Cell 61: Feature prep + OOF TE ──────────────────────────────────────────
cell61 = '''\
# ── Feature preparation and Target Encoding ───────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

train_final = raw_train_df.copy()
test_final  = raw_test_df.copy()

# Impute missing values with training statistics
for col in ["credit_score", "acc_balance", "prod_count"]:
    med = train_final[col].median()
    train_final[col] = train_final[col].fillna(med)
    test_final[col]  = test_final[col].fillna(med)
mode_country = train_final["country"].mode()[0]
train_final["country"] = train_final["country"].fillna(mode_country)
test_final["country"]  = test_final["country"].fillna(mode_country)

# ── Feature Engineering ────────────────────────────────────────────────────────
for df in [train_final, test_final]:
    df["zero_balance"] = (df["acc_balance"] == 0).astype(int)
    df["age_x_active"] = df["age"] * df["is_active"]
    df["balance_salary_ratio"] = df["acc_balance"] / (df["estimated_salary"] + 1)
    df["products_per_tenure"] = df["prod_count"] / (df["tenure"] + 1)
    df["is_senior"] = (df["age"] >= 60).astype(int)
    df["credit_active"] = df["credit_score"] * df["is_active"]
    df["is_active_products"] = df["is_active"] * df["prod_count"]

# ── Conditional Target Encoding (30 splits) ───────────────────────────────────
N_FOLDS     = 30
LN_SMOOTH   = 20
CID_SMOOTH  = 5

skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
y_final     = train_final["exit_status"].copy()
global_mean = y_final.mean()

train_final["last_name_te"]   = np.nan
train_final["customer_id_te"] = np.nan

for train_idx, val_idx in skf.split(train_final, y_final):
    tr = train_final.iloc[train_idx]

    # last_name TE (smoothing=20)
    ln    = tr.groupby("last_name")["exit_status"].agg(["count", "mean"])
    ln_te = (ln["count"] * ln["mean"] + LN_SMOOTH * global_mean) / (ln["count"] + LN_SMOOTH)
    train_final.loc[val_idx, "last_name_te"] = (
        train_final.iloc[val_idx]["last_name"].map(ln_te).fillna(global_mean)
    )

    # Conditional customer_id TE (group by customer_id, last_name, country, gender to avoid synthetic duplicates noise)
    cid    = tr.groupby(["customer_id", "last_name", "country", "gender"])["exit_status"].agg(["count", "mean"])
    cid_te = (cid["count"] * cid["mean"] + CID_SMOOTH * global_mean) / (cid["count"] + CID_SMOOTH)
    
    val_df = train_final.iloc[val_idx]
    val_keys = val_df[["customer_id", "last_name", "country", "gender"]]
    mapped = val_keys.set_index(["customer_id", "last_name", "country", "gender"]).index.map(cid_te).fillna(global_mean)
    train_final.loc[val_idx, "customer_id_te"] = mapped

# Full-train encodings for test set
ln_full  = train_final.groupby("last_name")["exit_status"].agg(["count", "mean"])
ln_fte   = (ln_full["count"] * ln_full["mean"] + LN_SMOOTH * global_mean) / (ln_full["count"] + LN_SMOOTH)
test_final["last_name_te"] = test_final["last_name"].map(ln_fte).fillna(global_mean)

cid_full = train_final.groupby(["customer_id", "last_name", "country", "gender"])["exit_status"].agg(["count", "mean"])
cid_fte  = (cid_full["count"] * cid_full["mean"] + CID_SMOOTH * global_mean) / (cid_full["count"] + CID_SMOOTH)

test_keys = test_final[["customer_id", "last_name", "country", "gender"]]
test_final["customer_id_te"] = test_keys.set_index(["customer_id", "last_name", "country", "gender"]).index.map(cid_fte).fillna(global_mean)

# ── Categorical One-Hot Encoding ─────────────────────────────────────────────
num_cols  = ["credit_score", "age", "tenure", "acc_balance", "prod_count", "has_card", "is_active", "estimated_salary",
             "zero_balance", "age_x_active", "balance_salary_ratio", "products_per_tenure", "is_senior", "credit_active", "is_active_products",
             "last_name_te", "customer_id_te"]
cat_cols  = ["country", "gender"]

train_ohe = pd.get_dummies(train_final[cat_cols], drop_first=True)
test_ohe  = pd.get_dummies(test_final[cat_cols],  drop_first=True)

X_final = pd.concat([train_final[num_cols].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
X_final_test = pd.concat([test_final[num_cols].reset_index(drop=True), test_ohe.reindex(columns=train_ohe.columns, fill_value=0).reset_index(drop=True)], axis=1)

print("Train features shape:", X_final.shape)
print("Test  features shape:", X_final_test.shape)
'''
nb.cells[61] = nbformat.v4.new_code_cell(cell61)

# ── Cell 62: Model training + Blending + Threshold optimization ───────────────
cell62 = '''\
# ── Fit 30-fold Ensemble of LightGBM, XGBoost, and CatBoost ──────────────────
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

oof_lgb = np.zeros(len(X_final))
oof_xgb = np.zeros(len(X_final))
oof_cat = np.zeros(len(X_final))

test_lgb = np.zeros(len(X_final_test))
test_xgb = np.zeros(len(X_final_test))
test_cat = np.zeros(len(X_final_test))

skf_final = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print("Training ensemble models across 30 folds...")
for fold, (train_idx, val_idx) in enumerate(skf_final.split(X_final, y_final)):
    X_tr, y_tr = X_final.iloc[train_idx], y_final.iloc[train_idx]
    X_va = X_final.iloc[val_idx]

    # LightGBM Classifier
    lgb = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42 + fold, verbose=-1)
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
    test_lgb += lgb.predict_proba(X_final_test)[:, 1] / N_FOLDS

    # XGBoost Classifier
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42 + fold, verbosity=0)
    xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
    test_xgb += xgb.predict_proba(X_final_test)[:, 1] / N_FOLDS

    # CatBoost Classifier
    cat = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, random_state=42 + fold, verbose=0)
    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]
    test_cat += cat.predict_proba(X_final_test)[:, 1] / N_FOLDS

# ── Blending probabilities ───────────────────────────────────────────────────
oof_blend = 0.4 * oof_lgb + 0.2 * oof_xgb + 0.4 * oof_cat
test_blend = 0.4 * test_lgb + 0.2 * test_xgb + 0.4 * test_cat

# ── Optimize F1 Threshold on OOF blend probabilities ────────────────────────
best_thresh, best_f1 = 0.5, 0.0
for t in np.linspace(0.10, 0.90, 161):
    s = f1_score(y_final, (oof_blend >= t).astype(int))
    if s > best_f1:
        best_f1, best_thresh = s, t

n_pos       = int(0.22 * len(oof_blend))
topn_thresh = sorted(oof_blend, reverse=True)[n_pos]
topn_f1     = f1_score(y_final, (oof_blend >= topn_thresh).astype(int))
if topn_f1 > best_f1:
    best_f1, best_thresh = topn_f1, topn_thresh

print("OOF F1 at default 0.5:    %.4f" % f1_score(y_final, (oof_blend >= 0.5).astype(int)))
print("OOF Best threshold:       %.4f" % best_thresh)
print("OOF Best F1 (CV):         %.4f" % best_f1)
print("Positive prediction rate: %.2f%%" % ((oof_blend >= best_thresh).mean() * 100))
'''
nb.cells[62] = nbformat.v4.new_code_cell(cell62)

# ── Cell 63: Save Submission ──────────────────────────────────────────────────
cell63 = '''\
# ── Save Predictions ──────────────────────────────────────────────────────────
import glob, re

test_preds = (test_blend >= best_thresh).astype(int)
print("Test predicted exits:      %d" % test_preds.sum())
print("Test predicted exit rate:  %.2f%%" % (test_preds.mean() * 100))

existing = glob.glob("submission*.csv")
counters = [int(m.group(1)) for f in existing
            for m in [re.search(r"submission(\\d+)\\.csv", f)] if m]
counter  = max(counters) + 1 if counters else 1
fname    = "submission%d.csv" % counter

id_col = "id" if "id" in raw_test_df.columns else raw_test_df.columns[0]
submission = pd.DataFrame({id_col: raw_test_df[id_col], "exit_status": test_preds})
submission.to_csv(fname, index=False)
submission.to_csv("submission.csv", index=False)
print("Saved: %s  (and submission.csv)" % fname)
print(submission.head(10))
'''
nb.cells[63] = nbformat.v4.new_code_cell(cell63)

nb.cells = nb.cells[:64]

with open('ka2_21.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("ka2_21.ipynb patched with conditional ensemble. Cells: %d" % len(nb.cells))
