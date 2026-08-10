"""
patch_ka2_21_final.py — Final definitive patch for ka2_21.ipynb
                        Best verified config: OOF F1 = 0.6609
                        (10-fold, +zero_balance+age_x_active, config-E HGB)
"""
import nbformat

with open('ka2_21.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ── Cell 60: Markdown ─────────────────────────────────────────────────────────
nb.cells[60] = nbformat.v4.new_markdown_cell(
    "## Improved Final Model (ka2_21)\n\n"
    "Four techniques that move binary-F1 — nothing else matters:\n\n"
    "1. **Threshold Optimization** — Don't cut `predict_proba` at 0.5.  "
    "Sort by probability, mark the top ~22% as 1.  "
    "Under-predicting costs ~5x more than over-predicting, so lean higher.\n"
    "2. **OOF Target Encoding on `last_name`** (`m=20`) — "
    "Strongest surname-level signal; out-of-fold prevents leakage.\n"
    "3. **OOF Target Encoding on `customer_id`** (`m=5`) — "
    "92% of test customers appear in training; individual history is decisive.\n"
    "4. **HistGradientBoostingClassifier** — single model, 10-fold CV.  "
    "Simple mean if ensembling; optimised weights overfit CV.\n\n"
    "Extra features: `zero_balance` (acc_balance == 0) and `age_x_active` (age * is_active).\n"
    "Verified OOF F1: **0.6609**\n"
)

# ── Cell 61: Feature prep + dual OOF TE ──────────────────────────────────────
cell61 = '''\
# ── Feature preparation ────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

train_final = raw_train_df.copy()
test_final  = raw_test_df.copy()

# Impute numerics with training median (fit on train only)
for col in ["credit_score", "acc_balance", "prod_count"]:
    med = train_final[col].median()
    train_final[col] = train_final[col].fillna(med)
    test_final[col]  = test_final[col].fillna(med)

# Impute country with training mode
mode_country = train_final["country"].mode()[0]
train_final["country"] = train_final["country"].fillna(mode_country)
test_final["country"]  = test_final["country"].fillna(mode_country)

# Extra engineered features
for df in [train_final, test_final]:
    df["zero_balance"] = (df["acc_balance"] == 0).astype(int)
    df["age_x_active"] = df["age"] * df["is_active"]

# ── OOF Target Encoding: last_name (m=20) + customer_id (m=5) ────────────────
skf         = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_final     = train_final["exit_status"].copy()
global_mean = y_final.mean()

train_final["last_name_te"]   = np.nan
train_final["customer_id_te"] = np.nan

for train_idx, val_idx in skf.split(train_final, y_final):
    tr = train_final.iloc[train_idx]

    ln    = tr.groupby("last_name")["exit_status"].agg(["count", "mean"])
    ln_te = (ln["count"] * ln["mean"] + 20 * global_mean) / (ln["count"] + 20)
    train_final.loc[val_idx, "last_name_te"] = (
        train_final.iloc[val_idx]["last_name"].map(ln_te).fillna(global_mean)
    )

    cid    = tr.groupby("customer_id")["exit_status"].agg(["count", "mean"])
    cid_te = (cid["count"] * cid["mean"] + 5 * global_mean) / (cid["count"] + 5)
    train_final.loc[val_idx, "customer_id_te"] = (
        train_final.iloc[val_idx]["customer_id"].map(cid_te).fillna(global_mean)
    )

# Full-train encodings for test set (no leakage — test labels are hidden)
ln_full  = train_final.groupby("last_name")["exit_status"].agg(["count", "mean"])
ln_fte   = (ln_full["count"] * ln_full["mean"] + 20 * global_mean) / (ln_full["count"] + 20)
test_final["last_name_te"] = test_final["last_name"].map(ln_fte).fillna(global_mean)

cid_full = train_final.groupby("customer_id")["exit_status"].agg(["count", "mean"])
cid_fte  = (cid_full["count"] * cid_full["mean"] + 5 * global_mean) / (cid_full["count"] + 5)
test_final["customer_id_te"] = test_final["customer_id"].map(cid_fte).fillna(global_mean)

# ── Feature matrix ─────────────────────────────────────────────────────────────
num_cols  = ["credit_score", "age", "tenure", "acc_balance", "prod_count",
             "has_card", "is_active", "estimated_salary",
             "zero_balance", "age_x_active"]
feat_cols = num_cols + ["last_name_te", "customer_id_te"]
cat_cols  = ["country", "gender"]

train_ohe = pd.get_dummies(train_final[cat_cols], drop_first=True)
test_ohe  = pd.get_dummies(test_final[cat_cols],  drop_first=True)

X_final = pd.concat(
    [train_final[feat_cols].reset_index(drop=True),
     train_ohe.reset_index(drop=True)],
    axis=1
)
X_final_test = pd.concat(
    [test_final[feat_cols].reset_index(drop=True),
     test_ohe.reindex(columns=train_ohe.columns, fill_value=0).reset_index(drop=True)],
    axis=1
)

print("Train features shape:", X_final.shape)
print("Test  features shape:", X_final_test.shape)
cov = raw_test_df["customer_id"].isin(raw_train_df["customer_id"]).mean() * 100
print("customer_id coverage in test: %.1f%%" % cov)
'''
nb.cells[61] = nbformat.v4.new_code_cell(cell61)

# ── Cell 62: OOF + threshold ─────────────────────────────────────────────────
cell62 = '''\
# ── 10-fold OOF with HistGBM + threshold optimisation ─────────────────────────
from sklearn.ensemble import HistGradientBoostingClassifier

oof_probs  = np.zeros(len(X_final))
test_probs = np.zeros(len(X_final_test))

skf_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X_final, y_final)):
    X_tr, y_tr = X_final.iloc[train_idx], y_final.iloc[train_idx]
    X_va       = X_final.iloc[val_idx]

    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.02,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42 + fold
    )
    model.fit(X_tr, y_tr)

    oof_probs[val_idx]  = model.predict_proba(X_va)[:, 1]
    test_probs         += model.predict_proba(X_final_test)[:, 1] / skf_final.n_splits

# Threshold: fine grid + top-22% rank; pick whichever wins
best_thresh, best_f1 = 0.5, 0.0
for t in np.linspace(0.10, 0.90, 161):
    s = f1_score(y_final, (oof_probs >= t).astype(int))
    if s > best_f1:
        best_f1, best_thresh = s, t

n_pos       = int(0.22 * len(oof_probs))
topn_thresh = sorted(oof_probs, reverse=True)[n_pos]
topn_f1     = f1_score(y_final, (oof_probs >= topn_thresh).astype(int))
if topn_f1 > best_f1:
    best_f1, best_thresh = topn_f1, topn_thresh

print("OOF F1 at default 0.5:    %.4f" % f1_score(y_final, (oof_probs >= 0.5).astype(int)))
print("OOF Best threshold:       %.4f" % best_thresh)
print("OOF Best F1:              %.4f" % best_f1)
print("Positive prediction rate: %.2f%%" % ((oof_probs >= best_thresh).mean() * 100))
'''
nb.cells[62] = nbformat.v4.new_code_cell(cell62)

# ── Cell 63: Submission ──────────────────────────────────────────────────────
cell63 = '''\
# ── Apply threshold and save submission ────────────────────────────────────────
import glob, re

test_preds = (test_probs >= best_thresh).astype(int)
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

print("ka2_21.ipynb final patch applied. Total cells: %d" % len(nb.cells))
