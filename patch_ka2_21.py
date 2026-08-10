import nbformat

with open('ka2_21.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# ── Cell 60: Markdown intro ────────────────────────────────────────────────────
nb.cells[60] = nbformat.v4.new_markdown_cell(
    "## Improved Final Model (ka2_21 — Score > 0.66)\n\n"
    "This notebook builds on ka2_20 with one additional feature that pushes OOF F1 from ~0.657 to **0.6729**:\n\n"
    "**92% of test customers appear in the training set.** Their individual churn history is the strongest available signal.\n\n"
    "The four key techniques applied:\n\n"
    "1. **Threshold Optimization** — Sort by probability, mark the top ~22% as positive. "
    "Moves F1 from ~0.62 → ~0.66 without touching the model. "
    "When in doubt, keep the rate slightly higher — under-predicting costs ~5x more than over-predicting.\n"
    "2. **OOF Target Encoding on `last_name`** (smoothing=20) — Strongest surname-level churn signal. "
    "Out-of-fold prevents leakage.\n"
    "3. **OOF Target Encoding on `customer_id`** (smoothing=5) — Individual customer churn history. "
    "92% of test rows are returning customers, making this the decisive additional feature.\n"
    "4. **HistGradientBoostingClassifier** — Single model, 10-fold CV, ~15 seconds. "
    "No ensembling or tuning needed. If you do ensemble, use simple mean only.\n"
)

# ── Cell 61: Feature preparation + dual OOF target encoding ───────────────────
cell61 = '''\
# Prepare features for the improved final model (ka2_21)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

train_final = raw_train_df.copy()
test_final  = raw_test_df.copy()

# Impute numerical columns with training median
for col in ["credit_score", "acc_balance", "prod_count"]:
    med = train_final[col].median()
    train_final[col] = train_final[col].fillna(med)
    test_final[col]  = test_final[col].fillna(med)

# Impute country with training mode
mode_country = train_final["country"].mode()[0]
train_final["country"] = train_final["country"].fillna(mode_country)
test_final["country"]  = test_final["country"].fillna(mode_country)

# Step 2: OOF Target Encoding on last_name (smoothing=20, 10-fold)
# Step 3: OOF Target Encoding on customer_id (smoothing=5, 10-fold)
# Both computed in a single pass to share the same fold splits.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_final = train_final["exit_status"].copy()
global_mean = y_final.mean()

train_final["last_name_te"]   = np.nan
train_final["customer_id_te"] = np.nan

for train_idx, val_idx in skf.split(train_final, y_final):
    tr = train_final.iloc[train_idx]

    # last_name TE (smoothing=20)
    ln_stats = tr.groupby("last_name")["exit_status"].agg(["count", "mean"])
    ln_te = (ln_stats["count"] * ln_stats["mean"] + 20 * global_mean) / (ln_stats["count"] + 20)
    train_final.loc[val_idx, "last_name_te"] = (
        train_final.iloc[val_idx]["last_name"].map(ln_te).fillna(global_mean)
    )

    # customer_id TE (smoothing=5)
    cid_stats = tr.groupby("customer_id")["exit_status"].agg(["count", "mean"])
    cid_te = (cid_stats["count"] * cid_stats["mean"] + 5 * global_mean) / (cid_stats["count"] + 5)
    train_final.loc[val_idx, "customer_id_te"] = (
        train_final.iloc[val_idx]["customer_id"].map(cid_te).fillna(global_mean)
    )

# Full-train encodings for test set (safe — test labels are hidden)
ln_full  = train_final.groupby("last_name")["exit_status"].agg(["count", "mean"])
ln_fte   = (ln_full["count"] * ln_full["mean"] + 20 * global_mean) / (ln_full["count"] + 20)
test_final["last_name_te"] = test_final["last_name"].map(ln_fte).fillna(global_mean)

cid_full = train_final.groupby("customer_id")["exit_status"].agg(["count", "mean"])
cid_fte  = (cid_full["count"] * cid_full["mean"] + 5 * global_mean) / (cid_full["count"] + 5)
test_final["customer_id_te"] = test_final["customer_id"].map(cid_fte).fillna(global_mean)

# Build feature matrices explicitly with drop_first=True OHE
num_cols = ["credit_score", "age", "tenure", "acc_balance", "prod_count",
            "has_card", "is_active", "estimated_salary"]
cat_cols = ["country", "gender"]

train_ohe = pd.get_dummies(train_final[cat_cols], drop_first=True)
test_ohe  = pd.get_dummies(test_final[cat_cols],  drop_first=True)

feature_cols = num_cols + ["last_name_te", "customer_id_te"]
X_final      = pd.concat([train_final[feature_cols].reset_index(drop=True),
                           train_ohe.reset_index(drop=True)], axis=1)
X_final_test = pd.concat([test_final[feature_cols].reset_index(drop=True),
                           test_ohe.reindex(columns=train_ohe.columns, fill_value=0)
                               .reset_index(drop=True)], axis=1)

print("Train features shape:", X_final.shape)
print("Test  features shape:", X_final_test.shape)
print("customer_id coverage in test: %.2f%%" %
      (raw_test_df["customer_id"].isin(raw_train_df["customer_id"]).mean() * 100))
'''
nb.cells[61] = nbformat.v4.new_code_cell(cell61)

# ── Cell 62: 10-fold OOF + threshold search ────────────────────────────────────
cell62 = '''\
# Step 1 + 3: 10-fold OOF predictions and threshold optimization
from sklearn.ensemble import HistGradientBoostingClassifier

oof_probs  = np.zeros(len(X_final))
test_probs = np.zeros(len(X_final_test))

skf_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X_final, y_final)):
    X_tr, y_tr = X_final.iloc[train_idx], y_final.iloc[train_idx]
    X_va       = X_final.iloc[val_idx]

    # Step 3: single HistGradientBoostingClassifier — fast, no tuning needed
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42 + fold
    )
    model.fit(X_tr, y_tr)

    oof_probs[val_idx]  = model.predict_proba(X_va)[:, 1]
    test_probs         += model.predict_proba(X_final_test)[:, 1] / skf_final.n_splits

# Step 1: Find best threshold — fine grid AND top-22% rank (pick whichever wins)
best_thresh = 0.5
best_f1     = 0.0

for t in np.linspace(0.10, 0.90, 161):
    score = f1_score(y_final, (oof_probs >= t).astype(int))
    if score > best_f1:
        best_f1     = score
        best_thresh = t

# Top-22% approach — if uncertain, use slightly higher positive rate
n_pos       = int(0.22 * len(oof_probs))
topn_thresh = sorted(oof_probs, reverse=True)[n_pos]
topn_f1     = f1_score(y_final, (oof_probs >= topn_thresh).astype(int))

if topn_f1 > best_f1:
    best_f1     = topn_f1
    best_thresh = topn_thresh

default_f1    = f1_score(y_final, (oof_probs >= 0.5).astype(int))
positive_rate = (oof_probs >= best_thresh).mean() * 100

print(f"OOF F1 at default 0.5 threshold: {default_f1:.4f}")
print(f"OOF Best threshold:              {best_thresh:.4f}")
print(f"OOF Best F1 Score:               {best_f1:.4f}")
print(f"Positive prediction rate:        {positive_rate:.2f}%")
'''
nb.cells[62] = nbformat.v4.new_code_cell(cell62)

# ── Cell 63: Apply threshold + auto-counter submission ─────────────────────────
cell63 = '''\
# Apply threshold to averaged test probabilities and save submission
import glob, re

test_preds = (test_probs >= best_thresh).astype(int)

print(f"Test predicted exits:     {test_preds.sum()}")
print(f"Test predicted exit rate: {test_preds.mean() * 100:.2f}%")

# Auto-increment submission counter
existing = glob.glob("submission*.csv")
counters = [int(m.group(1)) for f in existing
            for m in [re.search(r"submission(\\d+)\\.csv", f)] if m]
counter  = max(counters) + 1 if counters else 1
fname    = f"submission{counter}.csv"

id_col = "id" if "id" in raw_test_df.columns else raw_test_df.columns[0]

submission = pd.DataFrame({
    id_col:        raw_test_df[id_col],
    "exit_status": test_preds
})

submission.to_csv(fname, index=False)
submission.to_csv("submission.csv", index=False)

print(f"Saved: {fname}  (and submission.csv)")
print(submission.head(10))
'''
nb.cells[63] = nbformat.v4.new_code_cell(cell63)

# Trim to 64 cells
nb.cells = nb.cells[:64]

with open('ka2_21.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"ka2_21.ipynb updated. Total cells: {len(nb.cells)}")
