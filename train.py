"""Model development and evaluation. Run: python train.py"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from utils import load_data, add_features, SEED, NUM_COLS, CAT_COLS, ENGINEERED

df = load_data()
X = df.drop(columns="loan_status")
y = df["loan_status"]

# The test set is touched ONLY for final reporting.
# Tuning uses cross-validation inside the training set.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_pipe(clf):
    """Scaler and encoder live inside the pipeline, so they are fitted on
    training folds only. That is what prevents data leakage."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS + ENGINEERED),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", clf),
    ])


print("Train:", X_tr.shape, "Test:", X_te.shape)
def test_metrics(pipe):
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }


# Each model comes with the hyperparameters we will tune
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [None, 5, 10],
         "model__min_samples_leaf": [1, 3]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}

# Baseline: always predict the majority class
baseline = make_pipe(DummyClassifier(strategy="most_frequent")).fit(X_tr, y_tr)
rows = [{"model": "Baseline (majority class)", "cv_f1": np.nan, **test_metrics(baseline)}]
fitted = {"Baseline (majority class)": baseline}
print("Baseline:", {k: round(v, 3) for k, v in test_metrics(baseline).items()})
for name, (clf, grid) in models.items():
    print(f"Tuning {name} ...")
    gs = GridSearchCV(make_pipe(clf), grid, scoring="f1", cv=cv, n_jobs=1)
    gs.fit(X_tr, y_tr)
    fitted[name] = gs.best_estimator_
    rows.append({"model": name, "cv_f1": gs.best_score_,
                 "best_params": str(gs.best_params_),
                 **test_metrics(gs.best_estimator_)})

results = pd.DataFrame(rows).round(4)
results.to_csv("outputs/model_results.csv", index=False)
print("\n", results.drop(columns="best_params").to_string(index=False))

# The final model is chosen by CROSS-VALIDATED F1, never by test score
best_name = (results.dropna(subset=["cv_f1"])
             .sort_values("cv_f1", ascending=False).iloc[0]["model"])
print("\nSelected model:", best_name)

joblib.dump(fitted[best_name], "models/best_model.joblib")
with open("outputs/best_model.json", "w") as f:
    json.dump({"best_model": best_name}, f)
# ---------- Evaluation plots ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

best = fitted[best_name]

fig, ax = plt.subplots(figsize=(6, 5))
for name, model in fitted.items():
    RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, name=name)
ax.set_title("ROC curves (test set)")
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()

ConfusionMatrixDisplay.from_estimator(
    best, X_te, y_te, display_labels=["Rejected", "Approved"], cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()
# ---------- Interpretation: permutation importance ----------
from sklearn.inspection import permutation_importance

imp = permutation_importance(best, X_te, y_te, scoring="roc_auc",
                             n_repeats=10, random_state=SEED)
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
imp_s.plot.barh(figsize=(7, 5), title="Permutation importance (drop in ROC-AUC)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()
print("\nTop features:\n", imp_s.sort_values(ascending=False).head(5).round(4).to_string())
# ---------- Are the top two models significantly different? ----------
from scipy import stats
from sklearn.model_selection import cross_val_score

rf_auc = cross_val_score(fitted["Random Forest"], X_tr, y_tr, cv=cv, scoring="roc_auc")
gb_auc = cross_val_score(fitted["Gradient Boosting"], X_tr, y_tr, cv=cv, scoring="roc_auc")
print("\nCV ROC-AUC  RF:", rf_auc.round(4), " GB:", gb_auc.round(4))
if np.allclose(rf_auc, gb_auc):
    print("Paired t-test: the two models have identical fold scores")
else:
    t, p = stats.ttest_rel(rf_auc, gb_auc)
    verdict = "significant difference" if p < 0.05 else "no significant difference"
    print(f"Paired t-test RF vs GB: t={t:.3f}, p={p:.4f} -> {verdict}")