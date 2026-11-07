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