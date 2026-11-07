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