"""Shared helpers: data loading, cleaning and feature engineering."""
import pandas as pd

DATA_PATH = "data/loan_approval_dataset.csv"
SEED = 42

ASSET_COLS = ["residential_assets_value", "commercial_assets_value",
              "luxury_assets_value", "bank_asset_value"]
CAT_COLS = ["education", "self_employed"]
NUM_COLS = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
            "cibil_score"] + ASSET_COLS
ENGINEERED = ["total_assets", "loan_to_income"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and clean it."""
    # sep=None lets pandas detect the delimiter (comma, semicolon, tab);
    # utf-8-sig removes an invisible BOM character if the file has one
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

    # The raw file has leading spaces in column names and text values
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    # loan_id is just an identifier, so it carries no predictive signal
    df = df.drop(columns=["loan_id"], errors="ignore")

    # Target: 1 = Approved, 0 = Rejected
    df["loan_status"] = (df["loan_status"].str.lower() == "approved").astype(int)
    return df


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create new features. Each row is computed on its own, so nothing
    can leak between the training and test sets."""
    X = X.copy()
    X["total_assets"] = X[ASSET_COLS].sum(axis=1)
    X["loan_to_income"] = X["loan_amount"] / X["income_annum"]
    return X


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
    print(df["loan_status"].value_counts())
    df = add_features(df)
    print(df[["total_assets", "loan_to_income"]].describe())