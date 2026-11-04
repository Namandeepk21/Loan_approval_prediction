"""Shared helpers: data loading and cleaning."""
import pandas as pd

DATA_PATH = "data/loan_approval_dataset.csv"
SEED = 42

ASSET_COLS = ["residential_assets_value", "commercial_assets_value",
              "luxury_assets_value", "bank_asset_value"]
CAT_COLS = ["education", "self_employed"]
NUM_COLS = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
            "cibil_score"] + ASSET_COLS


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and clean it."""
    df = pd.read_csv(path)

    # The raw file has leading spaces in column names and text values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    # loan_id is just an identifier, so it carries no predictive signal
    df = df.drop(columns=["loan_id"], errors="ignore")

    # Target: 1 = Approved, 0 = Rejected
    df["loan_status"] = (df["loan_status"] == "Approved").astype(int)
    return df