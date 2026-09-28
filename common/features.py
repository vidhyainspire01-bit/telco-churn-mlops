"""Shared by training, evaluation, batch scoring and serving - one source of truth for features."""
import hashlib

import pandas as pd

TARGET = "label"
ID_COL = "customerID"

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL + NUMERIC

TEST_FRACTION = 0.2


def is_test(customer_id: str, test_fraction: float = TEST_FRACTION) -> bool:
    """Deterministic split by hashing the customer ID.

    The same customer always lands in the same split, on every run, so a new
    challenger and the current champion are always compared on the same holdout.
    (A random split with a new seed each run would make the promotion gate unfair.)
    """
    bucket = int(hashlib.md5(customer_id.encode()).hexdigest(), 16) % 100
    return bucket < test_fraction * 100


def split(df: pd.DataFrame):
    mask = df[ID_COL].map(is_test)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)