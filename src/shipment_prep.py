# part_a_cleaning/clean.py
import numpy as np
import pandas as pd

MEDIAN_COLS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_parcels(df: pd.DataFrame, medians: dict | None = None):
    """Clean raw parcel data.

    Returns (clean_df, medians). If `medians` is None, medians are computed
    from the non-missing values left after steps 1-6; otherwise the dict
    passed in is used (e.g. to clean new parcels with training medians).
    """
    df = df.copy()

    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: trim + title case -> Express / Standard / Economy
    df["service_type"] = df["service_type"].astype("string").str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: strip rupee sign and commas -> numeric; drop text column
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # 6. weight > 100 kg is a scanner-fault sentinel -> missing
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce")
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. median imputation (medians come from steps 1-6 output only)
    if medians is None:
        medians = {c: float(df[c].median()) for c in MEDIAN_COLS}
    for c in MEDIAN_COLS:
        df[c] = df[c].fillna(medians[c])

    return df, medians
# usage
raw = pd.read_csv("data/parcels_raw.csv")
clean, medians = clean_parcels(raw)           # training: computes medians
# later, for new parcels (Part E scoring script):
new_clean, _ = clean_parcels(new_raw, medians)  # reuses training medians
