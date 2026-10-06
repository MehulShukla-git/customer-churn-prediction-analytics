"""
data_cleaning.py
----------------
Handles all data quality problems found in the raw customer churn dataset.
Each function is documented with WHY the decision was made — not just what it does.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# STEP 1: Load Raw Data
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV file into a DataFrame.
    We never modify the original file — only the in-memory copy.
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────
# STEP 2: Data Quality Report
# ─────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a summary of data quality issues.
    Returns a dict that can be used in the web UI and documentation.
    """
    report = {}

    report["shape"] = df.shape
    report["missing_values"] = df.isnull().sum().to_dict()
    report["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_customer_ids"] = int(df["customer_id"].duplicated().sum())
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Categorical value issues
    report["plan_values"] = df["Plan"].value_counts(dropna=False).to_dict()
    report["contract_values"] = df["ContractType"].value_counts(dropna=False).to_dict()
    report["autorenew_values"] = df["AutoRenew"].value_counts(dropna=False).to_dict()
    report["region_values"] = df["Region"].value_counts(dropna=False).to_dict()
    report["churn_values"] = df["Churn"].value_counts(dropna=False).to_dict()

    # Numerical range issues
    age_num = pd.to_numeric(df["Age"], errors="coerce")
    spend_num = pd.to_numeric(df["MonthlySpend"], errors="coerce")

    report["age_range"] = {"min": float(age_num.min()), "max": float(age_num.max()),
                           "negatives": int((age_num < 0).sum()),
                           "over_100": int((age_num > 100).sum())}
    report["spend_range"] = {"min": float(spend_num.min()), "max": float(spend_num.max()),
                             "negatives": int((spend_num < 0).sum())}
    report["tenure_range"] = {"min": float(df["Tenure_Months"].min()),
                              "max": float(df["Tenure_Months"].max()),
                              "negatives": int((df["Tenure_Months"] < 0).sum())}
    report["discount_range"] = {"min": float(df["DiscountPercent"].min()),
                                "max": float(df["DiscountPercent"].max())}
    report["payment_score_range"] = {"min": float(df["PaymentHistory_Score"].dropna().min()),
                                     "max": float(df["PaymentHistory_Score"].dropna().max())}

    return report


# ─────────────────────────────────────────────
# STEP 3: Remove Exact Duplicate Rows
# ─────────────────────────────────────────────

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows (same values across all columns).
    These provide no additional information and inflate counts.
    We keep the first occurrence.

    For duplicate customer_ids with different data (e.g. updated records),
    we keep the LAST record — assuming it is the most recent entry.
    """
    before = len(df)

    # Remove exact row duplicates
    df = df.drop_duplicates(keep="first")

    # For duplicate customer_ids with different row content, keep last occurrence
    df = df.sort_values("customer_id").drop_duplicates(subset="customer_id", keep="last")

    after = len(df)
    print(f"[remove_duplicate_rows] Removed {before - after} duplicate records. Rows: {after:,}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 4: Clean Numerical Columns
# ─────────────────────────────────────────────

def clean_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix invalid numerical values.
    WHY: Negative ages, impossible tenures, and extreme spend values
    would distort the model's understanding of customer behaviour.
    """

    # Age: convert mixed-type column, handle negatives and unrealistic values
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    # Age < 0 or Age > 100 is physiologically impossible — replace with NaN for imputation
    df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = np.nan

    # MonthlySpend: convert, fix negatives and extreme outliers
    df["MonthlySpend"] = pd.to_numeric(df["MonthlySpend"], errors="coerce")
    # Negative spend is impossible — replace with NaN
    df.loc[df["MonthlySpend"] < 0, "MonthlySpend"] = np.nan
    # Cap extreme outliers using 99th percentile to avoid distortion
    cap_99 = df["MonthlySpend"].quantile(0.99)
    df.loc[df["MonthlySpend"] > cap_99, "MonthlySpend"] = cap_99

    # Tenure_Months: negative tenure is impossible
    df.loc[df["Tenure_Months"] < 0, "Tenure_Months"] = np.nan
    # Tenure > 99th percentile is unrealistic
    tenure_cap = df["Tenure_Months"].quantile(0.99)
    df.loc[df["Tenure_Months"] > tenure_cap, "Tenure_Months"] = np.nan

    # DiscountPercent: must be 0-100
    df.loc[df["DiscountPercent"] < 0, "DiscountPercent"] = np.nan
    df.loc[df["DiscountPercent"] > 100, "DiscountPercent"] = 100.0  # cap at 100%

    # PaymentHistory_Score: typically 0-100
    df.loc[df["PaymentHistory_Score"] < 0, "PaymentHistory_Score"] = np.nan
    df.loc[df["PaymentHistory_Score"] > 100, "PaymentHistory_Score"] = 100.0

    # MonthlyUsage_Hours, LoginFrequency_30D, FailedPayments_90D,
    # Complaints_90D, SupportTickets_90D: cannot be negative
    for col in ["MonthlyUsage_Hours", "LoginFrequency_30D",
                "FailedPayments_90D", "Complaints_90D", "SupportTickets_90D"]:
        df.loc[df[col] < 0, col] = np.nan

    # DaysSinceLastContact: cannot be negative
    df.loc[df["DaysSinceLastContact"] < 0, "DaysSinceLastContact"] = np.nan

    print("[clean_numerical] Numerical columns cleaned.")
    return df


# ─────────────────────────────────────────────
# STEP 5: Normalize Categorical Columns
# ─────────────────────────────────────────────

def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize inconsistent categorical values.
    WHY: Models treat 'basic' and 'Basic' as different categories,
    which would create unnecessary dummy variables and hurt performance.
    """

    # --- Plan ---
    plan_map = {
        "basic": "Basic", "BASIC": "Basic",
        "Std": "Standard", "std": "Standard", "STANDARD": "Standard",
        "premium": "Premium", "PREMIUM": "Premium",
        "enterprise": "Enterprise", "ENTERPRISE": "Enterprise",
        "Unknown": "Unknown",
    }
    df["Plan"] = df["Plan"].str.strip()
    df["Plan"] = df["Plan"].replace(plan_map)
    df["Plan"] = df["Plan"].fillna("Unknown")
    valid_plans = ["Basic", "Standard", "Premium", "Enterprise", "Unknown"]
    df.loc[~df["Plan"].isin(valid_plans), "Plan"] = "Unknown"

    # --- ContractType ---
    contract_map = {
        "monthly": "Monthly", "MONTHLY": "Monthly", "M": "Monthly",
        "annual": "Annual", "ANNUAL": "Annual",
        "quarterly": "Quarterly", "QUARTERLY": "Quarterly",
        "yearly": "Annual", "Yearly": "Annual", "YEARLY": "Annual",
    }
    df["ContractType"] = df["ContractType"].str.strip()
    df["ContractType"] = df["ContractType"].replace(contract_map)
    df["ContractType"] = df["ContractType"].fillna("Unknown")
    valid_contracts = ["Monthly", "Annual", "Quarterly", "Unknown"]
    df.loc[~df["ContractType"].isin(valid_contracts), "ContractType"] = "Unknown"

    # --- AutoRenew ---
    autorenew_map = {
        "yes": "Yes", "YES": "Yes", "y": "Yes", "Y": "Yes",
        "no": "No", "NO": "No", "n": "No", "N": "No",
    }
    df["AutoRenew"] = df["AutoRenew"].str.strip()
    df["AutoRenew"] = df["AutoRenew"].replace(autorenew_map)
    df["AutoRenew"] = df["AutoRenew"].fillna("Unknown")
    valid_autorenew = ["Yes", "No", "Unknown"]
    df.loc[~df["AutoRenew"].isin(valid_autorenew), "AutoRenew"] = "Unknown"

    # --- Region ---
    region_map = {
        "north": "North", "NORTH": "North",
        "south": "South", "SOUTH": "South",
        "east": "East", "EAST": "East",
        "west": "West", "WEST": "West",
    }
    df["Region"] = df["Region"].str.strip()
    df["Region"] = df["Region"].replace(region_map)
    df["Region"] = df["Region"].fillna("Unknown")
    valid_regions = ["North", "South", "East", "West", "Unknown"]
    df.loc[~df["Region"].isin(valid_regions), "Region"] = "Unknown"

    print("[normalize_categoricals] Categorical columns normalized.")
    return df


# ─────────────────────────────────────────────
# STEP 6: Clean Churn Target Column
# ─────────────────────────────────────────────

def clean_churn_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map all equivalent churn representations to binary 0/1.
    WHY: The target must be numeric (0/1) for sklearn classification.
    Rows with ambiguous or missing targets are dropped because we cannot
    reliably assign a class — including them would introduce noise.

    1 = Churned
    0 = Active (Not Churned)
    """
    churn_map = {
        # Positive (churned)
        "Yes": 1, "YES": 1, "yes": 1, "Y": 1, "y": 1,
        "1": 1, "Churned": 1, "churned": 1,
        # Negative (active)
        "No": 0, "NO": 0, "no": 0, "N": 0, "n": 0,
        "0": 0, "Active": 0, "active": 0,
    }

    df["Churn"] = df["Churn"].astype(str).str.strip()
    df["Churn"] = df["Churn"].map(churn_map)

    # Drop rows where Churn is NaN (was 'Unknown', missing, or unmappable)
    before = len(df)
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    after = len(df)

    print(f"[clean_churn_target] Dropped {before - after} rows with ambiguous/missing Churn.")
    print(f"[clean_churn_target] Churn distribution:\n{df['Churn'].value_counts()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 7: Clean Dates
# ─────────────────────────────────────────────

def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse SignupDate and extract useful features.
    WHY: Raw date strings are not usable by ML models;
    year and month can capture seasonality or cohort effects.
    """
    df["SignupDate"] = pd.to_datetime(df["SignupDate"], errors="coerce")
    median_year = df["SignupDate"].dt.year.median()
    median_month = df["SignupDate"].dt.month.median()
    df["SignupYear"] = df["SignupDate"].dt.year.fillna(median_year).astype(int)
    df["SignupMonth"] = df["SignupDate"].dt.month.fillna(median_month).astype(int)
    df = df.drop(columns=["SignupDate"])
    print("[clean_dates] SignupDate parsed and converted to year/month features.")
    return df


# ─────────────────────────────────────────────
# STEP 8: Impute Missing Values
# ─────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining NaN values after cleaning.

    For numerical columns: median imputation.
    WHY: Median is robust to outliers. We already capped extremes
    so mean would also work, but median is the safer default.

    For categorical columns: already filled with 'Unknown' in normalize step.
    """
    numerical_cols = [
        "Age", "Tenure_Months", "MonthlySpend", "SupportTickets_90D",
        "MonthlyUsage_Hours", "PaymentHistory_Score", "LoginFrequency_30D",
        "FailedPayments_90D", "DiscountPercent", "DaysSinceLastContact",
        "Complaints_90D", "SignupYear", "SignupMonth",
    ]

    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[impute_missing] {col}: filled NaN with median={median_val:.2f}")

    return df


# ─────────────────────────────────────────────
# STEP 9: Drop Non-Predictive Columns
# ─────────────────────────────────────────────

def drop_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop customer_id — it is an identifier, not a feature.
    WHY: Using IDs as features would cause the model to memorize training
    customers instead of learning patterns (overfitting / leakage).
    """
    df = df.drop(columns=["customer_id"], errors="ignore")
    print("[drop_non_predictive] Dropped customer_id.")
    return df


# ─────────────────────────────────────────────
# MASTER CLEAN FUNCTION
# ─────────────────────────────────────────────

def clean_data(filepath: str) -> pd.DataFrame:
    """
    Run all cleaning steps in order and return a clean DataFrame.
    The original file is never modified.
    """
    df = load_data(filepath)

    print("\n--- Starting Data Cleaning Pipeline ---")
    df = remove_duplicate_rows(df)
    df = clean_numerical(df)
    df = normalize_categoricals(df)
    df = clean_churn_target(df)
    df = clean_dates(df)
    df = impute_missing(df)
    df = drop_non_predictive(df)
    print(f"\n--- Cleaning Complete. Final shape: {df.shape} ---\n")
    return df


if __name__ == "__main__":
    df_clean = clean_data("data/customer_churn_messy_raw.csv")
    df_clean.to_csv("data/customer_churn_clean.csv", index=False)
    print("Clean dataset saved to data/customer_churn_clean.csv")
