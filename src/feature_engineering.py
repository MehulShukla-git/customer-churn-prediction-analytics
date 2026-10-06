"""
feature_engineering.py
-----------------------
Creates new features from the cleaned dataset to improve model performance.
Every feature has a clear business reason — we do not add features blindly.
"""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the cleaned DataFrame.
    These features capture business logic that raw columns may not express clearly.
    """
    df = df.copy()

    # ── Feature 1: Tenure Group ──────────────────────────────────────────────
    # WHY: Churn behaviour often differs by customer lifecycle stage.
    # New customers may leave early; long-term ones are more loyal.
    def tenure_group(months):
        if months <= 6:
            return "New"          # 0-6 months
        elif months <= 24:
            return "Growing"      # 7-24 months
        elif months <= 60:
            return "Established"  # 25-60 months
        else:
            return "Loyal"        # 60+ months

    df["TenureGroup"] = df["Tenure_Months"].apply(tenure_group)

    # ── Feature 2: High Value Customer ───────────────────────────────────────
    # WHY: High-spending customers are business-critical to retain.
    # A binary flag makes this explicit and easy for the model to use.
    spend_75th = df["MonthlySpend"].quantile(0.75)
    df["HighValueCustomer"] = (df["MonthlySpend"] >= spend_75th).astype(int)

    # ── Feature 3: Low Usage Indicator ───────────────────────────────────────
    # WHY: Customers who barely use the service are at higher churn risk.
    # Low engagement is a strong churn signal.
    usage_25th = df["MonthlyUsage_Hours"].quantile(0.25)
    df["LowUsage"] = (df["MonthlyUsage_Hours"] <= usage_25th).astype(int)

    # ── Feature 4: Payment Risk Indicator ────────────────────────────────────
    # WHY: Failed payments signal financial or satisfaction issues
    # that often precede cancellation.
    df["PaymentRisk"] = (df["FailedPayments_90D"] >= 2).astype(int)

    # ── Feature 5: Support Interaction Level ─────────────────────────────────
    # WHY: Very high support ticket volume may indicate a frustrated customer
    # who is about to churn. We bin into levels for interpretability.
    def support_level(tickets):
        if tickets == 0:
            return "None"
        elif tickets <= 2:
            return "Low"
        elif tickets <= 5:
            return "Medium"
        else:
            return "High"

    df["SupportLevel"] = df["SupportTickets_90D"].apply(support_level)

    # ── Feature 6: Has Complaint ──────────────────────────────────────────────
    # WHY: Any complaint is a red flag. A binary indicator is simpler
    # for the model than the raw count in many cases.
    df["HasComplaint"] = (df["Complaints_90D"] >= 1).astype(int)

    # ── Feature 7: Days Since Contact Group ──────────────────────────────────
    # WHY: Customers not contacted for a long time may feel neglected
    # and are at higher churn risk.
    def contact_group(days):
        if days <= 30:
            return "Recent"
        elif days <= 90:
            return "Moderate"
        else:
            return "Neglected"

    df["ContactGroup"] = df["DaysSinceLastContact"].apply(contact_group)

    # ── Feature 8: Engagement Score ──────────────────────────────────────────
    # WHY: Combines login frequency and usage hours into a single score
    # representing overall product engagement.
    df["EngagementScore"] = (
        df["LoginFrequency_30D"] * 0.5 +
        df["MonthlyUsage_Hours"] * 0.5
    )

    # ── Feature 9: Discount Effectiveness Flag ────────────────────────────────
    # WHY: Some customers get discounts but still churn — this flag helps
    # identify whether discounts are being used at all.
    df["HasDiscount"] = (df["DiscountPercent"] > 0).astype(int)

    print("[create_features] Engineered features added:")
    new_features = [
        "TenureGroup", "HighValueCustomer", "LowUsage", "PaymentRisk",
        "SupportLevel", "HasComplaint", "ContactGroup", "EngagementScore", "HasDiscount"
    ]
    for f in new_features:
        print(f"  - {f}")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_clean.csv")
    df = create_features(df)
    df.to_csv("data/customer_churn_features.csv", index=False)
    print(f"\nDataset with features saved. Shape: {df.shape}")
