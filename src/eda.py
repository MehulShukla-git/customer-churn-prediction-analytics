"""
eda.py
------
Performs Exploratory Data Analysis (EDA) on the cleaned dataset.
Each plot answers a specific business question about customer churn.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Style Configuration ────────────────────────────────────────────────────
ACCENT = "#2563EB"   # Professional blue accent
CHURN_COLORS = ["#22C55E", "#EF4444"]  # Green = Active, Red = Churned
OUTPUT_DIR = "eda_plots"


def _save(fig, name: str):
    """Save figure to the eda_plots directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_churn_distribution(df: pd.DataFrame) -> str:
    """
    Business Q: What percentage of customers churned?
    WHY: Understand class imbalance before modelling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["Churn"].value_counts()
    labels = ["Active", "Churned"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    axes[0].bar(labels, values, color=CHURN_COLORS, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Churn Count", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(values):
        axes[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(values, labels=labels, colors=CHURN_COLORS,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Churn Distribution", fontweight="bold")

    fig.suptitle("Customer Churn Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "01_churn_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> str:
    """
    Business Q: Are month-to-month customers more likely to churn?
    WHY: Contract length is a common indicator of commitment.
    """
    ct = df.groupby("ContractType")["Churn"].mean().reset_index()
    ct["ChurnRate"] = ct["Churn"] * 100
    ct = ct.sort_values("ChurnRate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(ct["ContractType"], ct["ChurnRate"],
                  color=ACCENT, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_churn_by_contract.png")


def plot_churn_by_plan(df: pd.DataFrame) -> str:
    """
    Business Q: Does the subscription plan affect churn probability?
    """
    plan_churn = df.groupby("Plan")["Churn"].mean().reset_index()
    plan_churn["ChurnRate"] = plan_churn["Churn"] * 100
    plan_churn = plan_churn.sort_values("ChurnRate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [ACCENT if i == 0 else "#93C5FD" for i in range(len(plan_churn))]
    bars = ax.bar(plan_churn["Plan"], plan_churn["ChurnRate"],
                  color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Plan", fontsize=13, fontweight="bold")
    ax.set_xlabel("Plan")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_churn_by_plan.png")


def plot_churn_by_autorenew(df: pd.DataFrame) -> str:
    """
    Business Q: Are customers without auto-renewal more likely to churn?
    WHY: Customers who opt out of auto-renewal often don't intend to continue.
    """
    ar_churn = df.groupby("AutoRenew")["Churn"].mean().reset_index()
    ar_churn["ChurnRate"] = ar_churn["Churn"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = {"Yes": "#22C55E", "No": "#EF4444", "Unknown": "#94A3B8"}
    colors = [bar_colors.get(v, "#94A3B8") for v in ar_churn["AutoRenew"]]
    bars = ax.bar(ar_churn["AutoRenew"], ar_churn["ChurnRate"],
                  color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Auto-Renewal Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Auto-Renewal")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_churn_by_autorenew.png")


def plot_churn_by_region(df: pd.DataFrame) -> str:
    """
    Business Q: Are certain geographic regions experiencing higher churn?
    """
    region_churn = df.groupby("Region")["Churn"].mean().reset_index()
    region_churn["ChurnRate"] = region_churn["Churn"] * 100
    region_churn = region_churn.sort_values("ChurnRate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(region_churn["Region"], region_churn["ChurnRate"],
           color=ACCENT, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Region", fontsize=13, fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return _save(fig, "05_churn_by_region.png")


def plot_monthly_spend_dist(df: pd.DataFrame) -> str:
    """
    Business Q: How is monthly spend distributed, and does it differ by churn?
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for churn_val, label, color in [(0, "Active", "#22C55E"), (1, "Churned", "#EF4444")]:
        subset = df[df["Churn"] == churn_val]["MonthlySpend"]
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_title("Monthly Spend Distribution by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Spend")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "06_monthly_spend_dist.png")


def plot_tenure_distribution(df: pd.DataFrame) -> str:
    """
    Business Q: Do newer customers churn more than long-term ones?
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for churn_val, label, color in [(0, "Active", "#22C55E"), (1, "Churned", "#EF4444")]:
        subset = df[df["Churn"] == churn_val]["Tenure_Months"]
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "07_tenure_distribution.png")


def plot_usage_vs_churn(df: pd.DataFrame) -> str:
    """
    Business Q: Do low-usage customers churn more often?
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    churn_labels = {0: "Active", 1: "Churned"}
    data_to_plot = [
        df[df["Churn"] == 0]["MonthlyUsage_Hours"].dropna(),
        df[df["Churn"] == 1]["MonthlyUsage_Hours"].dropna()
    ]
    bp = ax.boxplot(data_to_plot, patch_artist=True,
                    tick_labels=["Active", "Churned"])
    for patch, color in zip(bp["boxes"], CHURN_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Monthly Usage Hours vs Churn", fontsize=13, fontweight="bold")
    ax.set_ylabel("Monthly Usage Hours")
    fig.tight_layout()
    return _save(fig, "08_usage_vs_churn.png")


def plot_failed_payments_vs_churn(df: pd.DataFrame) -> str:
    """
    Business Q: Do customers with more failed payments churn more?
    """
    fp_churn = df.groupby("FailedPayments_90D")["Churn"].mean().reset_index()
    fp_churn = fp_churn[fp_churn["FailedPayments_90D"] <= 10]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fp_churn["FailedPayments_90D"], fp_churn["Churn"] * 100,
           color=ACCENT, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Failed Payments (Last 90 Days)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Failed Payments")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    return _save(fig, "09_failed_payments_vs_churn.png")


def plot_support_tickets_vs_churn(df: pd.DataFrame) -> str:
    """
    Business Q: Does high support ticket volume predict churn?
    """
    st_churn = df.groupby("SupportTickets_90D")["Churn"].mean().reset_index()
    st_churn = st_churn[st_churn["SupportTickets_90D"] <= 15]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(st_churn["SupportTickets_90D"], st_churn["Churn"] * 100,
           color="#7C3AED", edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Support Tickets (Last 90 Days)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Support Tickets")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    return _save(fig, "10_support_tickets_vs_churn.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Business Q: Which numerical features correlate most with churn?
    WHY: Helps identify the most informative raw features before modelling.
    """
    numerical_cols = [
        "Age", "Tenure_Months", "MonthlySpend", "SupportTickets_90D",
        "MonthlyUsage_Hours", "PaymentHistory_Score", "LoginFrequency_30D",
        "FailedPayments_90D", "DiscountPercent", "DaysSinceLastContact",
        "Complaints_90D", "Churn"
    ]
    corr_df = df[[c for c in numerical_cols if c in df.columns]].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_correlation_heatmap.png")


def plot_churn_by_tenure_group(df: pd.DataFrame) -> str:
    """
    Business Q: Which customer lifecycle stage has the highest churn rate?
    """
    if "TenureGroup" not in df.columns:
        from src.feature_engineering import create_features
        df = create_features(df)

    order = ["New", "Growing", "Established", "Loyal"]
    tg = df.groupby("TenureGroup")["Churn"].mean().reset_index()
    tg["ChurnRate"] = tg["Churn"] * 100
    tg["TenureGroup"] = pd.Categorical(tg["TenureGroup"], categories=order, ordered=True)
    tg = tg.sort_values("TenureGroup")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tg["TenureGroup"].astype(str), tg["ChurnRate"],
           color=ACCENT, edgecolor="white", linewidth=1.5)
    ax.set_title("Churn Rate by Tenure Group", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure Group")
    ax.set_ylabel("Churn Rate (%)")
    fig.tight_layout()
    return _save(fig, "12_churn_by_tenure_group.png")


def run_eda(df: pd.DataFrame):
    """
    Execute all EDA plots and print a summary.
    """
    print("\n=== Running EDA ===")

    print("\n1. Dataset Shape:", df.shape)
    print("2. Data Types:\n", df.dtypes)
    print("3. Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("4. Churn Distribution:\n", df["Churn"].value_counts())
    print("5. Numerical Summary:\n", df.describe().round(2))

    print("\n6. Generating plots...")
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_plan(df)
    plot_churn_by_autorenew(df)
    plot_churn_by_region(df)
    plot_monthly_spend_dist(df)
    plot_tenure_distribution(df)
    plot_usage_vs_churn(df)
    plot_failed_payments_vs_churn(df)
    plot_support_tickets_vs_churn(df)
    plot_correlation_heatmap(df)
    plot_churn_by_tenure_group(df)

    print("\n=== EDA Complete ===")


if __name__ == "__main__":
    df = pd.read_csv("data/customer_churn_clean.csv")
    run_eda(df)
