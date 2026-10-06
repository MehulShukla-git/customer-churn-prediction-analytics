"""
run_training.py
---------------
Master script: cleans data, runs EDA, trains models, evaluates,
selects the best, and saves everything.

Run this ONCE before starting the Flask app:
    python run_training.py
"""

import os
import sys
import json
import joblib
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split

from src.data_cleaning import clean_data, data_quality_report, load_data
from src.feature_engineering import create_features
from src.eda import run_eda
from src.train_model import prepare_data, build_preprocessor, train_models, save_best_model
from src.evaluate_model import evaluate_model, select_best_model, plot_confusion_matrices, build_comparison_table
from src.explain_model import explain_model


RAW_DATA_PATH = "data/customer_churn_messy_raw.csv"
CLEAN_DATA_PATH = "data/customer_churn_clean.csv"
FEATURES_DATA_PATH = "data/customer_churn_features.csv"


def main():
    print("=" * 60)
    print("  Customer Churn Prediction — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Clean data ───────────────────────────────────────
    print("\n[STEP 1] Generating Data Quality Report and Cleaning data...")
    df_raw = load_data(RAW_DATA_PATH)
    dq_report = data_quality_report(df_raw)
    with open("models/data_quality.json", "w") as f:
        json.dump(dq_report, f, indent=2)

    df_clean = clean_data(RAW_DATA_PATH)
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Clean dataset saved: {CLEAN_DATA_PATH} ({len(df_clean):,} rows)")

    # ── Step 2: Feature engineering ─────────────────────────────
    print("\n[STEP 2] Engineering features...")
    df_features = create_features(df_clean)
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Feature dataset saved: {FEATURES_DATA_PATH}")

    # ── Step 3: EDA ──────────────────────────────────────────────
    print("\n[STEP 3] Running EDA...")
    run_eda(df_clean)

    # ── Step 4: Prepare data for modeling ───────────────────────
    print("\n[STEP 4] Preparing modeling data...")
    X, y, feature_cols = prepare_data(RAW_DATA_PATH)

    # Stratified split preserves churn ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Step 5: Build preprocessor ──────────────────────────────
    preprocessor = build_preprocessor()

    # ── Step 6: Train models ─────────────────────────────────────
    print("\n[STEP 5] Training models...")
    lr_model, rf_model = train_models(X_train, y_train, preprocessor)

    # ── Step 7: Evaluate models ──────────────────────────────────
    print("\n[STEP 6] Evaluating models...")
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # ── Step 8: Comparison table ─────────────────────────────────
    all_results = [lr_results, rf_results]
    comparison_df = build_comparison_table(all_results)
    comparison_df.to_csv("models/model_comparison.csv", index=False)

    # Save results as JSON for Flask to load
    results_for_json = []
    for r in all_results:
        r_copy = dict(r)
        r_copy["confusion_matrix"] = r["confusion_matrix"]
        results_for_json.append(r_copy)
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results_for_json, f, indent=2)

    # ── Step 9: Plot confusion matrices ──────────────────────────
    plot_confusion_matrices(all_results)

    # ── Step 10: Select best model ───────────────────────────────
    print("\n[STEP 7] Selecting best model...")
    best_model, best_name = select_best_model(
        [lr_model, rf_model], all_results, ["Logistic Regression", "Random Forest"]
    )
    save_best_model(best_model, feature_cols)

    # Save model name and metrics for the Flask app
    best_result = lr_results if best_name == "Logistic Regression" else rf_results
    model_info = {
        "model_name": best_name,
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1": best_result["f1"],
        "roc_auc": best_result["roc_auc"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_features": len(feature_cols),
    }
    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved: {model_info}")

    # ── Step 11: Explain model ───────────────────────────────────
    print("\n[STEP 8] Generating feature importance...")
    explain_model()

    # ── Summary ──────────────────────────────────────────────────
    churn_rate = y.mean()
    total_customers = len(df_clean)
    churned_customers = int(y.sum())
    avg_spend = float(df_clean["MonthlySpend"].mean())

    dataset_summary = {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "active_customers": total_customers - churned_customers,
        "churn_rate": round(float(churn_rate) * 100, 2),
        "avg_monthly_spend": round(avg_spend, 2),
    }
    with open("models/dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training Pipeline Complete!")
    print("=" * 60)
    print(f"  Best Model    : {best_name}")
    print(f"  Accuracy      : {best_result['accuracy']:.4f}")
    print(f"  Recall        : {best_result['recall']:.4f}")
    print(f"  F1-Score      : {best_result['f1']:.4f}")
    print(f"  ROC-AUC       : {best_result['roc_auc']:.4f}")
    print(f"  Total Customers: {total_customers:,}")
    print(f"  Churn Rate    : {churn_rate:.2%}")
    print("\nAll files saved. Run 'python app.py' to start the web app.")


if __name__ == "__main__":
    main()
