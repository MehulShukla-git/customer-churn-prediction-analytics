"""
train_model.py
--------------
Trains Logistic Regression and Random Forest classifiers.
Uses a sklearn Pipeline to prevent data leakage.
Models are compared and the best one is saved.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_cleaning import clean_data
from src.feature_engineering import create_features


# ─────────────────────────────────────────────
# DEFINE FEATURE SETS
# ─────────────────────────────────────────────

# These are the raw + engineered features used for training.
# customer_id is already dropped; Churn is the target.
NUMERICAL_FEATURES = [
    "Age", "Tenure_Months", "MonthlySpend", "SupportTickets_90D",
    "MonthlyUsage_Hours", "PaymentHistory_Score", "LoginFrequency_30D",
    "FailedPayments_90D", "DiscountPercent", "DaysSinceLastContact",
    "Complaints_90D", "SignupYear", "SignupMonth",
    "HighValueCustomer", "LowUsage", "PaymentRisk",
    "HasComplaint", "EngagementScore", "HasDiscount",
]

CATEGORICAL_FEATURES = [
    "Plan", "ContractType", "AutoRenew", "Region",
    "TenureGroup", "SupportLevel", "ContactGroup",
]

TARGET = "Churn"


def prepare_data(filepath: str):
    """
    Load, clean, and engineer features. Return X and y.
    """
    df = clean_data(filepath)
    df = create_features(df)

    # Select only the columns we need
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    # Keep only columns that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET]

    print(f"[prepare_data] Features: {len(feature_cols)}, Samples: {len(X)}")
    print(f"[prepare_data] Churn rate: {y.mean():.2%}")
    return X, y, feature_cols


def build_preprocessor():
    """
    Build a ColumnTransformer that handles numerical and categorical columns.

    WHY Pipeline?
    - Ensures all transforms are fit on TRAINING data only
    - Prevents data leakage from test set
    - Makes prediction on new data straightforward

    Numerical:
    - SimpleImputer (median): handles any remaining NaN after cleaning
    - StandardScaler: Logistic Regression is sensitive to feature scale

    Categorical:
    - SimpleImputer (constant='Unknown'): safety net for missing values
    - OneHotEncoder: converts categories to numbers the model can use
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Use the actual feature lists (matching what prepare_data returns)
    num_cols = [c for c in NUMERICAL_FEATURES]
    cat_cols = [c for c in CATEGORICAL_FEATURES]

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop")

    return preprocessor


def train_models(X_train, y_train, preprocessor):
    """
    Train both models using the same preprocessor.
    WHY two models?
    - Logistic Regression: simple, fast, interpretable baseline
    - Random Forest: handles nonlinear relationships, usually more accurate
    """

    # Model 1: Logistic Regression
    # max_iter=1000 to ensure convergence on our dataset
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"  # handles class imbalance
        )),
    ])

    # Model 2: Random Forest
    # n_estimators=200 gives stable results without being too slow
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",  # handles class imbalance
            n_jobs=-1,
        )),
    ])

    print("\n[train_models] Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)

    print("[train_models] Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)

    return lr_pipeline, rf_pipeline


def save_best_model(best_model, feature_cols):
    """
    Save the best model and the feature column list.
    WHY: We need to reload the same pipeline for predictions.
    The feature_cols list ensures new data is structured the same way.
    """
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    print("[save_best_model] Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    from src.evaluate_model import evaluate_model, select_best_model

    filepath = "data/customer_churn_messy_raw.csv"
    X, y, feature_cols = prepare_data(filepath)

    # WHY stratify=y?
    # Preserves the churn ratio in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    lr_model, rf_model = train_models(X_train, y_train, preprocessor)

    print("\n--- Evaluating Models ---")
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    best_model, best_name = select_best_model(
        [lr_model, rf_model],
        [lr_results, rf_results],
        ["Logistic Regression", "Random Forest"]
    )

    save_best_model(best_model, feature_cols)
    print(f"\nBest Model: {best_name}")
