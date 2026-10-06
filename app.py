"""
app.py
------
Flask web application for Customer Churn Prediction.
Serves the dashboard, prediction form, analytics page, and REST API.
"""

import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort, send_from_directory

from pathlib import Path

# Add project root to path so src imports work
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.train_model import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

app = Flask(__name__)

# ── Jinja2 filters and globals ─────────────────────────────────
@app.template_filter('format_number')
def format_number(value):
    """Format large integers with commas."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)

app.jinja_env.globals['enumerate'] = enumerate

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RAW_DATA_PATH = BASE_DIR / "data" / "customer_churn_messy_raw.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "customer_churn_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.pkl"
DB_PATH = BASE_DIR / "database" / "predictions.db"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"
DATASET_SUMMARY_PATH = BASE_DIR / "models" / "dataset_summary.json"
EVAL_RESULTS_PATH = BASE_DIR / "models" / "evaluation_results.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"
DATA_QUALITY_PATH = BASE_DIR / "models" / "data_quality.json"


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    """
    Initialize SQLite database with the predictions table.
    SQLite is used for simplicity — no separate database server needed.
    """
    os.makedirs(BASE_DIR / "database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            tenure_months REAL,
            plan TEXT,
            monthly_spend REAL,
            support_tickets INTEGER,
            monthly_usage REAL,
            payment_score REAL,
            login_frequency REAL,
            failed_payments INTEGER,
            contract_type TEXT,
            auto_renew TEXT,
            discount_percent REAL,
            days_since_contact REAL,
            complaints INTEGER,
            region TEXT,
            signup_date TEXT,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Always ensure database table exists when app starts (under Gunicorn or python app.py)
init_db()


def save_prediction_to_db(input_data: dict, prediction: str,
                          probability: float, risk_level: str):
    """Store prediction in SQLite for history tracking."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions (
            age, tenure_months, plan, monthly_spend, support_tickets,
            monthly_usage, payment_score, login_frequency, failed_payments,
            contract_type, auto_renew, discount_percent, days_since_contact,
            complaints, region, signup_date, prediction, probability, risk_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        input_data.get("Age"),
        input_data.get("Tenure_Months"),
        input_data.get("Plan"),
        input_data.get("MonthlySpend"),
        input_data.get("SupportTickets_90D"),
        input_data.get("MonthlyUsage_Hours"),
        input_data.get("PaymentHistory_Score"),
        input_data.get("LoginFrequency_30D"),
        input_data.get("FailedPayments_90D"),
        input_data.get("ContractType"),
        input_data.get("AutoRenew"),
        input_data.get("DiscountPercent"),
        input_data.get("DaysSinceLastContact"),
        input_data.get("Complaints_90D"),
        input_data.get("Region"),
        input_data.get("SignupDate"),
        prediction,
        round(probability, 4),
        risk_level,
    ))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────

def load_model():
    """Load trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


def load_json(path: str) -> dict:
    """Load a JSON file safely."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_dataset_stats():
    """Load pre-computed dataset statistics for the dashboard."""
    clean_path = CLEAN_DATA_PATH
    if not os.path.exists(clean_path):
        return {}

    df = pd.read_csv(clean_path)
    stats = {
        "total_customers": len(df),
        "churned_customers": int(df["Churn"].sum()),
        "active_customers": int((df["Churn"] == 0).sum()),
        "churn_rate": round(df["Churn"].mean() * 100, 1),
        "avg_monthly_spend": round(df["MonthlySpend"].mean(), 2),
        "contract_churn": df.groupby("ContractType")["Churn"].mean().round(4).to_dict(),
        "plan_churn": df.groupby("Plan")["Churn"].mean().round(4).to_dict(),
        "autorenew_churn": df.groupby("AutoRenew")["Churn"].mean().round(4).to_dict(),
        "region_churn": df.groupby("Region")["Churn"].mean().round(4).to_dict(),
        "tenure_group_churn": {},
        "avg_tenure": round(df["Tenure_Months"].mean(), 1),
        "high_risk_count": 0, # Will be calculated if possible, else 0
    }

    # Tenure group requires feature engineering
    try:
        df_feat = create_features(df)
        tg_order = ["New", "Growing", "Established", "Loyal"]
        tg_churn = df_feat.groupby("TenureGroup")["Churn"].mean().round(4).to_dict()
        stats["tenure_group_churn"] = {k: tg_churn.get(k, 0) for k in tg_order if k in tg_churn}
    except Exception:
        pass

    return stats


# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────

def determine_risk(probability: float) -> str:
    """Map probability to a human-readable risk level."""
    if probability < 0.40:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    else:
        return "High"


def validate_input(data: dict) -> tuple:
    """
    Validate incoming JSON data.
    Returns (cleaned_dict, error_message).
    error_message is None if validation passes.
    """
    required_fields = [
        "Age", "Tenure_Months", "Plan", "MonthlySpend",
        "SupportTickets_90D", "MonthlyUsage_Hours", "PaymentHistory_Score",
        "LoginFrequency_30D", "FailedPayments_90D", "ContractType",
        "AutoRenew", "DiscountPercent", "DaysSinceLastContact",
        "Complaints_90D", "Region",
    ]

    for field in required_fields:
        if field not in data:
            return None, f"Missing required field: {field}"

    # Type validation
    try:
        age = float(data["Age"])
        if age < 0 or age > 120:
            return None, "Age must be between 0 and 120"
    except (ValueError, TypeError):
        return None, "Age must be a number"

    try:
        spend = float(data["MonthlySpend"])
        if spend < 0:
            return None, "MonthlySpend cannot be negative"
    except (ValueError, TypeError):
        return None, "MonthlySpend must be a number"

    try:
        discount = float(data["DiscountPercent"])
        if discount < 0 or discount > 100:
            return None, "DiscountPercent must be between 0 and 100"
    except (ValueError, TypeError):
        return None, "DiscountPercent must be a number"

    return data, None


def predict_customer(input_data: dict, model, feature_cols: list) -> dict:
    """
    Transform raw input data and generate a churn prediction.

    Steps:
    1. Convert input dict to DataFrame
    2. Apply same feature engineering used during training
    3. Select the trained feature columns
    4. Run model prediction
    5. Return prediction, probability, and risk level
    """
    # Convert to DataFrame (single row)
    row = {k: [v] for k, v in input_data.items()}
    df_input = pd.DataFrame(row)

    # Parse SignupDate to year/month (same as training)
    if "SignupDate" in df_input.columns:
        df_input["SignupDate"] = pd.to_datetime(df_input["SignupDate"], errors="coerce")
        df_input["SignupYear"] = df_input["SignupDate"].dt.year.fillna(2022).astype(int)
        df_input["SignupMonth"] = df_input["SignupDate"].dt.month.fillna(1).astype(int)
        df_input = df_input.drop(columns=["SignupDate"])

    # Apply feature engineering (same transformations as training)
    df_input = create_features(df_input)

    # Select only the feature columns used during training
    available_cols = [c for c in feature_cols if c in df_input.columns]
    missing_cols = [c for c in feature_cols if c not in df_input.columns]

    # Fill missing columns with 0 (safe default for engineered features)
    for col in missing_cols:
        df_input[col] = 0

    X = df_input[feature_cols]

    # Generate prediction
    prediction_int = model.predict(X)[0]
    probability = float(model.predict_proba(X)[0][1])

    prediction_label = "Churn" if prediction_int == 1 else "No Churn"
    risk_level = determine_risk(probability)

    # Simple explanation engine based on input data
    top_factors = []
    retention_actions = []

    if "ContractType" in df_input.columns and str(df_input["ContractType"].iloc[0]).strip().capitalize() == "Monthly":
        top_factors.append({"feature": "Month-to-Month Contract", "importance": 0.8})
        retention_actions.append("Offer a discounted upgrade to an Annual plan.")
    
    if "FailedPayments_90D" in df_input.columns and df_input["FailedPayments_90D"].iloc[0] > 0:
        top_factors.append({"feature": "Recent Failed Payments", "importance": 0.7})
        retention_actions.append("Contact customer to update payment method.")

    if "SupportTickets_90D" in df_input.columns and df_input["SupportTickets_90D"].iloc[0] > 2:
        top_factors.append({"feature": "High Support Ticket Volume", "importance": 0.6})
        retention_actions.append("Assign a dedicated success manager to resolve outstanding issues.")
        
    if "Tenure_Months" in df_input.columns and df_input["Tenure_Months"].iloc[0] < 6:
        top_factors.append({"feature": "Short Tenure (New Customer)", "importance": 0.5})
        retention_actions.append("Send onboarding tutorial and check-in email.")

    if not top_factors:
        top_factors.append({"feature": "Overall Engagement Profile", "importance": 0.3})
        retention_actions.append("Send a standard engagement survey to gather feedback.")

    return {
        "prediction": prediction_label,
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100, 1),
        "risk_level": risk_level,
        "top_factors": top_factors,
        "retention_actions": retention_actions,
    }


# ─────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard page — shows key metrics and charts."""
    model, _ = load_model()
    model_ready = model is not None

    stats = load_dataset_stats()
    model_info = load_json(MODEL_INFO_PATH)

    return render_template("index.html",
                           model_ready=model_ready,
                           stats=stats,
                           model_info=model_info)


@app.route("/prediction")
def prediction_page():
    """Customer prediction form."""
    model, _ = load_model()
    if not model:
        return render_template("prediction.html", model_ready=False)
    return render_template("prediction.html", model_ready=True)


@app.route("/analytics")
def analytics_page():
    """Analytics page with detailed charts."""
    stats = load_dataset_stats()
    eval_results = load_json(EVAL_RESULTS_PATH) if os.path.exists(EVAL_RESULTS_PATH) else []
    model_info = load_json(MODEL_INFO_PATH)

    # Load feature importance
    feature_importance = []
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        feature_importance = fi_df.head(10).to_dict("records")

    return render_template("analytics.html",
                           stats=stats,
                           eval_results=eval_results,
                           model_info=model_info,
                           feature_importance=feature_importance)


@app.route("/results")
def results_page():
    """Results page (shown after prediction)."""
    return render_template("results.html")

@app.route("/data-quality")
def data_quality_page():
    """Data Quality page."""
    dq_report = load_json(DATA_QUALITY_PATH)
    return render_template("data_quality.html", dq_report=dq_report)

@app.route("/eda")
def eda_page():
    """EDA Insights page."""
    return render_template("eda.html")

@app.route('/static/eda_plots/<path:filename>')
def serve_eda_plots(filename):
    """Serve generated plots from the eda_plots directory."""
    return send_from_directory(BASE_DIR / 'eda_plots', filename)

@app.route("/history")
def history_page():
    """Prediction History page."""
    return render_template("history.html")


# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Accept customer data JSON → return churn prediction.

    Example request:
    {
      "Age": 35,
      "Tenure_Months": 18,
      "Plan": "Premium",
      "MonthlySpend": 2500,
      ...
    }

    Example response:
    {
      "prediction": "Churn",
      "probability": 0.82,
      "risk_level": "High"
    }
    """
    model, feature_cols = load_model()
    if model is None:
        return jsonify({"error": "Model not trained yet. Run run_training.py first."}), 503

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    cleaned_data, error = validate_input(data)
    if error:
        return jsonify({"error": error}), 422

    try:
        result = predict_customer(cleaned_data, model, feature_cols)
        save_prediction_to_db(cleaned_data, result["prediction"],
                              result["probability"], result["risk_level"])
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return dataset statistics as JSON for the dashboard charts."""
    stats = load_dataset_stats()
    return jsonify(stats)


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Return model performance info."""
    model_info = load_json(MODEL_INFO_PATH)
    return jsonify(model_info)


@app.route("/api/feature-importance", methods=["GET"])
def api_feature_importance():
    """Return top 10 feature importance scores."""
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return jsonify({"error": "Feature importance not computed yet."}), 404
    fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    result = fi_df.head(10).to_dict("records")
    return jsonify(result)


@app.route("/api/evaluation", methods=["GET"])
def api_evaluation():
    """Return model evaluation results."""
    results = load_json(EVAL_RESULTS_PATH) if os.path.exists(EVAL_RESULTS_PATH) else {}
    return jsonify(results)


@app.route("/api/predictions-history", methods=["GET"])
def api_predictions_history():
    """Return last 20 stored predictions from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("""
            SELECT id, prediction, probability, risk_level, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        conn.close()
        data = [{"id": r[0], "prediction": r[1], "probability": r[2],
                 "risk_level": r[3], "created_at": r[4]} for r in rows]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("\n Customer Churn Prediction App")
    print(" -----------------------------------------")

    model, _ = load_model()
    if model is None:
        print(" WARNING: No trained model found.")
        print(" Please run:  python run_training.py")
    else:
        print(" Model loaded successfully.")

    print(" Starting Flask server at http://127.0.0.1:5000")
    print(" -----------------------------------------\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
