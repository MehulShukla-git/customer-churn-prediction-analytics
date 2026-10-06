# Customer Churn Prediction & Analytics

A complete end-to-end machine learning project that predicts customer churn using real customer data. Built for a technical interview to demonstrate Data Analytics, Machine Learning, Model Evaluation, Explainability, and API Development.

---

## Project Structure

```
customer-churn-prediction/
│
├── app.py                  ← Flask web application (main entry point)
├── run_training.py         ← Master training script (run this first)
├── requirements.txt
├── README.md
├── PROJECT_EXPLANATION.md  ← Interview Q&A
│
├── data/
│   ├── customer_churn_messy_raw.csv   ← Original dataset (untouched)
│   ├── customer_churn_clean.csv       ← Cleaned dataset (generated)
│   └── customer_churn_features.csv    ← With engineered features (generated)
│
├── notebooks/
│   └── churn_analysis.ipynb           ← Complete Jupyter notebook
│
├── src/
│   ├── data_cleaning.py       ← Data quality handling
│   ├── eda.py                 ← Exploratory data analysis
│   ├── feature_engineering.py ← Feature creation
│   ├── train_model.py         ← Model training pipeline
│   ├── evaluate_model.py      ← Metrics & model selection
│   └── explain_model.py       ← Feature importance
│
├── models/
│   ├── best_model.pkl          ← Saved best model (generated)
│   ├── feature_cols.pkl        ← Feature column names (generated)
│   └── feature_importance.csv  ← Top features (generated)
│
├── templates/
│   ├── base.html, index.html, analytics.html
│   ├── prediction.html, results.html
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── database/
│   └── predictions.db         ← SQLite database (generated)
│
└── eda_plots/                 ← Generated charts (generated)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python run_training.py
```

This will:
- Clean the raw dataset
- Run EDA and generate plots
- Train Logistic Regression and Random Forest
- Compare and select the best model
- Save the model and generate feature importance

### 3. Start the web app locally

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Deployment (Render)

This repository is pre-configured for seamless deployment on **Render**:

- **Runtime:** `Python`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Root Directory:** *(leave blank if repository root is the project folder)*

---

## Features

| Feature | Description |
|---|---|
| Data Cleaning | Handles missing values, duplicates, inconsistent categories, outliers |
| EDA | 12 business-driven visualizations |
| Feature Engineering | 9 new features based on business logic |
| Model Training | Logistic Regression + Random Forest |
| Model Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Model Selection | Automated selection using composite F1+AUC score |
| Explainability | Feature importance visualization |
| Prediction API | `POST /api/predict` returns churn probability + risk level |
| Web Dashboard | Interactive charts, analytics, prediction form |
| SQLite Storage | Every prediction stored in database |

---

## API Usage

**Endpoint:** `POST /api/predict`

**Request:**
```json
{
  "Age": 35,
  "Tenure_Months": 18,
  "Plan": "Premium",
  "MonthlySpend": 2500,
  "SupportTickets_90D": 2,
  "MonthlyUsage_Hours": 50,
  "PaymentHistory_Score": 85,
  "LoginFrequency_30D": 12,
  "FailedPayments_90D": 1,
  "ContractType": "Monthly",
  "AutoRenew": "No",
  "DiscountPercent": 10,
  "DaysSinceLastContact": 60,
  "Complaints_90D": 1,
  "Region": "North",
  "SignupDate": "2024-01-01"
}
```

**Response:**
```json
{
  "prediction": "Churn",
  "probability": 0.82,
  "probability_pct": 82.0,
  "risk_level": "High"
}
```

**Risk Levels:**
- `0% – 39%` → Low
- `40% – 69%` → Medium
- `70% – 100%` → High

---

## Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Data Processing | Pandas, NumPy | Industry standard, beginner-friendly |
| Visualization | Matplotlib, Seaborn | Clean static plots for EDA |
| Machine Learning | Scikit-learn | Complete ML toolkit, no GPU needed |
| Model Persistence | Joblib | Efficient serialization for sklearn |
| Web Backend | Flask | Lightweight, simple Python web framework |
| Frontend Charts | Chart.js | Interactive browser charts without React |
| UI Framework | Bootstrap 5 | Responsive design, no custom CSS framework |
| Database | SQLite | No server required, built into Python |

---

## Dataset

- **File:** `customer_churn_messy_raw.csv`
- **Rows:** ~6,625 customers
- **Columns:** 18 features
- **Target:** `Churn` (0 = Active, 1 = Churned)

---

## Author Notes

This project is designed to be:
- **Simple** — no unnecessary complexity
- **Clean** — well-named functions, clear comments
- **Explainable** — every decision is documented
- **Testable** — can be run end-to-end in minutes
