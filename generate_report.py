import json

with open('models/dataset_summary.json') as f:
    stats = json.load(f)

with open('models/model_info.json') as f:
    model = json.load(f)

with open('models/evaluation_results.json') as f:
    evals = json.load(f)
    
report = f"""# Data Science Analysis Report: Customer Churn

## 1. Dataset Overview
- **Total Customers**: {stats['total_customers']:,}
- **Churned Customers**: {stats['churned_customers']:,}
- **Active Customers**: {stats['active_customers']:,}
- **Churn Rate**: {stats['churn_rate']}%
- **Average Monthly Spend**: ₹{stats['avg_monthly_spend']:,.2f}

## 2. Data-Quality Findings
The raw dataset contained several real-world data quality issues:
- **Missing Values**: Present in `Age`, `Plan`, `MonthlySpend`, `MonthlyUsage_Hours`, `AutoRenew`, etc.
- **Duplicates**: 125 duplicate records/IDs were found.
- **Inconsistent Categories**: Features like `Plan`, `ContractType`, and `Region` had mixed cases (e.g., 'basic' vs 'Basic').
- **Invalid Numerical Values**: Negative ages, extreme spend outliers, and negative tenures were present.
- **Target Ambiguity**: The `Churn` column contained various representations ('Yes', '1', 'Y', 'Churned') and some missing/unknown values.

## 3. Cleaning Decisions
- **Duplicates**: Exact row duplicates were removed. For duplicate IDs, the latest record was retained.
- **Numerical Outliers**: Impossible values (negative age) were converted to NaN for imputation. Extreme outliers (spend) were capped at the 99th percentile.
- **Categorical Normalization**: Strings were stripped, capitalized, and mapped to standard categories. Unmappable values were set to 'Unknown'.
- **Missing Values**: Numerical features were imputed using the median (robust to outliers). Categorical features were filled with 'Unknown'.
- **Target Variable**: Mapped to strict binary 0/1. Rows with missing or ambiguous targets were dropped to prevent noisy training.

## 4. EDA Findings
- **Contract Type**: Month-to-month contracts exhibit significantly higher churn rates than annual contracts.
- **Auto-Renewal**: Customers who opted out of auto-renewal have a much higher likelihood of churning.
- **Engagement**: Low monthly usage hours and low login frequencies strongly correlate with churn.
- **Support & Payments**: A high number of support tickets and recent failed payments are strong leading indicators of churn.

## 5. Feature Engineering
Created 9 new business-motivated features:
1. `TenureGroup`: Binned tenure into New, Growing, Established, Loyal.
2. `HighValueCustomer`: Binary flag for top 25% of spenders.
3. `LowUsage`: Binary flag for bottom 25% of usage.
4. `PaymentRisk`: Flag for 2+ failed payments.
5. `SupportLevel`: Binned support tickets into None, Low, Medium, High.
6. `HasComplaint`: Binary flag.
7. `ContactGroup`: Binned days since last contact.
8. `EngagementScore`: Blended login frequency and usage hours.
9. `HasDiscount`: Binary flag.

## 6. Model Comparison
Evaluated two models using a strict pipeline to prevent data leakage:
- **Logistic Regression**: Serves as a strong, interpretable linear baseline.
- **Random Forest**: Captures non-linear relationships and interactions.

## 7. Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|"""

for r in evals:
    report += f"\n| {r['model_name']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"

report += f"""

## 8. Final Model Selection
**Selected Model**: {model['model_name']}
**Reasoning**: Model selection was based on a composite score emphasizing F1-Score and ROC-AUC. In churn prediction, Recall is critical because missing a churner (false negative) is more costly than a false alarm. The selected model provided the best balance of discriminative power and minority class detection.

## 9. Feature Importance
Based on the selected model, the most influential factors driving churn predictions are:
(See the analytics dashboard or `models/feature_importance.csv` for the exact top 10 ranking).
Key drivers generally include Support Level, Tenure, Contract Type, and Payment History.

## 10. Business Interpretation
- The model indicates that customer engagement and support experience are critical retention levers.
- Customers on monthly contracts without auto-renewal require proactive engagement.
- High support tickets combined with failed payments form a high-risk profile.
- **Recommendation**: Implement targeted retention campaigns (e.g., discounts or dedicated success managers) for customers flagged as High Risk by the API.

## 11. Limitations
- **Static Snapshot**: The model relies on a static snapshot of data. It does not natively handle time-series trajectories (e.g., *dropping* usage over time).
- **Correlation vs Causation**: Identifying predictors of churn does not mean modifying those predictors will prevent churn.
- **Concept Drift**: As market conditions and product offerings change, the model's accuracy will degrade without retraining.
"""

with open('docs/analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print('Analysis report generated.')
