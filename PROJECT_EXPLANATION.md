# Customer Churn Prediction: Interview Explanation Guide

This document is designed to help you explain every aspect of this project in a technical interview clearly, confidently, and without unnecessary jargon.

---

## 1. What is customer churn?
**Definition**: Customer churn occurs when a customer stops doing business with a company or cancels their subscription.
**Why I care**: It's usually more expensive to acquire a new customer than to keep an existing one.
**Simple Example**: If Netflix loses a subscriber because they canceled their membership, that is churn.

## 2. What problem does the project solve?
**Definition**: Companies don't know *which* customers are going to cancel until it's too late. This project uses historical data to predict churn *before* it happens.
**Why I built it**: So the business can intervene (e.g., offer a discount or provide better support) to save high-risk customers.
**Simple Example**: A dashboard flags that Customer X has an 82% risk of leaving next month. Customer Success calls them today and saves the account.

## 3. Why is churn prediction useful?
**Definition**: It allows proactive rather than reactive customer retention.
**Why it's useful**: Instead of guessing who might leave, the company can efficiently allocate its retention budget to the right people, improving revenue.

## 4. What dataset was used?
**Definition**: A messy, realistic dataset containing ~6,600 customer records with demographic, subscription, engagement, and payment data.
**Why this data**: These features are the most common indicators of user behavior in a SaaS or subscription business.

## 5. What data-quality problems were found?
**Definition**: Real-world data is never perfect. The dataset contained missing values, duplicate rows, duplicate IDs, inconsistent categorical names (e.g., "Basic" vs "basic"), and numerical outliers (e.g., negative age or extreme spend).
**Why it matters**: Feeding garbage data into a machine learning model produces garbage predictions.

## 6. How were missing values handled?
**Definition**: For numerical columns (like `Age` or `MonthlySpend`), I used **Median Imputation**. For categorical columns, I filled gaps with an "Unknown" category.
**Why I used it**: The median is robust to outliers (unlike the mean). Creating an "Unknown" category ensures we don't lose rows simply because a user didn't fill out one form field.

## 7. How were duplicates handled?
**Definition**: Exact duplicate rows were removed. For duplicate `customer_id`s with varying data, I kept the *last* occurrence.
**Why I used it**: Exact duplicates inflate model confidence artificially. For differing duplicates, the last occurrence usually represents the most up-to-date customer record.

## 8. How were inconsistent categories handled?
**Definition**: Strings like "basic", "BASIC", and "Std" were normalized to "Basic" and "Standard".
**Why I used it**: To a computer, "basic" and "Basic" are two completely different things. Normalizing prevents the model from splitting its learning across redundant categories.

## 9. How were outliers handled?
**Definition**: Impossible values (like negative age) were set to NaN and imputed. Extreme values (like an absurdly high monthly spend) were capped at the 99th percentile (winsorization).
**Why I used it**: A data entry error of $999,999 monthly spend would distort the algorithm. Capping it brings it back into a realistic range without deleting the entire customer record.

## 10. What is EDA?
**Definition**: Exploratory Data Analysis (EDA) is the process of visually and statistically investigating the dataset before building models.
**Why I used it**: To understand the underlying patterns, detect anomalies, and check relationships between features and the target variable (Churn).

## 11. What insights were found from EDA?
**Definition**: By plotting the data, several trends emerged:
1. Month-to-month contracts churn significantly more than annual ones.
2. Customers who opt out of auto-renewal are high flight risks.
3. High numbers of support tickets and failed payments strongly correlate with churn.
**Why it's useful**: These insights confirm business intuition and tell us which features the model will likely rely on most.

## 12. What feature engineering was performed?
**Definition**: Feature engineering is creating new data columns from existing ones to make it easier for the model to learn.
**Examples**:
- `TenureGroup`: Grouped raw months into "New", "Growing", "Established", "Loyal".
- `PaymentRisk`: A binary flag if the user had 2+ failed payments.
- `EngagementScore`: A blend of login frequency and usage hours.
**Why I used it**: Sometimes raw numbers are too noisy. Grouping them or flagging specific thresholds explicitly highlights business logic for the algorithm.

## 13. Why Logistic Regression?
**Definition**: A statistical model that predicts a binary outcome (0 or 1) using a linear combination of features.
**Why I used it**: It serves as an excellent baseline. It is fast, mathematically simple, and highly interpretable (you can look at the coefficients to see exactly what drives the prediction).

## 14. Why Random Forest?
**Definition**: An ensemble model that builds many decision trees and merges their predictions to get a more accurate and stable result.
**Why I used it**: It captures non-linear relationships (e.g., age might only matter if tenure is also low) and handles complex interactions between features well.

## 15. Why were these two models compared?
**Definition**: It is standard practice to compare a simple, interpretable linear model (Logistic Regression) against a more complex, non-linear tree model (Random Forest).
**Why I used it**: You never know which algorithm will perform better on a specific dataset until you try. Sometimes the simple model wins, avoiding unnecessary complexity.

## 16. What is Accuracy?
**Definition**: The percentage of total predictions the model got exactly right.
**Why we must be careful**: If a dataset has 90% active customers and 10% churners, a "dumb" model that always guesses "Active" will be 90% accurate, but completely useless for finding churners.

## 17. What is Precision?
**Definition**: Of all the customers the model *predicted* would churn, what percentage *actually* churned? (Quality of positive predictions).
**Simple Example**: If we predict 100 people will churn, and 30 do, our precision is 30%.

## 18. What is Recall?
**Definition**: Of all the customers who *actually* churned, what percentage did our model *successfully identify*? (Quantity of positive predictions).
**Simple Example**: If 100 people actually churn, and our model only spots 40 of them, our recall is 40%.

## 19. What is F1-score?
**Definition**: The harmonic mean of Precision and Recall. It provides a single metric that balances both concerns.
**Why I used it**: It is a much better metric than accuracy for imbalanced datasets like churn.

## 20. What is ROC-AUC?
**Definition**: Receiver Operating Characteristic - Area Under the Curve. It measures the model's ability to distinguish between classes across all possible probability thresholds.
**Why I used it**: An AUC of 0.5 means the model is guessing randomly. An AUC of 1.0 is perfect. It gives a broad view of model quality independent of the specific 50% cutoff line.

## 21. Why is recall important for churn?
**Definition**: In churn prediction, a **False Negative** (model says they will stay, but they leave) is usually much worse than a **False Positive** (model says they will leave, but they stay).
**Why**: If we have a false positive, we might accidentally give a loyal customer a 10% discount. If we have a false negative, we lose the customer entirely. Therefore, we want a model with high Recall to ensure we catch as many true churners as possible.

## 22. Which model performed better?
**Definition**: (Refer to the `models/model_comparison.csv` or dashboard). Typically, Random Forest might have better accuracy, but Logistic Regression often provides more balanced Recall.
**Why**: The evaluation script calculates a composite score (`0.5 * F1 + 0.5 * ROC-AUC`) to make a data-driven choice between the two.

## 23. Why was the final model selected?
**Definition**: The script automatically selected the model with the highest composite score of F1 and ROC-AUC.
**Why**: We want a model that has strong overall discriminative power (ROC-AUC) but specifically performs well on the minority churn class (F1-score).

## 24. What is feature importance?
**Definition**: A ranking of which data columns had the most influence on the model's final predictions.
**Why I used it**: To explain the "black box" to business stakeholders. If `AutoRenew_No` is the top feature, we can advise the product team to make auto-renewal more attractive.

## 25. What does the prediction API do?
**Definition**: A Flask REST API endpoint (`POST /api/predict`) that takes a JSON payload of one customer's details and returns their churn probability.
**Why I built it**: Machine learning is useless if it's trapped in a Jupyter Notebook. An API allows frontend applications, CRM systems, or mobile apps to query the model in real-time.

## 26. How does the frontend communicate with Flask?
**Definition**: The frontend HTML uses JavaScript (`fetch` API) to send the form data as a JSON payload to the Flask backend. Flask cleans the data, runs it through the saved model, and sends a JSON response back to the browser.

## 27. What are the limitations?
**Definition**: 
1. **Static Data**: The model assumes the data represents current behavior; if market conditions change, the model drifts and becomes inaccurate.
2. **Correlation vs Causation**: The model knows that "No Auto-Renew" correlates with churn, but forcing users to auto-renew might just make them angry, not loyal.

## 28. What future improvements can be made?
**Definition**:
1. **XGBoost/LightGBM**: Try advanced gradient boosting models which often outperform Random Forest on tabular data.
2. **Hyperparameter Tuning**: Use GridSearch or RandomizedSearch to find the mathematically optimal settings for the models.
3. **Time-Series Features**: Instead of a static "Usage Hours", track the *trend* (e.g., "Usage Hours Drop in Last 30 Days").
