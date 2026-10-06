"""
evaluate_model.py
-----------------
Calculates classification metrics and compares models.
Selects the best model based on F1-score and ROC-AUC.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model pipeline on the test set.

    WHY these metrics?
    - Accuracy: overall correctness (can be misleading with class imbalance)
    - Precision: of all predicted churners, how many actually churned?
    - Recall: of all actual churners, how many did we catch?
    - F1-score: harmonic mean of precision and recall (balanced metric)
    - ROC-AUC: overall discrimination ability across all thresholds

    For churn prediction, Recall is especially important:
    Missing a churner (false negative) is costly — the company loses the customer.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    return results


def select_best_model(models: list, results: list, names: list):
    """
    Select the best model based on a composite score.

    WHY composite score?
    We care about both capturing churners (Recall) and overall
    discriminative ability (ROC-AUC). Using both avoids selecting a model
    that is good at one metric but poor at the other.

    Score = 0.5 * F1 + 0.5 * ROC-AUC
    """
    best_idx = 0
    best_score = -1

    for i, r in enumerate(results):
        # Weighted composite: F1 and ROC-AUC both matter
        composite = 0.5 * r["f1"] + 0.5 * r["roc_auc"]
        print(f"\n[select_best_model] {names[i]} composite score: {composite:.4f}")
        if composite > best_score:
            best_score = composite
            best_idx = i

    print(f"\n[select_best_model] Selected: {names[best_idx]} (score={best_score:.4f})")
    return models[best_idx], names[best_idx]


def plot_confusion_matrices(results: list, output_dir: str = "eda_plots"):
    """
    Plot confusion matrices for all evaluated models side by side.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Active", "Churned"],
                    yticklabels=["Active", "Churned"],
                    linewidths=1, cbar=False)
        ax.set_title(f"{r['model_name']}\nAccuracy={r['accuracy']:.3f}  F1={r['f1']:.3f}",
                     fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "13_confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Build a clean DataFrame comparing all models on all metrics.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1']:.4f}",
            "ROC-AUC": f"{r['roc_auc']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n--- Model Comparison ---")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("Run train_model.py to train and evaluate models.")
