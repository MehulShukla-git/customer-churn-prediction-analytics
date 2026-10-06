"""
explain_model.py
----------------
Provides feature importance / explainability for the selected model.
Uses built-in sklearn methods — no external libraries needed.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_feature_names(model_pipeline) -> list:
    """
    Extract feature names after preprocessing (including one-hot encoded names).
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    num_transformer = preprocessor.transformers_[0]  # ("num", pipeline, cols)
    cat_transformer = preprocessor.transformers_[1]  # ("cat", pipeline, cols)

    num_feature_names = num_transformer[2]  # list of numerical column names

    # Get OHE feature names
    ohe = cat_transformer[1].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_transformer[2]))

    all_features = list(num_feature_names) + cat_feature_names
    return all_features


def get_feature_importance(model_pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from the trained model.

    For Random Forest: uses feature_importances_ attribute.
    For Logistic Regression: uses absolute coefficient values.

    WHY absolute values for LR?
    A large negative coefficient means the feature strongly predicts NO churn.
    We care about magnitude (how much the feature influences the prediction),
    not direction alone.

    IMPORTANT NOTE:
    Feature importance shows which features are most useful to the model.
    It does NOT prove causation — it shows correlation within the model's logic.
    """
    classifier = model_pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        # Random Forest
        importances = classifier.feature_importances_
        method = "Feature Importance (Random Forest)"
    elif hasattr(classifier, "coef_"):
        # Logistic Regression
        importances = np.abs(classifier.coef_[0])
        method = "Coefficient Magnitude (Logistic Regression)"
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")

    # Ensure lengths match
    min_len = min(len(feature_names), len(importances))
    feature_names = feature_names[:min_len]
    importances = importances[:min_len]

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n[get_feature_importance] Method: {method}")
    print("Top 10 Features:")
    print(df.head(10).to_string(index=False))

    return df, method


def plot_feature_importance(importance_df: pd.DataFrame, method: str,
                            top_n: int = 10, output_dir: str = "eda_plots") -> str:
    """
    Plot top N most important features as a horizontal bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_df = importance_df.head(top_n).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_df["Feature"], top_df["Importance"],
                   color="#2563EB", edgecolor="white", linewidth=1.2)

    ax.set_title(f"Top {top_n} Most Important Features\n({method})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")

    # Add value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "14_feature_importance.png")
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def explain_model(model_path: str = "models/best_model.pkl",
                  output_dir: str = "eda_plots"):
    """
    Load saved model, extract and plot feature importance.
    Returns the importance DataFrame for use in the web API.
    """
    if not os.path.exists(model_path):
        print(f"[explain_model] Model not found at {model_path}. Train first.")
        return None

    model = joblib.load(model_path)
    feature_names = get_feature_names(model)
    importance_df, method = get_feature_importance(model, feature_names)
    plot_feature_importance(importance_df, method, output_dir=output_dir)

    # Save importance to CSV for later use
    importance_df.to_csv("models/feature_importance.csv", index=False)
    print("[explain_model] Feature importance saved to models/feature_importance.csv")

    return importance_df


if __name__ == "__main__":
    explain_model()
