"""Model evaluation script for the InsureRisk project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.train_model import load_data, preprocess_data


def split_data(df: pd.DataFrame):
    """Recreate the train/test split used during training."""
    X = df.drop(columns=["charges", "high_cost"])
    y = df["high_cost"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def align_features(X_test: pd.DataFrame, model) -> pd.DataFrame:
    """Match evaluation features to the trained model input schema."""
    if hasattr(model, "feature_names_in_"):
        return X_test.reindex(columns=model.feature_names_in_, fill_value=0)
    return X_test


def compute_metrics(y_true: pd.Series, y_pred) -> dict:
    """Compute core classification metrics."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "classification_report": report,
    }


def save_metrics(metrics: dict, output_dir: Path) -> None:
    """Save evaluation metrics to JSON."""
    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def save_confusion_matrix(y_true: pd.Series, y_pred, output_dir: Path) -> None:
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Cost", "High Cost"],
    ).plot(cmap="Blues", ax=ax, colorbar=False)

    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to: {output_path}")


def save_feature_importance(model, X_test: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save a feature importance plot for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature importances. Skipping feature importance plot.")
        return

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_path = output_dir / "feature_importance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance plot to: {output_path}")


def main() -> None:
    """Evaluate the saved model on the held-out test set."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "clean_data.csv"
    model_path = base_dir / "models" / "model.pkl"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    df = preprocess_data(df)
    _, X_test, _, y_test = split_data(df)

    model = joblib.load(model_path)
    X_test = align_features(X_test, model)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    print("Model Evaluation")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_metrics(metrics, reports_dir)
    save_confusion_matrix(y_test, y_pred, reports_dir)
    save_feature_importance(model, X_test, reports_dir)


if __name__ == "__main__":
    main()