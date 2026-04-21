"""Model training script for the InsureRisk project."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the processed insurance dataset."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables and create the target variable."""
    df = df.copy()

    # Encode binary categorical variables
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

    # One-hot encode region
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Create binary target: high-cost vs low-cost
    df["high_cost"] = (df["charges"] > df["charges"].median()).astype(int)

    return df


def split_data(df: pd.DataFrame):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=["charges", "high_cost"])
    y = df["high_cost"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a random forest classifier."""
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> None:
    """Print classification metrics for a trained model."""
    predictions = model.predict(X_test)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, predictions))


def save_model(model, model_path: Path) -> None:
    """Save the trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


def main() -> None:
    """Run the full model training pipeline."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "clean_data.csv"
    model_path = base_dir / "models" / "model.pkl"

    df = load_data(data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    log_reg = train_logistic_regression(X_train, y_train)
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    save_model(rf_model, model_path)


if __name__ == "__main__":
    main()