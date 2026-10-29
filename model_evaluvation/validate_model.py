
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Run: uv add xgboost"
    )


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = MODEL_DIR / "validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "model_train.csv"
CONFIG_PATH = DATA_DIR / "feature_config.json"


def load_data():
    print("Loading training dataset...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    train = pd.read_csv(TRAIN_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    features = config.get("features")

    if not features:
        raise ValueError(
            "No features found in feature_config.json"
        )

    missing = [
        feature for feature in features
        if feature not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training features: {missing}"
        )

    if "is_hired" not in train.columns:
        raise ValueError("Target column is_hired not found.")

    X = train[features].copy()
    y = train["is_hired"].astype(int)

    if X.isnull().any().any():
        raise ValueError(
            "Missing feature values found. "
            "Run the preprocessing pipeline again."
        )

    print(f"Training records: {len(train)}")
    print(f"Features: {len(features)}")
    print(f"Hired: {(y == 1).sum()}")
    print(f"Not hired: {(y == 0).sum()}")

    return X, y, features


def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_fold(y_true, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),
        "brier_score": brier_score_loss(
            y_true, probabilities
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    }


def run_validation(X, y, models):
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    results = []

    total_folds = cv.get_n_splits(X, y)

    print(f"\nTotal validation folds: {total_folds}")
    print("Starting repeated cross-validation...\n")

    for model_name, base_model in models.items():

        print(f"Evaluating: {model_name}")

        for fold_number, (train_idx, val_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            calibrated_model = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=3
            )

            calibrated_model.fit(
                X_train, y_train
            )

            probabilities = calibrated_model.predict_proba(
                X_val
            )[:, 1]

            metrics = evaluate_fold(
                y_val, probabilities
            )

            results.append({
                "model": model_name,
                "fold": fold_number,
                **metrics
            })

            print(
                f"  Fold {fold_number}/{total_folds} "
                f"| ROC-AUC: {metrics['roc_auc']:.4f} "
                f"| Accuracy: {metrics['accuracy']:.4f}"
            )

    return pd.DataFrame(results)


def summarize_results(results):
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss"
    ]

    summary = results.groupby("model")[
        metric_columns
    ].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary = summary.sort_values(
        by="roc_auc_mean",
        ascending=False
    )

    return summary


def save_results(results, summary):
    results_path = OUTPUT_DIR / "cross_validation_results.csv"
    summary_path = OUTPUT_DIR / "validation_summary.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


def main():
    print("=" * 65)
    print("EMPLOYER-CANDIDATE PROPENSITY MATCHER")
    print("REPEATED CROSS-VALIDATION")
    print("=" * 65)

    X, y, features = load_data()

    models = build_models()

    results = run_validation(
        X, y, models
    )

    summary = summarize_results(results)

    save_results(results, summary)

    print("\nVALIDATION SUMMARY")
    print("=" * 65)

    display_columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "brier_score_mean",
        "log_loss_mean"
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                column: "{:.4f}".format
                for column in display_columns
                if column != "model"
            }
        )
    )

    best_model = summary.iloc[0]["model"]

    print("\nPROVISIONAL BEST MODEL")
    print("-" * 65)
    print(best_model)

    print(
        "\nNote: This is cross-validation on synthetic training data."
    )
    print(
        "The untouched test dataset must be used for final evaluation."
    )

    print("\nVALIDATION COMPLETED")


if __name__ == "__main__":
    main()
