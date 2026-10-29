
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = MODEL_DIR / "final_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_hiring_propensity_model.pkl"
CONFIG_PATH = DATA_DIR / "feature_config.json"

MODEL_TRAIN_PATH = DATA_DIR / "model_train.csv"
MODEL_TEST_PATH = DATA_DIR / "model_test.csv"

CLEANED_TRAIN_PATH = DATA_DIR / "cleaned_train.csv"
CLEANED_TEST_PATH = DATA_DIR / "cleaned_test.csv"

THRESHOLD = 0.50


def load_model():
    print("\nLoading selected model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    return joblib.load(MODEL_PATH)


def load_features():
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    features = config.get("features")

    if not features:
        raise ValueError(
            "Features not found in feature_config.json"
        )

    return features


def remove_overlapping_pairs():
    print("\nChecking train-test overlap...")

    train = pd.read_csv(CLEANED_TRAIN_PATH)
    test = pd.read_csv(CLEANED_TEST_PATH)

    model_train = pd.read_csv(MODEL_TRAIN_PATH)
    model_test = pd.read_csv(MODEL_TEST_PATH)

    if len(train) != len(model_train):
        raise ValueError(
            "Training data row counts do not match. "
            "Cannot safely align candidate-job pairs."
        )

    if len(test) != len(model_test):
        raise ValueError(
            "Test data row counts do not match. "
            "Cannot safely align candidate-job pairs."
        )

    required_columns = [
        "candidate_text",
        "job_text"
    ]

    for column in required_columns:
        if column not in train.columns or column not in test.columns:
            raise ValueError(
                f"Missing overlap-check column: {column}"
            )

    train_pairs = set(
        zip(
            train["candidate_text"].fillna(""),
            train["job_text"].fillna("")
        )
    )

    test_pairs = list(
        zip(
            test["candidate_text"].fillna(""),
            test["job_text"].fillna("")
        )
    )

    overlap_mask = np.array([
        pair in train_pairs
        for pair in test_pairs
    ])

    overlap_count = int(overlap_mask.sum())

    model_test = model_test.loc[
        ~overlap_mask
    ].reset_index(drop=True)

    print(f"Original test records: {len(test)}")
    print(f"Overlapping pairs removed: {overlap_count}")
    print(f"Final evaluation records: {len(model_test)}")

    return model_test, overlap_count


def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1]
            )
        ),
        "classification_threshold": THRESHOLD,
        "test_records": int(len(y_true))
    }

    return metrics, predictions


def save_confusion_matrix(y_true, predictions):
    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not Hired", "Hired"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("XGBoost - Test Confusion Matrix")
    plt.tight_layout()

    path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def save_roc_curve(y_true, probabilities):
    display = RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name="XGBoost"
    )

    plt.title("XGBoost - ROC Curve")
    plt.tight_layout()

    path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def save_precision_recall_curve(y_true, probabilities):
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name="XGBoost"
    )

    plt.title("XGBoost - Precision Recall Curve")
    plt.tight_layout()

    path = OUTPUT_DIR / "precision_recall_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def save_calibration_curve(y_true, probabilities):
    CalibrationDisplay.from_predictions(
        y_true,
        probabilities,
        n_bins=10,
        strategy="uniform",
        name="XGBoost"
    )

    plt.title("XGBoost - Probability Calibration")
    plt.tight_layout()

    path = OUTPUT_DIR / "calibration_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def main():
    print("=" * 65)
    print("EMPLOYER-CANDIDATE PROPENSITY MATCHER")
    print("FINAL MODEL EVALUATION")
    print("=" * 65)

    model = load_model()
    features = load_features()

    test, overlap_count = remove_overlapping_pairs()

    missing_features = [
        feature for feature in features
        if feature not in test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing test features: {missing_features}"
        )

    X_test = test[features].copy()
    y_test = test["is_hired"].astype(int)

    if X_test.isnull().any().any():
        raise ValueError(
            "Missing feature values found in test dataset."
        )

    if y_test.nunique() != 2:
        raise ValueError(
            "Test dataset must contain both target classes."
        )

    print("\nGenerating hiring probabilities...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics, predictions = calculate_metrics(
        y_test,
        probabilities
    )

    metrics["overlapping_pairs_removed"] = overlap_count
    metrics["model"] = "Calibrated XGBoost"

    print("\nGenerating evaluation reports...")

    save_confusion_matrix(
        y_test, predictions
    )

    save_roc_curve(
        y_test, probabilities
    )

    save_precision_recall_curve(
        y_test, probabilities
    )

    save_calibration_curve(
        y_test, probabilities
    )

    # Save individual predictions
    prediction_df = pd.DataFrame({
        "actual_is_hired": y_test.to_numpy(),
        "predicted_is_hired": predictions,
        "hiring_propensity": probabilities,
        "hiring_propensity_percentage": (
            probabilities * 100
        ).round(2)
    })

    prediction_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path, index=False
    )

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Hired", "Hired"],
        output_dict=True,
        zero_division=0
    )

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w"
    ) as file:
        json.dump(report, file, indent=4)

    # Save evaluation metrics
    with open(
        OUTPUT_DIR / "final_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Print results
    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 65)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:35s}: {value:.4f}")
        else:
            print(f"{metric:35s}: {value}")

    print("\nCONFUSION MATRIX")
    print("-" * 65)
    print(
        confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not Hired", "Hired"],
            zero_division=0
        )
    )

    print("\nSaved evaluation files:")
    print(OUTPUT_DIR)

    print("\nFINAL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
