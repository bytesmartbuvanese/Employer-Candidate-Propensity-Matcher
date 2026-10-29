
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
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
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "model_train.csv"
TEST_FILE = DATA_DIR / "model_test.csv"

CLEAN_TRAIN_FILE = DATA_DIR / "cleaned_train.csv"
CLEAN_TEST_FILE = DATA_DIR / "cleaned_test.csv"

CONFIG_FILE = DATA_DIR / "feature_config.json"

TARGET = "is_hired"
RANDOM_STATE = 42


def load_data():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    clean_train = pd.read_csv(CLEAN_TRAIN_FILE)
    clean_test = pd.read_csv(CLEAN_TEST_FILE)

    with open(CONFIG_FILE) as file:
        config = json.load(file)

    features = config["features"]

    if len(train) != len(clean_train):
        raise ValueError("Training row alignment mismatch.")

    if len(test) != len(clean_test):
        raise ValueError("Testing row alignment mismatch.")

    # Remove exact candidate-job pairs already seen in training.
    train_pairs = set(zip(
        clean_train["candidate_text"].fillna("").astype(str),
        clean_train["job_text"].fillna("").astype(str),
    ))

    test_pairs = list(zip(
        clean_test["candidate_text"].fillna("").astype(str),
        clean_test["job_text"].fillna("").astype(str),
    ))

    overlap_mask = [
        pair in train_pairs for pair in test_pairs
    ]

    overlap_count = sum(overlap_mask)

    test = test.loc[
        ~np.array(overlap_mask)
    ].reset_index(drop=True)

    print("Overlapping test pairs excluded:", overlap_count)

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test, features


def get_models():

    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "XGBoost": xgboost_model,
    }


def calibrate_model(model, X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method="sigmoid",
            cv=cv,
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=model,
            method="sigmoid",
            cv=cv,
        )

    calibrated.fit(X_train, y_train)

    return calibrated


def evaluate_model(name, model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "brier_score": brier_score_loss(
            y_test, probabilities
        ),
        "log_loss": log_loss(
            y_test, probabilities
        ),
    }

    return metrics, probabilities, predictions


def main():

    print("Loading datasets...")

    X_train, y_train, X_test, y_test, features = load_data()

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of features:", len(features))

    models = get_models()

    results = []
    trained_models = {}
    prediction_results = {}

    for name, base_model in models.items():

        print("\n" + "=" * 55)
        print("TRAINING:", name)
        print("=" * 55)

        calibrated_model = calibrate_model(
            base_model,
            X_train,
            y_train,
        )

        metrics, probabilities, predictions = evaluate_model(
            name,
            calibrated_model,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained_models[name] = calibrated_model

        prediction_results[name] = {
            "probabilities": probabilities,
            "predictions": predictions,
        }

        print("\nEvaluation metrics:")

        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Not Hired", "Hired"],
            zero_division=0,
        ))

    # Compare all models.
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by=["roc_auc", "pr_auc", "brier_score"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    comparison_file = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))

    # Select the best model according to ranking.
    best_model_name = comparison.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    best_model_file = MODEL_DIR / "best_hiring_propensity_model.pkl"

    joblib.dump(best_model, best_model_file)

    # Save the predictions of the selected model.
    best_probabilities = prediction_results[
        best_model_name
    ]["probabilities"]

    best_predictions = prediction_results[
        best_model_name
    ]["predictions"]

    predictions_df = pd.DataFrame({
        "actual_is_hired": y_test.to_numpy(),
        "predicted_is_hired": best_predictions,
        "hiring_propensity": best_probabilities,
        "hiring_propensity_percent": (
            best_probabilities * 100
        ),
    })

    predictions_df.to_csv(
        MODEL_DIR / "best_model_predictions.csv",
        index=False,
    )

    # Save model comparison report.
    report = {
        "selected_model": best_model_name,
        "selection_criteria": [
            "ROC-AUC descending",
            "PR-AUC descending",
            "Brier score ascending",
        ],
        "features": features,
        "comparison": comparison.to_dict(orient="records"),
    }

    with open(
        MODEL_DIR / "model_selection_report.json", "w"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("BEST MODEL SELECTED:", best_model_name)
    print("=" * 70)

    print("Saved model:", best_model_file)
    print("Comparison:", comparison_file)
    print("Predictions:", MODEL_DIR / "best_model_predictions.csv")
    print("Selection report:", MODEL_DIR / "model_selection_report.json")


if __name__ == "__main__":
    main()
