
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

MODEL_PATH = MODEL_DIR / "best_hiring_propensity_model.pkl"
CONFIG_PATH = DATA_DIR / "feature_config.json"

OUTPUT_DIR = MODEL_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading selected hiring propensity model...")
    return joblib.load(MODEL_PATH)


def load_feature_names():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Feature configuration not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    features = config.get("features")

    if not features:
        raise ValueError(
            "Feature names were not found in feature_config.json"
        )

    return features


def get_base_estimators(model):
    """
    Extract estimators from CalibratedClassifierCV or
    return the original estimator.
    """

    if hasattr(model, "calibrated_classifiers_"):
        estimators = []

        for calibrated_classifier in model.calibrated_classifiers_:
            estimator = getattr(
                calibrated_classifier, "estimator", None
            )

            if estimator is None:
                estimator = getattr(
                    calibrated_classifier, "base_estimator", None
                )

            if estimator is not None:
                estimators.append(estimator)

        return estimators

    return [model]


def extract_feature_importance(model, feature_names):
    estimators = get_base_estimators(model)

    importance_values = []

    for estimator in estimators:
        if hasattr(estimator, "feature_importances_"):
            importance_values.append(
                estimator.feature_importances_
            )

        elif hasattr(estimator, "coef_"):
            importance_values.append(
                np.abs(estimator.coef_).flatten()
            )

        elif hasattr(estimator, "named_steps"):
            # Handles sklearn pipelines such as
            # StandardScaler + LogisticRegression.
            final_estimator = list(
                estimator.named_steps.values()
            )[-1]

            if hasattr(final_estimator, "coef_"):
                importance_values.append(
                    np.abs(final_estimator.coef_).flatten()
                )

    if not importance_values:
        raise ValueError(
            "Could not extract feature importance from the model."
        )

    mean_importance = np.mean(
        np.array(importance_values), axis=0
    )

    if len(mean_importance) != len(feature_names):
        raise ValueError(
            "Feature count does not match model importance count. "
            f"Model: {len(mean_importance)}, "
            f"Features: {len(feature_names)}"
        )

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_importance
    })

    result = result.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    total = result["importance"].sum()

    if total > 0:
        result["importance_percentage"] = (
            result["importance"] / total * 100
        )
    else:
        result["importance_percentage"] = 0

    return result


def plot_feature_importance(importance_df):
    plot_data = importance_df.sort_values(
        "importance_percentage", ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["feature"],
        plot_data["importance_percentage"]
    )

    plt.xlabel("Relative importance (%)")
    plt.ylabel("Features")
    plt.title("Hiring Propensity Model - Feature Importance")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def analyze_feature_statistics():
    train_path = DATA_DIR / "model_train.csv"

    if not train_path.exists():
        print("Training dataset not found. Skipping statistics.")
        return

    df = pd.read_csv(train_path)

    feature_columns = [
        column for column in df.columns
        if column not in ["is_hired"]
    ]

    statistics = df[feature_columns].describe().T
    output_path = OUTPUT_DIR / "feature_statistics.csv"

    statistics.to_csv(output_path)

    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("EMPLOYER-CANDIDATE PROPENSITY MATCHER")
    print("MODEL ANALYSIS")
    print("=" * 60)

    model = load_model()
    feature_names = load_feature_names()

    importance_df = extract_feature_importance(
        model, feature_names
    )

    importance_path = (
        OUTPUT_DIR / "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path, index=False
    )

    plot_feature_importance(importance_df)
    analyze_feature_statistics()

    print("\nTOP 10 IMPORTANT FEATURES")
    print("-" * 60)

    print(
        importance_df.head(10).to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format,
                "importance_percentage": "{:.2f}%".format
            }
        )
    )

    print("\nMODEL ANALYSIS COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
