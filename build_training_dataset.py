
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_FILE = DATA_DIR / "matching_features_train.csv"
TEST_FILE = DATA_DIR / "matching_features_test.csv"

# Features approved for the initial XGBoost baseline.
BASE_FEATURES = [
    "skill_overlap_count",
    "skill_match_ratio",
    "candidate_skill_count",
    "job_skill_count",
    "experience_years",
    "candidate_text_length",
    "job_text_length",
    "text_similarity",
]

TARGET = "is_hired"


def extract_required_experience(text):
    """Extract required years of experience from job text."""

    if pd.isna(text):
        return np.nan

    text = str(text).lower()

    patterns = [
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*"
        r"(?:years?|yrs?)\s+(?:of\s+)?"
        r"(?:relevant\s+)?experience\b",

        r"\bexperience\s+(?:of|:)\s*"
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return float(match.group(1))

    return np.nan


def prepare_dataset(df, dataset_name):
    """Validate and prepare model features."""

    df = df.copy()

    required_columns = BASE_FEATURES + [
        "candidate_text",
        "job_text",
        TARGET,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: Missing columns: {missing}"
        )

    # Validate target.
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="raise"
    )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: Target must contain only 0 and 1."
        )

    # Extract job experience requirement.
    df["required_experience_years"] = (
        df["job_text"].apply(extract_required_experience)
    )

    df["experience_requirement_found"] = (
        df["required_experience_years"].notna().astype(int)
    )

    df["experience_gap"] = (
        df["experience_years"]
        - df["required_experience_years"]
    )

    # Final numerical feature list.
    feature_columns = BASE_FEATURES + [
        "required_experience_years",
        "experience_gap",
        "experience_requirement_found",
    ]

    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf], np.nan
        )

    # Return only model features and target.
    return df[feature_columns + [TARGET]], feature_columns


def main():

    print("Loading feature datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Validating and selecting features...")

    train_data, feature_columns = prepare_dataset(
        train_df, "TRAIN"
    )

    test_data, test_features = prepare_dataset(
        test_df, "TEST"
    )

    if feature_columns != test_features:
        raise ValueError("Train/test feature mismatch.")

    # Detect exact candidate-job text pairs appearing in both splits.
    train_pairs = set(
        zip(
            train_df["candidate_text"].fillna("").astype(str),
            train_df["job_text"].fillna("").astype(str),
        )
    )

    test_pairs = set(
        zip(
            test_df["candidate_text"].fillna("").astype(str),
            test_df["job_text"].fillna("").astype(str),
        )
    )

    overlapping_pairs = len(train_pairs.intersection(test_pairs))

    # Calculate imputation values using training data only.
    medians = {}

    for col in feature_columns:
        median_value = train_data[col].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Feature '{col}' has no usable training values."
            )

        medians[col] = float(median_value)

        train_data[col] = train_data[col].fillna(median_value)
        test_data[col] = test_data[col].fillna(median_value)

    # Final validation.
    if train_data.isnull().any().any():
        raise ValueError("Missing values remain in training data.")

    if test_data.isnull().any().any():
        raise ValueError("Missing values remain in test data.")

    # Save final datasets.
    train_output = DATA_DIR / "model_train.csv"
    test_output = DATA_DIR / "model_test.csv"

    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)

    # Save feature configuration for future inference.
    config = {
        "features": feature_columns,
        "target": TARGET,
        "imputation_medians": medians,
        "scaling_required": False,
    }

    config_output = DATA_DIR / "feature_config.json"

    with open(config_output, "w") as file:
        json.dump(config, file, indent=4)

    # Generate summary.
    summary = [
        "TRAINING DATASET BUILD SUMMARY",
        "=" * 55,
        "",
        f"Training rows: {len(train_data)}",
        f"Testing rows: {len(test_data)}",
        f"Number of features: {len(feature_columns)}",
        f"Overlapping candidate-job pairs: {overlapping_pairs}",
        "",
        "FEATURES",
        "-" * 55,
    ]

    summary.extend(feature_columns)

    summary.extend([
        "",
        "TRAIN TARGET DISTRIBUTION",
        "-" * 55,
        str(train_data[TARGET].value_counts().sort_index()),
        "",
        "TEST TARGET DISTRIBUTION",
        "-" * 55,
        str(test_data[TARGET].value_counts().sort_index()),
        "",
        "MISSING VALUES AFTER IMPUTATION",
        "-" * 55,
        f"Train: {train_data.isnull().sum().sum()}",
        f"Test: {test_data.isnull().sum().sum()}",
        "",
        "OUTPUT FILES",
        "-" * 55,
        str(train_output),
        str(test_output),
        str(config_output),
    ])

    summary_text = "\n".join(summary)

    summary_output = DATA_DIR / "training_dataset_summary.txt"
    summary_output.write_text(summary_text)

    print("\n" + summary_text)

    print("\nTRAINING DATASET BUILD COMPLETED.")


if __name__ == "__main__":
    main()
