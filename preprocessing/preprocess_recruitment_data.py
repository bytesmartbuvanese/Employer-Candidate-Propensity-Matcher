
import pandas as pd
import re
import ast
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "data/recruitment_data_train.csv"
TEST_FILE = BASE_DIR / "data/recruitment_data_test.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "recruitment_date",
    "opening_position",
    "description",
    "requirement",
    "keywords_for_recruitment",
    "cv",
    "experience_years",
    "keywords_from_applicants",
    "result",
]

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(value):
    """Normalize text while preserving useful technical terms."""

    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove unusual symbols but retain common technical punctuation.
    text = re.sub(r"[^\w\s+#./-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_keywords(value):
    """
    Convert keyword lists or strings into normalized text.
    Handles Python-list representations and ordinary text.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            text = " ".join(str(item) for item in parsed)

    except (ValueError, SyntaxError):
        pass

    return clean_text(text)


# =========================================================
# TARGET ENCODING
# =========================================================

def encode_target(value):

    value = str(value).strip().lower()

    mapping = {
        "accepted": 1,
        "rejected": 0,
    }

    if value not in mapping:
        raise ValueError(
            f"Unexpected recruitment outcome: {value}"
        )

    return mapping[value]


# =========================================================
# DATASET PREPROCESSING
# =========================================================

def preprocess_dataset(file_path, dataset_name):

    print(f"\nProcessing {dataset_name}...")

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("Original shape:", df.shape)

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} missing columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # Remove exact duplicate records
    # -----------------------------------------------------

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    # -----------------------------------------------------
    # Handle missing values
    # -----------------------------------------------------

    text_columns = [
        "opening_position",
        "description",
        "requirement",
        "keywords_for_recruitment",
        "cv",
        "keywords_from_applicants",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("")

    df["experience_years"] = pd.to_numeric(
        df["experience_years"],
        errors="coerce"
    )

    missing_experience = df["experience_years"].isna().sum()

    # Do not silently invent missing experience values.
    # Preserve missing values for explicit downstream handling.

    df["recruitment_date"] = pd.to_datetime(
        df["recruitment_date"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Clean candidate and job text
    # -----------------------------------------------------

    df["opening_position_clean"] = (
        df["opening_position"].apply(clean_text)
    )

    df["description_clean"] = (
        df["description"].apply(clean_text)
    )

    df["requirement_clean"] = (
        df["requirement"].apply(clean_text)
    )

    df["cv_clean"] = df["cv"].apply(clean_text)

    df["recruitment_keywords_clean"] = (
        df["keywords_for_recruitment"].apply(normalize_keywords)
    )

    df["applicant_keywords_clean"] = (
        df["keywords_from_applicants"].apply(normalize_keywords)
    )

    # -----------------------------------------------------
    # Encode target
    # -----------------------------------------------------

    df["result"] = df["result"].astype(str).str.strip()

    df["is_hired"] = df["result"].apply(encode_target)

    # -----------------------------------------------------
    # Build candidate and job text representations
    # -----------------------------------------------------

    df["candidate_text"] = (
        df["cv_clean"] + " " +
        df["applicant_keywords_clean"]
    ).str.strip()

    df["job_text"] = (
        df["opening_position_clean"] + " " +
        df["description_clean"] + " " +
        df["requirement_clean"] + " " +
        df["recruitment_keywords_clean"]
    ).str.strip()

    # Combined text for future NLP feature extraction.
    df["candidate_job_text"] = (
        "candidate: " + df["candidate_text"] +
        " job: " + df["job_text"]
    )

    # -----------------------------------------------------
    # Prevent target leakage
    # -----------------------------------------------------

    # reason_for_result may explain the hiring decision.
    # Never use it as a model input.
    if "reason_for_result" in df.columns:
        df = df.drop(columns=["reason_for_result"])

    # -----------------------------------------------------
    # Validate processed data
    # -----------------------------------------------------

    if df["is_hired"].isna().any():
        raise ValueError("Missing target values detected.")

    if not df["is_hired"].isin([0, 1]).all():
        raise ValueError("Invalid target encoding.")

    # -----------------------------------------------------
    # Save processed dataset
    # -----------------------------------------------------

    output_file = OUTPUT_DIR / f"cleaned_{dataset_name}.csv"

    df.to_csv(output_file, index=False)

    print("Processed shape:", df.shape)
    print("Duplicates removed:", duplicates)
    print("Missing experience:", missing_experience)
    print("Target distribution:")
    print(df["is_hired"].value_counts().to_string())
    print("Saved:", output_file)

    return {
        "dataset": dataset_name,
        "original_rows": len(pd.read_csv(file_path)),
        "processed_rows": len(df),
        "duplicates_removed": duplicates,
        "missing_experience": missing_experience,
        "hired": int((df["is_hired"] == 1).sum()),
        "not_hired": int((df["is_hired"] == 0).sum()),
        "output_file": str(output_file),
    }


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    print("=" * 65)
    print("EMPLOYER-CANDIDATE PROPENSITY MATCHER")
    print("RECRUITMENT DATA PREPROCESSING")
    print("=" * 65)

    results = []

    results.append(
        preprocess_dataset(TRAIN_FILE, "train")
    )

    results.append(
        preprocess_dataset(TEST_FILE, "test")
    )

    # -----------------------------------------------------
    # Generate preprocessing report
    # -----------------------------------------------------

    report_file = OUTPUT_DIR / "preprocessing_summary.txt"

    with open(report_file, "w", encoding="utf-8") as file:

        file.write("RECRUITMENT DATA PREPROCESSING SUMMARY\n")
        file.write("=" * 65 + "\n\n")

        for result in results:

            file.write(f"Dataset: {result['dataset']}\n")
            file.write("-" * 45 + "\n")

            for key, value in result.items():
                file.write(f"{key}: {value}\n")

            file.write("\n")

        file.write("\nTARGET DEFINITION\n")
        file.write("Accepted = 1 (Hired)\n")
        file.write("Rejected = 0 (Not hired)\n")

        file.write("\nLEAKAGE PREVENTION\n")
        file.write(
            "reason_for_result excluded from model inputs.\n"
        )
        file.write(
            "result retained for auditing; use is_hired as target.\n"
        )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETED")
    print("=" * 65)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary report: {report_file}")


if __name__ == "__main__":
    main()
