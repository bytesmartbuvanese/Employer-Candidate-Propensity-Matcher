
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "dataset_analysis_report.txt"

FILES = {
    "candidate_matching_resumes": BASE_DIR / "candidate_matching_resumes.csv",
    "recruitment_data_train": BASE_DIR / "recruitment_data_train.csv",
    "recruitment_data_test": BASE_DIR / "recruitment_data_test.csv",
    "applications": BASE_DIR / "job_board_applications_and_interviews" /
        "job-board-applications-and-interviews-dataset-v1-csv-applications.csv",
    "applicants": BASE_DIR / "job_board_applications_and_interviews" /
        "job-board-applications-and-interviews-dataset-v1-csv-applicants.csv",
    "jobs": BASE_DIR / "job_board_applications_and_interviews" /
        "job-board-applications-and-interviews-dataset-v1-csv-jobs.csv",
    "application_stages": BASE_DIR / "job_board_applications_and_interviews" /
        "job-board-applications-and-interviews-dataset-v1-csv-application_stages.csv",
    "companies": BASE_DIR / "job_board_applications_and_interviews" /
        "job-board-applications-and-interviews-dataset-v1-csv-companies.csv",
}


def write_line(file, text=""):
    file.write(str(text) + "\n")


def load_csv(path):
    """Read CSV while attempting to detect its delimiter."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)


def classify_column(column):
    name = column.lower()

    if any(x in name for x in ["shortlist", "short_list", "screening"]):
        return "SHORTLIST_OUTCOME"

    if any(x in name for x in ["selected", "hired", "hire", "offer"]):
        return "FINAL_SELECTION_OUTCOME"

    if any(x in name for x in ["reject", "status", "stage", "outcome", "decision"]):
        return "RECRUITMENT_STATUS_OR_OUTCOME"

    if any(x in name for x in ["skill", "technology", "competency"]):
        return "SKILL_INFORMATION"

    if any(x in name for x in ["experience", "years", "seniority"]):
        return "EXPERIENCE"

    if any(x in name for x in ["education", "degree", "qualification"]):
        return "EDUCATION"

    if any(x in name for x in ["resume", "cv", "description", "profile", "text"]):
        return "TEXT_OR_PROFILE_INFORMATION"

    if any(x in name for x in ["salary", "compensation", "pay"]):
        return "SALARY_INFORMATION"

    if any(x in name for x in ["date", "time", "created", "updated"]):
        return "DATE_OR_TIME"

    if name == "id" or name.endswith("_id") or name.startswith("id_"):
        return "POSSIBLE_IDENTIFIER"

    return "OTHER_FEATURE"


def analyze_dataframe(name, df, report):
    write_line(report, "\n" + "=" * 90)
    write_line(report, f"DATASET: {name}")
    write_line(report, "=" * 90)

    rows, columns = df.shape

    write_line(report, f"Rows: {rows:,}")
    write_line(report, f"Columns: {columns}")
    write_line(report, f"Duplicate rows: {df.duplicated().sum():,}")
    write_line(report, f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    write_line(report, "\n--- COLUMN INFORMATION ---")

    for col in df.columns:
        series = df[col]

        missing = series.isna().sum()
        missing_pct = (missing / len(df) * 100) if len(df) else 0

        write_line(
            report,
            f"\nColumn: {col}"
            f"\n  Type: {series.dtype}"
            f"\n  Missing: {missing:,} ({missing_pct:.2f}%)"
            f"\n  Unique values: {series.nunique(dropna=True):,}"
            f"\n  Possible role: {classify_column(col)}"
        )

        if series.nunique(dropna=True) <= 20:
            write_line(report, "  Value distribution:")
            counts = series.value_counts(dropna=False).head(20)

            for value, count in counts.items():
                write_line(report, f"    {value}: {count}")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols):
        write_line(report, "\n--- NUMERIC STATISTICS ---")
        write_line(report, df[numeric_cols].describe().to_string())

    write_line(report, "\n--- FIRST 5 ROWS ---")
    write_line(report, df.head(5).to_string(index=False))

    write_line(report, "\n--- POSSIBLE IDENTIFIER COLUMNS ---")

    id_cols = [
        col for col in df.columns
        if col.lower() == "id"
        or col.lower().endswith("_id")
        or col.lower().startswith("id_")
    ]

    write_line(report, ", ".join(id_cols) if id_cols else "No obvious identifier columns")

    write_line(report, "\n--- POSSIBLE MODEL TARGET COLUMNS ---")

    target_cols = [
        col for col in df.columns
        if classify_column(col) in [
            "SHORTLIST_OUTCOME",
            "FINAL_SELECTION_OUTCOME",
            "RECRUITMENT_STATUS_OR_OUTCOME",
        ]
    ]

    write_line(report, ", ".join(target_cols) if target_cols else "No obvious target columns")


def analyze_relationships(dataframes, report):
    write_line(report, "\n\n" + "#" * 90)
    write_line(report, "CROSS-DATASET RELATIONSHIP ANALYSIS")
    write_line(report, "#" * 90)

    pairs = [
        ("applications", "applicants"),
        ("applications", "jobs"),
        ("applications", "application_stages"),
        ("jobs", "companies"),
    ]

    for left_name, right_name in pairs:
        if left_name not in dataframes or right_name not in dataframes:
            continue

        left = dataframes[left_name]
        right = dataframes[right_name]

        write_line(report, f"\n{left_name} <-> {right_name}")

        common = sorted(set(left.columns) & set(right.columns))

        write_line(report, f"Common columns: {common}")

        possible_keys = [
            col for col in common
            if col.lower() == "id"
            or col.lower().endswith("_id")
            or col.lower().startswith("id_")
        ]

        if not possible_keys:
            write_line(report, "No obvious common identifier found.")
            continue

        for key in possible_keys:
            left_values = set(left[key].dropna().astype(str))
            right_values = set(right[key].dropna().astype(str))

            overlap = left_values & right_values

            write_line(
                report,
                f"Key: {key}"
                f"\n  Unique values in {left_name}: {len(left_values):,}"
                f"\n  Unique values in {right_name}: {len(right_values):,}"
                f"\n  Matching values: {len(overlap):,}"
            )


def analyze_propensity_readiness(dataframes, report):
    write_line(report, "\n\n" + "#" * 90)
    write_line(report, "EMPLOYER-CANDIDATE PROPENSITY MODEL READINESS")
    write_line(report, "#" * 90)

    for name, df in dataframes.items():
        columns = [c.lower() for c in df.columns]

        shortlist_candidates = [
            c for c in columns
            if "shortlist" in c or "short_list" in c
        ]

        selection_candidates = [
            c for c in columns
            if any(x in c for x in ["selected", "hired", "hire", "offer"])
        ]

        application_candidates = [
            c for c in columns
            if "application" in c
        ]

        write_line(report, f"\nDataset: {name}")
        write_line(report, f"Shortlist-related columns: {shortlist_candidates}")
        write_line(report, f"Selection-related columns: {selection_candidates}")
        write_line(report, f"Application-related columns: {application_candidates}")

        if shortlist_candidates:
            write_line(
                report,
                "Potential use: investigate for a shortlisting prediction target."
            )

        if selection_candidates:
            write_line(
                report,
                "Potential use: investigate for a final selection prediction target."
            )

        if not shortlist_candidates and not selection_candidates:
            write_line(
                report,
                "Warning: no obvious shortlist or final selection target detected."
            )

    write_line(report, "\nIMPORTANT:")
    write_line(
        report,
        "- Application, click, bookmark, and interview activity do not automatically "
        "represent a shortlist outcome."
    )
    write_line(
        report,
        "- Verify the actual meaning of every outcome/status column before using it as a target."
    )
    write_line(
        report,
        "- Do not use information recorded after a recruitment decision as model input."
    )
    write_line(
        report,
        "- Synthetic data can test the pipeline but cannot establish real-world "
        "shortlisting probabilities."
    )
    write_line(
        report,
        "- A calibrated propensity score requires representative historical outcomes."
    )


def main():
    dataframes = {}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as report:
        write_line(report, "EMPLOYER-CANDIDATE PROPENSITY MATCHER")
        write_line(report, "DATASET ANALYSIS REPORT")
        write_line(report, "=" * 90)

        write_line(report, f"Project directory: {BASE_DIR}")
        write_line(report, f"Total expected datasets: {len(FILES)}")

        for name, path in FILES.items():
            if not path.exists():
                write_line(report, f"\n[NOT FOUND] {name}: {path}")
                continue

            try:
                df = load_csv(path)
                dataframes[name] = df

                write_line(
                    report,
                    f"\n[LOADED] {name}: {df.shape[0]:,} rows x {df.shape[1]} columns"
                )

                analyze_dataframe(name, df, report)

            except Exception as error:
                write_line(report, f"\n[ERROR] {name}: {error}")

        analyze_relationships(dataframes, report)
        analyze_propensity_readiness(dataframes, report)

        write_line(report, "\n\n" + "=" * 90)
        write_line(report, "ANALYSIS SUMMARY")
        write_line(report, "=" * 90)
        write_line(report, f"Successfully loaded: {len(dataframes)} / {len(FILES)} datasets")

        for name, df in dataframes.items():
            write_line(report, f"{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("\nDataset analysis completed.")
    print(f"Successfully loaded: {len(dataframes)} / {len(FILES)} datasets")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
