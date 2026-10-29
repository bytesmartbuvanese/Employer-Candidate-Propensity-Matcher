
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_FILE = DATA_DIR / "cleaned_train.csv"
TEST_FILE = DATA_DIR / "cleaned_test.csv"

# Initial skill vocabulary.
# Expand this list as more skills appear in your recruitment data.
SKILL_ALIASES = {
    "automation testing": ["automation testing", "automationtesting"],
    "manual testing": ["manual testing", "manualtesting"],
    "selenium": ["selenium"],
    "cypress": ["cypress"],
    "bug tracking": ["bug tracking", "bugtracking"],
    "jenkins": ["jenkins", "jenkinsci"],
    "ci/cd": ["ci/cd", "cicd"],
    "terraform": ["terraform"],
    "kubernetes": ["kubernetes"],
    "aws": ["aws"],
    "linux": ["linux"],
    "software testing": ["software testing", "softwaretesting"],
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "sql": ["sql"],
    "excel": ["excel"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "business analysis": ["business analysis", "businessanalysis"],
    "requirements gathering": [
        "requirements gathering",
        "requirementsgathering"
    ],
    "process modeling": ["process modeling", "processmodeling"],
    "figma": ["figma"],
    "sketch": ["sketch"],
    "wireframing": ["wireframing"],
    "prototyping": ["prototyping"],
    "user research": ["user research", "userresearch"],
    "leadership": ["leadership"],
    "auditing": ["auditing"],
    "financial reporting": ["financial reporting", "financialreporting"],
    "tax compliance": ["tax compliance", "taxcompliance"],
}


def normalize_for_matching(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    normalized = normalize_for_matching(text)

    found = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            alias_normalized = normalize_for_matching(alias)

            # Match complete normalized phrases or concatenated forms.
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias_normalized).replace(r"\ ", r"\s*")
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, normalized):
                found.add(canonical_skill)
                break

    return sorted(found)


def extract_features(df):
    df = df.copy()

    df["candidate_skills"] = df["candidate_text"].apply(extract_skills)
    df["job_skills"] = df["job_text"].apply(extract_skills)

    df["candidate_skill_count"] = df["candidate_skills"].apply(len)
    df["job_skill_count"] = df["job_skills"].apply(len)

    df["matched_skills"] = df.apply(
        lambda row: sorted(
            set(row["candidate_skills"]).intersection(
                set(row["job_skills"])
            )
        ),
        axis=1
    )

    df["skill_overlap_count"] = df["matched_skills"].apply(len)

    df["skill_match_ratio"] = df.apply(
        lambda row: (
            row["skill_overlap_count"] / row["job_skill_count"]
            if row["job_skill_count"] > 0
            else 0.0
        ),
        axis=1
    )

    df["experience_years"] = pd.to_numeric(
        df["experience_years"], errors="coerce"
    )

    df["candidate_text_length"] = (
        df["candidate_text"].fillna("").astype(str).str.len()
    )

    df["job_text_length"] = (
        df["job_text"].fillna("").astype(str).str.len()
    )

    return df


def calculate_similarity(train_df, test_df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    # Fit vocabulary on training data only.
    train_documents = (
        train_df["candidate_text"].fillna("").astype(str)
        + " "
        + train_df["job_text"].fillna("").astype(str)
    )

    test_documents = (
        test_df["candidate_text"].fillna("").astype(str)
        + " "
        + test_df["job_text"].fillna("").astype(str)
    )

    vectorizer.fit(train_documents)

    train_candidate = vectorizer.transform(
        train_df["candidate_text"].fillna("").astype(str)
    )
    train_job = vectorizer.transform(
        train_df["job_text"].fillna("").astype(str)
    )

    test_candidate = vectorizer.transform(
        test_df["candidate_text"].fillna("").astype(str)
    )
    test_job = vectorizer.transform(
        test_df["job_text"].fillna("").astype(str)
    )

    train_df["text_similarity"] = cosine_similarity(
        train_candidate, train_job
    ).diagonal()

    test_df["text_similarity"] = cosine_similarity(
        test_candidate, test_job
    ).diagonal()

    return train_df, test_df


def main():
    print("Loading cleaned datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Extracting normalized skills...")

    train_df = extract_features(train_df)
    test_df = extract_features(test_df)

    print("Calculating TF-IDF similarity...")

    train_df, test_df = calculate_similarity(
        train_df, test_df
    )

    train_output = DATA_DIR / "matching_features_train.csv"
    test_output = DATA_DIR / "matching_features_test.csv"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    feature_columns = [
        "skill_overlap_count",
        "skill_match_ratio",
        "candidate_skill_count",
        "job_skill_count",
        "experience_years",
        "candidate_text_length",
        "job_text_length",
        "text_similarity",
    ]

    print("\nFEATURE EXTRACTION COMPLETED")
    print("-----------------------------------")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nFEATURE SUMMARY")
    print(train_df[feature_columns].describe().round(3))

    print("\nEXAMPLE SKILL MATCH")
    example = train_df.iloc[0]

    print("Candidate skills:", example["candidate_skills"])
    print("Job skills:", example["job_skills"])
    print("Matched skills:", example["matched_skills"])
    print("Match ratio:", example["skill_match_ratio"])

    print("\nTARGET DISTRIBUTION")
    print(train_df["is_hired"].value_counts())

    print("\nOUTPUT FILES")
    print(train_output)
    print(test_output)


if __name__ == "__main__":
    main()
