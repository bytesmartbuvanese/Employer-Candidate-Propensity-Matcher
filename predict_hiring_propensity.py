
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "best_hiring_propensity_model.pkl"
CONFIG_PATH = DATA_DIR / "feature_config.json"
TRAIN_PATH = DATA_DIR / "cleaned_train.csv"


# ---------------------------------------------------------
# 1. Skill vocabulary
# ---------------------------------------------------------

SKILL_ALIASES = {
    "automation testing": [
        "automation testing", "automationtesting"
    ],
    "manual testing": [
        "manual testing", "manualtesting"
    ],
    "selenium": ["selenium"],
    "cypress": ["cypress"],
    "bug tracking": [
        "bug tracking", "bugtracking"
    ],
    "jenkins": ["jenkins"],
    "ci/cd": [
        "ci/cd", "cicd", "continuous integration",
        "continuous deployment"
    ],
    "terraform": ["terraform"],
    "kubernetes": ["kubernetes"],
    "aws": ["aws", "amazon web services"],
    "linux": ["linux"],
    "software testing": [
        "software testing", "softwaretesting"
    ],
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "sql": ["sql"],
    "excel": ["excel", "microsoft excel"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "business analysis": [
        "business analysis", "businessanalysis"
    ],
    "requirements gathering": [
        "requirements gathering",
        "requirementsgathering"
    ],
    "process modeling": [
        "process modeling", "processmodeling"
    ],
    "figma": ["figma"],
    "sketch": ["sketch"],
    "wireframing": ["wireframing"],
    "prototyping": ["prototyping"],
    "user research": [
        "user research", "userresearch"
    ],
    "leadership": ["leadership"],
    "auditing": ["auditing"],
    "financial reporting": [
        "financial reporting", "financialreporting"
    ],
    "tax compliance": [
        "tax compliance", "taxcompliance"
    ]
}


# ---------------------------------------------------------
# 2. Text processing
# ---------------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_matching(text):
    text = clean_text(text)
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 3. Skill extraction
# ---------------------------------------------------------

def extract_skills(text):
    normalized_text = normalize_for_matching(text)

    found_skills = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            normalized_alias = normalize_for_matching(alias)

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(normalized_alias).replace(r"\ ", r"\s*")
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, normalized_text):
                found_skills.add(canonical_skill)
                break

    return sorted(found_skills)


# ---------------------------------------------------------
# 4. Required experience extraction
# ---------------------------------------------------------

def extract_required_experience(job_text):

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*"
        r"(?:years?|yrs?)\s*(?:of\s+)?"
        r"(?:relevant\s+)?experience",

        r"experience\s*(?:of|:)\s*"
        r"(\d+(?:\.\d+)?)\+?\s*"
        r"(?:years?|yrs?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            job_text,
            flags=re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    return np.nan


# ---------------------------------------------------------
# 5. Load model and training configuration
# ---------------------------------------------------------

def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    training_data = pd.read_csv(TRAIN_PATH)

    required_columns = [
        "candidate_text",
        "job_text"
    ]

    for column in required_columns:
        if column not in training_data.columns:
            raise ValueError(
                f"Missing column in cleaned_train.csv: {column}"
            )

    return model, config, training_data


# ---------------------------------------------------------
# 6. Rebuild TF-IDF vectorizer
# ---------------------------------------------------------

def build_vectorizer(training_data):

    candidate_documents = (
        training_data["candidate_text"]
        .fillna("")
        .apply(clean_text)
        .tolist()
    )

    job_documents = (
        training_data["job_text"]
        .fillna("")
        .apply(clean_text)
        .tolist()
    )

    # Match the original training vectorizer corpus:
    # one combined candidate-job document per training row.
    training_documents = [
        candidate + " " + job
        for candidate, job in zip(
            candidate_documents,
            job_documents
        )
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    vectorizer.fit(training_documents)

    return vectorizer


# ---------------------------------------------------------
# 7. Generate the 11 model features
# ---------------------------------------------------------

def generate_features(
    resume_text,
    job_description,
    experience_years,
    vectorizer
):

    candidate_text = clean_text(resume_text)
    job_text = clean_text(job_description)

    candidate_skills = extract_skills(candidate_text)
    job_skills = extract_skills(job_text)

    matched_skills = sorted(
        set(candidate_skills).intersection(job_skills)
    )

    skill_overlap_count = len(matched_skills)

    candidate_skill_count = len(candidate_skills)
    job_skill_count = len(job_skills)

    skill_match_ratio = (
        skill_overlap_count / job_skill_count
        if job_skill_count > 0
        else 0.0
    )

    required_experience_years = (
        extract_required_experience(job_text)
    )

    experience_requirement_found = (
        1 if not np.isnan(required_experience_years)
        else 0
    )

    experience_gap = (
        float(experience_years) - required_experience_years
        if experience_requirement_found
        else np.nan
    )

    candidate_vector = vectorizer.transform(
        [candidate_text]
    )

    job_vector = vectorizer.transform(
        [job_text]
    )

    text_similarity = float(
        cosine_similarity(
            candidate_vector,
            job_vector
        )[0][0]
    )

    features = {
        "skill_overlap_count": skill_overlap_count,
        "skill_match_ratio": skill_match_ratio,
        "candidate_skill_count": candidate_skill_count,
        "job_skill_count": job_skill_count,
        "experience_years": float(experience_years),
        "candidate_text_length": len(candidate_text),
        "job_text_length": len(job_text),
        "text_similarity": text_similarity,
        "required_experience_years": required_experience_years,
        "experience_gap": experience_gap,
        "experience_requirement_found": experience_requirement_found
    }

    return features, candidate_skills, job_skills, matched_skills


# ---------------------------------------------------------
# 8. Apply training-set imputation medians
# ---------------------------------------------------------

def prepare_model_input(features, config):

    feature_names = config["features"]

    medians = config.get("imputation_medians", {})

    input_data = {}

    for feature in feature_names:

        value = features[feature]

        if pd.isna(value):

            if feature not in medians:
                raise ValueError(
                    f"No training median available for {feature}"
                )

            value = medians[feature]

        input_data[feature] = value

    return pd.DataFrame(
        [input_data],
        columns=feature_names
    )


# ---------------------------------------------------------
# 9. Predict hiring propensity
# ---------------------------------------------------------

def predict_hiring_propensity(
    resume_text,
    job_description,
    experience_years
):

    if not str(resume_text).strip():
        raise ValueError("Resume text cannot be empty.")

    if not str(job_description).strip():
        raise ValueError("Job description cannot be empty.")

    if float(experience_years) < 0:
        raise ValueError("Experience cannot be negative.")

    model, config, training_data = load_artifacts()

    vectorizer = build_vectorizer(training_data)

    features, candidate_skills, job_skills, matched_skills = (
        generate_features(
            resume_text,
            job_description,
            experience_years,
            vectorizer
        )
    )

    model_input = prepare_model_input(
        features,
        config
    )

    probability = float(
        model.predict_proba(model_input)[0][1]
    )

    result = {
        "predicted_outcome": (
            "Hired" if probability >= 0.5
            else "Not Hired"
        ),
        "hiring_propensity": round(probability * 100, 2),
        "probability": round(probability, 4),
        "classification_threshold": 0.50,
        "candidate_skills": candidate_skills,
        "job_required_skills": job_skills,
        "matched_skills": matched_skills,
        "unmatched_job_skills": sorted(
            set(job_skills) - set(candidate_skills)
        ),
        "experience_years": float(experience_years),
        "required_experience_years": (
            None
            if np.isnan(features["required_experience_years"])
            else features["required_experience_years"]
        ),
        "feature_values": {
            key: (
                None if pd.isna(value) else value
            )
            for key, value in features.items()
        }
    }

    return result

# ---------------------------------------------------------
# 10. reader function
# ---------------------------------------------------------
def read_multiline_input(prompt):
    print(prompt)
    print("Type END on a new line when finished.\n")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines)
# ---------------------------------------------------------
# 10. Interactive execution
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("EMPLOYER-CANDIDATE PROPENSITY MATCHER")
    print("CANDIDATE-JOB HIRING PROPENSITY PREDICTION")
    print("=" * 65)

    print("\nEnter candidate information.")

    resume_text = read_multiline_input(
        "\nPaste candidate resume/profile text:"
    )

    experience_years = float(
        input("\nCandidate experience in years: ")
    )

    print("\nEnter job information.")

    job_description = read_multiline_input(
        "\nPaste job description:"
    )

    print("\nGenerating prediction...")

    result = predict_hiring_propensity(
        resume_text,
        job_description,
        experience_years
    )

    print("\n" + "=" * 65)
    print("HIRING PROPENSITY RESULT")
    print("=" * 65)

    print(
        f"\nPredicted outcome: {result['predicted_outcome']}"
    )

    print(
        f"Hiring propensity: "
        f"{result['hiring_propensity']}%"
    )

    print("\nCANDIDATE SKILLS")
    print(result["candidate_skills"])

    print("\nJOB REQUIRED SKILLS")
    print(result["job_required_skills"])

    print("\nMATCHED SKILLS")
    print(result["matched_skills"])

    print("\nUNMATCHED JOB SKILLS")
    print(result["unmatched_job_skills"])

    print("\nFEATURE VALUES")
    for feature, value in result["feature_values"].items():
        print(f"{feature}: {value}")

    output_path = (
        MODEL_DIR / "latest_hiring_prediction.json"
    )

    with open(output_path, "w") as file:
        json.dump(result, file, indent=4)

    print(f"\nPrediction saved: {output_path}")


if __name__ == "__main__":
    main()
