# Employer-Candidate Propensity Matcher

A Python-based hiring propensity prediction project that estimates how well a candidate matches a job description using resume text, job description text, and years of experience.

The project combines text feature extraction, skill matching, and a trained machine learning model to produce a hiring propensity score and explainable match details.

## Overview

This repository contains:

- a resume-to-job matching pipeline
- feature engineering for skill overlap and text similarity
- a trained hiring propensity model
- a FastAPI service for prediction requests
- CLI utilities for running predictions locally

## Project Goals

- Assess whether a candidate is likely to be hired for a role
- Compare candidate skills against required job skills
- Surface matched and unmatched skills
- Incorporate experience, skill overlap, and text similarity into the score

## Repository Structure

```text
.
├── api/
│   ├── main.py          # FastAPI app
│   └── schemas.py       # Request/response models
├── data/
│   ├── processed/       # processed model data and artifacts
│   ├── analyze_datasets.py
│   ├── candidate_matching_resumes.csv
│   ├── recruitment_data_train.csv
│   └── recruitment_data_test.csv
├── model_evaluvation/   # model evaluation assets
├── models/              # trained model artifacts
├── preprocessing/       # preprocessing utilities
├── src/
│   └── employer_candidate_propensity_matcher/
├── build_training_dataset.py
├── download_datasets.py
├── predict_hiring_propensity.py
├── pyproject.toml
├── sample_input_.txt
├── uv.lock
├── README.md
└── .python-version
```

## Tech Stack

- Python 3.12+
- scikit-learn
- pandas
- numpy
- joblib
- FastAPI
- Uvicorn
- XGBoost
- matplotlib
- shap
- datasets

## Setup

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the project metadata provided in `pyproject.toml`, the package can also be installed with:

```bash
pip install -e .
```

## Run the prediction CLI

The core prediction logic lives in `predict_hiring_propensity.py`.

```bash
python predict_hiring_propensity.py
```

The script will prompt for:

- candidate resume/profile text
- candidate years of experience
- job description text

It then prints:

- predicted outcome
- hiring propensity percentage
- candidate skills
- job-required skills
- matched skills
- unmatched job skills
- model feature values

## Run the API

Start the FastAPI app:

```bash
uvicorn api.main:app --reload
```

The service exposes:

- `GET /` - API metadata
- `GET /health` - health check
- `POST /predict` - predict hiring propensity

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with 3 years of experience. Skilled in Python, SQL, Django, REST APIs, PostgreSQL, Docker and AWS.",
    "job_description": "We are hiring a Python Backend Developer with 2+ years of experience. Required skills include Python, Django, REST API development, SQL, PostgreSQL, Docker and AWS.",
    "experience_years": 3
  }'
```

## Prediction Input Fields

The prediction endpoint expects:

- `resume_text`: candidate resume or profile summary
- `job_description`: full job description text
- `experience_years`: candidate experience in years

## Model Behavior

The prediction pipeline:

1. normalizes resume and job text
2. extracts skills using canonical aliases and keyword patterns
3. measures skill overlap and unmatched required skills
4. computes text similarity between candidate and job text
5. derives experience-related metrics
6. loads the trained model and feature configuration
7. returns a predicted hiring propensity and breakdown

## Sample Data

A sample request payload is available in `sample_input_.txt`.

## Notes

- The model expects the processed data and artifact files to be available under `data/processed` and `models/`.
- The project is configured for Python 3.12 and uses the local package structure under `src/`.
- The API is designed for local or lightweight backend deployment and can be extended for web or dashboard integration.

## License

This project does not currently declare a license in `pyproject.toml`. If you plan to share or distribute the repository publicly, consider adding an explicit license file.

## Contact

Developer: bytesmartbuvanese

Email: bytesmart.buvanese@gmail.com
