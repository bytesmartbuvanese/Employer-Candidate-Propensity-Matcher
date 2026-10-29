
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictionRequest(BaseModel):
    resume_text: str = Field(
        ..., min_length=20,
        description="Candidate resume or profile text"
    )

    job_description: str = Field(
        ..., min_length=20,
        description="Complete job description"
    )

    experience_years: float = Field(
        ..., ge=0,
        description="Candidate experience in years"
    )


class PredictionResponse(BaseModel):
    predicted_outcome: str
    hiring_propensity: float
    probability: float
    classification_threshold: float

    candidate_skills: List[str]
    job_required_skills: List[str]
    matched_skills: List[str]
    unmatched_job_skills: List[str]

    experience_years: float
    required_experience_years: Optional[float]

    feature_values: Dict[str, Any]
