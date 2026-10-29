
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictionRequest,
    PredictionResponse
)

from predict_hiring_propensity import (
    predict_hiring_propensity
)


app = FastAPI(
    title="Employer-Candidate Propensity Matcher",
    description=(
        "Predicts candidate hiring propensity "
        "based on resume and job description."
    ),
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Employer-Candidate Propensity Matcher API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Calibrated XGBoost"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:
        result = predict_hiring_propensity(
            resume_text=request.resume_text,
            job_description=request.job_description,
            experience_years=request.experience_years
        )

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Required model artifact not found: {error}"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
