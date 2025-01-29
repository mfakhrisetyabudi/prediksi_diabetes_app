from domain.domain import DiabetesRequest, DiabetesResponse
from service.diabetes_service import DiabetesService
from fastapi import FastAPI

prediction_app = FastAPI()

@prediction_app.post("predict")
async def predict_diabetes(request: DiabetesRequest)->DiabetesResponse:
    return DiabetesService.predict_diabetes(request=request)