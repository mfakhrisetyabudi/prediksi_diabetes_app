from pydantic import BaseModel

class DiabetesRequest(BaseModel):
    gender: str
    age: int
    hypertension: int  
    heart_disease: int 
    smoking_history: int 
    bmi: float 
    HbA1c_level: float 
    blood_glucose_level: int

class DiabetesResponse(BaseModel):
    diabetes: int