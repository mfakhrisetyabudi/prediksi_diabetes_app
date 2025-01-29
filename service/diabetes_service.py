import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

from domain.domain import DiabetesRequest, DiabetesResponse

class DiabetesService():
    def __init__(self):
        self.path_model = "artifacts/naive_bayes_diabetes.pkl"
        self.path_encoder = "artifacts/gender_encoder.pkl"
        self.model = self.load_artifact(self.path_model)
        self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self, path_to_artifact):
        '''memuat model dari prediksi NB pickle file'''
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact 

    def preprocess_input(self, request: DiabetesRequest)->pd.DataFrame:
        data_dict = {
            "gender": request.gender,
            "age": request.age,
            "hypertension": request.hypertension,
            "heart_disease": request.heart_disease,
            "smoking_history": request.smoking_history ,
            "bmi": request.bmi,"HbA1c_level": request.HbA1c_level ,
            "blood_glucose_level": request.blood_glucose_level }
        data_df = pd.DataFrame.from_dict([data_dict])

        data_df.gender = self.le.transform(data_df.gender)
        
        return data_df
    
    def predict_diabetes(self, request: DiabetesRequest)->DiabetesResponse:
        input_df = self.preprocess_input(request)
        #prediksi
        diabetes_prediction = self.model.predict(input_df)[0]

        response = DiabetesResponse
        response.diabetes = diabetes_prediction
        return response

if __name__ == "__main__":
    test_request = DiabetesRequest(gender="Female", age=44, hypertension=0, heart_disease=0, smoking_history=0, bmi=19.31, HbA1c_level=6.5 ,blood_glucose_level=200)
    
    diabetes_serv = DiabetesService()
    res = diabetes_serv.predict_diabetes(request=test_request)
    print(res.diabetes)