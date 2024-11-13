from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the pre-trained model and scaler
model = joblib.load(r"C:\Users\96653\Desktop\Tuwaiq\Usecase-4-main\ML\SL\Classification\knn_model.joblib")
scaler = joblib.load(r"C:\Users\96653\Desktop\Tuwaiq\Usecase-4-main\ML\SL\Classification\scaler.joblib")

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Year: int
    Engine_Size: float
    Mileage: float
    Type: str
    Make: str
    Options: str

# Preprocess the input data
def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Year': input_features.Year,
        'Engine_Size': input_features.Engine_Size,
        'Mileage': input_features.Mileage,
        'Type_Accent': input_features.Type == 'Accent',
        'Type_Land Cruiser': input_features.Type == 'LandCruiser',
        'Make_Hyundai': input_features.Make == 'Hyundai',
        'Make_Mercedes': input_features.Make == 'Mercedes',
        'Options_Full': input_features.Options == 'Full',
        'Options_Standard': input_features.Options == 'Standard'
    }
    # Ensure features are ordered correctly
    features_list = [dict_f[key] for key in sorted(dict_f)]
    scaled_features = scaler.transform([features_list])
    return scaled_features

# POST endpoint for predictions
@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        data = preprocessing(input_features)
        y_pred = model.predict(data)
        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}
