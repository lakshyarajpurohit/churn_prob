from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

# ---------- Load model artifacts at startup ----------
artifacts = joblib.load("churn_model_rf.joblib")
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]

# ---------- Create FastAPI app ----------
app = FastAPI()

# Mount the static folder for frontend files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Input schema ----------
class ChurnInput(BaseModel):
    CreditScore: int
    Geography: str        # "France", "Germany", "Spain"
    Gender: str           # "Male", "Female"
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# ---------- Serve the main frontend page ----------
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# ---------- Prediction endpoint ----------
@app.post("/predict")
def predict_churn(input_data: ChurnInput):
    # 1. Convert input to DataFrame
    df_input = pd.DataFrame([input_data.dict()])

    # 2. Apply same encoding as during training
    df_encoded = pd.get_dummies(df_input, drop_first=True).astype(int)

    # 3. Align columns with training data
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale data
    X_scaled = scaler.transform(df_encoded)

    # 5. Predict
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]  # churn probability

    return {
        "churn_prediction": int(pred),
        "churn_probability": float(proba)
    }
