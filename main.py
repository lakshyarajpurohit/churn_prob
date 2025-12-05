# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# -----------------------------
# Load model artifacts at startup
# -----------------------------
MODEL_PATH = "churn_model_rf.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file '{MODEL_PATH}' not found. "
        f"Make sure it is in the same folder as main.py."
    )

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]

# -----------------------------
# FastAPI app initialization
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI backend serving a churn prediction model with HTML frontend.",
    version="1.0.0",
)

# Mount static folder for HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------
# Pydantic model for request body
# -----------------------------
class ChurnInput(BaseModel):
    CreditScore: int
    Geography: str        # "France", "Germany", "Spain"
    Gender: str           # "Male", "Female"
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int        # 0 or 1
    IsActiveMember: int   # 0 or 1
    EstimatedSalary: float


# -----------------------------
# Root route - serve frontend
# -----------------------------
@app.get("/", include_in_schema=False)
def read_root():
    """
    Serve the main HTML page.
    This will also make HEAD / return 200 on most platforms (Render health checks).
    """
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        # Fallback JSON if HTML not found
        return JSONResponse(
            {"message": "index.html not found in /static. API is running."}
        )
    return FileResponse(index_path)


# Optional health check route
@app.get("/health", include_in_schema=False)
def health_check():
    return {"status": "ok"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(input_data: ChurnInput):
    """
    Accepts customer data and returns churn prediction and probability.
    """
    # 1. Convert input to DataFrame
    df_input = pd.DataFrame([input_data.dict()])

    # 2. Apply same encoding as training (get_dummies + drop_first)
    df_encoded = pd.get_dummies(df_input, drop_first=True).astype(int)

    # 3. Align columns with training data (missing columns -> 0)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Scale data using saved scaler
    X_scaled = scaler.transform(df_encoded)

    # 5. Predict
    pred = model.predict(X_scaled)[0]           # 0 or 1
    proba = model.predict_proba(X_scaled)[0][1] # probability of churn (class 1)

    return {
        "churn_prediction": int(pred),
        "churn_probability": float(proba),
    }

# --------------------------------------------------------
# NOTE:
# Do NOT put `uvicorn.run(...)` here when deploying.
# Run with: `uvicorn main:app --reload` (locally)
# or       `uvicorn main:app --host 0.0.0.0 --port $PORT` (Render)
# --------------------------------------------------------
