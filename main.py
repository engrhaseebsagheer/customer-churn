# ==============================
# 1. Import libraries
# ==============================
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# 2. Initialize FastAPI app
# ==============================
app = FastAPI()

# Enable CORS if frontend is on another domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 3. Load trained pipeline once
# ==============================
pipeline = joblib.load("customer_churn_pipeline.joblib")

# ==============================
# 4. Define input data schema
# ==============================
class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# ==============================
# 5. Value mapping helper
# ==============================
def preprocess_input(data: CustomerData):
    """
    Converts user-friendly values (Yes/No, etc.) to the format
    used during training before sending to the pipeline.
    """
    d = data.dict()

    # Map Yes/No to 1/0 for binary columns
    binary_map = {"Yes": 1, "No": 0}
    binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    for col in binary_columns:
        d[col] = binary_map.get(d[col], d[col])

    # SeniorCitizen already 0/1, categorical handled by OneHotEncoder
    return d

# ==============================
# 6. Prediction endpoint
# ==============================
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Preprocess user input
    processed_data = preprocess_input(customer)

    # Convert dict → DataFrame with correct column names
    input_df = pd.DataFrame([processed_data])

    # Make prediction and probability
    prediction = pipeline.predict(input_df)[0]
    churn_probability = pipeline.predict_proba(input_df)[0][1] * 100  # probability of churn

    # Prepare human-readable labels
    if prediction == 1:
        risk_label = "High Risk - Likely to Churn"
    else:
        risk_label = "Low Risk - Unlikely to Churn"

    return {
        "Prediction Result": risk_label,
        "Churn Probability": f"{churn_probability:.2f}%",
        "Model Used": "Random Forest",
        "Note": "This is a real prediction from FastAPI backend."
    }


# ==============================
# 7. Run the app
# ==============================
# Run this in terminal:
# uvicorn main:app --reload
