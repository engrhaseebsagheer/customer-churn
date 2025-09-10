# ===========================
# 1. Import required libraries
# ===========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===========================
# 2. Load dataset
# ===========================
data = pd.read_csv("/Users/haseebsagheer/Documents/Python Learning/Customer-Churn/Datasets/WA_Fn-UseC_-Telco-Customer-Churn 3.csv")

# ===========================
# 3. Drop unnecessary columns
# ===========================
data.drop(["customerID", "gender"], axis=1, inplace=True)

# ===========================
# 4. Convert TotalCharges to numeric & handle missing
# ===========================
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

# ===========================
# 5. Map Yes/No binary columns → 1/0
# ===========================
binary_columns = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
]
for col in binary_columns:
    data[col] = data[col].map({"Yes": 1, "No": 0})

# SeniorCitizen is already 0/1 in the dataset
binary_features = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# ===========================
# 6. Define categorical & numeric features
# ===========================
categorical_features = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# ===========================
# 7. Train-test split
# ===========================
X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# 8. Preprocessing
# ===========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
        ("bin", "passthrough", binary_features)
    ]
)

# ===========================
# 9. Full pipeline: Preprocessing + Random Forest
# ===========================
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ))
])

# ===========================
# 10. Train pipeline
# ===========================
pipeline.fit(X_train, y_train)

# ===========================
# 11. Save pipeline as joblib
# ===========================
joblib.dump(pipeline, "customer_churn_pipeline.joblib")
print("✅ Pipeline trained and saved as customer_churn_pipeline.joblib")

# ===========================
# 12. Optional: Evaluate accuracy
# ===========================
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
