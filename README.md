# Telco Customer Churn Prediction Project

### ***Customer Churn Reason***

In this project, I predicted **why customers churn** in a telecom company. *(Churn basically means "losing a customer" who stops using your services.)*

I built this end-to-end project from scratch: from data import and cleaning, to feature engineering, model building, evaluation, and interpretation.

---

## 📌 Business Understanding

### ❓ What is the main problem?

The telecom company is **losing potential customers**, and they want to understand the **reasons** behind this churn. Using historical data of previously churned customers, we aim to predict **which new customers are likely to churn**.

### 🎯 Why is this important?

* Churn reduces revenue and increases customer acquisition cost.
* By predicting churn, the company can **take proactive steps** (e.g., discounts, retention offers).

### ✅ Business Questions I Considered

1. What does churn mean to your business?
2. Why do you want to eliminate churn?
3. Will this system help improve revenue?
4. What actions will be taken when churn is predicted?
5. What is the end goal of building this predictive model?

---

## 📚 Project Understanding

This is a **binary classification** problem. We are predicting whether a customer will churn or not (Yes/No).

* ✅ **Target variable:** `Churn`
* ✅ **Type of ML problem:** Supervised learning - classification
* ✅ **Example model used:** Logistic Regression, Random Forest Classifier

---

## 📥 Data Collection

The data contains:

* User demographic details (e.g., gender, senior citizen)
* Account info (e.g., contract type, payment method)
* Services used (e.g., streaming, internet type)
* Billing details (e.g., total charges, monthly charges)
* Churn label: `Yes` or `No`

Fortunately, I didn’t need to scrape or collect data manually. I used an **open-source dataset from Kaggle**:

🔗 [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The data is already labeled — i.e., it's a **supervised learning dataset**.

---

## 📊 Data Understanding & Preprocessing

I explored the dataset deeply to understand trends and detect any cleaning requirements:

### ✔ Key Steps:

* Checked for **missing/null/NaN** values
* Cleaned `TotalCharges` (some non-numeric entries)
* **Encoded** categorical variables:

  * Binary columns (`Yes`/`No`) → 1/0
  * Nominal categorical columns → One-Hot Encoding (OHE)
* Dropped irrelevant columns (`customerID`, `gender`)
* Handled **class imbalance** using stratified train-test split

---

## 📈 Visualizations

I created several initial visualizations to understand:

* Churn rate distribution
* Churn by gender
* Churn by contract type, internet service, payment method, and tenure
* Boxplots for numeric features

These helped me gain **business-level insights** even before modeling.

---

## 🧠 Machine Learning Models

### ✅ **Logistic Regression**

* Easy to understand and interpret
* Achieved **80% accuracy**
* Plotted **ROC Curve** with **AUC = 0.84**
* Evaluated using: Precision, Recall, F1-score, Confusion Matrix

### ✅ **Random Forest Classifier**

* Tuned and evaluated on the same test set
* Also achieved **80% accuracy**
* AUC = 0.84
* Generated **feature importance chart** to identify what drives churn

---

## 🔍 Feature Importance Insights

Top predictors of churn:

* **Tenure**: Shorter tenure → more likely to churn
* **Total Charges**: Low total → new customer → high risk
* **Contract Type**: Month-to-month contracts → high churn
* **Fiber Optic Internet**: Higher churn rate compared to DSL

These features tell a **story about customer behavior** and what kind of customers are more likely to leave.

---

## 🧾 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-Score
* ROC-AUC Curve
* Confusion Matrix

Each metric helped me understand performance from a different angle (e.g., business cost of false positives vs. false negatives).

---

## 🚀 Final Thoughts

I gained **hands-on experience** with the full data science cycle:

* Business understanding → Data wrangling → Model building → Evaluation
* I focused on **interpretability and impact**, not just accuracy

This project has become an essential part of my **portfolio**. I plan to extend it with:

* XGBoost & hyperparameter tuning
* SHAP values for customer-specific explanations
* Deployment in a web dashboard for business use

---

### ✅ Completed: August 2025

**Author:** Haseeb Sagheer
**Project Type:** Supervised Machine Learning / Binary Classification
**Tools Used:** Python, Pandas, Seaborn, Scikit-learn, Matplotlib

---

> 💼 "Data without interpretation is noise. Data with direction is power."
