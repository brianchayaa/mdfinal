pip install streamlit
import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model(model_path="loan_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

expected_columns = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file"
]

st.title("Loan Approval Prediction")

def user_input():
    return {
        "person_age": st.number_input("Age", 18, 100, 30),
        "person_gender": st.selectbox("Gender", ["male", "female"]),
        "person_education": st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"]),
        "person_income": st.number_input("Income", 0, 1000000, 50000),
        "person_emp_exp": st.number_input("Employment Experience (years)", 0, 50, 5),
        "person_home_ownership": st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]),
        "loan_amnt": st.number_input("Loan Amount", 1000, 100000, 15000),
        "loan_intent": st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "PERSONAL"]),
        "loan_int_rate": st.number_input("Interest Rate (%)", 0.0, 100.0, 10.5),
        "loan_percent_income": st.number_input("Loan Percent Income", 0.0, 1.0, 0.25),
        "cb_person_cred_hist_length": st.number_input("Credit History Length (years)", 1, 50, 5),
        "credit_score": st.number_input("Credit Score", 300, 850, 700),
        "previous_loan_defaults_on_file": st.selectbox("Previous Loan Default", ["Yes", "No"])
    }

input_data = user_input()

if st.button("Predict"):
    try:
        df = pd.DataFrame([input_data])
        df = df[expected_columns]
        
        prediction = model.predict(df)[0]
        result = "✅ Approved" if prediction == 1 else "❌ Denied"
        st.success(f"Loan prediction result: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
