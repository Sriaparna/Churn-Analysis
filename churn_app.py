import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")
st.title(" Customer Churn Risk Calculator")
st.write("Predict the probability of a customer leaving and get an intervention strategy.")
model = joblib.load('churn_rf_model.joblib')
st.sidebar.header("Customer Characteristics")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0)
contract_type = st.sidebar.selectbox("Is the Contract Month-to-Month?", ["No", "Yes"])
contract_val = 1 if contract_type == "Yes" else 0
online_security = st.sidebar.selectbox("Does the customer have Online Security?", ["No", "Yes"])
security_val = 1 if online_security == "No" else 0
all_columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes']

input_data = pd.DataFrame(np.zeros((1, 40)), columns=all_columns)
input_data['TotalCharges'] = total_charges
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['Contract_Month-to-month'] = contract_val
input_data['OnlineSecurity_No'] = security_val
churn_prob = model.predict_proba(input_data)[0][1]
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Churn Risk Score", value=f"{churn_prob*100:.1f}%")
    if churn_prob > 0.7:
        st.error("🚨 HIGH RISK: Urgent intervention needed.")
    elif churn_prob > 0.3:
        st.warning("⚠️ MEDIUM RISK: Targeted engagement recommended.")
    else:
        st.success("✅ LOW RISK: Customer is stable.")

with col2:
    st.info("**Recommended Action Strategy:**")
    if churn_prob > 0.7:
        st.write("- **Immediate Call:** Schedule a retention call today.")
        st.write("- **Save Offer:** Offer a 30% discount for a 1-year contract.")
    elif churn_prob > 0.3:
        st.write("- **Email Feedback:** Send a customer satisfaction survey.")
        st.write("- **Soft Incentive:** Send a loyalty reward voucher.")
    else:
        st.write("- **Maintain Service:** Include in standard newsletters.")