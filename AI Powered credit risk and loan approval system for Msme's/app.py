import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
with open("msme_credit_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

st.set_page_config(page_title="MSME Credit Risk AI", layout="centered")

st.title("🏦 MSME Credit Risk & Loan Approval System")
st.write("AI-powered cashflow-based lending support for Indian MSMEs")

st.header("📋 Enter MSME Details")

monthly_revenue = st.number_input("Monthly Revenue (₹)", min_value=0)
monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0)
revenue_volatility = st.slider("Revenue Volatility", 0.0, 1.0)
avg_monthly_transactions = st.number_input("Avg Monthly Transactions", min_value=0)
loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0)
loan_tenure = st.number_input("Loan Tenure (months)", min_value=1)
existing_emi = st.number_input("Existing EMI (₹)", min_value=0)
years_in_business = st.number_input("Years in Business", min_value=0)

if st.button("Assess Credit Risk"):

    input_data = np.array([[ 
        monthly_revenue,
        monthly_expenses,
        revenue_volatility,
        avg_monthly_transactions,
        loan_amount,
        loan_tenure,
        existing_emi,
        years_in_business
    ]])

    input_scaled = scaler.transform(input_data)

    pd_default = model.predict_proba(input_scaled)[0][1]

    if pd_default < 0.3:
        risk = "LOW RISK"
        decision = "APPROVE"
        color = "green"
    elif pd_default < 0.6:
        risk = "MEDIUM RISK"
        decision = "REVIEW"
        color = "orange"
    else:
        risk = "HIGH RISK"
        decision = "REJECT"
        color = "red"

    monthly_surplus = monthly_revenue - monthly_expenses - existing_emi
    safe_emi = max(0, 0.3 * monthly_surplus)

    st.subheader("📊 Credit Assessment Result")

    st.markdown(f"**Probability of Default:** `{round(pd_default*100,2)}%`")
    st.markdown(f"**Risk Category:** :{color}[{risk}]")
    st.markdown(f"**Decision:** **{decision}**")
    st.markdown(f"**Recommended Safe EMI:** ₹ `{round(safe_emi,2)}`")

    if decision == "REJECT":
        st.warning("High default risk. Credit not recommended.")
    elif decision == "REVIEW":
        st.info("Manual credit officer review suggested.")
    else:
        st.success("Loan can be approved under safe EMI limits.")
