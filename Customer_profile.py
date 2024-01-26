"""
Main Frontend code using streamlit
"""
# import pickle

import pandas as pd
import requests
import streamlit as st

# API params
BACK_URL = "http://127.0.0.1:8000"

# Page params
st.set_page_config(
    page_title="Home Credit Loan Validation",
    page_icon="💰",
)
st.title("Home Credit Loan Validation")


# response = requests.get(BACK_URL)
# st.write(response.json()[0])


# Choose from customer list
with st.sidebar:
    customer_list = requests.get(BACK_URL + "/customers").json()
    customer_list.insert(0, "All")
    selectedID = st.selectbox("Select customer", customer_list)

if str(selectedID) == "All":
    # All customers stats
    st.header("All customers stats")
    all_customers_stats = requests.get(BACK_URL + "/customers_stats/").json()
    st.dataframe(all_customers_stats)
else:
    # Customer profile
    st.header("Profile of customer " + str(selectedID))
    customer_profile = requests.get(BACK_URL + "/customers/" + str(selectedID)).json()
    st.dataframe(customer_profile)

    # Predict for one customer
    st.header("Scoring for customer " + str(selectedID))
    predict = requests.get(BACK_URL + "/predict/" + str(selectedID)).json()
    st.table(predict)
    st.success("Loan granted", icon="✅")
    st.error("Loan denied", icon="❌")

    # Shap values
    st.header("Shap values for customer " + str(selectedID))
    shap = requests.get(BACK_URL + "/shap/" + str(selectedID)).json()
    st.write(shap)
