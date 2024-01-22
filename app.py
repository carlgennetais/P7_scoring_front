"""
Main Frontend code using streamlit
"""
# import pickle

import pandas as pd
import requests
import streamlit as st

# API params
BACK_URL = "http://127.0.0.1:8000"

st.title("Home Credit Scoring")


# response = requests.get(BACK_URL)
# st.write(response.json()[0])


# Choose from customer list
st.header("Select a customer")
customer_list = requests.get(BACK_URL + "/customers").json()

# Customer profile
selectedID = st.selectbox("Please select a customer", customer_list)
st.header("Profile of customer " + str(selectedID))
customer_profile = requests.get(BACK_URL + "/customers/" + str(selectedID)).json()
st.dataframe(customer_profile)

# All customers stats
st.header("All customers stats")
all_customers_stats = requests.get(BACK_URL + "/customers_stats/").json()
st.dataframe(all_customers_stats)

# Predict for one customer
st.header("Scoring for customer " + str(selectedID))
predict = requests.get(BACK_URL + "/predict/" + str(selectedID)).json()
st.table(predict)

# Shap values
st.header("Shap values for customer " + str(selectedID))
shap = requests.get(BACK_URL + "/shap/" + str(selectedID)).json()
st.write(shap)
