"""
Main Frontend code using streamlit
"""

import numpy as np
import requests
import shap
import streamlit as st
from streamlit_shap import st_shap


# API params
API_URL = "https://p7-scoring-back.onrender.com"
# API_URL = "http://127.0.0.1:8000"

# Page params
st.set_page_config(
    page_title="Home Credit Loan Validation",
    page_icon="💰",
)
st.title("Home Credit Loan Validation")


# Choose from customer list
with st.sidebar:
    customer_list = requests.get(f"{API_URL}/customers").json()
    customer_list.insert(0, "All")
    selectedID = st.selectbox("Select customer", customer_list)

if str(selectedID) == "All":
    # All customers stats
    st.header("All customers stats")
    all_customers_stats = requests.get(f"{API_URL}/customers_stats/").json()
    st.dataframe(all_customers_stats)
    with st.sidebar:
        st.metric("Customer count", len(customer_list) - 1)

else:
    # Predict for one customer
    st.header(f"Loan status for customer {str(selectedID)}")
    predict = requests.get(f"{API_URL}/predict/{str(selectedID)}").json()
    st.table(predict)
    st.success("Loan granted", icon="✅")
    st.error("Loan denied", icon="❌")

    # Customer profile
    st.header(f"Profile of customer {str(selectedID)}")
    with st.expander("Display full profile"):
        customer_profile = requests.get(f"{API_URL}/customers/{str(selectedID)}").json()
        st.dataframe(customer_profile)

    # Shap values
    st.header(f"Shap values for customer {str(selectedID)}")
    shap_dict = requests.get(f"{API_URL}/shap/{str(selectedID)}").json()
    keys = np.fromiter(shap_dict.keys(), dtype=object)
    values = np.fromiter(shap_dict.values(), dtype=float)
    st_shap(
        shap.bar_plot(
            values,
            feature_names=keys,
        ),
    )
