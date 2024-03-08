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
    page_title="Prêt à dépenser",
    page_icon="💰",
)
st.title("Prêt à dépenser")


# Choose from customer list
with st.sidebar:
    customer_list = requests.get(f"{API_URL}/customers").json()
    customer_list.insert(0, "Tous")
    selectedID = st.selectbox("Sélectionner un client", customer_list)

if str(selectedID) == "Tous":
    # All customers stats
    st.header("Statistiques sur l'ensemble des clients")
    all_customers_stats = requests.get(f"{API_URL}/customers_stats/").json()
    st.dataframe(all_customers_stats)
    with st.sidebar:
        st.metric("Nombre de clients", len(customer_list) - 1)

else:
    # Predict for one customer
    st.header(f"Résultat de la demande de prêt {str(selectedID)}")
    predict = requests.get(f"{API_URL}/predict/{str(selectedID)}").json()
    st.table(predict)
    # TODO: display loan application result
    st.success("Prêt accordé", icon="✅")
    st.error("Prêt refusé", icon="❌")

    # Customer profile
    st.header(f"Profil du client {str(selectedID)}")
    with st.expander("Afficher le profil complet"):
        # TODO: break features into categories for readability (work, house etc)
        customer_profile = requests.get(f"{API_URL}/customers/{str(selectedID)}").json()
        st.dataframe(customer_profile)

    # Shap values
    # TODO: add description
    st.header(f"Critères décisifs pour le client {str(selectedID)}")
    shap_dict = requests.get(f"{API_URL}/shap/{str(selectedID)}").json()
    keys = np.fromiter(shap_dict.keys(), dtype=object)
    values = np.fromiter(shap_dict.values(), dtype=float)
    st_shap(
        shap.bar_plot(
            values,
            feature_names=keys,
        ),
    )
