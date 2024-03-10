"""
Main Frontend code using streamlit
"""

import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from streamlit_shap import st_shap
import seaborn as sns

sns.set_theme()


def dict_to_exp(dico: dict) -> shap._explanation.Explanation:
    """
    Convert dict to Shap explanation
    """
    return shap.Explanation(
        values=np.array(list(dico["values"].values())),
        base_values=dico["base_values"],
        data=np.array(list(dico["data"].values())),
        display_data=pd.Series(dico["display_data"]),
    )


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
    predict = requests.get(f"{API_URL}/predict/{str(selectedID)}").json()["loan_result"]
    if predict == 0:
        st.success("Prêt accordé", icon="✅")
    else:
        st.error("Prêt refusé", icon="❌")

    # Customer profile
    st.header(f"Profil du client {str(selectedID)}")
    with st.expander("Afficher le profil complet"):
        # TODO: break features into categories for readability (work, house etc)
        customer_profile = requests.get(f"{API_URL}/customers/{str(selectedID)}").json()
        st.dataframe(customer_profile)

    # Shap values
    # TODO: add feature dictionnary
    st.header(f"Critères décisifs pour le prêt du client {str(selectedID)}")
    shap_dict = requests.get(f"{API_URL}/shap/{str(selectedID)}").json()
    exp = dict_to_exp(shap_dict)
    st_shap(shap.plots.waterfall(exp), height=600, width=1200)
