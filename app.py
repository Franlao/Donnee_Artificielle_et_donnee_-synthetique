import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
from PIL import Image
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Démo - Données Synthétiques en Santé",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTANT: Initialize session state variables at the beginning
if 'variables' not in st.session_state:
    st.session_state.variables = {}
    
if 'correlations' not in st.session_state:
    st.session_state.correlations = {}
    
if 'artificial_data' not in st.session_state:
    st.session_state.artificial_data = None
    
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
    
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

# Titre principal
st.title("Démonstration - Données Synthétiques et Artificielles en Santé")

# Création du package AI_Methods
# Assurez-vous que le répertoire existe
os.makedirs("AI_Methods", exist_ok=True)

# Importer les fonctions depuis demo_app.py
from Avec_AI.demo_app import (
    generate_artificial_data,
    generate_synthetic_data_bootstrap,
    generate_synthetic_data_gaussian_copula,
    compare_distributions,
    plot_numeric_comparison,
    plot_categorical_comparison,
    plot_correlation_comparison,
    plot_pca_comparison,
    render_dynamic_artificial_data_tab,
    render_synthetic_data_tab,
    render_about_page
)

# Importer la fonction render_ai_methods_tab depuis le package AI_Methods
from Avec_AI.AI_Method import render_ai_methods_tab

# Créer les onglets
tab1, tab2, tab3, tab4 = st.tabs(["Données Artificielles", "Données Synthétiques", "Méthodes d'IA", "À propos du projet"])

# Remplir chaque onglet avec le contenu correspondant
with tab1:
    render_dynamic_artificial_data_tab()

with tab2:
    render_synthetic_data_tab()

with tab3:
    render_ai_methods_tab()
    
with tab4:
    render_about_page()

# Instructions d'utilisation dans la barre latérale
with st.sidebar:
    st.title("Guide d'utilisation")
    
    st.markdown("""
   
    ### À propos du projet
    
    Consultez cet onglet pour comprendre:
    - Les concepts de données artificielles et synthétiques
    - Les fonctionnalités détaillées de chaque module
    - Les cas d'utilisation concrets
    - Les informations techniques
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Préparé pour:
    **Présentation d'Épidémiologie**
    
    *Données synthétiques et données artificielles en santé*
    """)
    
    # Section Installation et Prérequis
    with st.expander("Installation et Prérequis"):
        st.markdown("""
        ### Prérequis

        Cette application nécessite les bibliothèques Python suivantes:

        ```bash
        pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn
        ```

        Pour les fonctionnalités d'IA:

        ```bash
        pip install tensorflow mistralai
        ```

        ### Structure des fichiers

        Pour exécuter cette application, assurez-vous d'avoir les fichiers suivants:
        - `app.py` (ce fichier principal)
        - `demo_app.py` (fonctionnalités de base)
        - Package `AI_Methods` (fonctionnalités d'IA)

        ### Lancement

        ```bash
        streamlit run app.py
        ```
        """)