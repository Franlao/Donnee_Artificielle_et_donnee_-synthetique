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
import torch
if hasattr(torch, "classes"):
    del torch.classes
    
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
    render_synthetic_data_tab
)

# Importer la fonction render_ai_methods_tab depuis le package AI_Methods
from Avec_AI.AI_Method import render_ai_methods_tab

# Créer les onglets
tab1, tab2, tab3 = st.tabs(["Données Artificielles", "Données Synthétiques", "Méthodes d'IA"])

# Remplir chaque onglet avec le contenu correspondant
with tab1:
    render_dynamic_artificial_data_tab()

with tab2:
    render_synthetic_data_tab()

with tab3:
    render_ai_methods_tab()

# Instructions d'utilisation dans la barre latérale
with st.sidebar:
    st.title("Guide d'utilisation")
    
    st.markdown("""
    ### Données Artificielles
    
    Utilisez cet onglet pour générer des données à partir de distributions statistiques que vous définissez.
    
    1. Définissez les paramètres pour chaque variable
    2. Spécifiez les corrélations entre variables numériques
    3. Cliquez sur "Générer les Données Artificielles"
    4. Explorez les visualisations et téléchargez les données
    
    ### Données Synthétiques
    
    Utilisez cet onglet pour générer des données synthétiques basées sur des données réelles.
    
    1. Chargez un fichier CSV contenant vos données réelles
    2. Sélectionnez les colonnes à inclure
    3. Choisissez la méthode de génération
    4. Cliquez sur "Générer les Données Synthétiques"
    5. Comparez les distributions et téléchargez les données
    
    ### Méthodes d'IA
    
    Utilisez cet onglet pour explorer les approches avancées basées sur l'intelligence artificielle.
    
    #### IA pour Données Synthétiques:
    - GAN: Réseaux antagonistes génératifs
    - VAE: Auto-encodeurs variationnels
    
    #### IA pour Données Artificielles:
    - LLM: Grands modèles de langage (via API Mistral)
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