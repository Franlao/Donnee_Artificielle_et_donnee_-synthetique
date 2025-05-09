import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import json
from PIL import Image
import time
import tempfile
import pickle
from typing import Tuple, Dict, List, Union, Optional, Any

# Importer les modules décomposés
from .data_processor import DataProcessor, DatasetMetadata
from .llm_generator import MistralGenerator, MISTRAL_AVAILABLE
from .ui_components import render_llm_tab, visualize_categorical_comparison, visualize_numeric_comparison
from .load_csv_safe import load_csv_safely
from .missing_values_handler import missing_values_module
# Importer le module des modèles
try:
    from .models import (
        train_gan_model, 
        train_vae_model, 
        save_model, 
        load_model, 
        GAN_AVAILABLE, 
        VAE_AVAILABLE, 
        CUDA_AVAILABLE
    )
except ImportError:
    # Définir des valeurs par défaut si le module n'est pas disponible
    GAN_AVAILABLE = False
    VAE_AVAILABLE = False
    CUDA_AVAILABLE = False

# Essayer d'importer PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch n'est pas installé. Pour l'installer: pip install torch")

# Méthodes statistiques simples
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def generate_with_statistical_methods(preprocessed_data, metadata, method_type, n_components, n_samples):
    """
    Génère des données synthétiques en utilisant des méthodes statistiques simples.
    
    Args:
        preprocessed_data: Données prétraitées
        metadata: Métadonnées
        method_type: Type de méthode ('gmm' ou 'pca')
        n_components: Nombre de composantes
        n_samples: Nombre d'échantillons à générer
        
    Returns:
        Données synthétiques générées
    """
    if method_type == 'gmm':
        # Utiliser un mélange de gaussiennes
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(preprocessed_data)
        
        # Générer des échantillons
        samples, _ = gmm.sample(n_samples=n_samples)
        return samples
        
    elif method_type == 'pca':
        # Utiliser PCA pour la réduction de dimension + reconstruction
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(preprocessed_data)
        
        # Générer des échantillons en échantillonnant l'espace réduit
        random_reduced = np.random.normal(
            loc=np.mean(reduced, axis=0),
            scale=np.std(reduced, axis=0),
            size=(n_samples, n_components)
        )
        
        # Reconstruire les données
        reconstructed = pca.inverse_transform(random_reduced)
        return reconstructed
    
    else:
        raise ValueError(f"Méthode statistique non reconnue: {method_type}")

def render_ai_methods_tab():
    """Rendre l'onglet des méthodes d'IA dans Streamlit"""
    st.header("Génération de Données avec l'Intelligence Artificielle")
    
    subtab1, subtab2 = st.tabs(["IA pour Données Synthétiques", "IA pour Données Artificielles"])
    
    with subtab1:
        st.subheader("Génération de Données Synthétiques avec l'IA")
        st.write("""
        Cette section vous permet d'utiliser des techniques avancées d'intelligence artificielle 
        pour générer des données synthétiques à partir de données réelles.
        """)
        
        # Options: GAN, VAE ou méthodes simples
        ai_method = st.radio(
            "Sélectionnez la méthode d'IA:",
            [
                "GAN (Generative Adversarial Network)",
                "VAE (Variational Autoencoder)",
                "Méthodes statistiques simples (GMM, PCA)"
            ],
            index=0
        )
        
        # Avertissement PyTorch
        if not TORCH_AVAILABLE and (ai_method.startswith("GAN") or ai_method.startswith("VAE")):
            st.warning("""
            PyTorch n'est pas installé. Pour utiliser GAN ou VAE, installez PyTorch:
            
            ```bash
            pip install torch
            ```
            
            Puis redémarrez l'application. Vous pouvez toujours utiliser les méthodes statistiques simples.
            """)
        elif ai_method.startswith("GAN") and not GAN_AVAILABLE:
            st.warning("""
            Le module tabular_gan.py n'est pas disponible. 
            Assurez-vous qu'il est présent dans le répertoire du projet.
            """)
        elif ai_method.startswith("VAE") and not VAE_AVAILABLE:
            st.warning("""
            Le module tabular_vae.py n'est pas disponible. 
            Assurez-vous qu'il est présent dans le répertoire du projet.
            """)
        
        # Information sur l'accélération GPU
        if TORCH_AVAILABLE and CUDA_AVAILABLE and (ai_method.startswith("GAN") or ai_method.startswith("VAE")):
            st.success("L'accélération GPU (CUDA) est disponible et sera utilisée pour l'entraînement.")
        elif TORCH_AVAILABLE and (ai_method.startswith("GAN") or ai_method.startswith("VAE")):
            st.info("PyTorch est disponible mais l'accélération GPU (CUDA) n'est pas détectée. L'entraînement utilisera le CPU.")
        
        # Charger des données réelles
        uploaded_file = st.file_uploader("Charger un fichier CSV contenant des données réelles", 
                                        type="csv", key="ai_synth_upload")
        
        if uploaded_file is not None:
            # Charger les données
            try:
                # Utiliser la fonction sécurisée pour charger le CSV
                with st.spinner("Chargement et analyse du fichier CSV..."):
                    real_data = load_csv_safely(uploaded_file)
                
                if real_data is None:
                    st.error("Impossible de charger le fichier CSV. Veuillez vérifier le format.")
                else:
                    # Vérifier les valeurs manquantes
                    missing_values = real_data.isnull().sum().sum()
                    if missing_values > 0:
                        st.warning(f"⚠️ {missing_values} valeurs manquantes détectées dans le jeu de données.")
                        if st.button("Gérer les valeurs manquantes"):
                            real_data = missing_values_module(real_data)
                            st.success("Traitement des valeurs manquantes terminé.")
                    
                    # Afficher un aperçu des données réelles
                    st.subheader("Aperçu des Données Réelles")
                    st.write(real_data.head())
                    
                    # Analyse des données et détection automatique des types
                    analysis = DataProcessor.analyze_data(real_data)
                    
                    with st.expander("Analyse des Données et Détection des Types", expanded=True):
                        # Afficher les statistiques de base
                        st.write(f"Total des lignes: {analysis['total_rows']}")
                        st.write(f"Total des colonnes: {len(analysis['columns'])}")
                        
                        # Afficher les types détectés
                        st.subheader("Types de Colonnes Détectés Automatiquement")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Colonnes numériques:**")
                            for col in analysis["column_types"]["numeric"]:
                                is_int = col in analysis["column_types"]["integer"]
                                is_binary = col in analysis["column_types"]["binary"]
                                type_info = ""
                                if is_int: 
                                    type_info += " (entier)"
                                if is_binary:
                                    type_info += " (binaire 0/1)"
                                st.write(f"- {col}{type_info}")
                        
                        with col2:
                            st.write("**Colonnes catégorielles:**")
                            for col in analysis["column_types"]["categorical"]:
                                is_binary = col in analysis["column_types"]["binary"]
                                type_info = " (binaire 0/1)" if is_binary else ""
                                st.write(f"- {col}{type_info}")
                        
                        # Permettre à l'utilisateur de modifier les types détectés
                        st.subheader("Modifier les Types de Colonnes (si nécessaire)")
                        
                        # Utiliser des multi-selects pour choisir les types
                        all_columns = real_data.columns.tolist()
                        
                        categorical_override = st.multiselect(
                            "Colonnes à traiter comme catégorielles:",
                            all_columns,
                            default=analysis["column_types"]["categorical"],
                            help="Sélectionnez toutes les colonnes qui devraient être traitées comme catégorielles, même si elles contiennent des nombres (ex: codes, ID, variables binaires 0/1)"
                        )
                        
                        if analysis['missing_values']:
                            st.warning("Valeurs manquantes détectées dans ces colonnes:")
                            for col, count in analysis['missing_values'].items():
                                st.write(f"- {col}: {count} valeurs manquantes")
                    
                    try:
                        # Prétraiter les données avec les types spécifiés par l'utilisateur
                        with st.spinner("Prétraitement des données..."):
                            preprocessed_data, metadata = DataProcessor.preprocess_data(
                                real_data, 
                                categorical_cols_override=categorical_override
                            )
                        
                        # Paramètres d'entraînement
                        st.subheader("Paramètres d'Entraînement")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if ai_method.startswith("Méthodes statistiques"):
                                method_type = st.selectbox(
                                    "Méthode statistique",
                                    ["gmm", "pca"],
                                    index=0,
                                    format_func=lambda x: "Mélange de Gaussiennes (GMM)" if x == "gmm" else "Analyse en Composantes Principales (PCA)"
                                )
                                n_components = st.slider(
                                    "Nombre de composantes",
                                    min_value=1,
                                    max_value=20,
                                    value=5,
                                    step=1,
                                    help="Nombre de composantes/clusters pour la méthode statistique"
                                )
                            else:
                                latent_dim = st.slider(
                                    "Dimension de l'espace latent",
                                    min_value=2,
                                    max_value=50,
                                    value=16,
                                    step=1,
                                    help="Dimension de l'espace latent pour le modèle générateur"
                                )
                                
                                epochs = st.slider(
                                    "Nombre d'époques",
                                    min_value=10,
                                    max_value=1000,
                                    value=100,
                                    step=10,
                                    help="Nombre d'itérations complètes sur les données d'entraînement"
                                )
                        
                        with col2:
                            if not ai_method.startswith("Méthodes statistiques"):
                                batch_size = st.slider(
                                    "Taille du batch",
                                    min_value=8,
                                    max_value=128,
                                    value=32,
                                    step=8,
                                    help="Nombre d'exemples traités en une seule fois"
                                )
                            
                            n_samples = st.slider(
                                "Nombre d'échantillons à générer",
                                min_value=100,
                                max_value=5000,
                                value=len(real_data),
                                step=100,
                                help="Nombre d'échantillons synthétiques à générer"
                            )
                        
                        # Paramètres avancés
                        with st.expander("Paramètres Avancés"):
                            if ai_method.startswith("GAN"):
                                learning_rate = st.number_input(
                                    "Taux d'apprentissage",
                                    min_value=0.0001,
                                    max_value=0.01,
                                    value=0.0002,
                                    format="%.5f",
                                    help="Taille des pas pour les mises à jour du gradient"
                                )
                                
                                early_stopping = st.slider(
                                    "Patience pour l'arrêt anticipé",
                                    min_value=10,
                                    max_value=100,
                                    value=30,
                                    step=5,
                                    help="Nombre d'époques sans amélioration avant l'arrêt"
                                )
                            elif ai_method.startswith("VAE"):
                                beta_value = st.slider(
                                    "Valeur beta",
                                    min_value=0.1,
                                    max_value=5.0,
                                    value=1.0,
                                    step=0.1,
                                    help="Poids pour le terme de divergence KL (des valeurs plus élevées imposent un espace latent plus désenchevêtré)"
                                )
                                
                                learning_rate = st.number_input(
                                    "Taux d'apprentissage",
                                    min_value=0.00001,
                                    max_value=0.01,
                                    value=0.0001,
                                    format="%.5f",
                                    help="Taille des pas pour les mises à jour du gradient"
                                )
                                
                                early_stopping = st.slider(
                                    "Patience pour l'arrêt anticipé",
                                    min_value=5,
                                    max_value=50,
                                    value=10,
                                    step=1,
                                    help="Nombre d'époques sans amélioration avant l'arrêt"
                                )
                        
                        # Persistance du modèle
                        save_model_option = st.checkbox(
                            "Sauvegarder le modèle entraîné", 
                            value=False,
                            help="Sauvegarder le modèle entraîné pour une utilisation future"
                        )
                        
                        if save_model_option:
                            model_path = st.text_input(
                                "Chemin de sauvegarde du modèle", 
                                value=f"./models/{ai_method.split(' ')[0].lower()}_model",
                                help="Chemin où le modèle sera sauvegardé"
                            )
                        
                        # Avertissement sur le temps d'entraînement
                        st.markdown("""
                        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #fff8e1; border: 1px solid #ffc107;">
                        <b>⚠️ Note:</b> L'entraînement peut prendre plusieurs minutes selon la taille des données
                        et les paramètres choisis. Un arrêt anticipé sera effectué si aucune amélioration n'est observée
                        après plusieurs époques.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Bouton pour lancer l'entraînement
                        if st.button("Entraîner le modèle et générer des données", key="train_ai_model"):
                            try:
                                # Créer et entraîner le modèle
                                if ai_method.startswith("GAN") and GAN_AVAILABLE:
                                    st.info("Entraînement du GAN en cours...")
                                    
                                    # Configuration des paramètres pour GAN
                                    gan_params = {
                                        'latent_dim': latent_dim,
                                        'epochs': epochs,
                                        'batch_size': batch_size,
                                        'n_samples': n_samples,
                                        'learning_rate': learning_rate,
                                        'early_stopping': early_stopping
                                    }
                                    
                                    # Démarrer un indicateur de progression
                                    progress_bar = st.progress(0)
                                    
                                    # Entraîner le modèle GAN
                                    model, synthetic_data = train_gan_model(preprocessed_data, metadata, gan_params)
                                    
                                    # Mise à jour de l'indicateur de progression
                                    progress_bar.progress(100)
                                    
                                    # Sauvegarder le modèle si demandé
                                    if save_model_option:
                                        save_model(model, model_path, model_type='gan')
                                        st.success(f"Modèle GAN sauvegardé avec succès à {model_path}")
                                    
                                elif ai_method.startswith("VAE") and VAE_AVAILABLE:
                                    st.info("Entraînement du VAE en cours...")
                                    
                                    # Configuration des paramètres pour VAE
                                    vae_params = {
                                        'latent_dim': latent_dim,
                                        'epochs': epochs,
                                        'batch_size': batch_size,
                                        'n_samples': n_samples,
                                        'learning_rate': learning_rate,
                                        'beta': beta_value,
                                        'early_stopping': early_stopping
                                    }
                                    
                                    # Démarrer un indicateur de progression
                                    progress_bar = st.progress(0)
                                    
                                    # Entraîner le modèle VAE
                                    model, synthetic_data = train_vae_model(preprocessed_data, metadata, vae_params)
                                    
                                    # Mise à jour de l'indicateur de progression
                                    progress_bar.progress(100)
                                    
                                    # Sauvegarder le modèle si demandé
                                    if save_model_option:
                                        save_model(model, model_path, model_type='vae')
                                        st.success(f"Modèle VAE sauvegardé avec succès à {model_path}")
                                    
                                else:  # Méthodes statistiques simples
                                    st.info(f"Utilisation de la méthode statistique {method_type.upper()}...")
                                    
                                    # Générer des données avec les méthodes statistiques
                                    synthetic_data = generate_with_statistical_methods(
                                        preprocessed_data, 
                                        metadata, 
                                        method_type, 
                                        n_components, 
                                        n_samples
                                    )
                                
                                # Convertir les données synthétiques au format original
                                synthetic_df = DataProcessor.inverse_transform(synthetic_data, metadata)
                                
                                # Afficher les données générées
                                st.subheader("Données Synthétiques Générées")
                                
                                # Afficher les données générées
                                st.write("Aperçu des données synthétiques:")
                                st.write(synthetic_df.head())
                                
                                # Afficher les types de données
                                st.write("Types des colonnes générées:")
                                st.write(synthetic_df.dtypes)
                                
                                # Option de téléchargement
                                st.download_button(
                                    label="Télécharger les Données Synthétiques (CSV)",
                                    data=synthetic_df.to_csv(index=False).encode('utf-8'),
                                    file_name="donnees_synthetiques.csv",
                                    mime="text/csv"
                                )
                                
                                # Visualisations comparatives
                                st.subheader("Visualisations Comparatives")
                                
                                for col in metadata.categorical_cols:
                                    if synthetic_df[col].nunique() < 15:  # Seulement pour les colonnes avec un nombre raisonnable de catégories
                                        fig = visualize_categorical_comparison(real_data, synthetic_df, col)
                                        st.pyplot(fig)

                                # Pour les colonnes numériques
                                for col in metadata.numeric_cols:
                                    fig = visualize_numeric_comparison(real_data, synthetic_df, col)
                                    st.pyplot(fig)
                                
                                # Statistiques descriptives
                                st.subheader("Statistiques Descriptives")
                                
                                if metadata.numeric_cols:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Données réelles:**")
                                        st.write(real_data[metadata.numeric_cols].describe())
                                    
                                    with col2:
                                        st.write("**Données synthétiques:**")
                                        st.write(synthetic_df[metadata.numeric_cols].describe())
                                
                            except Exception as e:
                                st.error(f"Une erreur s'est produite: {str(e)}")
                                import traceback
                                st.exception(traceback.format_exc())
                    except Exception as e:
                        st.error(f"Erreur lors du prétraitement: {str(e)}")
                        import traceback
                        st.exception(traceback.format_exc())
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier CSV: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())
    
    with subtab2:
        # Code pour l'onglet des données artificielles (LLM)
        st.subheader("Génération de Données Artificielles avec LLM")
        render_llm_tab()