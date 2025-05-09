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

# Nouvelle fonction pour afficher l'interface de gestion des valeurs manquantes
def show_missing_values_interface():
    """Affiche une interface dédiée à la gestion des valeurs manquantes"""
    if not hasattr(st.session_state, 'original_data'):
        st.error("Aucune donnée à traiter. Veuillez d'abord charger un fichier.")
        return
    
    st.title("Gestion des Valeurs Manquantes")
    
    # Récupérer les données originales
    data = st.session_state.original_data.copy()
    
    # Nettoyer les données pour l'affichage
    clean_data = clean_dataframe_for_display(data)
    
    # Afficher un aperçu des données originales
    st.subheader("Aperçu des Données")
    st.dataframe(clean_data.head())
    
    # Analyser les valeurs manquantes
    missing_total = data.isnull().sum().sum()
    missing_by_col = data.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0].index.tolist()
    
    # Afficher les statistiques des valeurs manquantes
    st.subheader("Analyse des Valeurs Manquantes")
    st.write(f"Nombre total de valeurs manquantes: **{missing_total}**")
    
    # Créer un dataframe pour afficher les colonnes avec valeurs manquantes
    missing_df = pd.DataFrame({
        'Colonne': missing_cols,
        'Valeurs manquantes': [missing_by_col[col] for col in missing_cols],
        'Pourcentage': [f"{missing_by_col[col] / len(data) * 100:.2f}%" for col in missing_cols]
    })
    
    st.write("Valeurs manquantes par colonne:")
    st.dataframe(missing_df)
    
    # Visualiser les valeurs manquantes
    st.subheader("Visualisation des Valeurs Manquantes")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data[missing_cols].isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax)
    ax.set_title('Heatmap des valeurs manquantes')
    st.pyplot(fig)
    
    # Graphique à barres
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_by_col[missing_cols].sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Nombre de valeurs manquantes par colonne')
    ax.set_ylabel('Nombre de valeurs manquantes')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sélection des méthodes de traitement
    st.subheader("Traitement des Valeurs Manquantes")
    
    # Créer des onglets pour les différentes méthodes
    treatment_tabs = st.tabs([
        "Suppression", 
        "Imputation Simple", 
        "Imputation Avancée",
        "Résultats"
    ])
    
    # Variable pour stocker les données traitées
    treated_data = None
    
    # Onglet 1: Suppression
    with treatment_tabs[0]:
        st.write("### Suppression des lignes ou colonnes avec valeurs manquantes")
        
        removal_method = st.radio(
            "Méthode de suppression:",
            ["Supprimer les lignes", "Supprimer les colonnes"]
        )
        
        if removal_method == "Supprimer les lignes":
            threshold = st.slider(
                "Seuil de suppression (% maximum de valeurs manquantes par ligne)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Les lignes avec un pourcentage de valeurs manquantes supérieur à ce seuil seront supprimées"
            )
            
            if st.button("Appliquer la suppression de lignes"):
                # Calcul du pourcentage de valeurs manquantes par ligne
                missing_percentage = data.isnull().mean(axis=1) * 100
                
                # Filtrer les lignes avec moins de valeurs manquantes que le seuil
                filtered_data = data[missing_percentage <= threshold]
                
                # Afficher les résultats
                removed_rows = len(data) - len(filtered_data)
                st.info(f"{removed_rows} lignes supprimées sur {len(data)} ({removed_rows/len(data)*100:.2f}%)")
                
                treated_data = filtered_data
                st.session_state.processed_data = treated_data
                
                # Montrer un aperçu
                st.write("Aperçu des données après suppression:")
                st.dataframe(clean_dataframe_for_display(treated_data).head())
                
                # Mettre à jour les statistiques de valeurs manquantes
                new_missing = treated_data.isnull().sum().sum()
                st.write(f"Valeurs manquantes restantes: {new_missing} (sur {missing_total} initialement)")
        
        else:  # Supprimer les colonnes
            col_threshold = st.slider(
                "Seuil de suppression (% maximum de valeurs manquantes par colonne)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Les colonnes avec un pourcentage de valeurs manquantes supérieur à ce seuil seront supprimées"
            )
            
            if st.button("Appliquer la suppression de colonnes"):
                # Calculer le pourcentage de valeurs manquantes par colonne
                missing_percentage = data.isnull().mean() * 100
                
                # Sélectionner les colonnes à conserver
                columns_to_keep = missing_percentage[missing_percentage <= col_threshold].index.tolist()
                filtered_data = data[columns_to_keep]
                
                # Afficher les résultats
                removed_cols = len(data.columns) - len(filtered_data.columns)
                st.info(f"{removed_cols} colonnes supprimées sur {len(data.columns)} ({removed_cols/len(data.columns)*100:.2f}%)")
                
                treated_data = filtered_data
                st.session_state.processed_data = treated_data
                
                # Montrer un aperçu
                st.write("Aperçu des données après suppression:")
                st.dataframe(clean_dataframe_for_display(treated_data).head())
                
                # Mettre à jour les statistiques de valeurs manquantes
                new_missing = treated_data.isnull().sum().sum()
                st.write(f"Valeurs manquantes restantes: {new_missing} (sur {missing_total} initialement)")
    
    # Onglet 2: Imputation Simple
    with treatment_tabs[1]:
        st.write("### Imputation simple des valeurs manquantes")
        
        # Séparer les colonnes numériques et catégorielles
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        
        # Obtenir les colonnes avec valeurs manquantes
        numeric_missing = [col for col in numeric_cols if col in missing_cols]
        categorical_missing = [col for col in categorical_cols if col in missing_cols]
        
        # Méthodes d'imputation pour colonnes numériques
        if numeric_missing:
            st.write("#### Colonnes numériques avec valeurs manquantes:")
            st.write(", ".join(numeric_missing))
            
            numeric_method = st.selectbox(
                "Méthode d'imputation pour variables numériques:",
                ["Moyenne", "Médiane", "Constante"]
            )
            
            numeric_constant = None
            if numeric_method == "Constante":
                numeric_constant = st.number_input("Valeur constante pour l'imputation numérique:", value=0.0)
        
        # Méthodes d'imputation pour colonnes catégorielles
        if categorical_missing:
            st.write("#### Colonnes catégorielles avec valeurs manquantes:")
            st.write(", ".join(categorical_missing))
            
            categorical_method = st.selectbox(
                "Méthode d'imputation pour variables catégorielles:",
                ["Mode (valeur la plus fréquente)", "Constante"]
            )
            
            categorical_constant = None
            if categorical_method == "Constante":
                categorical_constant = st.text_input("Valeur constante pour l'imputation catégorielle:", value="Inconnu")
        
        if st.button("Appliquer l'imputation simple"):
            imputed_data = data.copy()
            
            # Imputer les colonnes numériques
            if numeric_missing:
                for col in numeric_missing:
                    if numeric_method == "Moyenne":
                        imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mean())
                    elif numeric_method == "Médiane":
                        imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())
                    else:  # Constante
                        imputed_data[col] = imputed_data[col].fillna(numeric_constant)
            
            # Imputer les colonnes catégorielles
            if categorical_missing:
                for col in categorical_missing:
                    if categorical_method == "Mode (valeur la plus fréquente)":
                        imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mode()[0])
                    else:  # Constante
                        imputed_data[col] = imputed_data[col].fillna(categorical_constant)
            
            treated_data = imputed_data
            st.session_state.processed_data = treated_data
            
            # Afficher les résultats
            st.write("Aperçu des données après imputation:")
            st.dataframe(clean_dataframe_for_display(treated_data).head())
            
            # Mettre à jour les statistiques de valeurs manquantes
            new_missing = treated_data.isnull().sum().sum()
            st.success(f"Valeurs manquantes comblées: {missing_total - new_missing} sur {missing_total}")
            st.write(f"Valeurs manquantes restantes: {new_missing}")
    
    # Onglet 3: Imputation Avancée
    with treatment_tabs[2]:
        st.write("### Méthodes d'imputation avancées")
        
        advanced_method = st.selectbox(
            "Méthode d'imputation avancée:",
            ["KNN (k plus proches voisins)", "MICE (Imputation multiple par équations chaînées)"]
        )
        
        # Ajout d'une option pour le mode rapide
        fast_mode = st.checkbox(
            "Mode rapide (recommandé pour grands jeux de données)", 
            value=True,
            help="Limite le nombre de colonnes et d'échantillons pour accélérer le traitement"
        )
        
        max_cols = None
        if fast_mode:
            st.info("⚡ Le mode rapide est activé. L'imputation sera plus rapide mais potentiellement moins précise.")
            # Permettre à l'utilisateur de définir une limite de colonnes à traiter
            max_cols = st.slider(
                "Nombre maximum de colonnes à traiter",
                min_value=5,
                max_value=30,
                value=15,
                help="Limiter le nombre de colonnes pour accélérer le traitement"
            )
        
        if advanced_method == "KNN (k plus proches voisins)":
            n_neighbors = st.slider(
                "Nombre de voisins (k)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Nombre de voisins à considérer pour l'imputation KNN"
            )
            
            if st.button("Appliquer l'imputation KNN"):
                # Utiliser le module missing_values_handler avec la méthode KNN
                from .missing_values_handler import MissingValuesHandler
                
                # Message d'information sur le traitement
                info_msg = st.info("Préparation des données pour l'imputation KNN... Veuillez patienter.")
                
                # Si mode rapide est activé, sélectionner un sous-ensemble de colonnes
                if fast_mode and max_cols and len(data.columns) > max_cols:
                    # Prioriser les colonnes avec valeurs manquantes
                    missing_counts = data.isnull().sum()
                    cols_to_process = list(missing_counts[missing_counts > 0].index)
                    
                    # Ajouter des colonnes sans valeurs manquantes jusqu'à atteindre max_cols
                    other_cols = [c for c in data.columns if c not in cols_to_process]
                    import random
                    random.shuffle(other_cols)
                    cols_to_process.extend(other_cols[:max(0, max_cols - len(cols_to_process))])
                    
                    # Utiliser seulement ce sous-ensemble
                    subset_data = data[cols_to_process].copy()
                    st.write(f"Traitement limité à {len(cols_to_process)} colonnes sur {len(data.columns)} pour des raisons de performance.")
                    handler = MissingValuesHandler(subset_data)
                else:
                    handler = MissingValuesHandler(data)
                
                # Mise à jour du message
                info_msg.info("Application de l'imputation KNN... Cette opération peut prendre du temps.")
                
                # Créer une barre de progression
                progress_bar = st.progress(0)
                
                # Fonction pour mettre à jour la progression
                def update_progress(progress):
                    progress_bar.progress(int(progress * 100))
                
                # Appliquer l'imputation KNN avec callback de progression
                try:
                    advanced_data = handler.apply_imputation('knn_imputation', n_neighbors=n_neighbors, progress_callback=update_progress)
                    
                    # Si on a utilisé un sous-ensemble, réintégrer les colonnes dans le jeu de données original
                    if fast_mode and max_cols and len(data.columns) > max_cols:
                        # Copier les données originales et mettre à jour seulement les colonnes traitées
                        full_data = data.copy()
                        for col in cols_to_process:
                            # Vérifier si la colonne existe encore dans les données traitées 
                            # (elle pourrait avoir été transformée par one-hot encoding)
                            if col in advanced_data.columns:
                                full_data[col] = advanced_data[col]
                            else:
                                st.info(f"La colonne {col} a été transformée ou supprimée pendant le traitement.")
                                # Chercher des colonnes dérivées (après one-hot encoding par exemple)
                                derived_cols = [c for c in advanced_data.columns if c.startswith(f"{col}_")]
                                if derived_cols:
                                    st.info(f"Colonnes dérivées trouvées: {', '.join(derived_cols)}")
                                    # Ajouter ces colonnes au DataFrame complet
                                    for derived_col in derived_cols:
                                        full_data[derived_col] = advanced_data[derived_col]
                        advanced_data = full_data
                    
                    # Finaliser
                    progress_bar.progress(100)
                    info_msg.success("Imputation KNN terminée avec succès!")
                    
                    treated_data = advanced_data
                    st.session_state.processed_data = treated_data
                    
                    # Afficher les résultats
                    st.write("Aperçu des données après imputation KNN:")
                    st.dataframe(clean_dataframe_for_display(treated_data).head())
                    
                    # Vérifier les valeurs manquantes restantes
                    missing_total = data.isnull().sum().sum()
                    new_missing = treated_data.isnull().sum().sum()
                    st.success(f"Valeurs manquantes comblées: {missing_total - new_missing} sur {missing_total}")
                    st.write(f"Valeurs manquantes restantes: {new_missing}")
                except Exception as e:
                    info_msg.error(f"Erreur lors de l'imputation KNN: {str(e)}")
                    st.exception(e)
        
        else:  # MICE
            max_iter = st.slider(
                "Nombre maximum d'itérations",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Nombre maximum d'itérations pour l'imputation itérative"
            )
            
            if st.button("Appliquer l'imputation MICE"):
                # Utiliser le module missing_values_handler avec la méthode MICE
                from .missing_values_handler import MissingValuesHandler
                
                # Message d'information sur le traitement
                info_msg = st.info("Préparation des données pour l'imputation MICE... Veuillez patienter.")
                
                # Si mode rapide est activé, sélectionner un sous-ensemble de colonnes
                if fast_mode and max_cols and len(data.columns) > max_cols:
                    # Prioriser les colonnes avec valeurs manquantes
                    missing_counts = data.isnull().sum()
                    cols_to_process = list(missing_counts[missing_counts > 0].index)
                    
                    # Ajouter des colonnes sans valeurs manquantes jusqu'à atteindre max_cols
                    other_cols = [c for c in data.columns if c not in cols_to_process]
                    import random
                    random.shuffle(other_cols)
                    cols_to_process.extend(other_cols[:max(0, max_cols - len(cols_to_process))])
                    
                    # Utiliser seulement ce sous-ensemble
                    subset_data = data[cols_to_process].copy()
                    st.write(f"Traitement limité à {len(cols_to_process)} colonnes sur {len(data.columns)} pour des raisons de performance.")
                    handler = MissingValuesHandler(subset_data)
                else:
                    handler = MissingValuesHandler(data)
                
                # Mise à jour du message
                info_msg.info("Application de l'imputation MICE... Cette opération peut prendre du temps.")
                
                # Créer une barre de progression
                progress_bar = st.progress(0)
                
                # Fonction pour mettre à jour la progression
                def update_progress(progress):
                    progress_bar.progress(int(progress * 100))
                
                # Appliquer l'imputation MICE avec callback de progression
                try:
                    advanced_data = handler.apply_imputation('iterative_imputation', max_iter=max_iter, progress_callback=update_progress)
                    
                    # Si on a utilisé un sous-ensemble, réintégrer les colonnes dans le jeu de données original
                    if fast_mode and max_cols and len(data.columns) > max_cols:
                        # Copier les données originales et mettre à jour seulement les colonnes traitées
                        full_data = data.copy()
                        for col in cols_to_process:
                            # Vérifier si la colonne existe encore dans les données traitées 
                            # (elle pourrait avoir été transformée par one-hot encoding)
                            if col in advanced_data.columns:
                                full_data[col] = advanced_data[col]
                            else:
                                st.info(f"La colonne {col} a été transformée ou supprimée pendant le traitement.")
                                # Chercher des colonnes dérivées (après one-hot encoding par exemple)
                                derived_cols = [c for c in advanced_data.columns if c.startswith(f"{col}_")]
                                if derived_cols:
                                    st.info(f"Colonnes dérivées trouvées: {', '.join(derived_cols)}")
                                    # Ajouter ces colonnes au DataFrame complet
                                    for derived_col in derived_cols:
                                        full_data[derived_col] = advanced_data[derived_col]
                        advanced_data = full_data
                    
                    # Finaliser
                    progress_bar.progress(100)
                    info_msg.success("Imputation MICE terminée avec succès!")
                    
                    treated_data = advanced_data
                    st.session_state.processed_data = treated_data
                    
                    # Afficher les résultats
                    st.write("Aperçu des données après imputation MICE:")
                    st.dataframe(clean_dataframe_for_display(treated_data).head())
                    
                    # Vérifier les valeurs manquantes restantes
                    missing_total = data.isnull().sum().sum()
                    new_missing = treated_data.isnull().sum().sum()
                    st.success(f"Valeurs manquantes comblées: {missing_total - new_missing} sur {missing_total}")
                    st.write(f"Valeurs manquantes restantes: {new_missing}")
                except Exception as e:
                    info_msg.error(f"Erreur lors de l'imputation MICE: {str(e)}")
                    st.exception(e)
    
    # Onglet 4: Résultats
    with treatment_tabs[3]:
        st.write("### Téléchargement et Finalisation")
        
        if hasattr(st.session_state, 'processed_data'):
            treated_data = st.session_state.processed_data
            
            # Statistiques après traitement
            st.write("#### Statistiques après traitement")
            
            # Vérifier les valeurs manquantes restantes
            new_missing = treated_data.isnull().sum().sum()
            new_missing_by_col = treated_data.isnull().sum()
            new_missing_cols = new_missing_by_col[new_missing_by_col > 0].index.tolist()
            
            # Créer un récapitulatif
            st.write(f"**Valeurs manquantes comblées:** {missing_total - new_missing} sur {missing_total}")
            st.write(f"**Taux de complétion:** {((missing_total - new_missing) / missing_total * 100):.2f}%")
            
            if new_missing > 0:
                st.warning(f"**Valeurs manquantes restantes:** {new_missing}")
                
                # Afficher les colonnes avec valeurs manquantes restantes
                st.write("Colonnes avec valeurs manquantes restantes:")
                st.dataframe(pd.DataFrame({
                    'Colonne': new_missing_cols,
                    'Valeurs manquantes': [new_missing_by_col[col] for col in new_missing_cols],
                    'Pourcentage': [f"{new_missing_by_col[col] / len(treated_data) * 100:.2f}%" for col in new_missing_cols]
                }))
            else:
                st.success("**Toutes les valeurs manquantes ont été traitées !**")
            
            # Téléchargement des données traitées
            st.write("#### Télécharger les données traitées")
            
            # Nettoyer et sauvegarder les données traitées en CSV
            clean_treated_data = clean_dataframe_for_display(treated_data)
            csv = clean_treated_data.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="donnees_traitees.csv">Télécharger les données traitées (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Bouton pour finaliser et retourner au traitement
            if st.button("Finaliser et retourner au traitement", type="primary"):
                if 'return_to' in st.session_state:
                    # Effacer le flag de page des valeurs manquantes
                    del st.session_state.show_missing_values_page
                    # Conserver les données traitées et le flag de retour
                    # Ils seront utilisés dans la page principale
                    st.rerun()
        else:
            st.info("Veuillez d'abord appliquer une méthode de traitement des valeurs manquantes.")

def clean_dataframe_for_display(df):
    """
    Nettoie un DataFrame pour s'assurer qu'il peut être affiché correctement dans Streamlit.
    
    Args:
        df: Le DataFrame à nettoyer
        
    Returns:
        DataFrame nettoyé avec des types compatibles
    """
    # Créer une copie pour éviter de modifier l'original
    clean_df = df.copy()
    
    # Parcourir toutes les colonnes
    for col in clean_df.columns:
        # Si la colonne est de type objet, convertir en string
        if clean_df[col].dtype == 'object':
            # Essayer de convertir les objets en string
            try:
                clean_df[col] = clean_df[col].apply(lambda x: str(x) if x is not None else None)
            except Exception:
                # En cas d'échec, remplacer par des chaînes vides
                clean_df[col] = clean_df[col].apply(lambda x: "" if pd.isna(x) else str(x))
        
        # Pour les booléens, convertir en entiers (0/1)
        elif clean_df[col].dtype == 'bool':
            clean_df[col] = clean_df[col].astype(int)
    
    return clean_df

def render_ai_methods_tab():
    """Rendre l'onglet des méthodes d'IA dans Streamlit"""
    # Vérifier si nous devons afficher la page de gestion des valeurs manquantes
    if hasattr(st.session_state, 'show_missing_values_page') and st.session_state.show_missing_values_page:
        show_missing_values_interface()
        return
    
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
                            # Créer une page dédiée pour la gestion des valeurs manquantes
                            st.session_state.show_missing_values_page = True
                            st.session_state.original_data = real_data.copy()
                            # Mettre un flag pour indiquer où revenir après traitement
                            st.session_state.return_to = "ai_methods_tab"
                            st.rerun()
                    
                    # Vérifier si nous revenons du traitement des valeurs manquantes
                    if hasattr(st.session_state, 'processed_data') and st.session_state.get('return_to') == "ai_methods_tab":
                        # Récupérer les données traitées
                        real_data = st.session_state.processed_data
                        # Nettoyer les variables de session
                        st.success("🎉 Les valeurs manquantes ont été traitées avec succès !")
                        del st.session_state.processed_data
                        del st.session_state.return_to
                    
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
                                
                                # Nettoyer le DataFrame pour l'affichage
                                clean_synthetic_df = clean_dataframe_for_display(synthetic_df)
                                
                                # Afficher les données générées
                                st.write("Aperçu des données synthétiques:")
                                st.write(clean_synthetic_df.head())
                                
                                # Afficher les types de données
                                st.write("Types des colonnes générées:")
                                st.write(clean_synthetic_df.dtypes)
                                
                                # Option de téléchargement
                                st.download_button(
                                    label="Télécharger les Données Synthétiques (CSV)",
                                    data=clean_synthetic_df.to_csv(index=False).encode('utf-8'),
                                    file_name="donnees_synthetiques.csv",
                                    mime="text/csv"
                                )
                                
                                # Visualisations comparatives
                                st.subheader("Visualisations Comparatives")
                                
                                for col in metadata.categorical_cols:
                                    if clean_synthetic_df[col].nunique() < 15:  # Seulement pour les colonnes avec un nombre raisonnable de catégories
                                        fig = visualize_categorical_comparison(real_data, clean_synthetic_df, col)
                                        st.pyplot(fig)

                                # Pour les colonnes numériques
                                for col in metadata.numeric_cols:
                                    fig = visualize_numeric_comparison(real_data, clean_synthetic_df, col)
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
                                        st.write(clean_synthetic_df[metadata.numeric_cols].describe())
                                
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