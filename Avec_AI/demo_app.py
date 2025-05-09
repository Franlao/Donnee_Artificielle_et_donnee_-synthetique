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
import json
from typing import Dict, List, Union, Tuple, Any, Optional
import uuid
from .missing_methods_tab import show_missing_values_tab

# Fonctions pour générer des données artificielles
def generate_artificial_data(params: Dict[str, Any], n_samples: int = 1000) -> pd.DataFrame:
    """
    Génère des données artificielles selon les paramètres fournis par l'utilisateur.
    
    Args:
        params: Dictionnaire contenant les paramètres de génération pour chaque variable
        n_samples: Nombre d'échantillons à générer
        
    Returns:
        DataFrame contenant les données générées
    """
    data = {}

    # Génération des variables selon leurs types et distributions
    for feature, feature_params in params.items():
        # On s'assure que feature_params est un dictionnaire et contient 'type'
        if not isinstance(feature_params, dict) or feature_params.get('type') is None:
            continue  # On ignore les clés non concernées (ex: 'correlations')
            
        if feature_params['type'] == 'numeric':
            if feature_params['distribution'] == 'normal':
                data[feature] = np.random.normal(
                    feature_params['mean'], 
                    feature_params['std'], 
                    n_samples
                )
            elif feature_params['distribution'] == 'uniform':
                data[feature] = np.random.uniform(
                    feature_params['min'], 
                    feature_params['max'], 
                    n_samples
                )
            elif feature_params['distribution'] == 'exponential':
                data[feature] = np.random.exponential(
                    feature_params['scale'], 
                    n_samples
                )
            elif feature_params['distribution'] == 'beta':
                data[feature] = np.random.beta(
                    feature_params['alpha'],
                    feature_params['beta'],
                    n_samples
                )
            elif feature_params['distribution'] == 'gamma':
                data[feature] = np.random.gamma(
                    feature_params['shape'],
                    feature_params['scale'],
                    n_samples
                )
            elif feature_params['distribution'] == 'poisson':
                data[feature] = np.random.poisson(
                    feature_params['lam'],
                    n_samples
                )
                
            # Application des contraintes sur les valeurs
            # 1. Contraindre dans une plage spécifique si demandé
            if feature_params.get('constrain_range', False):
                min_val = feature_params.get('range_min', None)
                max_val = feature_params.get('range_max', None)
                
                if min_val is not None and max_val is not None:
                    # Tronquer les valeurs en dehors de la plage
                    data[feature] = np.clip(data[feature], min_val, max_val)
            
            # 2. Forcer les valeurs à être strictement positives si demandé
            if feature_params.get('strictly_positive', False):
                # Remplacer les valeurs négatives par leur valeur absolue ou par une petite valeur positive
                data[feature] = np.maximum(data[feature], 0.0001)
                
        elif feature_params['type'] == 'categorical':
            categories = feature_params.get('categories', [])
            probabilities = feature_params.get('probabilities', [])

            if not categories or len(categories) != len(probabilities):
                st.error(f"Erreur : Problème avec les catégories ou probabilités pour '{feature}'.")
                st.write(f"Catégories: {categories}")
                st.write(f"Probabilités: {probabilities}")
                continue  # Passe à la prochaine feature

            # Normaliser les probabilités si nécessaire
            prob_sum = sum(probabilities)
            if not np.isclose(prob_sum, 1):
                st.warning(f"⚠️ Correction : Les probabilités pour '{feature}' ne somment pas à 1 ({prob_sum}). Normalisation en cours...")
                probabilities = [p / prob_sum for p in probabilities]

            data[feature] = np.random.choice(categories, n_samples, p=probabilities)

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Appliquer les corrélations si spécifiées
    if 'correlations' in params:
        # On ne considère que les features numériques
        numeric_features = [f for f, p in params.items() 
                            if isinstance(p, dict) and p.get('type') == 'numeric']
        
        if len(numeric_features) >= 2:
            X = df[numeric_features].values

            # Standardisation des données
            X = StandardScaler().fit_transform(X)

            # Construire la matrice de corrélation cible
            target_corr = np.eye(len(numeric_features))
            for i, feat1 in enumerate(numeric_features):
                for j, feat2 in enumerate(numeric_features):
                    if i != j and (feat1, feat2) in params['correlations']:
                        target_corr[i, j] = params['correlations'][(feat1, feat2)]
                        target_corr[j, i] = params['correlations'][(feat1, feat2)]

            # Décomposition en valeurs propres
            eigenvalues, eigenvectors = np.linalg.eigh(target_corr)
            # Sécuriser les valeurs propres pour éviter les problèmes numériques
            eigenvalues = np.maximum(eigenvalues, 1e-6)

            # Transformation des données pour introduire la corrélation
            L = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            X_corr = X @ L.T

            # Retransformation selon les distributions originales
            for i, feature in enumerate(numeric_features):
                if params[feature]['distribution'] == 'normal':
                    df[feature] = X_corr[:, i]
                    df[feature] = df[feature] * params[feature]['std'] + params[feature]['mean']
                elif params[feature]['distribution'] == 'uniform':
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    scale = params[feature]['max'] - params[feature]['min']
                    df[feature] = params[feature]['min'] + unif_values * scale
                elif params[feature]['distribution'] == 'exponential':
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    df[feature] = stats.expon.ppf(unif_values, scale=params[feature]['scale'])
                elif params[feature]['distribution'] == 'beta':
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    df[feature] = stats.beta.ppf(unif_values, a=params[feature]['alpha'], b=params[feature]['beta'])
                elif params[feature]['distribution'] == 'gamma':
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    df[feature] = stats.gamma.ppf(unif_values, a=params[feature]['shape'], scale=params[feature]['scale'])
                elif params[feature]['distribution'] == 'poisson':
                    # Pour Poisson, cette méthode est approximative car il s'agit d'une distribution discrète
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    df[feature] = stats.poisson.ppf(unif_values, mu=params[feature]['lam'])

    # Convertir les variables qui devraient être des entiers en entiers
    for feature, feature_params in params.items():
        if isinstance(feature_params, dict) and feature_params.get('type') == 'numeric':
            if feature_params.get('integer', False):
                df[feature] = np.round(df[feature]).astype(int)

    # Convertir les variables catégorielles en type catégoriel si spécifié
    for feature, feature_params in params.items():
        if isinstance(feature_params, dict) and feature_params.get('type') == 'categorical':
            if feature_params.get('use_categorical_type', False):
                df[feature] = pd.Categorical(df[feature])

    return df

# Fonctions pour la visualisation
def plot_numeric_comparison(real_data, synthetic_data, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(real_data[column], color='blue', alpha=0.5, label='Données réelles', kde=True, ax=ax)
    sns.histplot(synthetic_data[column], color='red', alpha=0.5, label='Données synthétiques', kde=True, ax=ax)
    
    ax.set_title(f'Comparaison des distributions pour {column}')
    ax.legend()
    
    return fig

def plot_categorical_comparison(real_data, synthetic_data, column):
    real_counts = real_data[column].value_counts(normalize=True).sort_index()
    synth_counts = synthetic_data[column].value_counts(normalize=True).sort_index()
    
    all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
    real_counts = real_counts.reindex(all_categories, fill_value=0)
    synth_counts = synth_counts.reindex(all_categories, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(all_categories))
    width = 0.35
    
    ax.bar(x - width/2, real_counts, width, label='Données réelles', color='blue', alpha=0.7)
    ax.bar(x + width/2, synth_counts, width, label='Données synthétiques', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.set_title(f'Comparaison des distributions pour {column}')
    ax.set_ylabel('Proportion')
    ax.legend()
    
    fig.tight_layout()
    
    return fig

def plot_correlation_comparison(real_data, synthetic_data):
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    real_corr = real_data[numeric_cols].corr()
    synthetic_corr = synthetic_data[numeric_cols].corr()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Corrélations - Données réelles')
    
    sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Corrélations - Données synthétiques')
    
    diff_corr = np.abs(real_corr - synthetic_corr)
    sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='Reds', ax=axes[2])
    axes[2].set_title('Différence absolue des corrélations')
    
    fig.tight_layout()
    
    return fig

def plot_pca_comparison(real_data, synthetic_data):
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    real_data_numeric = real_data[numeric_cols].copy()
    synthetic_data_numeric = synthetic_data[numeric_cols].copy()
    
    real_data_numeric['source'] = 'Réel'
    synthetic_data_numeric['source'] = 'Synthétique'
    
    combined_data = pd.concat([real_data_numeric, synthetic_data_numeric], ignore_index=True)
    
    sources = combined_data['source']
    combined_data = combined_data.drop('source', axis=1)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['source'] = sources
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for source, color in zip(['Réel', 'Synthétique'], ['blue', 'red']):
        subset = pca_df[pca_df['source'] == source]
        ax.scatter(subset['PC1'], subset['PC2'], c=color, alpha=0.5, label=source)
    
    feature_names = numeric_cols
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                 head_width=0.05, head_length=0.05, fc='k', ec='k')
        ax.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, feature)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance expliquée)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance expliquée)')
    ax.set_title('Analyse en Composantes Principales')
    ax.legend()
    ax.grid(True)
    
    return fig

# Initialisation de l'état de session au besoin
if 'variables' not in st.session_state:
    st.session_state.variables = {}
    
if 'correlations' not in st.session_state:
    st.session_state.correlations = {}

def render_dynamic_artificial_data_tab():
    """
    Interface utilisateur dynamique pour la génération de données artificielles
    permettant d'ajouter, configurer et supprimer des variables à volonté.
    """
    st.header("Génération de Données Artificielles")
    st.write("""
    Dans cette section, vous pouvez définir et configurer vos propres variables
    pour générer des données artificielles personnalisées selon vos besoins.
    """)
    
    st.subheader("Configuration des Variables")
    
    # Section pour ajouter une nouvelle variable
    with st.expander("Ajouter une nouvelle variable", expanded=len(st.session_state.variables) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            new_var_name = st.text_input("Nom de la variable", key="new_var_name")
            new_var_type = st.selectbox("Type de variable", ["numeric", "categorical"], key="new_var_type")
            
        with col2:
            if new_var_type == "numeric":
                new_var_dist = st.selectbox("Distribution", 
                                        ["normal", "uniform", "exponential", "beta", "gamma", "poisson"], 
                                        key="new_var_dist")
                is_integer = st.checkbox("Valeur entière (arrondie)", key="new_var_int")
            else:
                new_var_categorical = st.text_input("Catégories (séparées par des virgules)", 
                                               key="new_var_cat")
                use_categorical_type = st.checkbox("Utiliser le type Pandas Categorical", 
                                              key="new_var_cat_type", 
                                              value=True)
        
        if st.button("Ajouter la variable"):
            if not new_var_name:
                st.error("Le nom de la variable ne peut pas être vide.")
            elif new_var_name in st.session_state.variables:
                st.error(f"Une variable nommée '{new_var_name}' existe déjà.")
            else:
                var_id = str(uuid.uuid4())
                
                if new_var_type == "numeric":
                    st.session_state.variables[new_var_name] = {
                        "id": var_id,
                        "type": "numeric",
                        "distribution": new_var_dist,
                        "integer": is_integer,
                        "strictly_positive": False,
                        "constrain_range": False,
                        "range_min": 0.0,
                        "range_max": 100.0
                    }
                    
                    # Ajouter les paramètres selon la distribution
                    if new_var_dist == "normal":
                        st.session_state.variables[new_var_name].update({
                            "mean": 0.0,
                            "std": 1.0
                        })
                    elif new_var_dist == "uniform":
                        st.session_state.variables[new_var_name].update({
                            "min": 0.0,
                            "max": 1.0
                        })
                    elif new_var_dist == "exponential":
                        st.session_state.variables[new_var_name].update({
                            "scale": 1.0
                        })
                    elif new_var_dist == "beta":
                        st.session_state.variables[new_var_name].update({
                            "alpha": 2.0,
                            "beta": 2.0
                        })
                    elif new_var_dist == "gamma":
                        st.session_state.variables[new_var_name].update({
                            "shape": 2.0,
                            "scale": 1.0
                        })
                    elif new_var_dist == "poisson":
                        st.session_state.variables[new_var_name].update({
                            "lam": 5.0
                        })
                else:
                    categories = [c.strip() for c in new_var_categorical.split(",") if c.strip()]
                    if not categories:
                        categories = ["A", "B"]
                    
                    # Probabilités équiprobables
                    probabilities = [1.0 / len(categories)] * len(categories)
                    
                    st.session_state.variables[new_var_name] = {
                        "id": var_id,
                        "type": "categorical",
                        "categories": categories,
                        "probabilities": probabilities,
                        "use_categorical_type": use_categorical_type
                    }
                
                st.success(f"Variable '{new_var_name}' ajoutée avec succès!")
                st.rerun()
    
    # Affichage et configuration des variables existantes
    if st.session_state.variables:
        st.subheader("Variables configurées")
        
        for var_name, var_config in list(st.session_state.variables.items()):
            with st.expander(f"{var_name} ({var_config['type']})", expanded=False):
                col1, col2, col3 = st.columns([3, 3, 1])
                
                with col3:
                    if st.button("Supprimer", key=f"del_{var_config['id']}"):
                        del st.session_state.variables[var_name]
                        
                        # Supprimer toutes les corrélations impliquant cette variable
                        for corr_key in list(st.session_state.correlations.keys()):
                            if var_name in corr_key:
                                del st.session_state.correlations[corr_key]
                        
                        st.success(f"Variable '{var_name}' supprimée.")
                        st.rerun()
                        continue  # Skip the rest for this deleted variable
                
                if var_config['type'] == 'numeric':
                    with col1:
                        dist = var_config['distribution']
                        
                        if dist == "normal":
                            var_config['mean'] = st.slider(
                                f"Moyenne pour {var_name}", 
                                min_value=-100.0, 
                                max_value=100.0, 
                                value=float(var_config.get('mean', 0.0)),
                                step=0.1,
                                key=f"mean_{var_config['id']}"
                            )
                            
                            var_config['std'] = st.slider(
                                f"Écart-type pour {var_name}", 
                                min_value=0.1, 
                                max_value=50.0, 
                                value=float(var_config.get('std', 1.0)),
                                step=0.1,
                                key=f"std_{var_config['id']}"
                            )
                        
                        elif dist == "uniform":
                            var_config['min'] = st.slider(
                                f"Minimum pour {var_name}", 
                                min_value=-100.0, 
                                max_value=100.0, 
                                value=float(var_config.get('min', 0.0)),
                                step=0.1,
                                key=f"min_{var_config['id']}"
                            )
                            
                            var_config['max'] = st.slider(
                                f"Maximum pour {var_name}", 
                                min_value=-100.0, 
                                max_value=100.0, 
                                value=float(var_config.get('max', 1.0)),
                                step=0.1,
                                key=f"max_{var_config['id']}"
                            )
                            
                            if var_config['min'] >= var_config['max']:
                                st.error(f"Le minimum doit être inférieur au maximum pour {var_name}.")
                                var_config['max'] = var_config['min'] + 1.0
                        
                        elif dist == "exponential":
                            var_config['scale'] = st.slider(
                                f"Paramètre d'échelle (λ) pour {var_name}", 
                                min_value=0.1, 
                                max_value=50.0, 
                                value=float(var_config.get('scale', 1.0)),
                                step=0.1,
                                key=f"scale_{var_config['id']}"
                            )
                        
                        elif dist == "beta":
                            var_config['alpha'] = st.slider(
                                f"Alpha pour {var_name}", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(var_config.get('alpha', 2.0)),
                                step=0.1,
                                key=f"alpha_{var_config['id']}"
                            )
                            
                            var_config['beta'] = st.slider(
                                f"Beta pour {var_name}", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(var_config.get('beta', 2.0)),
                                step=0.1,
                                key=f"beta_{var_config['id']}"
                            )
                        
                        elif dist == "gamma":
                            var_config['shape'] = st.slider(
                                f"Paramètre de forme (k) pour {var_name}", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(var_config.get('shape', 2.0)),
                                step=0.1,
                                key=f"shape_{var_config['id']}"
                            )
                            
                            var_config['scale'] = st.slider(
                                f"Paramètre d'échelle (θ) pour {var_name}", 
                                min_value=0.1, 
                                max_value=10.0, 
                                value=float(var_config.get('scale', 1.0)),
                                step=0.1,
                                key=f"scale_gamma_{var_config['id']}"
                            )
                        
                        elif dist == "poisson":
                            var_config['lam'] = st.slider(
                                f"Lambda (λ) pour {var_name}", 
                                min_value=0.1, 
                                max_value=50.0, 
                                value=float(var_config.get('lam', 5.0)),
                                step=0.1,
                                key=f"lam_{var_config['id']}"
                            )
                        
                        # Options pour contraindre les valeurs
                        var_config['strictly_positive'] = st.checkbox(
                            f"Valeurs strictement positives pour {var_name}", 
                            value=var_config.get('strictly_positive', False),
                            key=f"pos_{var_config['id']}"
                        )
                        
                        var_config['constrain_range'] = st.checkbox(
                            f"Contraindre dans une plage pour {var_name}", 
                            value=var_config.get('constrain_range', False),
                            key=f"range_{var_config['id']}"
                        )
                        
                        if var_config.get('constrain_range', False):
                            range_col1, range_col2 = st.columns(2)
                            with range_col1:
                                var_config['range_min'] = st.number_input(
                                    f"Minimum plage pour {var_name}", 
                                    value=float(var_config.get('range_min', 0.0)),
                                    step=0.1,
                                    key=f"range_min_{var_config['id']}"
                                )
                            with range_col2:
                                var_config['range_max'] = st.number_input(
                                    f"Maximum plage pour {var_name}", 
                                    value=float(var_config.get('range_max', 100.0)),
                                    step=0.1,
                                    key=f"range_max_{var_config['id']}"
                                )
                            
                            if var_config['range_min'] >= var_config['range_max']:
                                st.error(f"Le minimum de la plage doit être inférieur au maximum pour {var_name}.")
                                var_config['range_max'] = var_config['range_min'] + 1.0
                    
                    with col2:
                        var_config['integer'] = st.checkbox(
                            f"Valeur entière pour {var_name}", 
                            value=var_config.get('integer', False),
                            key=f"int_{var_config['id']}"
                        )
                        
                        # Changer la distribution
                        new_dist = st.selectbox(
                            f"Distribution pour {var_name}",
                            ["normal", "uniform", "exponential", "beta", "gamma", "poisson"],
                            index=["normal", "uniform", "exponential", "beta", "gamma", "poisson"].index(dist),
                            key=f"dist_{var_config['id']}"
                        )
                        
                        if new_dist != dist:
                            var_config['distribution'] = new_dist
                            
                            # Réinitialiser avec des valeurs par défaut pour la nouvelle distribution
                            if new_dist == "normal":
                                var_config.update({
                                    "mean": 0.0,
                                    "std": 1.0
                                })
                            elif new_dist == "uniform":
                                var_config.update({
                                    "min": 0.0,
                                    "max": 1.0
                                })
                            elif new_dist == "exponential":
                                var_config.update({
                                    "scale": 1.0
                                })
                            elif new_dist == "beta":
                                var_config.update({
                                    "alpha": 2.0,
                                    "beta": 2.0
                                })
                            elif new_dist == "gamma":
                                var_config.update({
                                    "shape": 2.0,
                                    "scale": 1.0
                                })
                            elif new_dist == "poisson":
                                var_config.update({
                                    "lam": 5.0
                                })
                            
                            st.rerun()
                
                else:  # Categorical
                    with col1:
                        categories_str = st.text_input(
                            f"Catégories pour {var_name} (séparées par des virgules)",
                            value=",".join(var_config.get('categories', [])),
                            key=f"cats_{var_config['id']}"
                        )
                        
                        var_config['use_categorical_type'] = st.checkbox(
                            f"Utiliser le type Pandas Categorical pour {var_name}", 
                            value=var_config.get('use_categorical_type', True),
                            key=f"cat_type_{var_config['id']}"
                        )
                        
                        categories = [c.strip() for c in categories_str.split(",") if c.strip()]
                        
                        if not categories:
                            st.error(f"La liste des catégories ne peut pas être vide pour {var_name}.")
                            categories = ["A", "B"]
                        
                        var_config['categories'] = categories
                    
                    with col2:
                        st.write("Probabilités:")
                        
                        # Gérer les probabilités pour chaque catégorie
                        probabilities = var_config.get('probabilities', [])
                        
                        # Ajuster la longueur des probabilités si nécessaire
                        if len(probabilities) != len(categories):
                            # Si la liste des catégories a changé, réinitialiser les probabilités
                            probabilities = [1.0 / len(categories)] * len(categories)
                        
                        # Permettre la configuration des n-1 premières probabilités
                        new_probs = []
                        remaining_prob = 1.0
                        
                        for i, cat in enumerate(categories[:-1]):
                            default_val = probabilities[i] if i < len(probabilities) else 1.0 / len(categories)
                            prob = st.slider(
                                f"P({cat})", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=min(default_val, remaining_prob),
                                step=0.01,
                                key=f"prob_{var_config['id']}_{i}"
                            )
                            new_probs.append(prob)
                            remaining_prob -= prob
                        
                        # La dernière probabilité est calculée automatiquement
                        remaining_prob = max(0.0, remaining_prob)
                        new_probs.append(remaining_prob)
                        
                        st.write(f"P({categories[-1]}): {remaining_prob:.2f}")
                        
                        var_config['probabilities'] = new_probs
        
        # Configuration des corrélations
        numeric_vars = [name for name, config in st.session_state.variables.items() 
                        if config['type'] == 'numeric']
        
        if len(numeric_vars) >= 2:
            st.subheader("Configuration des Corrélations")
            
            with st.expander("Définir les corrélations entre variables numériques"):
                for i, var1 in enumerate(numeric_vars):
                    for var2 in numeric_vars[i+1:]:
                        corr_key = (var1, var2)
                        current_corr = st.session_state.correlations.get(corr_key, 0.0)
                        
                        new_corr = st.slider(
                            f"Corrélation entre {var1} et {var2}",
                            min_value=-1.0,
                            max_value=1.0,
                            value=current_corr,
                            step=0.05,
                            key=f"corr_{var1}_{var2}"
                        )
                        
                        st.session_state.correlations[corr_key] = new_corr
        
        # Nombre d'échantillons et génération des données
        st.subheader("Génération des Données")
        
        n_samples = st.slider(
            "Nombre d'échantillons à générer", 
            min_value=10, 
            max_value=10000, 
            value=1000, 
            step=10
        )
        
        if st.button("Générer les Données"):
            if not st.session_state.variables:
                st.error("Vous devez définir au moins une variable avant de générer des données.")
            else:
                with st.spinner("Génération des données en cours..."):
                    # Préparer les paramètres au format attendu par la fonction
                    params = {}
                    
                    # Ajouter les variables
                    for var_name, var_config in st.session_state.variables.items():
                        # Créer une copie pour éviter de modifier la configuration originale
                        params[var_name] = var_config.copy()
                    
                    # Ajouter les corrélations
                    if st.session_state.correlations:
                        params['correlations'] = {}
                        for (var1, var2), corr_value in st.session_state.correlations.items():
                            params['correlations'][(var1, var2)] = corr_value
                    
                    # Générer les données
                    artificial_data = generate_artificial_data(params, n_samples)
                    st.session_state.artificial_data = artificial_data

                    # Afficher les résultats
                    st.subheader("Aperçu des Données Générées")
                    st.write(artificial_data.head())

                    st.subheader("Statistiques Descriptives")
                    st.write(artificial_data.describe(include='all'))

                    st.subheader("Visualisations")
                    
                    # Visualisations des variables numériques
                    num_cols = artificial_data.select_dtypes(include=['float64', 'int64']).columns
                    if len(num_cols) > 0:
                        st.markdown("#### Distributions des Variables Numériques")
                        for col in num_cols:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(artificial_data[col], kde=True, ax=ax)
                            ax.set_title(f"Distribution de {col}")
                            st.pyplot(fig)
                    
                    # Visualisations des variables catégorielles
                    cat_cols = artificial_data.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) > 0:
                        st.markdown("#### Distributions des Variables Catégorielles")
                        for col in cat_cols:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            counts = artificial_data[col].value_counts().sort_index()
                            sns.barplot(x=counts.index, y=counts.values, ax=ax)
                            ax.set_title(f"Distribution de {col}")
                            ax.tick_params(axis='x', rotation=45)
                            fig.tight_layout()
                            st.pyplot(fig)
                    
                    # Matrice de corrélation
                    if len(num_cols) >= 2:
                        st.markdown("#### Matrice de Corrélation")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        corr_matrix = artificial_data[num_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                        ax.set_title("Matrice de Corrélation")
                        st.pyplot(fig)
                    
                    # Option de téléchargement
                    st.subheader("Télécharger les Données")
                    csv = artificial_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="donnees_artificielles.csv">Télécharger les données (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Sauvegarder la configuration
                    config_data = {
                        "variables": st.session_state.variables,
                        "correlations": {str(k): v for k, v in st.session_state.correlations.items()}
                    }
                    config_json = json.dumps(config_data, indent=2)
                    b64_config = base64.b64encode(config_json.encode()).decode()
                    href_config = f'<a href="data:file/json;base64,{b64_config}" download="configuration.json">Télécharger la configuration</a>'
                    st.markdown(href_config, unsafe_allow_html=True)
    else:
        st.info("Aucune variable n'a été définie. Utilisez le formulaire ci-dessus pour ajouter des variables.")
    
    # Charger une configuration précédemment sauvegardée
    st.subheader("Charger une Configuration")
    config_file = st.file_uploader("Charger un fichier de configuration JSON", type="json")
    
    if config_file is not None:
        try:
            config_data = json.load(config_file)
            
            if "variables" in config_data and "correlations" in config_data:
                st.session_state.variables = config_data["variables"]
                
                # Convertir les clés de corrélation de string à tuple
                correlations = {}
                for k, v in config_data["correlations"].items():
                    # Évaluer la chaîne en tant que tuple
                    try:
                        # Supprimer les parenthèses et diviser par la virgule
                        parts = k.strip("()").split(", ")
                        if len(parts) == 2:
                            key = (parts[0].strip("'\""), parts[1].strip("'\""))
                            correlations[key] = v
                    except:
                        st.warning(f"Impossible de parser la clé de corrélation: {k}")
                
                st.session_state.correlations = correlations
                
                st.success("Configuration chargée avec succès!")
                st.rerun()
            else:
                st.error("Format de fichier de configuration invalide.")
        except Exception as e:
            st.error(f"Erreur lors du chargement de la configuration: {str(e)}")

# Fonctions pour générer des données synthétiques
def generate_synthetic_data_bootstrap(real_data, n_samples=None):
    """
    Génère des données synthétiques par bootstrap avancé
    """
    if n_samples is None:
        n_samples = len(real_data)
    
    # Identifier les types de colonnes
    numeric_columns = real_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = real_data.select_dtypes(include=['object', 'category']).columns
    int_columns = [col for col in numeric_columns if real_data[col].dtype == 'int64']
    
    # Sélection aléatoire avec remise
    bootstrap_indices = np.random.choice(len(real_data), size=n_samples, replace=True)
    synthetic_data = real_data.iloc[bootstrap_indices].copy()
    
    # Ajouter un bruit pour éviter les duplications exactes (uniquement pour les colonnes numériques)
    for col in numeric_columns:
        if col not in categorical_columns:
            noise = np.random.normal(0, synthetic_data[col].std() * 0.05, size=len(synthetic_data))
            synthetic_data[col] = synthetic_data[col] + noise
    
    # Convertir les colonnes entières en entiers après ajout du bruit
    for col in int_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
    
    # S'assurer que les colonnes catégorielles restent bien catégorielles
    for col in categorical_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = pd.Categorical(synthetic_data[col])
    
    # Réinitialiser l'index
    synthetic_data.reset_index(drop=True, inplace=True)
    
    return synthetic_data

def generate_synthetic_data_gaussian_copula(real_data, n_samples=None):
    """
    Génère des données synthétiques en utilisant une copule gaussienne
    """
    if n_samples is None:
        n_samples = len(real_data)
    
    # Identifier les types de colonnes
    numeric_columns = real_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = real_data.select_dtypes(include=['object', 'category']).columns
    int_columns = [col for col in numeric_columns if real_data[col].dtype == 'int64']
    
    synthetic_data = pd.DataFrame(index=range(n_samples))
    
    # Traiter les variables numériques avec la copule gaussienne
    if len(numeric_columns) >= 2:
        try:
            # Conversion en rangs normalisés
            def to_normalized_ranks(x):
                ranks = stats.rankdata(x)
                return (ranks - 0.5) / len(x)
            
            # Obtenir les données uniformes à partir des rangs
            uniform_data = pd.DataFrame()
            for col in numeric_columns:
                uniform_data[col] = to_normalized_ranks(real_data[col])
            
            # Conversion en données normales
            normal_data = pd.DataFrame()
            for col in uniform_data.columns:
                normal_data[col] = stats.norm.ppf(uniform_data[col])
            
            # Calculer la matrice de corrélation
            corr_matrix = normal_data.corr()
            
            if not np.isnan(corr_matrix.values).any():
                # Générer les échantillons selon la distribution normale multivariée
                mvn_samples = np.random.multivariate_normal(
                    mean=np.zeros(len(numeric_columns)),
                    cov=corr_matrix,
                    size=n_samples
                )
                
                # Convertir en DataFrame
                synthetic_normal = pd.DataFrame(mvn_samples, columns=numeric_columns)
                
                # Retransformer en distribution uniforme
                synthetic_uniform = pd.DataFrame()
                for col in synthetic_normal.columns:
                    synthetic_uniform[col] = stats.norm.cdf(synthetic_normal[col])
                
                # Utiliser la fonction de quantile pour retrouver les distributions originales
                for col in numeric_columns:
                    synthetic_data[col] = np.quantile(real_data[col], synthetic_uniform[col])
                    
                    # Arrondir les colonnes entières
                    if col in int_columns:
                        synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
            else:
                # Fallback en cas de matrice de corrélation invalide
                st.warning("Matrice de corrélation invalide, utilisation du bootstrap comme méthode alternative")
                return generate_synthetic_data_bootstrap(real_data, n_samples)
        
        except Exception as e:
            st.warning(f"Erreur lors de la génération par copule gaussienne: {e}")
            st.info("Utilisation du bootstrap comme méthode alternative...")
            return generate_synthetic_data_bootstrap(real_data, n_samples)
    else:
        # Si moins de 2 variables numériques, utiliser bootstrap pour ces variables
        for col in numeric_columns:
            bootstrap_indices = np.random.choice(len(real_data), size=n_samples, replace=True)
            synthetic_data[col] = real_data[col].iloc[bootstrap_indices].values
            
            # Ajouter du bruit
            noise = np.random.normal(0, real_data[col].std() * 0.05, size=n_samples)
            synthetic_data[col] = synthetic_data[col] + noise
            
            # Arrondir si nécessaire
            if col in int_columns:
                synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
    
    # Générer les variables catégorielles
    for col in categorical_columns:
        value_counts = real_data[col].value_counts(normalize=True)
        categories = value_counts.index.tolist()
        probabilities = value_counts.values

        if len(categories) > 0:
            synthetic_data[col] = np.random.choice(
                categories,
                size=n_samples,
                p=probabilities
            )
            synthetic_data[col] = pd.Categorical(synthetic_data[col])
    
    return synthetic_data

# Fonctions d'évaluation des données synthétiques
def compare_distributions(real_data, synthetic_data):
    """
    Compare les distributions des données réelles et synthétiques
    """
    results = {}
    
    # Comparer les variables numériques
    for col in real_data.select_dtypes(include=['float64', 'int64']).columns:
        ks_stat, ks_pval = stats.ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
        
        results[col] = {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'real_mean': real_data[col].mean(),
            'synthetic_mean': synthetic_data[col].mean(),
            'real_std': real_data[col].std(),
            'synthetic_std': synthetic_data[col].std(),
            'real_min': real_data[col].min(),
            'synthetic_min': synthetic_data[col].min(),
            'real_max': real_data[col].max(),
            'synthetic_max': synthetic_data[col].max(),
        }
    
    # Comparer les variables catégorielles
    for col in real_data.select_dtypes(include=['object', 'category']).columns:
        real_counts = real_data[col].value_counts(normalize=True)
        synth_counts = synthetic_data[col].value_counts(normalize=True)
        
        all_categories = pd.Index(set(real_counts.index) | set(synth_counts.index))
        real_aligned = real_counts.reindex(all_categories, fill_value=0)
        synth_aligned = synth_counts.reindex(all_categories, fill_value=0)
        
        tv_distance = 0.5 * np.sum(np.abs(real_aligned - synth_aligned))
        
        results[col] = {
            'tv_distance': tv_distance,
            'categories': all_categories.tolist(),
            'real_proportions': real_aligned.values.tolist(),
            'synthetic_proportions': synth_aligned.values.tolist()
        }
    
    # Comparer les corrélations
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        real_corr = real_data[numeric_cols].corr()
        synthetic_corr = synthetic_data[numeric_cols].corr()
        
        corr_diff = np.abs(real_corr - synthetic_corr)
        results['correlation'] = {
            'mean_absolute_diff': corr_diff.mean().mean(),
            'max_diff': corr_diff.max().max()
        }
    
    return results

def render_synthetic_data_tab():
    """
    Interface utilisateur pour la génération de données synthétiques
    basées sur des données réelles chargées par l'utilisateur.
    """
    st.header("Génération de Données Synthétiques")
    st.write("""
    Dans cette section, vous pouvez charger un jeu de données réelles et générer
    des données synthétiques qui préservent les propriétés statistiques des données
    originales, sans correspondre à des individus réels.
    """)
    
    uploaded_file = st.file_uploader("Charger un fichier CSV contenant des données réelles", type="csv")
    
    if uploaded_file is not None:
        real_data = pd.read_csv(uploaded_file)
        
        # Vérifier les valeurs manquantes
        missing_values = real_data.isnull().sum().sum()
        if missing_values > 0:
            st.warning(f"⚠️ {missing_values} valeurs manquantes détectées dans le jeu de données.")
            if st.button("Gérer les valeurs manquantes"):
                real_data = show_missing_values_tab(real_data)
        
        st.subheader("Aperçu des Données Réelles")
        st.write(real_data.head())
        
        # Mettre à jour les données dans la session state
        st.session_state.data = real_data
        
        st.subheader("Sélectionner les Colonnes à Utiliser")
        all_columns = real_data.columns.tolist()
        selected_columns = st.multiselect("Colonnes", all_columns, default=all_columns)
        
        if selected_columns:
            real_data = real_data[selected_columns]
            
            # Identifier les types de colonnes
            numeric_cols = real_data.select_dtypes(include=['number']).columns
            int_cols = [col for col in numeric_cols if real_data[col].apply(lambda x: float(x).is_integer()).all()]
            
            # Convertir les colonnes entières
            for col in int_cols:
                real_data[col] = pd.to_numeric(real_data[col], errors='coerce')
                real_data[col] = real_data[col].fillna(real_data[col].median())
                real_data[col] = real_data[col].astype(int)
            
            # Convertir les colonnes catégorielles
            cat_cols = real_data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                real_data[col] = pd.Categorical(real_data[col])
            
            st.subheader("Paramètres de Génération")
            
            n_samples = st.slider("Nombre d'échantillons synthétiques", 
                                  min_value=100, 
                                  max_value=max(10000, len(real_data)*2), 
                                  value=len(real_data),
                                  step=100)
            
            generation_method = st.selectbox(
                "Méthode de génération",
                ["Bootstrap Avancé", "Copule Gaussienne"]
            )
            
            if st.button("Générer les Données Synthétiques"):
                with st.spinner("Génération des données synthétiques en cours..."):
                    try:
                        if generation_method == "Bootstrap Avancé":
                            synthetic_data = generate_synthetic_data_bootstrap(real_data, n_samples)
                        else:
                            synthetic_data = generate_synthetic_data_gaussian_copula(real_data, n_samples)
                    
                        st.session_state.real_data = real_data
                        st.session_state.synthetic_data = synthetic_data
                        
                        comparison_results = compare_distributions(real_data, synthetic_data)
                        
                        st.subheader("Comparaison des Données Réelles et Synthétiques")
                        
                        st.markdown("#### Aperçu des Données Synthétiques")
                        st.write(synthetic_data.head())
                        
                        st.markdown("#### Statistiques Comparatives")
                        
                        num_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
                        if len(num_cols) > 0:
                            numeric_comparison = pd.DataFrame()
                            
                            for col in num_cols:
                                new_row = {
                                    'Variable': col,
                                    'Moyenne (Réel)': f"{comparison_results[col]['real_mean']:.2f}",
                                    'Moyenne (Synth.)': f"{comparison_results[col]['synthetic_mean']:.2f}",
                                    'Écart-type (Réel)': f"{comparison_results[col]['real_std']:.2f}",
                                    'Écart-type (Synth.)': f"{comparison_results[col]['synthetic_std']:.2f}",
                                    'KS p-value': f"{comparison_results[col]['ks_pvalue']:.4f}"
                                }
                                numeric_comparison = pd.concat([numeric_comparison, pd.DataFrame([new_row])], ignore_index=True)
                            
                            st.write(numeric_comparison)
                        
                        cat_cols = real_data.select_dtypes(include=['object', 'category']).columns
                        if len(cat_cols) > 0:
                            st.markdown("#### Distance de Variation Totale (Variables Catégorielles)")
                            
                            for col in cat_cols:
                                st.write(f"{col}: {comparison_results[col]['tv_distance']:.4f}")
                        
                        if 'correlation' in comparison_results:
                            st.markdown("#### Différence des Corrélations")
                            st.write(f"Différence moyenne absolue: {comparison_results['correlation']['mean_absolute_diff']:.4f}")
                            st.write(f"Différence maximale: {comparison_results['correlation']['max_diff']:.4f}")
                        
                        st.subheader("Visualisations Comparatives")
                        
                        if len(num_cols) > 0:
                            st.markdown("#### Distributions des Variables Numériques")
                            for col in num_cols:
                                fig = plot_numeric_comparison(real_data, synthetic_data, col)
                                st.pyplot(fig)
                        
                        if len(cat_cols) > 0:
                            st.markdown("#### Distributions des Variables Catégorielles")
                            for col in cat_cols:
                                fig = plot_categorical_comparison(real_data, synthetic_data, col)
                                st.pyplot(fig)
                        
                        if len(num_cols) >= 2:
                            st.markdown("#### Comparaison des Matrices de Corrélation")
                            fig = plot_correlation_comparison(real_data, synthetic_data)
                            st.pyplot(fig)
                        
                        if len(num_cols) >= 2:
                            st.markdown("#### Analyse en Composantes Principales")
                            fig = plot_pca_comparison(real_data, synthetic_data)
                            st.pyplot(fig)
                        
                        st.subheader("Télécharger les Données Synthétiques")
                        csv = synthetic_data.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="donnees_synthetiques.csv">Télécharger les données synthétiques (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des données: {str(e)}")
        else:
            st.warning("Veuillez sélectionner au moins une colonne.")
    else:
        st.info("""
        Vous n'avez pas encore chargé de données réelles. 
        
        Vous pouvez télécharger l'exemple de données ci-dessous pour tester l'application :
        """)
        
        # Génération d'un exemple simple de données
        example_data = pd.DataFrame({
            'Age': np.random.normal(35, 10, 200).round().astype(int),
            'Sexe': np.random.choice(['H', 'F'], size=200, p=[0.5, 0.5]),
            'Revenu': np.random.normal(45000, 15000, 200).round(-2),
            'Satisfaction': np.random.choice(['Faible', 'Moyenne', 'Élevée'], size=200, p=[0.2, 0.5, 0.3])
        })
        
        csv = example_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="exemple_donnees.csv">Télécharger un exemple de données (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

def main():
    """
    Fonction principale qui configure l'application Streamlit.
    """
    st.set_page_config(
        page_title="Générateur de Données Artificielles et Synthétiques",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Générateur de Données Artificielles et Synthétiques")
    
    # Initialisation de la session state pour stocker les données
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs([
        "Données Artificielles", 
        "Données Synthétiques",
        "Gestion des Valeurs Manquantes"
    ])
    
    with tab1:
        render_dynamic_artificial_data_tab()
    
    with tab2:
        render_synthetic_data_tab()
        
    with tab3:
        if st.session_state.data is not None:
            st.session_state.data = show_missing_values_tab(st.session_state.data)
        else:
            st.warning("Veuillez d'abord charger ou générer un jeu de données dans les autres onglets.")

if __name__ == "__main__":
    main()