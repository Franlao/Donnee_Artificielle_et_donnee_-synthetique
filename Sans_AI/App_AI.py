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

# Pour mesurer la similarité des distributions
from scipy.stats import ks_2samp, chisquare

# Configuration de la page
st.set_page_config(
    page_title="Démo - Données Synthétiques en Santé",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions pour générer des données artificielles
def generate_artificial_data(params, n_samples=1000):
    """
    Génère des données artificielles selon les paramètres fournis.
    """
    data = {}

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
        elif feature_params['type'] == 'categorical':
            categories = feature_params.get('categories', [])
            probabilities = feature_params.get('probabilities', [])

            if not categories or len(categories) != len(probabilities):
                print(f"Erreur : Problème avec les catégories ou probabilités pour '{feature}'.")
                print(f"Categories: {categories}")
                print(f"Probabilities: {probabilities}")
                continue  # Passe à la prochaine feature

            # Normaliser les probabilités si nécessaire
            prob_sum = sum(probabilities)
            if not np.isclose(prob_sum, 1):
                print(f"⚠️ Correction : Les probabilités pour '{feature}' ne somment pas à 1 ({prob_sum}). Normalisation en cours...")
                probabilities = [p / prob_sum for p in probabilities]

            data[feature] = np.random.choice(categories, n_samples, p=probabilities)

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Appliquer les corrélations si spécifiées
    if 'correlations' in params:
        # On ne considère que les features numériques (en s'assurant qu'elles contiennent 'type')
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
                    scale = np.float64(params[feature]['max'] - params[feature]['min'])
                    df[feature] = params[feature]['min'] + unif_values * scale
                elif params[feature]['distribution'] == 'exponential':
                    unif_values = stats.norm.cdf(X_corr[:, i])
                    df[feature] = stats.expon.ppf(unif_values, scale=params[feature]['scale'])

    return df
# Fonctions pour générer des données synthétiques
def generate_synthetic_data_bootstrap(real_data, n_samples=None):
    """
    Génère des données synthétiques par bootstrap avancé
    """
    if n_samples is None:
        n_samples = len(real_data)
    
    # Sélection aléatoire avec remise
    bootstrap_indices = np.random.choice(len(real_data), size=n_samples, replace=True)
    synthetic_data = real_data.iloc[bootstrap_indices].copy()
    
    # Ajouter un bruit pour éviter les duplications exactes
    for col in synthetic_data.select_dtypes(include=['float64', 'int64']).columns:
        noise = np.random.normal(0, synthetic_data[col].std() * 0.05, size=len(synthetic_data))
        synthetic_data[col] = synthetic_data[col] + noise
    
    # Réinitialiser l'index
    synthetic_data.reset_index(drop=True, inplace=True)
    
    return synthetic_data

def generate_synthetic_data_gaussian_copula(real_data, n_samples=None):
    """
    Génère des données synthétiques en utilisant une copule gaussienne
    """
    if n_samples is None:
        n_samples = len(real_data)
    
    # Séparer les variables numériques et catégorielles
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convertir les données en rangs normalisés (entre 0 et 1)
    def to_normalized_ranks(x):
        ranks = stats.rankdata(x)
        return (ranks - 0.5) / len(x)
    
    # Transformer les variables numériques en distributions uniformes
    uniform_data = pd.DataFrame()
    for col in numeric_cols:
        uniform_data[col] = to_normalized_ranks(real_data[col])
    
    # Transformer en distribution normale standard
    normal_data = pd.DataFrame()
    for col in uniform_data.columns:
        normal_data[col] = stats.norm.ppf(uniform_data[col])
    
    # Calculer la matrice de corrélation
    corr_matrix = normal_data.corr()
    
    # Générer des données gaussiennes multivariées
    mvn_samples = np.random.multivariate_normal(
        mean=np.zeros(len(numeric_cols)),
        cov=corr_matrix,
        size=n_samples
    )
    
    # Convertir en DataFrame
    synthetic_normal = pd.DataFrame(mvn_samples, columns=numeric_cols)
    
    # Transformer en distributions uniformes
    synthetic_uniform = pd.DataFrame()
    for col in synthetic_normal.columns:
        synthetic_uniform[col] = stats.norm.cdf(synthetic_normal[col])
    
    # Transformer en distributions originales
    synthetic_data = pd.DataFrame()
    for col in numeric_cols:
        if real_data[col].dtype == 'int64':
            # Pour les entiers, utiliser l'inverse de la fonction de répartition empirique
            synthetic_data[col] = np.quantile(real_data[col], synthetic_uniform[col])
            synthetic_data[col] = synthetic_data[col].round().astype(int)
        else:
            # Pour les flottants, utiliser l'inverse de la fonction de répartition empirique
            synthetic_data[col] = np.quantile(real_data[col], synthetic_uniform[col])
    
    # Générer les variables catégorielles
    for col in categorical_cols:
        # Calculer les probabilités des catégories
        value_counts = real_data[col].value_counts(normalize=True)
        categories = value_counts.index.tolist()
        probabilities = value_counts.values
        
        # Générer des échantillons selon ces probabilités
        synthetic_data[col] = np.random.choice(
            categories,
            size=n_samples,
            p=probabilities
        )
    
    return synthetic_data

# Fonctions d'évaluation des données synthétiques
def compare_distributions(real_data, synthetic_data):
    """
    Compare les distributions des données réelles et synthétiques
    """
    results = {}
    
    # Pour chaque colonne numérique
    for col in real_data.select_dtypes(include=['float64', 'int64']).columns:
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pval = ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
        
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
    
    # Pour chaque colonne catégorielle
    for col in real_data.select_dtypes(include=['object', 'category']).columns:
        real_counts = real_data[col].value_counts(normalize=True)
        synth_counts = synthetic_data[col].value_counts(normalize=True)
        
        # Aligner les catégories
        all_categories = pd.Index(set(real_counts.index) | set(synth_counts.index))
        real_aligned = real_counts.reindex(all_categories, fill_value=0)
        synth_aligned = synth_counts.reindex(all_categories, fill_value=0)
        
        # Calculer la distance de variation totale
        tv_distance = 0.5 * np.sum(np.abs(real_aligned - synth_aligned))
        
        results[col] = {
            'tv_distance': tv_distance,
            'categories': all_categories.tolist(),
            'real_proportions': real_aligned.values.tolist(),
            'synthetic_proportions': synth_aligned.values.tolist()
        }
    
    # Comparer les matrices de corrélation
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

# Fonctions de visualisation
def plot_numeric_comparison(real_data, synthetic_data, column):
    """
    Crée un histogramme comparatif pour une colonne numérique
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramme des données réelles
    sns.histplot(real_data[column], color='blue', alpha=0.5, label='Données réelles', kde=True, ax=ax)
    
    # Histogramme des données synthétiques
    sns.histplot(synthetic_data[column], color='red', alpha=0.5, label='Données synthétiques', kde=True, ax=ax)
    
    ax.set_title(f'Comparaison des distributions pour {column}')
    ax.legend()
    
    return fig

def plot_categorical_comparison(real_data, synthetic_data, column):
    """
    Crée un graphique à barres comparatif pour une colonne catégorielle
    """
    real_counts = real_data[column].value_counts(normalize=True).sort_index()
    synth_counts = synthetic_data[column].value_counts(normalize=True).sort_index()
    
    # Combiner les catégories
    all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
    
    # Réindexer avec toutes les catégories
    real_counts = real_counts.reindex(all_categories, fill_value=0)
    synth_counts = synth_counts.reindex(all_categories, fill_value=0)
    
    # Créer le graphique
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
    """
    Crée une visualisation comparative des matrices de corrélation
    """
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    real_corr = real_data[numeric_cols].corr()
    synthetic_corr = synthetic_data[numeric_cols].corr()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Matrice de corrélation des données réelles
    sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Corrélations - Données réelles')
    
    # Matrice de corrélation des données synthétiques
    sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Corrélations - Données synthétiques')
    
    # Différence absolue entre les matrices
    diff_corr = np.abs(real_corr - synthetic_corr)
    sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='Reds', ax=axes[2])
    axes[2].set_title('Différence absolue des corrélations')
    
    fig.tight_layout()
    
    return fig

def plot_pca_comparison(real_data, synthetic_data):
    """
    Visualise la comparaison par ACP des données réelles et synthétiques
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        return None
    
    # Combiner les données pour l'ACP
    real_data_numeric = real_data[numeric_cols].copy()
    synthetic_data_numeric = synthetic_data[numeric_cols].copy()
    
    # Ajouter une colonne pour identifier la source
    real_data_numeric['source'] = 'Réel'
    synthetic_data_numeric['source'] = 'Synthétique'
    
    # Combiner les données
    combined_data = pd.concat([real_data_numeric, synthetic_data_numeric], ignore_index=True)
    
    # Extraire les étiquettes de source
    sources = combined_data['source']
    combined_data = combined_data.drop('source', axis=1)
    
    # Standardiser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    # Appliquer l'ACP
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    # Créer un DataFrame pour la visualisation
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['source'] = sources
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracer les points
    for source, color in zip(['Réel', 'Synthétique'], ['blue', 'red']):
        subset = pca_df[pca_df['source'] == source]
        ax.scatter(subset['PC1'], subset['PC2'], c=color, alpha=0.5, label=source)
    
    # Ajouter les flèches des composantes
    feature_names = numeric_cols
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                 head_width=0.05, head_length=0.05, fc='k', ec='k')
        ax.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, feature)
    
    # Ajouter des étiquettes et une légende
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance expliquée)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance expliquée)')
    ax.set_title('Analyse en Composantes Principales')
    ax.legend()
    ax.grid(True)
    
    return fig

# Interface utilisateur Streamlit
st.title("Démonstration - Données Synthétiques et Artificielles en Santé")

# Créer les onglets
tab1, tab2 = st.tabs(["Données Artificielles", "Données Synthétiques"])

with tab1:
    st.header("Génération de Données Artificielles")
    st.write("""
    Dans cette section, vous pouvez définir les distributions statistiques
    à utiliser pour générer des données artificielles. Ces données ne sont pas
    dérivées de données réelles mais suivent les distributions que vous spécifiez.
    """)
    
    # Paramètres pour les données artificielles
    st.subheader("Paramètres des Variables")
    
    # Nombre d'échantillons
    n_samples = st.slider("Nombre d'échantillons à générer", 100, 10000, 1000, 100)
    
    # Interface pour définir les variables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variables Numériques")
        
        # Paramètres pour l'âge
        st.markdown("#### Âge")
        age_distribution = st.selectbox("Distribution pour l'âge", ["normal", "uniform"], key="age_dist")
        
        if age_distribution == "normal":
            age_mean = st.slider("Moyenne de l'âge", 0, 100, 45, 1)
            age_std = st.slider("Écart-type de l'âge", 1, 30, 15, 1)
        else:  # uniform
            age_min = st.slider("Âge minimum", 0, 100, 18, 1)
            age_max = st.slider("Âge maximum", 0, 100, 90, 1)
        
        # Paramètres pour la glycémie
        st.markdown("#### Glycémie (mg/dL)")
        glucose_distribution = st.selectbox("Distribution pour la glycémie", ["normal", "uniform", "exponential"], key="glucose_dist")
        
        if glucose_distribution == "normal":
            glucose_mean = st.slider("Moyenne de la glycémie", 70, 300, 100, 1)
            glucose_std = st.slider("Écart-type de la glycémie", 1, 100, 25, 1)
        elif glucose_distribution == "uniform":
            glucose_min = st.slider("Glycémie minimum", 70, 300, 70, 1)
            glucose_max = st.slider("Glycémie maximum", 70, 300, 200, 1)
        else:  # exponential
            glucose_scale = st.slider("Paramètre d'échelle (λ) pour la glycémie", 1, 100, 30, 1)
        
        # Paramètres pour la tension artérielle systolique
        st.markdown("#### Tension Artérielle Systolique (mmHg)")
        bp_distribution = st.selectbox("Distribution pour la tension", ["normal", "uniform"], key="bp_dist")
        
        if bp_distribution == "normal":
            bp_mean = st.slider("Moyenne de la tension", 80, 200, 120, 1)
            bp_std = st.slider("Écart-type de la tension", 1, 50, 15, 1)
        else:  # uniform
            bp_min = st.slider("Tension minimum", 80, 200, 90, 1)
            bp_max = st.slider("Tension maximum", 80, 200, 180, 1)
    
    with col2:
        st.subheader("Variables Catégorielles")
        
        # Paramètres pour le sexe
        st.markdown("#### Sexe")
        gender_categories = st.text_input("Catégories (séparées par des virgules)", "M,F")
        gender_categories = [c.strip() for c in gender_categories.split(",")]
        
        # Créer des sliders pour les probabilités
        gender_probs = []
        for i, cat in enumerate(gender_categories):
            if i < len(gender_categories) - 1:
                prob = st.slider(f"Probabilité pour {cat}", 0.0, 1.0, 1.0/len(gender_categories), 0.01, key=f"gender_prob_{i}")
                gender_probs.append(prob)
        
        # Calculer la dernière probabilité
        if gender_categories:
            remaining_prob = 1.0 - sum(gender_probs[:-1]) if len(gender_probs) > 0 else 1.0
            gender_probs.append(remaining_prob)
            st.write(f"Probabilité pour {gender_categories[-1]}: {remaining_prob:.2f}")
        
        # Paramètres pour le diabète
        st.markdown("#### Diabète")
        diabetes_categories = st.text_input("Catégories (séparées par des virgules)", "0,1", key="diabetes_cats")
        diabetes_categories = [c.strip() for c in diabetes_categories.split(",")]
        
        # Créer des sliders pour les probabilités
        diabetes_probs = []
        for i, cat in enumerate(diabetes_categories):
            if i < len(diabetes_categories) - 1:
                prob = st.slider(f"Probabilité pour {cat}", 0.0, 1.0, 1.0/len(diabetes_categories), 0.01, key=f"diabetes_prob_{i}")
                diabetes_probs.append(prob)
        
        # Calculer la dernière probabilité
        if diabetes_categories:
            remaining_prob = 1.0 - sum(diabetes_probs[:-1]) if len(diabetes_probs) > 0 else 1.0
            diabetes_probs.append(remaining_prob)
            st.write(f"Probabilité pour {diabetes_categories[-1]}: {remaining_prob:.2f}")
        
        # Corrélations entre variables
        st.subheader("Corrélations")
        
        # Corrélation entre âge et glycémie
        age_glucose_corr = st.slider("Corrélation Âge-Glycémie", -1.0, 1.0, 0.3, 0.05)
        
        # Corrélation entre âge et tension
        age_bp_corr = st.slider("Corrélation Âge-Tension", -1.0, 1.0, 0.4, 0.05)
        
        # Corrélation entre glycémie et tension
        glucose_bp_corr = st.slider("Corrélation Glycémie-Tension", -1.0, 1.0, 0.2, 0.05)
    
    # Construire les paramètres pour la génération
    params = {}
    
    # Âge
    params['age'] = {
        'type': 'numeric',
        'distribution': age_distribution
    }
    if age_distribution == 'normal':
        params['age']['mean'] = age_mean
        params['age']['std'] = age_std
    else:  # uniform
        params['age']['min'] = age_min
        params['age']['max'] = age_max
    
    # Glycémie
    params['glucose'] = {
        'type': 'numeric',
        'distribution': glucose_distribution
    }
    if glucose_distribution == 'normal':
        params['glucose']['mean'] = glucose_mean
        params['glucose']['std'] = glucose_std
    elif glucose_distribution == 'uniform':
        params['glucose']['min'] = glucose_min
        params['glucose']['max'] = glucose_max
    else:  # exponential
        params['glucose']['scale'] = glucose_scale
    
    # Tension
    params['blood_pressure'] = {
        'type': 'numeric',
        'distribution': bp_distribution
    }
    if bp_distribution == 'normal':
        params['blood_pressure']['mean'] = bp_mean
        params['blood_pressure']['std'] = bp_std
    else:  # uniform
        params['blood_pressure']['min'] = bp_min
        params['blood_pressure']['max'] = bp_max
    
    # Sexe
    params['gender'] = {
        'type': 'categorical',
        'categories': gender_categories,
        'probabilities': gender_probs
    }
    
    # Diabète
    params['diabetes'] = {
        'type': 'categorical',
        'categories': diabetes_categories,
        'probabilities': diabetes_probs
    }
    
    # Corrélations
    params['correlations'] = {
        ('age', 'glucose'): age_glucose_corr,
        ('age', 'blood_pressure'): age_bp_corr,
        ('glucose', 'blood_pressure'): glucose_bp_corr
    }
    
    # Bouton pour générer les données
    if st.button("Générer les Données Artificielles"):
        with st.spinner("Génération des données en cours..."):
            # Générer les données artificielles
            artificial_data = generate_artificial_data(params, n_samples)
            
            # Afficher les résultats
            st.session_state.artificial_data = artificial_data
            
            # Afficher un aperçu des données
            st.subheader("Aperçu des Données Générées")
            st.write(artificial_data.head())
            
            # Statistiques descriptives
            st.subheader("Statistiques Descriptives")
            st.write(artificial_data.describe())
            
            # Visualisations
            st.subheader("Visualisations")
            
            # Variables numériques
            st.markdown("#### Distributions des Variables Numériques")
            cols = artificial_data.select_dtypes(include=['float64', 'int64']).columns
            for col in cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(artificial_data[col], kde=True, ax=ax)
                ax.set_title(f"Distribution de {col}")
                st.pyplot(fig)
            
            # Variables catégorielles
            st.markdown("#### Distributions des Variables Catégorielles")
            cat_cols = artificial_data.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=artificial_data[col], ax=ax)
                ax.set_title(f"Distribution de {col}")
                st.pyplot(fig)
            
            # Matrice de corrélation
            st.markdown("#### Matrice de Corrélation")
            if len(cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(artificial_data[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title("Matrice de Corrélation")
                st.pyplot(fig)
            
            # Téléchargement des données
            st.subheader("Télécharger les Données")
            csv = artificial_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="donnees_artificielles.csv">Télécharger les données (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)

with tab2:
    st.header("Génération de Données Synthétiques")
    st.write("""
    Dans cette section, vous pouvez charger un jeu de données réelles et générer
    des données synthétiques qui préservent les propriétés statistiques des données
    originales, sans correspondre à des individus réels.
    """)
    
    # Charger des données réelles
    uploaded_file = st.file_uploader("Charger un fichier CSV contenant des données réelles", type="csv")
    
    if uploaded_file is not None:
        # Charger les données
        real_data = pd.read_csv(uploaded_file)
        
        # Afficher un aperçu des données réelles
        st.subheader("Aperçu des Données Réelles")
        st.write(real_data.head())
        
        # Sélectionner les colonnes à conserver
        st.subheader("Sélectionner les Colonnes à Utiliser")
        all_columns = real_data.columns.tolist()
        selected_columns = st.multiselect("Colonnes", all_columns, default=all_columns)
        
        if selected_columns:
            real_data = real_data[selected_columns]
            
            # Paramètres pour la génération
            st.subheader("Paramètres de Génération")
            
            # Nombre d'échantillons
            n_samples = st.slider("Nombre d'échantillons synthétiques", 
                                 min_value=100, 
                                 max_value=max(10000, len(real_data)*2), 
                                 value=len(real_data),
                                 step=100)
            
            # Méthode de génération
            generation_method = st.selectbox(
                "Méthode de génération",
                ["Bootstrap Avancé", "Copule Gaussienne"]
            )
            
            # Bouton pour générer les données
            if st.button("Générer les Données Synthétiques"):
                with st.spinner("Génération des données synthétiques en cours..."):
                    # Générer les données synthétiques
                    if generation_method == "Bootstrap Avancé":
                        synthetic_data = generate_synthetic_data_bootstrap(real_data, n_samples)
                    else:  # Copule Gaussienne
                        synthetic_data = generate_synthetic_data_gaussian_copula(real_data, n_samples)
                    
                    # Stocker dans la session
                    st.session_state.real_data = real_data
                    st.session_state.synthetic_data = synthetic_data
                    
                    # Comparer les distributions
                    comparison_results = compare_distributions(real_data, synthetic_data)
                    
                    # Afficher les résultats
                    st.subheader("Comparaison des Données Réelles et Synthétiques")
                    
                    # Tableau de comparaison
                    st.markdown("#### Statistiques Comparatives")
                    
                    # Pour les variables numériques
                    num_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
                    if len(num_cols) > 0:
                        numeric_comparison = pd.DataFrame()
                        
                        for col in num_cols:
                            numeric_comparison = numeric_comparison._append({
                                'Variable': col,
                                'Moyenne (Réel)': f"{comparison_results[col]['real_mean']:.2f}",
                                'Moyenne (Synth.)': f"{comparison_results[col]['synthetic_mean']:.2f}",
                                'Écart-type (Réel)': f"{comparison_results[col]['real_std']:.2f}",
                                'Écart-type (Synth.)': f"{comparison_results[col]['synthetic_std']:.2f}",
                                'KS p-value': f"{comparison_results[col]['ks_pvalue']:.4f}"
                            }, ignore_index=True)
                        
                        st.write(numeric_comparison)
                    
                    # Pour les variables catégorielles
                    cat_cols = real_data.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) > 0:
                        st.markdown("#### Distance de Variation Totale (Variables Catégorielles)")
                        
                        for col in cat_cols:
                            st.write(f"{col}: {comparison_results[col]['tv_distance']:.4f}")
                    
                    # Pour les corrélations
                    if 'correlation' in comparison_results:
                        st.markdown("#### Différence des Corrélations")
                        st.write(f"Différence moyenne absolue: {comparison_results['correlation']['mean_absolute_diff']:.4f}")
                        st.write(f"Différence maximale: {comparison_results['correlation']['max_diff']:.4f}")
                    
                    # Visualisations
                    st.subheader("Visualisations Comparatives")
                    
                    # Variables numériques
                    if len(num_cols) > 0:
                        st.markdown("#### Distributions des Variables Numériques")
                        for col in num_cols:
                            fig = plot_numeric_comparison(real_data, synthetic_data, col)
                            st.pyplot(fig)
                    
                    # Variables catégorielles
                    if len(cat_cols) > 0:
                        st.markdown("#### Distributions des Variables Catégorielles")
                        for col in cat_cols:
                            fig = plot_categorical_comparison(real_data, synthetic_data, col)
                            st.pyplot(fig)
                    
                    # Matrice de corrélation
                    if len(num_cols) >= 2:
                        st.markdown("#### Comparaison des Matrices de Corrélation")
                        fig = plot_correlation_comparison(real_data, synthetic_data)
                        st.pyplot(fig)
                    
                    # Visualisation par ACP
                    if len(num_cols) >= 2:
                        st.markdown("#### Analyse en Composantes Principales")
                        fig = plot_pca_comparison(real_data, synthetic_data)
                        st.pyplot(fig)
                    
                    # Téléchargement des données synthétiques
                    st.subheader("Télécharger les Données Synthétiques")
                    csv = synthetic_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="donnees_synthetiques.csv">Télécharger les données synthétiques (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Veuillez sélectionner au moins une colonne.")
    else:
        # Exemple de données à utiliser
        st.info("""
        Vous n'avez pas encore chargé de données réelles. 
        
        Vous pouvez télécharger l'exemple de données ci-dessous pour tester l'application :
        """)
        
        # Créer un exemple de jeu de données
        example_data = pd.DataFrame({
            'age': np.random.normal(65, 10, 200).astype(int),
            'gender': np.random.choice(['M', 'F'], size=200, p=[0.48, 0.52]),
            'glucose': np.round(np.random.normal(110, 30, 200)).astype(int),
            'blood_pressure':np.round(np.random.normal(135, 15, 200)).astype(int),
            'cholesterol': np.round(np.random.normal(210, 40, 200)).astype(int),
            'diabetes': np.random.choice(['0', '1'], size=200, p=[0.7, 0.3])
        })
        
        # Convertir les variables numériques en float
        #example_data['blood_pressure'] = example_data['blood_pressure'].astype(np.float64)
        #example_data['glucose'] = example_data['glucose'].astype(np.float64)
        # Créer quelques corrélations
        for i in range(len(example_data)):
            #Âge influence la tension artérielle
            example_data.loc[i, 'blood_pressure'] += example_data.loc[i, 'age'] * 0.3
            
            # Glucose élevé augmente probabilité de diabète
            if example_data.loc[i, 'glucose'] > 126:
                if np.random.random() < 0.8:
                    example_data.loc[i, 'diabetes'] = '1'
            
            # Diabète influence les niveaux de glucose
            if example_data.loc[i, 'diabetes'] == '1':
                example_data.loc[i, 'glucose'] += np.round(np.random.normal(50, 20)).astype(int)
        
        # Arrondir les valeurs
        example_data['blood_pressure'] = np.round(example_data['blood_pressure']).astype(int)
        example_data['glucose'] = np.round(example_data['glucose']).astype(int)
        
        # Convertir en CSV pour téléchargement
        csv = example_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="exemple_donnees_sante.csv">Télécharger un exemple de données de santé (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

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
    
    ### Méthodes de génération
    
    **Bootstrap Avancé**: Rééchantillonnage avec perturbation
    
    **Copule Gaussienne**: Modélise les dépendances entre variables via une distribution normale multivariée
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Préparé pour:
    **Présentation d'Épidémiologie**
    
    *Données synthétiques et données artificielles en santé*
    """)