import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fonction pour générer des données artificielles avec des paramètres spécifiques
def generate_artificial_medical_data(n_samples=1000, seed=42):
    """
    Génère un jeu de données médicales artificielles avec des distributions
    et des corrélations spécifiques.
    """
    np.random.seed(seed)
    
    # Générer les variables de base avec les corrélations souhaitées
    # Nous allons générer des données corrélées en utilisant la décomposition de Cholesky
    
    # Matrice de corrélation cible
    corr_matrix = np.array([
        [1.0, 0.35, 0.42, 0.25],  # âge
        [0.35, 1.0, 0.28, 0.65],  # glycémie
        [0.42, 0.28, 1.0, 0.18],  # tension artérielle
        [0.25, 0.65, 0.18, 1.0]   # IMC
    ])
    
    # Paramètres des distributions
    means = np.array([65, 110, 130, 27])  # moyenne d'âge, glycémie, tension, IMC
    stds = np.array([12, 30, 15, 5])      # écarts-types
    
    # Décomposition de Cholesky
    L = np.linalg.cholesky(corr_matrix)
    
    # Générer des variables normales indépendantes
    uncorrelated = np.random.normal(size=(n_samples, 4))
    
    # Transformer pour obtenir des variables corrélées
    correlated = uncorrelated @ L.T
    
    # Mettre à l'échelle selon les moyennes et écarts-types souhaités
    scaled_data = correlated * stds + means
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'age': np.round(np.clip(scaled_data[:, 0], 18, 100)).astype(int),
        'glucose': np.round(np.clip(scaled_data[:, 1], 70, 300)).astype(int),
        'blood_pressure': np.round(np.clip(scaled_data[:, 2], 90, 200)).astype(int),
        'bmi': np.round(scaled_data[:, 3], 1)
    })
    
    # Ajouter des variables catégorielles
    # Sexe (avec distribution 48% M, 52% F)
    data['gender'] = np.random.choice(['M', 'F'], size=n_samples, p=[0.48, 0.52])
    
    # Diabète (influencé par la glycémie)
    data['diabetes'] = '0'
    
    # Règle : glycémie > 126 mg/dL a une forte probabilité de diabète
    high_glucose_mask = data['glucose'] > 126
    data.loc[high_glucose_mask, 'diabetes'] = np.random.choice(
        ['0', '1'], 
        size=high_glucose_mask.sum(), 
        p=[0.2, 0.8]  # 80% de chance d'avoir le diabète si glycémie élevée
    )
    
    # Règle : glycémie entre 100 et 126 mg/dL a une probabilité modérée de diabète
    moderate_glucose_mask = (data['glucose'] > 100) & (data['glucose'] <= 126)
    data.loc[moderate_glucose_mask, 'diabetes'] = np.random.choice(
        ['0', '1'],
        size=moderate_glucose_mask.sum(),
        p=[0.7, 0.3]  # 30% de chance d'avoir le diabète si glycémie modérément élevée
    )
    
    # Ajouter cholestérol (corrélé avec âge et BMI)
    cholesterol_base = 150 + 0.3 * (data['age'] - means[0]) + 2 * (data['bmi'] - means[3])
    cholesterol_noise = np.random.normal(0, 25, n_samples)
    data['cholesterol'] = np.round(cholesterol_base + cholesterol_noise).astype(int)
    data['cholesterol'] = np.clip(data['cholesterol'], 120, 350)
    
    # Ajouter traitement (influencé par diabète, tension et cholestérol)
    data['treatment'] = '0'
    
    # Règles pour le traitement
    # Diabète + tension élevée OU cholestérol élevé → forte probabilité de traitement
    high_risk_mask = ((data['diabetes'] == '1') & (data['blood_pressure'] > 140)) | (data['cholesterol'] > 240)
    data.loc[high_risk_mask, 'treatment'] = np.random.choice(
        ['0', '1'],
        size=high_risk_mask.sum(),
        p=[0.1, 0.9]  # 90% de chance d'être sous traitement
    )
    
    # Tension élevée OU cholestérol élevé → probabilité modérée de traitement
    moderate_risk_mask = ((data['blood_pressure'] > 140) | (data['cholesterol'] > 240)) & ~high_risk_mask
    data.loc[moderate_risk_mask, 'treatment'] = np.random.choice(
        ['0', '1'],
        size=moderate_risk_mask.sum(),
        p=[0.5, 0.5]  # 50% de chance d'être sous traitement
    )
    
    return data

# Fonction pour générer des données pathologiques réelles et synthétiques
def generate_synthetic_data(real_data, method='copula', n_samples=None, seed=42):
    """
    Génère des données synthétiques à partir de données réelles.
    
    Paramètres:
    - real_data: DataFrame contenant les données réelles
    - method: 'copula' pour la méthode de copule gaussienne, 'bootstrap' pour bootstrap avancé
    - n_samples: nombre d'échantillons à générer (par défaut, même taille que les données réelles)
    - seed: graine pour la reproductibilité
    
    Retourne:
    - DataFrame contenant les données synthétiques
    """
    np.random.seed(seed)
    
    if n_samples is None:
        n_samples = len(real_data)
    
    if method == 'bootstrap':
        # Méthode de bootstrap avancé
        bootstrap_indices = np.random.choice(len(real_data), size=n_samples, replace=True)
        synthetic_data = real_data.iloc[bootstrap_indices].copy()
        
        # Ajouter du bruit aux variables numériques
        for col in synthetic_data.select_dtypes(include=['float64', 'int64']).columns:
            noise = np.random.normal(0, real_data[col].std() * 0.05, size=len(synthetic_data))
            synthetic_data[col] = synthetic_data[col] + noise
            
            # Arrondir les entiers
            if real_data[col].dtype == 'int64':
                synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
                
        # Réinitialiser l'index
        synthetic_data.reset_index(drop=True, inplace=True)
        
    elif method == 'copula':
        # Méthode de la copule gaussienne
        # Séparer les variables numériques et catégorielles
        numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Convertir en rangs normalisés (entre 0 et 1)
        uniform_data = pd.DataFrame()
        for col in numeric_cols:
            uniform_data[col] = stats.rankdata(real_data[col]) / len(real_data)
        
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
            # Utiliser l'inverse de la fonction de répartition empirique
            synthetic_data[col] = np.quantile(real_data[col], synthetic_uniform[col])
            
            # Arrondir les entiers
            if real_data[col].dtype == 'int64':
                synthetic_data[col] = np.round(synthetic_data[col]).astype(int)
        
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

# Fonction pour comparer les distributions statistiques
def compare_distributions(real_data, synthetic_data):
    """
    Compare les distributions statistiques entre données réelles et synthétiques
    et génère des visualisations.
    """
    # Créer un dossier pour les visualisations
    os.makedirs('visualizations', exist_ok=True)
    
    # Statistiques comparatives pour variables numériques
    numeric_cols = real_data.select_dtypes(include=['float64', 'int64']).columns
    
    stats_comparison = pd.DataFrame(columns=[
        'Variable', 'Type', 'Moyenne (Réel)', 'Moyenne (Synth.)', 
        'Écart-type (Réel)', 'Écart-type (Synth.)', 
        'Min (Réel)', 'Min (Synth.)', 'Max (Réel)', 'Max (Synth.)',
        'KS statistic', 'KS p-value'
    ])
    
    # Comparer les variables numériques
    for col in numeric_cols:
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.ks_2samp(real_data[col], synthetic_data[col])
        
        # Ajouter les statistiques
        stats_comparison = stats_comparison._append({
            'Variable': col,
            'Type': 'Numérique',
            'Moyenne (Réel)': f"{real_data[col].mean():.2f}",
            'Moyenne (Synth.)': f"{synthetic_data[col].mean():.2f}",
            'Écart-type (Réel)': f"{real_data[col].std():.2f}",
            'Écart-type (Synth.)': f"{synthetic_data[col].std():.2f}",
            'Min (Réel)': f"{real_data[col].min():.2f}",
            'Min (Synth.)': f"{synthetic_data[col].min():.2f}",
            'Max (Réel)': f"{real_data[col].max():.2f}",
            'Max (Synth.)': f"{synthetic_data[col].max():.2f}",
            'KS statistic': f"{ks_stat:.4f}",
            'KS p-value': f"{ks_pval:.4f}"
        }, ignore_index=True)
        
        # Visualisation comparative
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(real_data[col], kde=True, color='blue')
        plt.title(f'Distribution réelle: {col}')
        
        plt.subplot(1, 2, 2)
        sns.histplot(synthetic_data[col], kde=True, color='red')
        plt.title(f'Distribution synthétique: {col}')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/compare_{col}_hist.png')
        plt.close()
        
        # QQ plot
        plt.figure(figsize=(8, 8))
        stats.probplot(real_data[col], dist="norm", plot=plt)
        plt.title(f'QQ Plot: {col} (Données réelles)')
        plt.savefig(f'visualizations/qq_real_{col}.png')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        stats.probplot(synthetic_data[col], dist="norm", plot=plt)
        plt.title(f'QQ Plot: {col} (Données synthétiques)')
        plt.savefig(f'visualizations/qq_synth_{col}.png')
        plt.close()
    
    # Comparer les variables catégorielles
    cat_cols = real_data.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        real_counts = real_data[col].value_counts(normalize=True)
        synth_counts = synthetic_data[col].value_counts(normalize=True)
        
        # Aligner les catégories
        all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
        real_aligned = real_counts.reindex(all_categories, fill_value=0)
        synth_aligned = synth_counts.reindex(all_categories, fill_value=0)
        
        # Calculer la distance de variation totale
        tv_distance = 0.5 * sum(abs(real_aligned - synth_aligned))
        
        # Ajouter au tableau de comparaison
        stats_comparison = stats_comparison._append({
            'Variable': col,
            'Type': 'Catégorielle',
            'Moyenne (Réel)': '-',
            'Moyenne (Synth.)': '-',
            'Écart-type (Réel)': '-',
            'Écart-type (Synth.)': '-',
            'Min (Réel)': '-',
            'Min (Synth.)': '-',
            'Max (Réel)': '-',
            'Max (Synth.)': '-',
            'KS statistic': '-',
            'KS p-value': f"TV: {tv_distance:.4f}"
        }, ignore_index=True)
        
        # Visualisation comparative
        plt.figure(figsize=(12, 6))
        
        # Données réelles
        plt.subplot(1, 2, 1)
        sns.barplot(x=real_aligned.index, y=real_aligned.values, color='blue')
        plt.title(f'Distribution réelle: {col}')
        plt.xticks(rotation=45)
        
        # Données synthétiques
        plt.subplot(1, 2, 2)
        sns.barplot(x=synth_aligned.index, y=synth_aligned.values, color='red')
        plt.title(f'Distribution synthétique: {col}')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/compare_{col}_bar.png')
        plt.close()
    
    # Comparer les matrices de corrélation
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(18, 6))
        
        # Matrice de corrélation des données réelles
        plt.subplot(1, 3, 1)
        real_corr = real_data[numeric_cols].corr()
        sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Corrélations - Données réelles')
        
        # Matrice de corrélation des données synthétiques
        plt.subplot(1, 3, 2)
        synth_corr = synthetic_data[numeric_cols].corr()
        sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Corrélations - Données synthétiques')
        
        # Différence absolue entre les matrices
        plt.subplot(1, 3, 3)
        diff_corr = np.abs(real_corr - synth_corr)
        sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='Reds')
        plt.title('Différence absolue des corrélations')
        
        plt.tight_layout()
        plt.savefig('visualizations/correlation_comparison.png')
        plt.close()
    
    # Enregistrer les statistiques de comparaison
    stats_comparison.to_csv('visualizations/stats_comparison.csv', index=False)
    
    return stats_comparison

# Fonction principale
def main():
    """
    Fonction principale qui exécute l'ensemble du processus de génération
    et de comparaison des données.
    """
    # 1. Générer des données pathologiques artificielles (simulées)
    print("Génération des données artificielles...")
    artificial_data = generate_artificial_medical_data(n_samples=1000, seed=42)
    artificial_data.to_csv('artificial_health_data.csv', index=False)
    print(f"Données artificielles générées ({len(artificial_data)} échantillons)")
    
    # Résumé des données artificielles
    print("\nRésumé des données artificielles:")
    print(artificial_data.describe())
    
    # 2. Considérer les données artificielles comme "réelles" pour la démonstration
    # et générer des données synthétiques à partir de celles-ci
    print("\nGénération des données synthétiques (méthode de copule)...")
    synthetic_data_copula = generate_synthetic_data(
        artificial_data, 
        method='copula', 
        n_samples=1000, 
        seed=123
    )
    synthetic_data_copula.to_csv('synthetic_health_data_copula.csv', index=False)
    print(f"Données synthétiques (copule) générées ({len(synthetic_data_copula)} échantillons)")
    
    print("\nGénération des données synthétiques (méthode de bootstrap)...")
    synthetic_data_bootstrap = generate_synthetic_data(
        artificial_data, 
        method='bootstrap', 
        n_samples=1000, 
        seed=456
    )
    synthetic_data_bootstrap.to_csv('synthetic_health_data_bootstrap.csv', index=False)
    print(f"Données synthétiques (bootstrap) générées ({len(synthetic_data_bootstrap)} échantillons)")
    
    # 3. Comparer les distributions
    print("\nComparaison des distributions (copule)...")
    stats_copula = compare_distributions(artificial_data, synthetic_data_copula)
    print("Comparaison terminée et visualisations enregistrées dans le dossier 'visualizations'")
    
    print("\nComparaison des distributions (bootstrap)...")
    stats_bootstrap = compare_distributions(artificial_data, synthetic_data_bootstrap)
    print("Comparaison terminée et visualisations enregistrées dans le dossier 'visualizations'")
    
    # 4. Afficher un résumé des résultats
    print("\nRésumé des performances:")
    print("\nMéthode de copule gaussienne:")
    for idx, row in stats_copula.iterrows():
        if row['Type'] == 'Numérique':
            print(f"  - {row['Variable']}: KS p-value = {row['KS p-value']} (similitude: {'Haute' if float(row['KS p-value']) > 0.05 else 'Basse'})")
        else:
            print(f"  - {row['Variable']}: TV distance = {row['KS p-value']} (similitude: {'Haute' if float(row['KS p-value'].split(': ')[1]) < 0.1 else 'Basse'})")
    
    print("\nMéthode de bootstrap avancé:")
    for idx, row in stats_bootstrap.iterrows():
        if row['Type'] == 'Numérique':
            print(f"  - {row['Variable']}: KS p-value = {row['KS p-value']} (similitude: {'Haute' if float(row['KS p-value']) > 0.05 else 'Basse'})")
        else:
            print(f"  - {row['Variable']}: TV distance = {row['KS p-value']} (similitude: {'Haute' if float(row['KS p-value'].split(': ')[1]) < 0.1 else 'Basse'})")

if __name__ == "__main__":
    main()