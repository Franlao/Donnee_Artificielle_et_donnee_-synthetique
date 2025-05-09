import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import base64
import json

from .llm_generator import MistralGenerator, MISTRAL_AVAILABLE

def clean_dataframe_for_display(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un DataFrame pour s'assurer qu'il peut être affiché correctement dans Streamlit
    en convertissant les colonnes problématiques en types compatibles avec Arrow.
    """
    df = df_input.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convertit tous les éléments en chaîne, gérant None/NaN en chaînes vides
            # Puis s'assure que toute la colonne est de type string.
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else '').astype(str)
        elif df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        # Gérer d'autres types non standard si nécessaire à l'avenir
        # Par exemple, les colonnes de type 'category' peuvent parfois causer des soucis
        # if df[col].dtype.name == 'category':
        #     df[col] = df[col].astype(str)
    return df

def render_llm_tab():
    """Fonction pour rendre l'onglet de génération de données avec LLM"""
    st.write("""
    Cette section vous permet d'utiliser des grands modèles de langage (LLM) 
    pour générer des données de santé artificielles. Les données générées ne sont pas 
    basées sur des données réelles mais suivent les contraintes que vous spécifiez.
    """)
    
    if not MISTRAL_AVAILABLE:
        st.warning("""
        La bibliothèque Mistral AI n'est pas installée. Pour l'utiliser:
        
        ```bash
        pip install mistralai
        ```
        
        Puis redémarrez l'application.
        """)
    
    # Paramètres pour l'API Mistral
    st.subheader("Configuration de l'API Mistral")
    
    # Clé API
    api_key = st.text_input("Clé API Mistral", type="password")
    
    # Modèle à utiliser
    model = st.selectbox(
        "Modèle Mistral à utiliser",
        ["mistral-large-latest", "mistral-medium", "mistral-small"],
        index=0
    )
    
    # Description des données à générer
    st.subheader("Description des Données à Générer")
    
    # Sélection du type de données
    data_type = st.radio(
        "Type de données médicales à générer:",
        ["Dossiers patients standard", "Patients diabétiques", "Laboratoire médical", "Personnalisé"], 
        index=0
    )
    
    # Descriptions prédéfinies selon le type
    if data_type == "Dossiers patients standard":
        data_description = st.text_area(
            "Description des données (modifiable)",
            """Générer des dossiers de patients avec les caractéristiques suivantes:
- Âge entre 40 et 80 ans
- Sexe: homme ou femme
- IMC entre 18 et 40
- Niveau de glucose entre 70 et 300 mg/dL
- Tension artérielle entre 90/60 et 180/120 mmHg
- Cholestérol total entre 120 et 300 mg/dL
- Antécédents médicaux possibles: hypertension, diabète, hypercholestérolémie, maladie cardiaque
- Diagnostics primaires possibles: diabète de type 2, hypertension, syndrome métabolique, maladie coronarienne""",
            height=200
        )
        default_constraints = """- 30% des patients doivent avoir un diagnostic de diabète
- Les patients diabétiques doivent avoir une glycémie > 126 mg/dL
- 40% des patients doivent avoir une hypertension (tension > 140/90)
- La corrélation entre l'âge et la tension artérielle doit être d'environ 0.4
- Inclure au moins 20% de patients avec multiples comorbidités"""
    
    elif data_type == "Patients diabétiques":
        data_description = st.text_area(
            "Description des données (modifiable)",
            """Générer des dossiers de patients diabétiques avec les caractéristiques suivantes:
- Âge entre 50 et 75 ans
- Sexe: homme ou femme
- IMC entre 25 et 40
- Niveau de glucose entre 126 et 300 mg/dL (glycémie à jeun)
- HbA1c entre 6.5% et 12%
- Tension artérielle entre 120/70 et 180/120 mmHg
- Cholestérol total entre 150 et 350 mg/dL
- HDL entre 25 et 60 mg/dL
- LDL entre 70 et 200 mg/dL
- Triglycérides entre 100 et 500 mg/dL
- Antécédents médicaux incluant: hypertension, hypercholestérolémie, neuropathie, rétinopathie, néphropathie
- Médications: metformine, sulfonylurées, insuline, inhibiteurs SGLT2, etc.""",
            height=250
        )
        default_constraints = """- 100% des patients doivent avoir un diagnostic de diabète de type 2
- 60% doivent avoir une HbA1c > 7.5%
- 50% doivent avoir une tension artérielle élevée (>140/90)
- 40% doivent avoir au moins une complication (rétinopathie, neuropathie ou néphropathie)
- 70% doivent être sous metformine
- 30% doivent être sous insuline
- L'IMC doit être corrélé positivement avec la glycémie (r~0.3)"""
    
    elif data_type == "Laboratoire médical":
        data_description = st.text_area(
            "Description des données (modifiable)",
            """Générer des résultats de tests de laboratoire médicaux avec les caractéristiques suivantes:
- ID du patient
- Âge et sexe du patient
- Date du test
- Hémoglobine (g/dL): 10-18
- Globules blancs (10^9/L): 4.0-11.0
- Plaquettes (10^9/L): 150-450
- Glucose à jeun (mg/dL): 70-200
- Hémoglobine glyquée HbA1c (%): 4.5-12.0
- Urée (mg/dL): 15-50
- Créatinine (mg/dL): 0.5-2.0
- Sodium (mmol/L): 135-145
- Potassium (mmol/L): 3.5-5.0
- Chlore (mmol/L): 98-108
- TSH (µIU/mL): 0.4-4.0
- T4 libre (ng/dL): 0.8-1.8
- AST (U/L): 10-40
- ALT (U/L): 7-56
- Type d'échantillon (sang, urine)
- Médecin prescripteur""",
            height=300
        )
        default_constraints = """- 15% des patients doivent avoir une glycémie > 126 mg/dL
- 10% doivent avoir une anémie (hémoglobine < 12 g/dL pour les femmes, < 13 g/dL pour les hommes)
- 8% doivent avoir une leucocytose (globules blancs > 11.0 10^9/L)
- 5% doivent avoir une thrombocytopénie (plaquettes < 150 10^9/L)
- 12% doivent avoir une insuffisance rénale (créatinine > 1.2 mg/dL)
- 7% doivent avoir une hypothyroïdie (TSH > 4.0 µIU/mL)
- 10% doivent avoir des anomalies hépatiques (AST ou ALT > 40 U/L)"""
    
    else:  # Personnalisé
        data_description = st.text_area(
            "Description personnalisée des données",
            """Décrivez ici les données que vous souhaitez générer:
- Type de données
- Variables à inclure avec plages de valeurs
- Relations entre variables
- Autres caractéristiques importantes""",
            height=200
        )
        default_constraints = """Spécifiez ici vos contraintes:
- Distribution souhaitée des variables
- Corrélations attendues
- Pourcentages de cas spécifiques
- Autres contraintes"""
    
    # Contraintes spécifiques
    constraints = st.text_area(
        "Contraintes spécifiques (optionnel)",
        default_constraints,
        height=150
    )
    
    # Paramètres avancés
    with st.expander("Paramètres Avancés"):
        # Nombre d'échantillons
        n_samples = st.slider(
            "Nombre de dossiers patients à générer",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Température
        temperature = st.slider(
            "Température",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Des valeurs plus élevées rendent la sortie plus aléatoire, des valeurs plus basses plus déterministes"
        )
        
        # Nombre maximum de tokens
        max_tokens = st.slider(
            "Nombre maximum de tokens",
            min_value=1000,
            max_value=8000,
            value=4000,
            step=500,
            help="Nombre maximum de tokens dans la réponse"
        )
    
    # Générer les données
    if st.button("Générer avec LLM", key="generate_llm"):
        if not api_key:
            st.error("Veuillez entrer une clé API Mistral valide")
        else:
            with st.spinner("Génération de données avec Mistral LLM..."):
                try:
                    # Initialiser le générateur Mistral
                    mistral_gen = MistralGenerator(api_key=api_key, model=model)
                    
                    # Afficher le prompt qui sera envoyé
                    with st.expander("Voir le prompt envoyé au LLM"):
                        prompt = mistral_gen.create_prompt(
                            data_description=data_description,
                            constraints=constraints,
                            n_samples=n_samples
                        )
                        st.code(prompt)
                    
                    # Générer des données
                    start_time = time.time()
                    generated_df, error = mistral_gen.generate_data(
                        data_description=data_description,
                        constraints=constraints,
                        n_samples=n_samples,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    generation_time = time.time() - start_time
                    
                    if error:
                        st.error(f"Erreur lors de la génération des données: {error}")
                    else:
                        # Nettoyer le DataFrame avant de l'utiliser
                        generated_df_cleaned = clean_dataframe_for_display(generated_df)

                        # Afficher les données générées
                        st.subheader("Données Générées")
                        st.success(f"Génération réussie en {generation_time:.2f} secondes")
                        st.write(generated_df_cleaned) # Utiliser le DF nettoyé
                        
                        # Option de téléchargement
                        st.download_button(
                            label="Télécharger les Données Générées (CSV)",
                            data=generated_df_cleaned.to_csv(index=False).encode('utf-8'), # Utiliser le DF nettoyé
                            file_name="donnees_generees_llm.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisations
                        st.subheader("Analyse et Visualisations")
                        
                        # Détection automatique des types de colonnes à partir du DF nettoyé
                        numeric_cols = []
                        categorical_cols = []
                        for col in generated_df_cleaned.columns:
                            # Essayer de convertir en numérique pour la détection, au cas où le nettoyage l'aurait rendu string
                            try:
                                pd.to_numeric(generated_df_cleaned[col])
                                numeric_cols.append(col)
                            except ValueError:
                                categorical_cols.append(col)
                        
                        # Statistiques de base
                        if numeric_cols:
                            st.write("#### Statistiques descriptives")
                            # Assurez-vous que .describe() est aussi nettoyé avant l'affichage
                            described_data = generated_df_cleaned[numeric_cols].describe().reset_index()
                            st.write(clean_dataframe_for_display(described_data))
                        
                        # Créer des visualisations
                        if numeric_cols:
                            st.write("#### Distributions des variables numériques")
                            for col in numeric_cols:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                # Utiliser pd.to_numeric pour s'assurer que la colonne est traitable par sns.histplot
                                sns.histplot(pd.to_numeric(generated_df_cleaned[col], errors='coerce').dropna(), kde=True, ax=ax)
                                ax.set_title(f"Distribution de {col}")
                                st.pyplot(fig)
                            
                            if len(numeric_cols) > 1:
                                st.write("#### Matrice de corrélation")
                                fig, ax = plt.subplots(figsize=(10, 8))
                                # S'assurer que les données pour la corrélation sont numériques
                                numeric_df_for_corr = generated_df_cleaned[numeric_cols].apply(pd.to_numeric, errors='coerce')
                                correlation_matrix = numeric_df_for_corr.corr()
                                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                                st.pyplot(fig)
                        
                        if categorical_cols:
                            st.write("#### Distributions des variables catégorielles")
                            for col in categorical_cols:
                                if generated_df_cleaned[col].dtype == 'object' and generated_df_cleaned[col].apply(lambda x: isinstance(x, list)).any():
                                    all_items = []
                                    for items in generated_df_cleaned[col].dropna():
                                        if isinstance(items, list):
                                            all_items.extend(items)
                                        else:
                                            all_items.append(items)
                                    item_counts = pd.Series(all_items).value_counts()
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    item_counts.plot(kind='bar', ax=ax)
                                    ax.set_title(f"Fréquence des éléments dans {col}")
                                    ax.set_ylabel("Nombre d'occurrences")
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    value_counts = generated_df_cleaned[col].value_counts()
                                    # S'assurer que l'index est de type string pour le barplot
                                    sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax)
                                    ax.set_title(f"Distribution de {col}")
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        
                        if 'diabetes' in generated_df_cleaned.columns and 'glucose_level' in generated_df_cleaned.columns:
                            st.write("#### Analyse de la glycémie par statut diabétique")
                            # S'assurer que les colonnes sont des types corrects pour l'analyse
                            temp_df_analysis = generated_df_cleaned.copy()
                            temp_df_analysis['diabetes'] = pd.to_numeric(temp_df_analysis['diabetes'], errors='coerce').fillna(0).astype(int)
                            temp_df_analysis['glucose_level'] = pd.to_numeric(temp_df_analysis['glucose_level'], errors='coerce')

                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x='diabetes', y='glucose_level', data=temp_df_analysis, ax=ax)
                            ax.set_title("Niveaux de glucose par statut diabétique")
                            ax.set_xlabel("Diabétique (1=Oui, 0=Non)")
                            ax.set_ylabel("Niveau de glucose (mg/dL)")
                            st.pyplot(fig)
                            
                            glucose_by_diabetes = temp_df_analysis.groupby('diabetes')['glucose_level'].describe().reset_index()
                            st.write("Statistiques de glucose par statut diabétique:")
                            st.write(clean_dataframe_for_display(glucose_by_diabetes))
                            
                            diabetic_percent = temp_df_analysis['diabetes'].mean() * 100
                            st.write(f"Pourcentage de patients diabétiques: {diabetic_percent:.1f}%")
                            
                            diabetic_patients = temp_df_analysis[temp_df_analysis['diabetes'] == 1]
                            if not diabetic_patients.empty:
                                high_glucose_percent = (diabetic_patients['glucose_level'] > 126).mean() * 100
                                st.write(f"Pourcentage de diabétiques avec glucose > 126 mg/dL: {high_glucose_percent:.1f}%")
                                if high_glucose_percent < 80:
                                    st.warning("⚠️ Attention: Certains patients diabétiques ont une glycémie normale, ce qui est inhabituel.")
                
                except Exception as e:
                    st.error(f"Une erreur s'est produite: {str(e)}")
                    import traceback
                    st.exception(traceback.format_exc())
    else:
        # Afficher des informations sur la méthode
        st.markdown("""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: #e3f2fd; border: 1px solid #2196F3;">
        <h3>Comment fonctionnent les LLM pour la génération de données?</h3>
        <p>Les grands modèles de langage (LLM) comme Mistral peuvent générer des données médicales artificielles en suivant des instructions textuelles et en respectant des contraintes spécifiées.</p>
        
        <h4>Avantages:</h4>
        <ul>
            <li>Génération de données avec cohérence clinique grâce aux connaissances médicales du modèle</li>
            <li>Capacité à suivre des contraintes complexes exprimées en langage naturel</li>
            <li>Pas besoin de données réelles comme point de départ</li>
        </ul>
        
        <h4>Limitations:</h4>
        <ul>
            <li>Risque d'hallucinations ou d'informations médicalement incohérentes</li>
            <li>Nombre limité d'échantillons par requête</li>
            <li>La structure exacte des données est moins contrôlable qu'avec des méthodes statistiques</li>
        </ul>
        
        <h4>Applications:</h4>
        <ul>
            <li>Création de jeux de données pour prototypage rapide</li>
            <li>Génération d'exemples réalistes pour formation médicale</li>
            <li>Simulation de cas cliniques avec des caractéristiques spécifiques</li>
            <li>Test de systèmes d'information clinique sans utiliser de données réelles de patients</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def visualize_categorical_comparison(real_data, synthetic_df, col):
    """
    Crée une visualisation comparative pour une colonne catégorielle
    en s'assurant que les types de données sont compatibles.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convertir la colonne en chaîne dans les deux DataFrames pour éviter les problèmes de type
    real_col = real_data[col].astype(str)
    # Utiliser le DataFrame synthétique déjà nettoyé si possible, sinon nettoyer ici
    synth_col_cleaned = clean_dataframe_for_display(synthetic_df[[col]])[col].astype(str)
    
    # Compter les catégories
    real_counts = real_col.value_counts(normalize=True)
    synth_counts = synth_col_cleaned.value_counts(normalize=True)
    
    # S'assurer que les deux séries ont les mêmes indices
    all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
    
    # Limiter le nombre de catégories à afficher si trop nombreuses
    if len(all_categories) > 20:
        # Trouver les catégories les plus fréquentes
        top_real = set(real_counts.nlargest(10).index)
        top_synth = set(synth_counts.nlargest(10).index)
        # Combiner les catégories les plus fréquentes
        all_categories = sorted(list(top_real.union(top_synth))) # Ensure list for bar_positions
        
        # Ajouter une note au titre
        ax.set_title(f'Comparaison de la distribution (top catégories) pour {col}')
    else:
        ax.set_title(f'Comparaison de la distribution des catégories pour {col}')
    
    # Créer un DataFrame de comparaison en s'assurant que tous les indices sont présents
    compare_data = {
        'Catégorie': all_categories,
        'Réel': [real_counts.get(cat, 0) for cat in all_categories],
        'Synthétique': [synth_counts.get(cat, 0) for cat in all_categories]
    }
    
    compare_df = pd.DataFrame(compare_data)
    
    # Reformater pour seaborn
    compare_df_long = pd.melt(
        compare_df, 
        id_vars=['Catégorie'], 
        value_vars=['Réel', 'Synthétique'],
        var_name='Type de Données', 
        value_name='Proportion'
    )
    
    # Graphique avec positions fixes pour éviter les avertissements
    if all_categories: # Ensure all_categories is not empty
        bar_positions = np.arange(len(all_categories))
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
    
    # Utiliser barplot avec des positions explicites
    sns.barplot(x='Catégorie', y='Proportion', hue='Type de Données', data=compare_df_long, ax=ax)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    return fig

def visualize_numeric_comparison(real_data, synthetic_df, col):
    """
    Crée une visualisation comparative pour une colonne numérique.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # S'assurer que les données sont numériques
    try:
        real_values = pd.to_numeric(real_data[col], errors='coerce').dropna()
        # Utiliser le DataFrame synthétique déjà nettoyé si possible, sinon nettoyer ici
        synth_values_cleaned_series = pd.to_numeric(clean_dataframe_for_display(synthetic_df[[col]])[col], errors='coerce').dropna()

        if real_values.empty and synth_values_cleaned_series.empty:
            ax.text(0.5, 0.5, "Pas de données numériques valides à afficher.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Pas de données pour {col}")
            return fig

        real_mean = real_values.mean() if not real_values.empty else np.nan
        synth_mean = synth_values_cleaned_series.mean() if not synth_values_cleaned_series.empty else np.nan
        real_std = real_values.std() if not real_values.empty else np.nan
        synth_std = synth_values_cleaned_series.std() if not synth_values_cleaned_series.empty else np.nan
        
        # Histogramme des données réelles avec KDE
        if not real_values.empty:
            sns.histplot(real_values, color='blue', alpha=0.5, 
                        label=f'Données réelles (moy={real_mean:.2f}, std={real_std:.2f})', 
                        kde=True, ax=ax)
            ax.axvline(real_mean, color='blue', linestyle='--', alpha=0.7)

        # Histogramme des données synthétiques avec KDE
        if not synth_values_cleaned_series.empty:
            sns.histplot(synth_values_cleaned_series, color='red', alpha=0.5,
                        label=f'Données synthétiques (moy={synth_mean:.2f}, std={synth_std:.2f})', 
                        kde=True, ax=ax)
            ax.axvline(synth_mean, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Comparaison des distributions pour {col}')
        
        # Ajuster la légende pour qu'elle soit bien visible
        if not real_values.empty or not synth_values_cleaned_series.empty:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
    except Exception as e:
        # En cas d'erreur, afficher un graphique d'erreur
        ax.text(0.5, 0.5, f"Erreur de visualisation: {str(e)}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Impossible de visualiser la colonne {col}")
    
    return fig