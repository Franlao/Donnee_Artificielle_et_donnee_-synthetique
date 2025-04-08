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
from dataclasses import dataclass

# Vérifier si ce fichier est exécuté directement ou importé
is_imported = __name__ != "__main__"

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

# Pour l'API Mistral
try:
    from mistralai import Mistral,UserMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("La bibliothèque Mistral AI n'est pas installée. Pour l'installer: pip install mistralai")

# Alternative scikit-learn pour méthodes simples (toujours installée avec streamlit)
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ----------------- MODULE DE TRAITEMENT DES DONNÉES -----------------

@dataclass
class DatasetMetadata:
    """Métadonnées pour le prétraitement et la transformation des données"""
    numeric_cols: List[str]
    categorical_cols: List[str]
    numeric_scaler: Any
    categorical_encoder: Any
    input_dim: int
    numeric_stats: Dict[str, Dict[str, float]]
    is_integer: Dict[str, bool]  # Indique si une colonne numérique est un entier
    categorical_values: Dict[str, List]  # Stocke les valeurs possibles pour chaque catégorie

class DataProcessor:
    """Classe pour gérer le prétraitement et la transformation des données avec détection intelligente des types"""
    
    @staticmethod
    def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Analyser les données pour comprendre leur structure et leurs statistiques"""
        analysis = {
            "total_rows": len(data),
            "columns": {},
            "missing_values": {},
            "correlations": None,
            "column_types": {
                "numeric": [],
                "categorical": [],
                "integer": [],
                "binary": []
            }
        }
        
        # Détecter automatiquement les types de colonnes
        numeric_cols, categorical_cols, is_integer = DataProcessor.infer_column_types(data)
        
        # Trouver les colonnes binaires (0/1)
        binary_cols = []
        for col in data.columns:
            if set(data[col].dropna().astype(str).unique()).issubset({'0', '1', '0.0', '1.0'}):
                binary_cols.append(col)
        
        analysis["column_types"]["numeric"] = numeric_cols
        analysis["column_types"]["categorical"] = categorical_cols
        analysis["column_types"]["integer"] = [col for col, is_int in is_integer.items() if is_int]
        analysis["column_types"]["binary"] = binary_cols
        
        # Analyser chaque colonne
        for col in data.columns:
            series = data[col]
            col_type = str(series.dtype)
            
            analysis["columns"][col] = {
                "type": col_type,
                "unique_values": series.nunique(),
                "detected_type": "categorical" if col in categorical_cols else "numeric"
            }
            
            # Analyse supplémentaire pour les colonnes numériques
            if col in numeric_cols:
                analysis["columns"][col].update({
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                    "is_integer": col in is_integer and is_integer[col],
                    "is_binary": col in binary_cols
                })
            elif col in categorical_cols:
                analysis["columns"][col].update({
                    "categories": sorted(series.dropna().unique().tolist()),
                    "is_binary": col in binary_cols
                })
            
            # Compter les valeurs manquantes
            missing = series.isna().sum()
            if missing > 0:
                analysis["missing_values"][col] = int(missing)
        
        # Calculer les corrélations pour les colonnes numériques
        if numeric_cols:
            analysis["correlations"] = data[numeric_cols].corr().to_dict()
        
        return analysis
    
    @staticmethod
    def is_probably_categorical(series: pd.Series, threshold: int = 10) -> bool:
        """
        Détermine si une série numérique est probablement catégorielle
        basée sur le nombre de valeurs uniques et leur distribution
        
        Args:
            series: La série pandas à analyser
            threshold: Nombre maximum de valeurs uniques pour considérer comme catégorique
            
        Returns:
            bool: True si la série est probablement catégorielle
        """
        # Si déjà de type catégorie ou objet
        if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            return True
        
        # Si booléen
        if pd.api.types.is_bool_dtype(series):
            return True
        
        # Si contient des nombres entiers
        if pd.api.types.is_integer_dtype(series):
            unique_values = series.nunique()
            # Si peu de valeurs uniques, probablement catégorielle
            if unique_values <= threshold:
                return True
            
            # Vérifier si toutes les valeurs sont 0, 1 ou NA (binaire)
            values_set = set(series.dropna().unique())
            if values_set.issubset({0, 1}) or values_set.issubset({'0', '1'}):
                return True
        
        # Pour les flottants, vérifier si ce sont en fait des entiers avec peu de valeurs uniques
        if pd.api.types.is_float_dtype(series):
            # Vérifier si tous les flottants sont des entiers (comme 1.0, 2.0)
            is_integer_values = np.all(series.dropna() == series.dropna().astype(int))
            if is_integer_values:
                # Si ce sont des entiers (comme 1.0, 2.0) et peu nombreux
                unique_values = series.nunique()
                if unique_values <= threshold:
                    return True
        
        return False
    
    @staticmethod
    def infer_column_types(data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, bool]]:
        """
        Détermine intelligemment quelles colonnes sont numériques et lesquelles sont catégorielles
        
        Args:
            data: DataFrame à analyser
            
        Returns:
            Tuple contenant:
            - Liste des colonnes numériques
            - Liste des colonnes catégorielles
            - Dict indiquant quelles colonnes numériques doivent être traitées comme des entiers
        """
        numeric_cols = []
        categorical_cols = []
        is_integer = {}
        
        for col in data.columns:
            series = data[col]
            
            # Si la colonne est de type numérique
            if pd.api.types.is_numeric_dtype(series):
                if DataProcessor.is_probably_categorical(series):
                    categorical_cols.append(col)
                    # Convertir en chaîne pour l'encodage one-hot
                    data[col] = data[col].astype(str)
                else:
                    numeric_cols.append(col)
                    # Détecter si c'est un entier
                    is_integer[col] = pd.api.types.is_integer_dtype(series) or (
                        pd.api.types.is_float_dtype(series) and 
                        np.all(series.dropna() == series.dropna().astype(int))
                    )
            else:
                categorical_cols.append(col)
        
        return numeric_cols, categorical_cols, is_integer
    
    @staticmethod
    def preprocess_data(data: pd.DataFrame, categorical_cols_override: List[str] = None) -> Tuple[np.ndarray, DatasetMetadata]:
        """
        Prétraiter les données pour l'entraînement du modèle avec une détection intelligente des types
        
        Args:
            data: DataFrame à prétraiter
            categorical_cols_override: Liste optionnelle des colonnes à traiter comme catégorielles
        
        Returns:
            Tuple contenant:
            - Les données prétraitées
            - Les métadonnées du jeu de données
        """
        # Copier le dataframe pour éviter de modifier l'original
        df = data.copy()
        
        # Détection intelligente des types de colonnes
        if categorical_cols_override is not None:
            # Utiliser la liste fournie
            categorical_cols = categorical_cols_override
            numeric_cols = [col for col in df.columns if col not in categorical_cols]
            is_integer = {col: pd.api.types.is_integer_dtype(df[col]) for col in numeric_cols}
            
            # Convertir les colonnes catégorielles en chaînes
            for col in categorical_cols:
                df[col] = df[col].astype(str)
        else:
            # Détection automatique
            numeric_cols, categorical_cols, is_integer = DataProcessor.infer_column_types(df)
        
        print(f"Colonnes numériques: {numeric_cols}")
        print(f"Colonnes catégorielles: {categorical_cols}")
        print(f"Colonnes entières: {[col for col, is_int in is_integer.items() if is_int]}")
        
        # Stocker les statistiques pour chaque colonne numérique
        numeric_stats = {}
        for col in numeric_cols:
            numeric_stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }
        
        # Stocker les valeurs possibles pour chaque catégorie
        categorical_values = {}
        for col in categorical_cols:
            categorical_values[col] = sorted(df[col].dropna().unique().tolist())
        
        # Prétraiter les variables numériques
        if numeric_cols:
            numeric_data = df[numeric_cols].copy()
            # Gérer les valeurs manquantes avant la mise à l'échelle
            numeric_data = numeric_data.fillna(numeric_data.mean())
            numeric_scaler = StandardScaler()
            numeric_scaled = numeric_scaler.fit_transform(numeric_data)
        else:
            numeric_scaled = np.array([]).reshape(len(df), 0)
            numeric_scaler = None
        
        # Prétraiter les variables catégorielles
        if categorical_cols:
            categorical_data = df[categorical_cols].copy()
            # Gérer les valeurs manquantes dans les données catégorielles
            for col in categorical_cols:
                categorical_data[col] = categorical_data[col].fillna(categorical_data[col].mode().iloc[0])
            
            categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_encoded = categorical_encoder.fit_transform(categorical_data)
        else:
            categorical_encoded = np.array([]).reshape(len(df), 0)
            categorical_encoder = None
        
        # Combiner les données numériques et catégorielles
        if numeric_cols and categorical_cols:
            preprocessed_data = np.hstack([numeric_scaled, categorical_encoded])
        elif numeric_cols:
            preprocessed_data = numeric_scaled
        else:
            preprocessed_data = categorical_encoded
        
        # Créer les métadonnées du jeu de données
        metadata = DatasetMetadata(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_scaler=numeric_scaler,
            categorical_encoder=categorical_encoder,
            input_dim=preprocessed_data.shape[1],
            numeric_stats=numeric_stats,
            is_integer=is_integer,
            categorical_values=categorical_values
        )
        
        return preprocessed_data, metadata
    
    @staticmethod
    def inverse_transform(synthetic_data: np.ndarray, metadata: DatasetMetadata) -> pd.DataFrame:
        """
        Convertir les données synthétiques au format d'origine avec le bon typage
        """
        # Initialiser un DataFrame vide
        result_df = pd.DataFrame()
        
        # Traiter les variables numériques si elles existent
        if metadata.numeric_cols and metadata.numeric_scaler:
            # Déterminer la dimension pour les parties numériques
            numeric_dim = len(metadata.numeric_cols)
            
            # Extraire la partie numérique
            numeric_part = synthetic_data[:, :numeric_dim]
            
            # Transformer inversement les données numériques
            numeric_inverted = metadata.numeric_scaler.inverse_transform(numeric_part)
            
            # Créer un DataFrame pour les données numériques
            df_numeric = pd.DataFrame(numeric_inverted, columns=metadata.numeric_cols)
            
            # Appliquer les contraintes basées sur les statistiques des données d'origine
            for col in metadata.numeric_cols:
                stats = metadata.numeric_stats[col]
                
                # S'assurer que les valeurs sont dans la plage d'origine avec une petite marge
                df_numeric[col] = np.clip(
                    df_numeric[col], 
                    stats["min"] * 0.95, 
                    stats["max"] * 1.05
                )
                
                # Arrondir les colonnes entières
                if col in metadata.is_integer and metadata.is_integer[col]:
                    df_numeric[col] = np.round(df_numeric[col]).astype(int)
            
            result_df = df_numeric
        
        # Traiter les variables catégorielles si elles existent
        if metadata.categorical_cols and metadata.categorical_encoder:
            # Extraire la partie catégorielle
            start_idx = len(metadata.numeric_cols) if metadata.numeric_cols else 0
            categorical_part = synthetic_data[:, start_idx:]
            
            # Appliquer softmax pour assurer une distribution de probabilité appropriée
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)
            
            # Si l'encodeur a des catégories connues
            if hasattr(metadata.categorical_encoder, 'categories_'):
                # Obtenir les dimensions des catégories
                cat_dims = [len(c) for c in metadata.categorical_encoder.categories_]
                
                start_idx = 0
                cat_values = []
                cat_columns = []
                
                # Traiter chaque variable catégorielle
                for i, col in enumerate(metadata.categorical_cols):
                    dim = cat_dims[i]
                    
                    # Obtenir les probabilités encodées pour cette variable
                    cat_probs = categorical_part[:, start_idx:start_idx + dim]
                    
                    # Convertir en distribution de probabilité appropriée
                    cat_probs = softmax(cat_probs)
                    
                    # Obtenir la catégorie avec la plus haute probabilité
                    cat_idx = np.argmax(cat_probs, axis=1)
                    
                    # Convertir les indices en catégories d'origine
                    original_categories = metadata.categorical_encoder.categories_[i]
                    cat_column = np.array([original_categories[idx] for idx in cat_idx])
                    
                    cat_values.append(cat_column)
                    cat_columns.append(col)
                    start_idx += dim
                
                # Créer un DataFrame pour les données catégorielles
                if cat_values:
                    df_categorical = pd.DataFrame(np.column_stack(cat_values), columns=cat_columns)
                    
                    # Convertir les colonnes binaires (0/1) au format approprié
                    for col in df_categorical.columns:
                        # Si toutes les valeurs sont '0' ou '1', convertir en entier
                        if set(df_categorical[col].unique()).issubset({'0', '1', '0.0', '1.0'}):
                            df_categorical[col] = df_categorical[col].astype(int)
                    
                    # Combiner avec les données numériques s'il y en a
                    if not result_df.empty:
                        result_df = pd.concat([result_df, df_categorical], axis=1)
                    else:
                        result_df = df_categorical
            else:
                # Fallback si l'encodeur catégoriel n'a pas la structure attendue
                categorical_data = metadata.categorical_encoder.inverse_transform(categorical_part)
                df_categorical = pd.DataFrame(categorical_data, columns=metadata.categorical_cols)
                
                # Convertir les colonnes binaires (0/1) au format approprié
                for col in df_categorical.columns:
                    # Si toutes les valeurs sont '0' ou '1', convertir en entier
                    if set(df_categorical[col].unique()).issubset({'0', '1', '0.0', '1.0'}):
                        df_categorical[col] = df_categorical[col].astype(int)
                
                # Combiner avec les données numériques s'il y en a
                if not result_df.empty:
                    result_df = pd.concat([result_df, df_categorical], axis=1)
                else:
                    result_df = df_categorical
        
        return result_df
    
# ----------------- MODULE LLMS -----------------
class MistralGenerator:
    """Classe pour générer des données médicales en utilisant les LLM Mistral"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """Initialiser le générateur LLM Mistral"""
        self.client = Mistral(api_key=api_key)
        self.model = model
    
    def create_prompt(self, data_description: str, constraints: str = None,
                      n_samples: int = 10, output_format: str = "json") -> str:
        """Créer un prompt pour générer des données médicales"""
        prompt = f"""
Générez {n_samples} dossiers médicaux synthétiques au format {output_format.upper()}.

Description des données à générer:
{data_description}

Format attendu:
[
  {{
    "patient_id": "SYN_001",
    "age": 67,
    "gender": "F",
    "bmi": 28.4,
    "medical_history": ["hypertension", "osteoarthritis"],
    "glucose_level": 142,
    "blood_pressure": "145/88",
    "cholesterol": 210,
    "hdl": 45,
    "ldl": 130,
    "triglycerides": 180,
    "primary_diagnosis": "Type 2 Diabetes"
  }},
  ...
]
"""
        
        # Ajouter des contraintes spécifiques si fournies
        if constraints:
            prompt += f"\nContraintes spécifiques à respecter:\n{constraints}\n"
        
        # Ajouter des instructions pour la cohérence médicale
        prompt += """
Important: Les données doivent être médicalement cohérentes et refléter des corrélations réalistes:
- Les patients diabétiques devraient généralement avoir des taux de glucose plus élevés (>126 mg/dL à jeun)
- Les patients hypertendus devraient avoir des valeurs de tension artérielle élevées (>140/90)
- L'IMC devrait être cohérent avec le poids et la taille
- Les profils lipidiques (cholestérol, HDL, LDL) devraient être cohérents avec l'état de santé global
- L'âge devrait influencer les variables appropriées (par ex., tension artérielle plus élevée chez les patients plus âgés)
- Les valeurs de laboratoire doivent se situer dans des plages médicales réalistes

Retournez UNIQUEMENT le tableau JSON sans texte supplémentaire.
"""
        
        return prompt
    
    def generate_data(self, data_description: str, constraints: str = None,
                     n_samples: int = 10, max_tokens: int = 4096,
                     temperature: float = 0.7) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Générer des données médicales en utilisant l'API Mistral avec une gestion robuste des erreurs"""
        # Créer le prompt
        prompt = self.create_prompt(data_description, constraints, n_samples)
        
        # Créer des messages pour l'appel API
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Appeler l'API Mistral
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extraire le texte généré
            generated_text = response.choices[0].message.content
            
            # Essayer de parser la réponse JSON
            try:
                # Trouver le début et la fin du JSON dans la réponse
                start_idx = generated_text.find('[')
                end_idx = generated_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = generated_text[start_idx:end_idx]
                    data = json.loads(json_text)
                    
                    # Convertir en DataFrame
                    df = pd.DataFrame(data)
                    
                    # Post-traiter les données pour la qualité et la cohérence
                    df = self._post_process_data(df)
                    
                    return df, None
                else:
                    error_message = "Impossible de trouver un format JSON valide dans la réponse du modèle."
                    return None, error_message
                    
            except json.JSONDecodeError as e:
                # Essayer d'extraire un sous-ensemble JSON valide si possible
                import re
                json_pattern = r'\[\s*{.*}\s*\]'
                match = re.search(json_pattern, generated_text, re.DOTALL)
                
                if match:
                    try:
                        json_text = match.group(0)
                        data = json.loads(json_text)
                        df = pd.DataFrame(data)
                        df = self._post_process_data(df)
                        return df, None
                    except:
                        pass
                
                error_message = f"Erreur d'analyse JSON: {str(e)}\n\nTexte généré:\n{generated_text}"
                return None, error_message
                
        except Exception as e:
            error_message = f"Erreur lors de l'appel à l'API Mistral: {str(e)}"
            return None, error_message
    
    def _post_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-traiter les données générées pour la qualité et la cohérence"""
        # Assurer un format cohérent pour l'ID du patient
        if 'patient_id' in df.columns:
            if not df['patient_id'].str.contains('SYN_').all():
                df['patient_id'] = ['SYN_' + str(i+1).zfill(3) for i in range(len(df))]
        
        # Assurer que les données numériques sont correctement typées
        numeric_cols = ['age', 'bmi', 'glucose_level', 'cholesterol', 'hdl', 'ldl', 'triglycerides']
        for col in [c for c in numeric_cols if c in df.columns]:
            if df[col].dtype == object:
                # Convertir les nombres sous forme de chaîne en float
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remplir les valeurs manquantes avec des valeurs par défaut raisonnables
            if col == 'age':
                df[col] = df[col].fillna(65).astype(int)
            elif col == 'bmi':
                df[col] = df[col].fillna(25.0).astype(float)
            else:
                df[col] = df[col].fillna(df[col].mean()).astype(float)
        
        # Traiter la tension artérielle si elle existe
        if 'blood_pressure' in df.columns:
            if df['blood_pressure'].dtype == object:
                # Essayer de standardiser le format
                df['blood_pressure'] = df['blood_pressure'].apply(
                    lambda x: self._standardize_bp_format(x) if isinstance(x, str) else x
                )
        
        # Traiter l'historique médical s'il existe
        if 'medical_history' in df.columns:
            if df['medical_history'].dtype == object:
                # Convertir les listes sous forme de chaîne ou les listes de chaînes en listes
                df['medical_history'] = df['medical_history'].apply(
                    lambda x: self._parse_medical_history(x) if isinstance(x, str) else x
                )
        
        # Convertir les colonnes binaires (0/1) en entiers si nécessaire
        binary_cols = ['diabetes', 'hypertension', 'smoking']
        for col in [c for c in binary_cols if c in df.columns]:
            # Vérifier si la colonne contient principalement des valeurs 0/1 ou '0'/'1'
            if df[col].dtype == object:
                if set(df[col].dropna().astype(str).unique()).issubset({'0', '1', 'true', 'false', 'True', 'False'}):
                    # Standardiser en 0/1
                    df[col] = df[col].map({'0': 0, '1': 1, 'false': 0, 'true': 1, 'False': 0, 'True': 1}).astype(int)
        
        return df
    
    def _standardize_bp_format(self, bp_string: str) -> str:
        """Standardiser le format de la tension artérielle en systolique/diastolique"""
        import re
        
        # Vérifier si c'est déjà au bon format
        if re.match(r'\d+/\d+', bp_string):
            return bp_string
        
        # Essayer d'extraire les nombres
        numbers = re.findall(r'\d+', bp_string)
        if len(numbers) >= 2:
            return f"{numbers[0]}/{numbers[1]}"
        
        # Valeur par défaut
        return "120/80"
    
    def _parse_medical_history(self, history_str: str) -> List[str]:
        """Analyser l'historique médical de chaîne à liste"""
        import re
        
        # Vérifier si c'est déjà une chaîne de type liste
        if history_str.startswith('[') and history_str.endswith(']'):
            try:
                # Essayer de parser comme JSON
                return json.loads(history_str)
            except:
                pass
        
        # Diviser par virgules
        items = [item.strip() for item in re.split(r',|;', history_str) if item.strip()]
        return items

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
                        # Afficher les données générées
                        st.subheader("Données Générées")
                        st.success(f"Génération réussie en {generation_time:.2f} secondes")
                        st.write(generated_df)
                        
                        # Option de téléchargement
                        st.download_button(
                            label="Télécharger les Données Générées (CSV)",
                            data=generated_df.to_csv(index=False).encode('utf-8'),
                            file_name="donnees_generees_llm.csv",
                            mime="text/csv"
                        )
                        
                        # Visualisations
                        st.subheader("Analyse et Visualisations")
                        
                        # Détection automatique des types de colonnes
                        numeric_cols = []
                        categorical_cols = []
                        for col in generated_df.columns:
                            if pd.api.types.is_numeric_dtype(generated_df[col]):
                                numeric_cols.append(col)
                            else:
                                categorical_cols.append(col)
                        
                        # Statistiques de base
                        if numeric_cols:
                            st.write("#### Statistiques descriptives")
                            st.write(generated_df[numeric_cols].describe())
                        
                        # Créer des visualisations
                        if numeric_cols:
                            st.write("#### Distributions des variables numériques")
                            # Afficher des histogrammes pour les colonnes numériques
                            for col in numeric_cols:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(generated_df[col], kde=True, ax=ax)
                                ax.set_title(f"Distribution de {col}")
                                st.pyplot(fig)
                            
                            # Matrice de corrélation
                            if len(numeric_cols) > 1:
                                st.write("#### Matrice de corrélation")
                                fig, ax = plt.subplots(figsize=(10, 8))
                                correlation_matrix = generated_df[numeric_cols].corr()
                                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
                                st.pyplot(fig)
                        
                        # Colonnes catégorielles
                        if categorical_cols:
                            st.write("#### Distributions des variables catégorielles")
                            for col in categorical_cols:
                                # Vérifier si c'est une liste (comme medical_history)
                                if generated_df[col].dtype == 'object' and generated_df[col].apply(lambda x: isinstance(x, list)).any():
                                    # Aplatir les listes
                                    all_items = []
                                    for items in generated_df[col].dropna():
                                        if isinstance(items, list):
                                            all_items.extend(items)
                                        else:
                                            all_items.append(items)
                                    
                                    # Compter les occurrences
                                    item_counts = pd.Series(all_items).value_counts()
                                    
                                    # Créer un graphique à barres
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    item_counts.plot(kind='bar', ax=ax)
                                    ax.set_title(f"Fréquence des éléments dans {col}")
                                    ax.set_ylabel("Nombre d'occurrences")
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    # Graphique à barres pour les catégories
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    value_counts = generated_df[col].value_counts()
                                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                                    ax.set_title(f"Distribution de {col}")
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        
                        # Analyse spécifique pour les données médicales
                        if 'diabetes' in generated_df.columns and 'glucose_level' in generated_df.columns:
                            st.write("#### Analyse de la glycémie par statut diabétique")
                            
                            # Convertir en numérique si nécessaire
                            if generated_df['diabetes'].dtype == object:
                                generated_df['diabetes'] = generated_df['diabetes'].astype(str).map({'True': 1, 'true': 1, '1': 1, 'False': 0, 'false': 0, '0': 0}).astype(int)
                            
                            # Boxplot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x='diabetes', y='glucose_level', data=generated_df, ax=ax)
                            ax.set_title("Niveaux de glucose par statut diabétique")
                            ax.set_xlabel("Diabétique (1=Oui, 0=Non)")
                            ax.set_ylabel("Niveau de glucose (mg/dL)")
                            st.pyplot(fig)
                            
                            # Statistiques
                            glucose_by_diabetes = generated_df.groupby('diabetes')['glucose_level'].describe()
                            st.write("Statistiques de glucose par statut diabétique:")
                            st.write(glucose_by_diabetes)
                            
                            # Vérifier le pourcentage de diabétiques
                            diabetic_percent = generated_df['diabetes'].mean() * 100
                            st.write(f"Pourcentage de patients diabétiques: {diabetic_percent:.1f}%")
                            
                            # Vérifier si les diabétiques ont une glycémie > 126
                            diabetic_patients = generated_df[generated_df['diabetes'] == 1]
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


# ----------------- MODULE D'INTERFACE STREAMLIT -----------------

# Fonction pour visualiser les comparaisons des colonnes catégorielles
def visualize_categorical_comparison(real_data, synthetic_df, col):
    """
    Crée une visualisation comparative pour une colonne catégorielle
    en s'assurant que les types de données sont compatibles.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convertir la colonne en chaîne dans les deux DataFrames pour éviter les problèmes de type
    real_col = real_data[col].astype(str)
    synth_col = synthetic_df[col].astype(str)
    
    # Compter les catégories
    real_counts = real_col.value_counts(normalize=True)
    synth_counts = synth_col.value_counts(normalize=True)
    
    # S'assurer que les deux séries ont les mêmes indices
    all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
    
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
    
    # Graphique
    sns.barplot(x='Catégorie', y='Proportion', hue='Type de Données', data=compare_df_long, ax=ax)
    ax.set_title(f'Comparaison de la distribution des catégories pour {col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()    
    return fig

# Fonction pour visualiser les comparaisons des colonnes numériques
def visualize_numeric_comparison(real_data, synthetic_df, col):
    """
    Crée une visualisation comparative pour une colonne numérique.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogramme des données réelles
    sns.histplot(real_data[col], color='blue', alpha=0.5, 
                label='Données réelles', kde=True, ax=ax)
    
    # Histogramme des données synthétiques
    sns.histplot(synthetic_df[col], color='red', alpha=0.5,
                label='Données synthétiques', kde=True, ax=ax)
    
    ax.set_title(f'Comparaison des distributions pour {col}')
    ax.legend()
    
    return fig

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
        
        # Charger des données réelles
        uploaded_file = st.file_uploader("Charger un fichier CSV contenant des données réelles", 
                                        type="csv", key="ai_synth_upload")
        
        if uploaded_file is not None:
            # Charger les données
            try:
                real_data = pd.read_csv(uploaded_file)
                
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
                
                # Prétraiter les données avec les types spécifiés par l'utilisateur
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
                save_model = st.checkbox(
                    "Sauvegarder le modèle entraîné", 
                    value=False,
                    help="Sauvegarder le modèle entraîné pour une utilisation future"
                )
                
                if save_model:
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
                        if ai_method.startswith("GAN"):
                            # Code pour GAN...
                            st.info("Entraînement du GAN en cours...")
                        elif ai_method.startswith("VAE"):
                            # Code pour VAE...
                            st.info("Entraînement du VAE en cours...")
                        else:  # Méthodes statistiques simples
                            # Code pour méthodes statistiques...
                            st.info(f"Utilisation de la méthode statistique {method_type.upper()}...")
                        
                        # Génération de données synthétiques simulée pour démonstration
                        st.subheader("Données Synthétiques Générées")
                        
                        # Génération de données aléatoires pour le prototype
                        random_data = np.random.randn(*preprocessed_data.shape)
                        synthetic_df = DataProcessor.inverse_transform(random_data, metadata)
                        
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
                        
                    except Exception as e:
                        st.error(f"Une erreur s'est produite: {str(e)}")
                        import traceback
                        st.exception(traceback.format_exc())
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier CSV: {str(e)}")
    
    with subtab2:
        # Code pour l'onglet des données artificielles (LLM)
        st.subheader("Génération de Données Artificielles avec LLM")
        render_llm_tab()

# Si ce script est exécuté directement (pas importé)
if __name__ == "__main__":
    # Configuration de la page
    st.set_page_config(
        page_title="Méthodes d'IA pour Données Synthétiques",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Titre principal
    st.title("Génération de Données avec l'Intelligence Artificielle")
    
    # Appeler la fonction de rendu
    render_ai_methods_tab()