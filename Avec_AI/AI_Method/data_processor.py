import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, List, Union, Optional, Any
from dataclasses import dataclass
import warnings

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
    high_cardinality_cols: List[str]  # Colonnes à cardinalité élevée traitées différemment
    cardinality_limits: Dict[str, int]  # Limites de cardinalité appliquées

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
                # FIX: Trier de manière sécurisée en convertissant d'abord toutes les valeurs en chaînes
                try:
                    # Première méthode: trier après conversion en chaînes
                    categories = series.dropna().unique().tolist()
                    categories_as_str = [str(x) for x in categories]
                    sorted_categories = sorted(categories_as_str)
                    # Reconvertir au type d'origine si possible
                    if all(cat.isdigit() for cat in categories_as_str):
                        sorted_categories = [int(x) for x in sorted_categories]
                    
                    analysis["columns"][col].update({
                        "categories": sorted_categories[:100],  # Limiter à 100 catégories affichées
                        "is_binary": col in binary_cols,
                        "total_categories": len(sorted_categories)
                    })
                except Exception as e:
                    # En cas d'échec, stocker la liste non triée
                    analysis["columns"][col].update({
                        "categories": series.dropna().unique().tolist()[:100],  # Limiter à 100 catégories
                        "is_binary": col in binary_cols,
                        "sort_error": str(e),
                        "total_categories": series.nunique()
                    })
        
        # Compter les valeurs manquantes
        for col in data.columns:
            missing = data[col].isna().sum()
            if missing > 0:
                analysis["missing_values"][col] = int(missing)
        
        # Calculer les corrélations pour les colonnes numériques
        if numeric_cols:
            try:
                analysis["correlations"] = data[numeric_cols].corr().to_dict()
            except Exception as e:
                analysis["correlation_error"] = str(e)
        
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
    def preprocess_data(data: pd.DataFrame, 
                        categorical_cols_override: List[str] = None,
                        max_categories_per_feature: int = 100,
                        memory_efficient: bool = True) -> Tuple[np.ndarray, DatasetMetadata]:
        """
        Prétraiter les données pour l'entraînement du modèle avec une détection intelligente des types
        
        Args:
            data: DataFrame à prétraiter
            categorical_cols_override: Liste optionnelle des colonnes à traiter comme catégorielles
            max_categories_per_feature: Nombre maximum de catégories à conserver par variable (pour limiter la mémoire)
            memory_efficient: Si True, utilise des méthodes d'encodage économes en mémoire
        
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
        high_cardinality_cols = []
        cardinality_limits = {}
        
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
        
        # Analyser la cardinalité des variables catégorielles
        total_categories = 0
        for col in categorical_cols:
            unique_values = df[col].nunique()
            print(f"Variable {col}: {unique_values} valeurs uniques")
            total_categories += min(unique_values, max_categories_per_feature)
            
            # Stocker les valeurs uniques (limitées)
            unique_cats = sorted(df[col].dropna().unique().tolist())
            
            # Vérifier si la variable a trop de catégories
            if unique_values > max_categories_per_feature:
                warnings.warn(f"La variable '{col}' a {unique_values} catégories, au-delà de la limite de {max_categories_per_feature}. Les catégories les plus fréquentes seront conservées.")
                high_cardinality_cols.append(col)
                cardinality_limits[col] = max_categories_per_feature
                
                # Conserver uniquement les catégories les plus fréquentes
                top_categories = df[col].value_counts().nlargest(max_categories_per_feature).index.tolist()
                categorical_values[col] = top_categories
                
                # Remplacer les catégories moins fréquentes par "other"
                df.loc[~df[col].isin(top_categories), col] = "other"
            else:
                categorical_values[col] = unique_cats
        
        # Estimation de la mémoire requise (approx.)
        est_memory_mb = (len(df) * total_categories * 8) / (1024 * 1024)  # En Mo
        print(f"Mémoire estimée pour l'encodage: {est_memory_mb:.2f} Mo")
        
        # Avertissement si la mémoire estimée est supérieure à 2 Go
        if est_memory_mb > 2000:
            warnings.warn(f"L'encodage one-hot pourrait nécessiter environ {est_memory_mb:.2f} Mo. Considérez une méthode alternative.")
        
        # Prétraiter les variables catégorielles
        if categorical_cols:
            categorical_data = df[categorical_cols].copy()
            # Gérer les valeurs manquantes dans les données catégorielles
            for col in categorical_cols:
                categorical_data[col] = categorical_data[col].fillna(categorical_data[col].mode().iloc[0])
            
            if memory_efficient and (est_memory_mb > 1000 or len(high_cardinality_cols) > 0):
                # Approche alternative pour économiser de la mémoire
                print("Utilisation de l'encodage épargnant de la mémoire...")
                
                # Créer un encodeur par colonne pour éviter une matrice trop grande
                encoded_pieces = []
                categorical_encoder = {}
                
                for col in categorical_cols:
                    # Création d'un encodeur individuel par colonne
                    col_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
                    encoded_piece = col_encoder.fit_transform(categorical_data[[col]])
                    categorical_encoder[col] = col_encoder
                    encoded_pieces.append(encoded_piece)
                
                # Combiner les résultats
                if encoded_pieces:
                    from scipy import sparse
                    categorical_encoded = sparse.hstack(encoded_pieces).toarray()
                else:
                    categorical_encoded = np.array([]).reshape(len(df), 0)
            else:
                # Approche standard
                print("Utilisation de l'encodage one-hot standard...")
                categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                categorical_encoded = categorical_encoder.fit_transform(categorical_data)
        else:
            categorical_encoded = np.array([]).reshape(len(df), 0)
            categorical_encoder = None
        
        # Combiner les données numériques et catégorielles
        if numeric_cols and len(categorical_encoded) > 0:
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
            categorical_values=categorical_values,
            high_cardinality_cols=high_cardinality_cols,
            cardinality_limits=cardinality_limits
        )
        
        return preprocessed_data, metadata
    
    @staticmethod
    def inverse_transform(synthetic_data: Union[np.ndarray, 'torch.Tensor'], metadata: DatasetMetadata) -> pd.DataFrame:
        """
        Convertir les données synthétiques au format d'origine avec le bon typage
        
        Args:
            synthetic_data: Données synthétiques (numpy array ou tenseur PyTorch)
            metadata: Métadonnées du jeu de données
            
        Returns:
            DataFrame avec les données transformées
        """
        # Convertir le tenseur PyTorch en numpy array si nécessaire
        if not isinstance(synthetic_data, np.ndarray):
            try:
                # Tenter d'importer torch pour vérifier si c'est un tenseur PyTorch
                import torch
                if isinstance(synthetic_data, torch.Tensor):
                    try:
                        # Essayer de convertir en numpy
                        synthetic_data = synthetic_data.cpu().numpy()
                    except RuntimeError as e:
                        # Si NumPy n'est pas disponible, on utilise une approche alternative
                        if "Numpy is not available" in str(e):
                            print("Attention: NumPy n'est pas disponible pour la conversion PyTorch → NumPy.")
                            print("Tentative de conversion manuelle des données...")
                            
                            # Convertir manuellement (version simplifiée)
                            synthetic_data = synthetic_data.cpu().tolist()
                            synthetic_data = np.array(synthetic_data)
                        else:
                            raise e
            except ImportError:
                # Si torch n'est pas disponible mais qu'on a passé un objet qui n'est pas numpy
                raise ValueError("Les données synthétiques doivent être un tableau NumPy ou un tenseur PyTorch")
        
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
            
            # Gérer différents types d'encodeurs
            if isinstance(metadata.categorical_encoder, dict):
                # Cas de l'encodage économe en mémoire (un encodeur par colonne)
                cat_columns = []
                cat_values = []
                
                current_pos = 0
                for col in metadata.categorical_cols:
                    # Récupérer l'encodeur pour cette colonne
                    encoder = metadata.categorical_encoder[col]
                    
                    # Déterminer le nombre de catégories pour cette colonne
                    n_categories = len(encoder.categories_[0])
                    
                    # Extraire la partie correspondante
                    col_part = categorical_part[:, current_pos:current_pos + n_categories]
                    
                    # Appliquer softmax pour obtenir des probabilités
                    e_x = np.exp(col_part - np.max(col_part, axis=1, keepdims=True))
                    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
                    
                    # Obtenir la catégorie avec la plus haute probabilité
                    cat_idx = np.argmax(probs, axis=1)
                    
                    # Convertir les indices en catégories d'origine
                    original_categories = encoder.categories_[0]
                    cat_column = np.array([original_categories[idx] for idx in cat_idx])
                    
                    cat_values.append(cat_column)
                    cat_columns.append(col)
                    
                    # Mettre à jour la position courante
                    current_pos += n_categories
                
                # Créer un DataFrame pour les données catégorielles
                if cat_values:
                    df_categorical = pd.DataFrame(np.column_stack(cat_values), columns=cat_columns)
                else:
                    df_categorical = pd.DataFrame(index=range(len(synthetic_data)))
            else:
                # Cas standard (un seul encodeur)
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
                    else:
                        df_categorical = pd.DataFrame(index=range(len(synthetic_data)))
                else:
                    # Fallback si l'encodeur catégoriel n'a pas la structure attendue
                    try:
                        categorical_data = metadata.categorical_encoder.inverse_transform(categorical_part)
                        df_categorical = pd.DataFrame(categorical_data, columns=metadata.categorical_cols)
                    except Exception as e:
                        print(f"Erreur lors de la transformation inverse: {str(e)}")
                        # Créer un DataFrame vide avec les bonnes dimensions
                        df_categorical = pd.DataFrame(index=range(len(synthetic_data)), 
                                                     columns=metadata.categorical_cols)
            
            # Convertir les colonnes binaires (0/1) au format approprié
            for col in df_categorical.columns:
                # Si toutes les valeurs sont '0' ou '1', convertir en entier
                if set(df_categorical[col].astype(str).unique()).issubset({'0', '1', '0.0', '1.0'}):
                    df_categorical[col] = df_categorical[col].astype(int)
            
            # Combiner avec les données numériques s'il y en a
            if not result_df.empty:
                result_df = pd.concat([result_df, df_categorical], axis=1)
            else:
                result_df = df_categorical
        
        return result_df