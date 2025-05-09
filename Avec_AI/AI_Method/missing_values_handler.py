import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union

class MissingValuesHandler:
    """
    Module pour la gestion et le traitement des valeurs manquantes dans les jeux de données.
    Propose différentes techniques d'imputation avec leurs avantages et inconvénients.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialise le module avec un DataFrame.
        
        Args:
            data: DataFrame pandas contenant potentiellement des valeurs manquantes
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.missing_info = self._analyze_missing_values()
        
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyse les valeurs manquantes dans le jeu de données.
        
        Returns:
            Dictionnaire contenant des informations sur les valeurs manquantes
        """
        # Nombre total de valeurs manquantes
        total_missing = self.data.isna().sum().sum()
        
        # Calcul du pourcentage de valeurs manquantes
        total_cells = np.prod(self.data.shape)
        missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        # Information par colonne
        column_info = {}
        for column in self.data.columns:
            missing_count = self.data[column].isna().sum()
            missing_percent = (missing_count / len(self.data)) * 100
            column_info[column] = {
                'missing_count': missing_count,
                'missing_percent': missing_percent,
                'data_type': str(self.data[column].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(self.data[column])
            }
        
        # Identifier les colonnes avec valeurs manquantes
        columns_with_missing = [col for col, info in column_info.items() if info['missing_count'] > 0]
        
        return {
            'total_missing': total_missing,
            'missing_percentage': missing_percentage,
            'column_info': column_info,
            'columns_with_missing': columns_with_missing
        }
    
    def get_missing_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des valeurs manquantes.
        
        Returns:
            Dictionnaire contenant un résumé des valeurs manquantes
        """
        return self.missing_info
    
    def visualize_missing_values(self) -> None:
        """
        Visualise les valeurs manquantes dans le jeu de données.
        """
        if not self.missing_info['columns_with_missing']:
            st.info("Aucune valeur manquante détectée dans le jeu de données.")
            return
        
        # Afficher le pourcentage global
        st.write(f"**Pourcentage global de valeurs manquantes:** {self.missing_info['missing_percentage']:.2f}%")
        
        # Créer un graphique à barres pour les colonnes avec des valeurs manquantes
        missing_data = {col: self.missing_info['column_info'][col]['missing_percent'] 
                      for col in self.missing_info['columns_with_missing']}
        
        if missing_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_missing = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing Percentage'])
            df_missing = df_missing.sort_values('Missing Percentage', ascending=False)
            
            sns.barplot(x='Missing Percentage', y='Column', data=df_missing, hue='Column', palette='viridis', ax=ax, legend=False)
            ax.set_title('Pourcentage de valeurs manquantes par colonne')
            ax.set_xlim(0, 100)
            ax.grid(axis='x')
            
            st.pyplot(fig)
        
        # Afficher un heatmap des valeurs manquantes
        if len(self.data.columns) <= 50:  # Limiter pour éviter les graphiques trop grands
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(self.data.isna(), cmap='viridis', yticklabels=False, cbar=False, ax=ax)
            ax.set_title('Heatmap des valeurs manquantes')
            st.pyplot(fig)
    
    def get_imputation_methods(self) -> Dict[str, Dict[str, str]]:
        """
        Retourne un dictionnaire des méthodes d'imputation disponibles avec leurs descriptions,
        avantages et inconvénients.
        
        Returns:
            Dictionnaire des méthodes d'imputation
        """
        methods = {
            'remove_rows': {
                'name': 'Suppression des lignes',
                'description': 'Supprime les lignes contenant des valeurs manquantes.',
                'advantages': '• Simple et rapide\n• Préserve la distribution des variables\n• Pas de biais introduit par imputation',
                'disadvantages': '• Perte de données potentiellement importante\n• Biais si les données manquantes ne sont pas MCAR (Missing Completely At Random)\n• Réduction de la taille de l\'échantillon',
                'suitable_for': 'Jeux de données avec peu de valeurs manquantes (<5%) et distribution aléatoire des valeurs manquantes'
            },
            'remove_columns': {
                'name': 'Suppression des colonnes',
                'description': 'Supprime les colonnes avec un pourcentage élevé de valeurs manquantes.',
                'advantages': '• Simple et rapide\n• Élimine complètement le problème des valeurs manquantes\n• Utile pour les variables avec beaucoup de valeurs manquantes',
                'disadvantages': '• Perte d\'informations potentiellement importantes\n• Non applicable si toutes les variables sont importantes\n• Peut réduire la capacité prédictive des modèles',
                'suitable_for': 'Variables avec un pourcentage très élevé de valeurs manquantes (>50%) ou variables peu importantes'
            },
            'mean_imputation': {
                'name': 'Imputation par la moyenne',
                'description': 'Remplace les valeurs manquantes par la moyenne de la colonne (variables numériques uniquement).',
                'advantages': '• Simple et rapide\n• Préserve la moyenne de la variable\n• Facile à comprendre et à implémenter',
                'disadvantages': '• Réduit la variance\n• Ne prend pas en compte les relations entre variables\n• Peut créer des pics artificiels dans la distribution',
                'suitable_for': 'Variables numériques avec distribution symétrique et peu de valeurs aberrantes'
            },
            'median_imputation': {
                'name': 'Imputation par la médiane',
                'description': 'Remplace les valeurs manquantes par la médiane de la colonne (variables numériques uniquement).',
                'advantages': '• Robuste aux valeurs aberrantes\n• Préserve la médiane de la variable\n• Adapté aux distributions asymétriques',
                'disadvantages': '• Peut créer des pics artificiels dans la distribution\n• Ne prend pas en compte les relations entre variables\n• Réduit la variance',
                'suitable_for': 'Variables numériques avec valeurs aberrantes ou distribution asymétrique'
            },
            'mode_imputation': {
                'name': 'Imputation par le mode',
                'description': 'Remplace les valeurs manquantes par la valeur la plus fréquente (pour variables catégorielles ou numériques).',
                'advantages': '• Applicable aux variables catégorielles et numériques\n• Préserve la distribution des catégories\n• Simple à comprendre',
                'disadvantages': '• Peut créer un biais vers la catégorie majoritaire\n• Ne prend pas en compte les relations entre variables\n• Problématique si plusieurs modes existent',
                'suitable_for': 'Variables catégorielles ou variables numériques avec peu de valeurs discrètes'
            },
            'constant_imputation': {
                'name': 'Imputation par une constante',
                'description': 'Remplace les valeurs manquantes par une valeur constante spécifiée (ex: 0, "Inconnu").',
                'advantages': '• Simple et rapide\n• Facile à identifier les anciennes valeurs manquantes\n• Utile pour certains algorithmes spécifiques',
                'disadvantages': '• Introduit un biais potentiellement important\n• Peut créer des pics artificiels dans la distribution\n• Ne préserve pas les statistiques de la variable',
                'suitable_for': 'Cas spécifiques où les valeurs manquantes ont une signification particulière'
            },
            'knn_imputation': {
                'name': 'Imputation par k plus proches voisins (KNN)',
                'description': 'Prédit les valeurs manquantes à partir des k observations les plus similaires.',
                'advantages': '• Prend en compte la similarité entre observations\n• Préserve les relations entre variables\n• Robuste pour différents types de données',
                'disadvantages': '• Computationnellement intensif pour grands jeux de données\n• Sensible au choix de k et à la distance utilisée\n• Moins efficace avec beaucoup de dimensions',
                'suitable_for': 'Jeux de données de taille moyenne avec corrélations entre variables'
            },
            'iterative_imputation': {
                'name': 'Imputation itérative (MICE)',
                'description': 'Modélise chaque variable avec valeurs manquantes en fonction des autres variables, et itère plusieurs fois.',
                'advantages': '• Préserve les relations complexes entre variables\n• Très précis pour les données MNAR (Missing Not At Random)\n• Utilise toute l\'information disponible',
                'disadvantages': '• Computationnellement intensif\n• Complexe à paramétrer\n• Peut converger lentement',
                'suitable_for': 'Jeux de données avec fortes corrélations entre variables et valeurs manquantes non aléatoires'
            }
        }
        
        return methods
    
    def display_method_selection(self) -> str:
        """
        Affiche les méthodes d'imputation disponibles et permet à l'utilisateur
        de choisir une méthode.
        
        Returns:
            Clé de la méthode sélectionnée
        """
        methods = self.get_imputation_methods()
        
        st.subheader("Sélection de la méthode pour traiter les valeurs manquantes")
        
        # Afficher les méthodes sous forme d'expanseurs
        selected_method = None
        
        method_tabs = st.tabs([methods[method_key]['name'] for method_key in methods])
        
        for i, method_key in enumerate(methods):
            method = methods[method_key]
            with method_tabs[i]:
                st.markdown(f"### {method['name']}")
                st.markdown(f"**Description:** {method['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Avantages:**")
                    st.markdown(method['advantages'])
                with col2:
                    st.markdown("**Inconvénients:**")
                    st.markdown(method['disadvantages'])
                
                st.markdown(f"**Recommandé pour:** {method['suitable_for']}")
                
                if st.button("Sélectionner cette méthode", key=f"select_{method_key}"):
                    selected_method = method_key
        
        return selected_method
    
    def apply_imputation(self, method_key: str, **kwargs) -> pd.DataFrame:
        """
        Applique la méthode d'imputation sélectionnée aux données.
        
        Args:
            method_key: Clé de la méthode à appliquer
            **kwargs: Paramètres supplémentaires pour la méthode
            
        Returns:
            DataFrame avec les valeurs manquantes traitées
        """
        # Vérifier s'il y a des valeurs manquantes
        if self.missing_info['total_missing'] == 0:
            st.success("Aucune valeur manquante à traiter!")
            return self.data
        
        # Créer une copie des données pour éviter de modifier l'original
        imputed_data = self.data.copy()
        
        if method_key == 'remove_rows':
            # Obtenir le seuil de suppression (% de valeurs manquantes par ligne)
            threshold = kwargs.get('threshold', 0)
            
            # Calculer le pourcentage de valeurs manquantes par ligne
            missing_percentage = imputed_data.isna().mean(axis=1) * 100
            
            # Filtrer les lignes avec moins de valeurs manquantes que le seuil
            rows_to_keep = missing_percentage <= threshold
            imputed_data = imputed_data[rows_to_keep]
            
            # Afficher des statistiques
            removed_rows = sum(~rows_to_keep)
            st.info(f"{removed_rows} lignes supprimées sur {len(self.data)} ({removed_rows/len(self.data)*100:.2f}%)")
            
        elif method_key == 'remove_columns':
            # Obtenir le seuil de suppression (% de valeurs manquantes par colonne)
            threshold = kwargs.get('threshold', 50)
            
            # Calculer le pourcentage de valeurs manquantes par colonne
            missing_percentage = imputed_data.isna().mean() * 100
            
            # Sélectionner les colonnes à conserver
            columns_to_keep = missing_percentage[missing_percentage <= threshold].index.tolist()
            imputed_data = imputed_data[columns_to_keep]
            
            # Afficher des statistiques
            removed_cols = len(self.data.columns) - len(columns_to_keep)
            st.info(f"{removed_cols} colonnes supprimées sur {len(self.data.columns)} ({removed_cols/len(self.data.columns)*100:.2f}%)")
            
        elif method_key == 'mean_imputation':
            # Séparer les variables numériques et catégorielles
            numeric_cols = imputed_data.select_dtypes(include=['number']).columns
            categorical_cols = imputed_data.select_dtypes(exclude=['number']).columns
            
            # Appliquer l'imputation par la moyenne aux variables numériques
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='mean')
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])
            
            # Pour les variables catégorielles, utiliser le mode
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[categorical_cols] = cat_imputer.fit_transform(imputed_data[categorical_cols])
                st.info(f"Note: Les variables catégorielles ont été imputées avec le mode (valeur la plus fréquente).")
            
        elif method_key == 'median_imputation':
            # Séparer les variables numériques et catégorielles
            numeric_cols = imputed_data.select_dtypes(include=['number']).columns
            categorical_cols = imputed_data.select_dtypes(exclude=['number']).columns
            
            # Appliquer l'imputation par la médiane aux variables numériques
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                imputed_data[numeric_cols] = imputer.fit_transform(imputed_data[numeric_cols])
            
            # Pour les variables catégorielles, utiliser le mode
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[categorical_cols] = cat_imputer.fit_transform(imputed_data[categorical_cols])
                st.info(f"Note: Les variables catégorielles ont été imputées avec le mode (valeur la plus fréquente).")
            
        elif method_key == 'mode_imputation':
            # Appliquer l'imputation par le mode à toutes les colonnes
            imputer = SimpleImputer(strategy='most_frequent')
            imputed_data = pd.DataFrame(
                imputer.fit_transform(imputed_data),
                columns=imputed_data.columns
            )
            
        elif method_key == 'constant_imputation':
            # Obtenir la valeur constante à utiliser
            fill_value = kwargs.get('fill_value', 0)
            
            # Appliquer l'imputation par une constante
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            imputed_data = pd.DataFrame(
                imputer.fit_transform(imputed_data),
                columns=imputed_data.columns
            )
            
        elif method_key == 'knn_imputation':
            # Obtenir le nombre de voisins
            n_neighbors = kwargs.get('n_neighbors', 5)
            
            # Séparer les variables numériques et catégorielles
            numeric_cols = imputed_data.select_dtypes(include=['number']).columns
            categorical_cols = imputed_data.select_dtypes(exclude=['number']).columns
            
            # Pour KNN, nous devons d'abord encoder les variables catégorielles
            if len(categorical_cols) > 0:
                # Imputer d'abord les valeurs manquantes dans les variables catégorielles avec le mode
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[categorical_cols] = cat_imputer.fit_transform(imputed_data[categorical_cols])
                
                # Encoder les variables catégorielles (one-hot encoding)
                for col in categorical_cols:
                    dummies = pd.get_dummies(imputed_data[col], prefix=col, dummy_na=False)
                    imputed_data = pd.concat([imputed_data.drop(col, axis=1), dummies], axis=1)
            
            # Maintenant, appliquer KNN sur l'ensemble du DataFrame
            if len(imputed_data.select_dtypes(include=['number']).columns) > 0:
                numeric_data = imputed_data.select_dtypes(include=['number'])
                
                # Normaliser les données pour KNN
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Appliquer KNN
                knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed_numeric = knn_imputer.fit_transform(scaled_data)
                
                # Dénormaliser et remettre dans le DataFrame
                imputed_numeric = scaler.inverse_transform(imputed_numeric)
                imputed_data[numeric_data.columns] = imputed_numeric
            
        elif method_key == 'iterative_imputation':
            # Obtenir le nombre maximum d'itérations
            max_iter = kwargs.get('max_iter', 10)
            
            # Séparer les variables numériques et catégorielles
            numeric_cols = imputed_data.select_dtypes(include=['number']).columns
            categorical_cols = imputed_data.select_dtypes(exclude=['number']).columns
            
            # Pour l'imputation itérative, nous devons d'abord encoder les variables catégorielles
            if len(categorical_cols) > 0:
                # Imputer d'abord les valeurs manquantes dans les variables catégorielles avec le mode
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[categorical_cols] = cat_imputer.fit_transform(imputed_data[categorical_cols])
                
                # Encoder les variables catégorielles (one-hot encoding)
                for col in categorical_cols:
                    dummies = pd.get_dummies(imputed_data[col], prefix=col, dummy_na=False)
                    imputed_data = pd.concat([imputed_data.drop(col, axis=1), dummies], axis=1)
            
            # Maintenant, appliquer l'imputation itérative sur l'ensemble du DataFrame
            if len(imputed_data.select_dtypes(include=['number']).columns) > 0:
                numeric_data = imputed_data.select_dtypes(include=['number'])
                
                # Normaliser les données pour l'imputation itérative
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Appliquer l'imputation itérative
                mice_imputer = IterativeImputer(max_iter=max_iter, random_state=42)
                imputed_numeric = mice_imputer.fit_transform(scaled_data)
                
                # Dénormaliser et remettre dans le DataFrame
                imputed_numeric = scaler.inverse_transform(imputed_numeric)
                imputed_data[numeric_data.columns] = imputed_numeric
        
        # Vérifier si toutes les valeurs manquantes ont été traitées
        remaining_missing = imputed_data.isna().sum().sum()
        if remaining_missing > 0:
            st.warning(f"Il reste {remaining_missing} valeurs manquantes dans le jeu de données après imputation.")
        else:
            st.success("Toutes les valeurs manquantes ont été traitées avec succès!")
        
        return imputed_data
    
    def compare_before_after(self, imputed_data: pd.DataFrame) -> None:
        """
        Compare les statistiques avant et après imputation.
        
        Args:
            imputed_data: DataFrame après imputation
        """
        if len(self.missing_info['columns_with_missing']) == 0:
            st.info("Aucune valeur manquante à comparer.")
            return
        
        st.subheader("Comparaison avant et après imputation")
        
        # Pour chaque colonne numérique avec des valeurs manquantes
        numeric_cols_with_missing = [
            col for col in self.missing_info['columns_with_missing'] 
            if self.missing_info['column_info'][col]['is_numeric'] and col in imputed_data.columns
        ]
        
        if numeric_cols_with_missing:
            st.write("### Comparaison des statistiques descriptives (colonnes numériques)")
            
            for col in numeric_cols_with_missing:
                st.write(f"#### {col}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Avant imputation:**")
                    stats_before = self.original_data[col].describe()
                    st.write(stats_before)
                
                with col2:
                    st.write("**Après imputation:**")
                    stats_after = imputed_data[col].describe()
                    st.write(stats_after)
                
                # Créer un graphique comparatif
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # Distribution avant imputation
                sns.histplot(self.original_data[col].dropna(), kde=True, ax=ax[0])
                ax[0].set_title(f"Distribution avant imputation\n(n={self.original_data[col].count()})")
                
                # Distribution après imputation
                sns.histplot(imputed_data[col], kde=True, ax=ax[1])
                ax[1].set_title(f"Distribution après imputation\n(n={imputed_data[col].count()})")
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Pour chaque colonne catégorielle avec des valeurs manquantes
        categorical_cols_with_missing = [
            col for col in self.missing_info['columns_with_missing'] 
            if not self.missing_info['column_info'][col]['is_numeric'] and col in imputed_data.columns
        ]
        
        if categorical_cols_with_missing:
            st.write("### Comparaison des distributions (colonnes catégorielles)")
            
            for col in categorical_cols_with_missing:
                st.write(f"#### {col}")
                
                # Créer un graphique comparatif
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))
                
                # Distribution avant imputation
                before_counts = self.original_data[col].value_counts(normalize=True, dropna=False)
                # Trier par valeur (ou par index pour les catégories)
                if pd.api.types.is_numeric_dtype(before_counts.index):
                    before_counts = before_counts.sort_index()
                
                sns.barplot(x=before_counts.index.astype(str), y=before_counts.values, ax=ax[0])
                ax[0].set_title(f"Distribution avant imputation\n(n={self.original_data[col].count()})")
                ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
                
                # Distribution après imputation
                after_counts = imputed_data[col].value_counts(normalize=True)
                # Trier par valeur (ou par index pour les catégories)
                if pd.api.types.is_numeric_dtype(after_counts.index):
                    after_counts = after_counts.sort_index()
                
                sns.barplot(x=after_counts.index.astype(str), y=after_counts.values, ax=ax[1])
                ax[1].set_title(f"Distribution après imputation\n(n={imputed_data[col].count()})")
                ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)

def missing_values_module(data: pd.DataFrame) -> pd.DataFrame:
    """
    Module principal pour le traitement des valeurs manquantes.
    
    Args:
        data: DataFrame à traiter
        
    Returns:
        DataFrame avec les valeurs manquantes traitées
    """
    st.title("Module de traitement des valeurs manquantes")
    
    # Initialiser le gestionnaire
    handler = MissingValuesHandler(data)
    
    # Afficher l'analyse des valeurs manquantes
    st.header("Analyse des valeurs manquantes")
    
    missing_info = handler.get_missing_summary()
    
    # Afficher des statistiques de base
    st.write(f"**Nombre total de valeurs manquantes:** {missing_info['total_missing']}")
    st.write(f"**Pourcentage global de valeurs manquantes:** {missing_info['missing_percentage']:.2f}%")
    
    # Si des valeurs manquantes sont détectées
    if missing_info['total_missing'] > 0:
        # Afficher les colonnes avec des valeurs manquantes
        st.subheader("Colonnes avec valeurs manquantes")
        
        missing_cols_df = pd.DataFrame([
            {
                'Colonne': col,
                'Type de données': missing_info['column_info'][col]['data_type'],
                'Nombre de valeurs manquantes': missing_info['column_info'][col]['missing_count'],
                'Pourcentage de valeurs manquantes': f"{missing_info['column_info'][col]['missing_percent']:.2f}%"
            }
            for col in missing_info['columns_with_missing']
        ])
        
        st.dataframe(missing_cols_df)
        
        # Visualiser les valeurs manquantes
        st.subheader("Visualisation des valeurs manquantes")
        handler.visualize_missing_values()
        
        # Sélection de la méthode d'imputation
        st.header("Méthodes de traitement des valeurs manquantes")
        
        # Permettre à l'utilisateur de choisir une méthode
        selected_method = handler.display_method_selection()
        
        # Si une méthode a été sélectionnée
        if selected_method:
            st.subheader(f"Application de la méthode: {handler.get_imputation_methods()[selected_method]['name']}")
            
            # Paramètres selon la méthode choisie
            params = {}
            
            if selected_method == 'remove_rows':
                params['threshold'] = st.slider(
                    "Seuil de suppression (% maximum de valeurs manquantes par ligne)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Les lignes avec un pourcentage de valeurs manquantes supérieur à ce seuil seront supprimées"
                )
            
            elif selected_method == 'remove_columns':
                params['threshold'] = st.slider(
                    "Seuil de suppression (% maximum de valeurs manquantes par colonne)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Les colonnes avec un pourcentage de valeurs manquantes supérieur à ce seuil seront supprimées"
                )
            
            elif selected_method == 'constant_imputation':
                params['fill_value'] = st.text_input(
                    "Valeur constante pour l'imputation",
                    value="0",
                    help="Toutes les valeurs manquantes seront remplacées par cette valeur constante"
                )
                
                # Convertir en nombre si possible
                try:
                    params['fill_value'] = float(params['fill_value'])
                    if params['fill_value'] == int(params['fill_value']):
                        params['fill_value'] = int(params['fill_value'])
                except ValueError:
                    pass  # Garder la valeur comme chaîne
            
            elif selected_method == 'knn_imputation':
                params['n_neighbors'] = st.slider(
                    "Nombre de voisins (k)",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Nombre de voisins à considérer pour l'imputation KNN"
                )
            
            elif selected_method == 'iterative_imputation':
                params['max_iter'] = st.slider(
                    "Nombre maximum d'itérations",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    help="Nombre maximum d'itérations pour l'imputation itérative"
                )
            
            # Appliquer l'imputation lorsque l'utilisateur confirme
            if st.button("Appliquer la méthode d'imputation"):
                with st.spinner(f"Application de la méthode: {handler.get_imputation_methods()[selected_method]['name']}..."):
                    # Appliquer la méthode d'imputation
                    imputed_data = handler.apply_imputation(selected_method, **params)
                    
                    # Comparer les résultats avant et après
                    handler.compare_before_after(imputed_data)
                    
                    # Option de téléchargement
                    st.subheader("Télécharger les données traitées")
                    
                    csv = imputed_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger les données (CSV)",
                        data=csv,
                        file_name="donnees_sans_valeurs_manquantes.csv",
                        mime="text/csv"
                    )
                    
                    return imputed_data
    else:
        st.success("Aucune valeur manquante détectée dans le jeu de données!")
    
    # Si aucune imputation n'a été appliquée, retourner les données originales
    return data