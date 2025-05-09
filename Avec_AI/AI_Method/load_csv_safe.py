import pandas as pd
import io
import streamlit as st

def load_csv_safely(uploaded_file, **kwargs):
    """
    Charge un fichier CSV de manière sécurisée en gérant les erreurs courantes.
    
    Args:
        uploaded_file: Fichier téléchargé via st.file_uploader
        **kwargs: Arguments supplémentaires à passer à pd.read_csv
        
    Returns:
        DataFrame pandas ou None en cas d'erreur
    """
    try:
        # Première tentative: lecture standard
        return pd.read_csv(uploaded_file, **kwargs)
    except Exception as e1:
        try:
            # Réinitialiser la position du fichier
            uploaded_file.seek(0)
            
            # Deuxième tentative: lecture avec tous les types en str
            return pd.read_csv(uploaded_file, dtype=str, **kwargs)
        except Exception as e2:
            try:
                # Réinitialiser la position du fichier
                uploaded_file.seek(0)
                
                # Lire les premières lignes pour détecter le séparateur
                sample = uploaded_file.read(1024).decode('utf-8')
                uploaded_file.seek(0)
                
                # Détecter le séparateur
                if sample.count(';') > sample.count(','):
                    sep = ';'
                else:
                    sep = ','
                
                # Troisième tentative: lecture avec séparateur détecté et tous les types en str
                df = pd.read_csv(uploaded_file, dtype=str, sep=sep, **kwargs)
                
                # Convertir les colonnes numériques si possible
                for col in df.columns:
                    try:
                        # Essayer de convertir en numérique en ignorant les erreurs
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        # Si peu de NaN après conversion, c'est probablement une colonne numérique
                        if numeric_values.isna().mean() < 0.2:  # moins de 20% de NaN
                            df[col] = numeric_values
                    except:
                        pass
                
                return df
            except Exception as e3:
                # Afficher les erreurs pour le débogage
                st.error(f"Erreur lors du chargement du fichier CSV:")
                st.error(f"1ère tentative: {str(e1)}")
                st.error(f"2ème tentative: {str(e2)}")
                st.error(f"3ème tentative: {str(e3)}")
                
                # Suggestions pour résoudre le problème
                st.info("""
                Suggestions pour résoudre ce problème:
                1. Vérifiez que votre fichier est bien au format CSV
                2. Vérifiez qu'il n'y a pas de lignes vides ou mal formatées
                3. Essayez d'ouvrir le fichier dans un éditeur de texte pour vérifier son contenu
                4. Essayez d'exporter à nouveau votre fichier CSV depuis sa source avec un encodage UTF-8
                """)
                
                return None