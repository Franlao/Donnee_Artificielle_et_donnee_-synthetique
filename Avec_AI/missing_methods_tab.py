import streamlit as st
import pandas as pd
from Avec_AI.AI_Method.missing_values_handler import missing_values_module

def show_missing_values_tab(data: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche l'onglet de gestion des valeurs manquantes.
    
    Args:
        data: DataFrame à traiter
        
    Returns:
        DataFrame traité
    """
    st.header("Gestion des valeurs manquantes")
    
    if data is None:
        st.warning("Veuillez d'abord charger un jeu de données.")
        return None
    
    # Afficher le module de gestion des valeurs manquantes
    processed_data = missing_values_module(data)
    
    return processed_data 