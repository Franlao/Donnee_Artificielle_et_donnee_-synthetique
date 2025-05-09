import pandas as pd
import json
import re
from typing import Tuple, List, Dict, Union, Optional, Any
import time

# Pour l'API Mistral
try:
    from mistralai import Mistral,UserMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("La bibliothèque Mistral AI n'est pas installée. Pour l'installer: pip install mistralai")

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