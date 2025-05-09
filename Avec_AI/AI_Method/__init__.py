# Importations pour faciliter l'accès aux classes et fonctions principales
from .data_processor import DataProcessor, DatasetMetadata
from .llm_generator import MistralGenerator, MISTRAL_AVAILABLE
from .ui_components import render_llm_tab, visualize_categorical_comparison, visualize_numeric_comparison
from .ai_methods_tab import render_ai_methods_tab, TORCH_AVAILABLE
from .missing_values_handler import missing_values_module

# Importations des modèles GAN et VAE
try:
    from .models import (
        train_gan_model,
        train_vae_model,
        save_model,
        load_model,
        GAN_AVAILABLE,
        VAE_AVAILABLE,
        CUDA_AVAILABLE
    )
except ImportError:
    # Définir des valeurs par défaut si le module n'est pas disponible
    GAN_AVAILABLE = False
    VAE_AVAILABLE = False
    CUDA_AVAILABLE = False

# Définir ce qui est exposé lors de l'importation du module
__all__ = [
    'DataProcessor', 
    'DatasetMetadata',
    'MistralGenerator',
    'MISTRAL_AVAILABLE',
    'TORCH_AVAILABLE',
    'render_llm_tab',
    'render_ai_methods_tab',
    'visualize_categorical_comparison',
    'visualize_numeric_comparison',
    # Modèles GAN/VAE
    'train_gan_model',
    'train_vae_model',
    'save_model',
    'load_model',
    'GAN_AVAILABLE',
    'VAE_AVAILABLE',
    'CUDA_AVAILABLE'
]