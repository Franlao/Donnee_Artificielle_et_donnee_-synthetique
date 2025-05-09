"""
Module d'intégration des modèles GAN et VAE pour la génération de données synthétiques.
Ce module importe les classes des fichiers tabular_gan.py et tabular_vae.py.
"""

import os
import numpy as np
import pandas as pd
import torch

# Importer les classes de GAN (directement depuis le fichier)
try:
    from Avec_AI.tabular_gan import TabularGAN, TabularGenerator, TabularDiscriminator, FeatureActivation
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    print("Module tabular_gan.py non trouvé. La fonctionnalité GAN ne sera pas disponible.")

# Importer les classes de VAE (directement depuis le fichier)
try:
    from Avec_AI.tabular_vae import TabularVAE, TabularEncoder, TabularDecoder
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("Module tabular_vae.py non trouvé. La fonctionnalité VAE ne sera pas disponible.")

# Vérifier si CUDA est disponible pour PyTorch
CUDA_AVAILABLE = torch.cuda.is_available() if 'torch' in globals() else False

def train_gan_model(preprocessed_data, metadata, params):
    """
    Entraîne un modèle GAN sur les données prétraitées.
    
    Args:
        preprocessed_data: Données prétraitées au format numpy array
        metadata: Métadonnées du prétraitement
        params: Paramètres d'entraînement (dict)
        
    Returns:
        Modèle GAN entraîné et les données synthétiques générées
    """
    if not GAN_AVAILABLE:
        raise ImportError("Le module tabular_gan.py n'est pas disponible.")
    
    # Déballez les paramètres
    latent_dim = params.get('latent_dim', 16)
    epochs = params.get('epochs', 300)
    batch_size = params.get('batch_size', 64)
    n_samples = params.get('n_samples', len(preprocessed_data))
    learning_rate_g = params.get('learning_rate', 0.0002)
    learning_rate_d = params.get('learning_rate', 0.0002)
    early_stopping = params.get('early_stopping', 30)
    
    # Définir le device (CPU/GPU)
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    
    # Initialiser le modèle GAN
    gan = TabularGAN(
        noise_dim=latent_dim,
        generator_hidden_dims=[256, 512, 256],
        discriminator_hidden_dims=[256, 128, 64],
        lr_generator=learning_rate_g,
        lr_discriminator=learning_rate_d,
        device=device
    )
    
    # Entraîner le modèle
    gan.train(
        preprocessed_data=preprocessed_data,
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping,
        verbose=True,
        progress_interval=10
    )
    
    # Générer des données synthétiques
    synthetic_data = gan.generate(n_samples)
    
    return gan, synthetic_data

def train_vae_model(preprocessed_data, metadata, params):
    """
    Entraîne un modèle VAE sur les données prétraitées.
    
    Args:
        preprocessed_data: Données prétraitées au format numpy array
        metadata: Métadonnées du prétraitement
        params: Paramètres d'entraînement (dict)
        
    Returns:
        Modèle VAE entraîné et les données synthétiques générées
    """
    if not VAE_AVAILABLE:
        raise ImportError("Le module tabular_vae.py n'est pas disponible.")
    
    # Déballez les paramètres
    latent_dim = params.get('latent_dim', 16)
    epochs = params.get('epochs', 300)
    batch_size = params.get('batch_size', 64)
    n_samples = params.get('n_samples', len(preprocessed_data))
    learning_rate = params.get('learning_rate', 0.001)
    beta = params.get('beta', 1.0)
    early_stopping = params.get('early_stopping', 30)
    
    # Définir le device (CPU/GPU)
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    
    # Initialiser le modèle VAE
    vae = TabularVAE(
        latent_dim=latent_dim,
        encoder_hidden_dims=[256, 128],
        decoder_hidden_dims=[128, 256],
        learning_rate=learning_rate,
        beta=beta,
        device=device
    )
    
    # Entraîner le modèle
    vae.train(
        preprocessed_data=preprocessed_data,
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping,
        verbose=True,
        progress_interval=10
    )
    
    # Générer des données synthétiques
    synthetic_data = vae.generate(n_samples)
    
    return vae, synthetic_data

def save_model(model, path, model_type='gan'):
    """
    Sauvegarde un modèle entraîné.
    
    Args:
        model: Le modèle à sauvegarder (GAN ou VAE)
        path: Chemin où sauvegarder le modèle
        model_type: Type du modèle ('gan' ou 'vae')
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarder le modèle
    model.save(path)
    
    print(f"Modèle {model_type.upper()} sauvegardé avec succès à {path}")

def load_model(path, input_dim, metadata, model_type='gan'):
    """
    Charge un modèle préalablement sauvegardé.
    
    Args:
        path: Chemin du modèle à charger
        input_dim: Dimension d'entrée des données
        metadata: Métadonnées du prétraitement
        model_type: Type du modèle ('gan' ou 'vae')
        
    Returns:
        Le modèle chargé
    """
    if model_type.lower() == 'gan':
        if not GAN_AVAILABLE:
            raise ImportError("Le module tabular_gan.py n'est pas disponible.")
        
        model = TabularGAN()
        model.load(path, input_dim, metadata)
        
    elif model_type.lower() == 'vae':
        if not VAE_AVAILABLE:
            raise ImportError("Le module tabular_vae.py n'est pas disponible.")
        
        model = TabularVAE()
        model.load(path, input_dim, metadata)
        
    else:
        raise ValueError(f"Type de modèle non reconnu: {model_type}")
    
    return model