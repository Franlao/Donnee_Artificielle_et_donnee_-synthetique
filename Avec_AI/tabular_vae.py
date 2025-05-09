import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
import time

class TabularEncoder(nn.Module):
    """
    Encoder network for Variational Autoencoder on tabular data.
    
    This model encodes input tabular data into parameters of a multivariate Gaussian
    distribution (mean and log variance) in the latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [256, 128]):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Dimension of the input tabular data
            latent_dim: Dimension of the latent space
            hidden_dims: List of hidden layer dimensions
        """
        super(TabularEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tabular data of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) for the latent space distribution
        """
        x = self.shared_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class TabularDecoder(nn.Module):
    """
    Decoder network for Variational Autoencoder on tabular data.
    
    This model decodes latent space vectors back to tabular data.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [128, 256]):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim: Dimension of the latent space
            output_dim: Dimension of the output tabular data
            hidden_dims: List of hidden layer dimensions
        """
        super(TabularDecoder, self).__init__()
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # No activation at the end to allow the model to generate any range of values
        # Activation will be applied separately for numerical and categorical features
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z: Latent space vector of shape (batch_size, latent_dim)
            
        Returns:
            Decoded tabular data of shape (batch_size, output_dim)
        """
        return self.model(z)

class TabularVAE:
    """
    Implementation of a Variational Autoencoder for generating synthetic tabular data.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        encoder_hidden_dims: List[int] = [256, 128],
        decoder_hidden_dims: List[int] = [128, 256],
        learning_rate: float = 0.001,
        beta: float = 1.0,
        device: str = None
    ):
        """
        Initialize the TabularVAE.
        
        Args:
            latent_dim: Dimension of the latent space
            encoder_hidden_dims: List of hidden layer dimensions for the encoder
            decoder_hidden_dims: List of hidden layer dimensions for the decoder
            learning_rate: Learning rate for the optimizer
            beta: Weight for the KL divergence term in the loss (beta-VAE)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Determine the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.learning_rate = learning_rate
        self.beta = beta
        
        # The input_dim and output_dim will be set during training
        self.encoder = None
        self.decoder = None
        self.metadata = None
        
        # Optimizer will be initialized during training
        self.optimizer = None
        
        # Training history
        self.loss_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': []
        }
    
    def _initialize_networks(self, input_dim: int):
        """
        Initialize the encoder and decoder networks.
        
        Args:
            input_dim: Dimension of the input tabular data
        """
        self.encoder = TabularEncoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.encoder_hidden_dims
        ).to(self.device)
        
        self.decoder = TabularDecoder(
            latent_dim=self.latent_dim,
            output_dim=input_dim,
            hidden_dims=self.decoder_hidden_dims
        ).to(self.device)
        
        # Initialize optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to enable backpropagation through the sampling process.
        
        Args:
            mu: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _apply_activations(
        self, 
        decoded_data: torch.Tensor, 
        sigmoid_temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Apply appropriate activations to the decoder output.
        
        Args:
            decoded_data: Raw output from the decoder
            sigmoid_temperature: Temperature parameter for sigmoid activation
            
        Returns:
            Processed decoded data with appropriate activations applied
        """
        result = decoded_data.clone()
        
        # Get indices for different feature types
        num_features = len(self.metadata.numeric_cols)
        cat_features_start = num_features
        
        # Apply activations for numerical features
        if num_features > 0:
            # No activation needed for numerical features as they can take any range of values
            # They will be transformed later based on the original data distribution
            pass
        
        # Apply softmax for categorical features
        if self.metadata.categorical_cols:
            cat_start_idx = cat_features_start
            
            # Process each categorical feature
            for i, col in enumerate(self.metadata.categorical_cols):
                # Get number of categories for this feature
                if hasattr(self.metadata.categorical_encoder, 'categories_'):
                    num_categories = len(self.metadata.categorical_encoder.categories_[i])
                    
                    # Extract the logits for this categorical feature
                    cat_end_idx = cat_start_idx + num_categories
                    logits = decoded_data[:, cat_start_idx:cat_end_idx]
                    
                    # Apply softmax with temperature
                    probabilities = F.softmax(logits * sigmoid_temperature, dim=1)
                    
                    # Update the result tensor
                    result[:, cat_start_idx:cat_end_idx] = probabilities
                    
                    # Update start index for next categorical feature
                    cat_start_idx = cat_end_idx
        
        return result
    
    def _calculate_loss(self, 
                       x: torch.Tensor, 
                       x_recon: torch.Tensor, 
                       mu: torch.Tensor, 
                       logvar: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the VAE loss, composed of reconstruction loss and KL divergence.
        
        Args:
            x: Original input data
            x_recon: Reconstructed data
            mu: Mean of the latent Gaussian
            logvar: Log variance of the latent Gaussian
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Get indices for different feature types
        num_features = len(self.metadata.numeric_cols)
        cat_features_start = num_features
        
        # Reconstruction loss
        # For numerical features: MSE loss
        recon_loss_num = 0
        if num_features > 0:
            recon_loss_num = F.mse_loss(
                x_recon[:, :num_features], 
                x[:, :num_features], 
                reduction='sum'
            )
        
        # For categorical features: Cross-entropy loss
        recon_loss_cat = 0
        if self.metadata.categorical_cols:
            cat_start_idx = cat_features_start
            
            for i, col in enumerate(self.metadata.categorical_cols):
                if hasattr(self.metadata.categorical_encoder, 'categories_'):
                    num_categories = len(self.metadata.categorical_encoder.categories_[i])
                    
                    cat_end_idx = cat_start_idx + num_categories
                    
                    # Get the target (one-hot encoded)
                    target = x[:, cat_start_idx:cat_end_idx]
                    
                    # Get the predictions (log-softmax)
                    logits = x_recon[:, cat_start_idx:cat_end_idx]
                    log_softmax = F.log_softmax(logits, dim=1)
                    
                    # Cross-entropy loss
                    cat_loss = -torch.sum(target * log_softmax)
                    recon_loss_cat += cat_loss
                    
                    cat_start_idx = cat_end_idx
        
        # Total reconstruction loss
        recon_loss = recon_loss_num + recon_loss_cat
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train(
        self,
        preprocessed_data: np.ndarray,
        metadata: Any,
        epochs: int = 300,
        batch_size: int = 64,
        early_stopping_patience: int = 30,
        verbose: bool = True,
        progress_interval: int = 10
    ):
        """
        Train the VAE model on preprocessed tabular data.
        
        Args:
            preprocessed_data: Preprocessed tabular data
            metadata: Metadata containing information about the features
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Number of epochs with no improvement to wait before stopping
            verbose: Whether to print progress during training
            progress_interval: Interval (in epochs) for printing progress
            
        Returns:
            Self, for method chaining
        """
        self.metadata = metadata
        input_dim = preprocessed_data.shape[1]
        
        # Initialize networks if not already initialized
        if self.encoder is None or self.decoder is None:
            self._initialize_networks(input_dim)
        
        # Create a DataLoader for batching
        tensor_x = torch.tensor(preprocessed_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0
            recon_loss_total = 0
            kl_loss_total = 0
            num_batches = 0
            
            for (x,) in dataloader:
                x = x.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Encode
                mu, logvar = self.encoder(x)
                
                # Sample from the latent distribution
                z = self._reparameterize(mu, logvar)
                
                # Decode
                x_recon = self.decoder(z)
                
                # Apply activations
                x_recon = self._apply_activations(x_recon)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self._calculate_loss(x, x_recon, mu, logvar)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                recon_loss_total += recon_loss.item()
                kl_loss_total += kl_loss.item()
                num_batches += 1
            
            # Calculate epoch loss
            epoch_loss = total_loss / num_batches
            epoch_recon_loss = recon_loss_total / num_batches
            epoch_kl_loss = kl_loss_total / num_batches
            
            # Store losses
            self.loss_history['total_loss'].append(epoch_loss)
            self.loss_history['reconstruction_loss'].append(epoch_recon_loss)
            self.loss_history['kl_loss'].append(epoch_kl_loss)
            
            # Print progress
            if verbose and (epoch % progress_interval == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch}/{epochs}] [Loss: {epoch_loss:.4f}] [Recon: {epoch_recon_loss:.4f}] [KL: {epoch_kl_loss:.4f}] [Time: {elapsed:.2f}s]")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if verbose:
            print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def generate(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate synthetic tabular data by sampling from the latent space.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated tabular data as a numpy array
        """
        # Check if the model has been trained
        if self.encoder is None or self.decoder is None:
            raise ValueError("The model must be trained before generating data")
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Generate data
        with torch.no_grad():
            # Sample from the latent space
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            
            # Create batches to avoid memory issues
            batch_size = 500
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            generated_data = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                current_batch_size = end_idx - start_idx
                
                z_batch = z[start_idx:end_idx]
                
                # Decode latent vectors
                batch_data = self.decoder(z_batch)
                
                # Apply appropriate activations
                batch_data = self._apply_activations(batch_data)
                
                # Move to CPU and convert to numpy
                generated_data.append(batch_data.cpu().numpy())
            
            # Concatenate batches
            generated_data = np.vstack(generated_data)
        
        return generated_data
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data into the latent space.
        
        Args:
            data: Preprocessed data to encode
            
        Returns:
            Latent space representation (mean vectors) as a numpy array
        """
        # Check if the model has been trained
        if self.encoder is None:
            raise ValueError("The model must be trained before encoding data")
        
        # Set encoder to evaluation mode
        self.encoder.eval()
        
        # Convert data to tensor
        tensor_x = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Encode data
        with torch.no_grad():
            mu, _ = self.encoder(tensor_x)
            
        return mu.cpu().numpy()
    
    def decode(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Decode latent vectors into data.
        
        Args:
            latent_vectors: Latent space vectors to decode
            
        Returns:
            Decoded data as a numpy array
        """
        # Check if the model has been trained
        if self.decoder is None:
            raise ValueError("The model must be trained before decoding data")
        
        # Set decoder to evaluation mode
        self.decoder.eval()
        
        # Convert data to tensor
        tensor_z = torch.tensor(latent_vectors, dtype=torch.float32).to(self.device)
        
        # Decode data
        with torch.no_grad():
            decoded = self.decoder(tensor_z)
            decoded = self._apply_activations(decoded)
            
        return decoded.cpu().numpy()
    
    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """
        Reconstruct data by encoding and then decoding.
        
        Args:
            data: Preprocessed data to reconstruct
            
        Returns:
            Reconstructed data as a numpy array
        """
        # Check if the model has been trained
        if self.encoder is None or self.decoder is None:
            raise ValueError("The model must be trained before reconstructing data")
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        
        # Convert data to tensor
        tensor_x = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Reconstruct data
        with torch.no_grad():
            mu, logvar = self.encoder(tensor_x)
            z = self._reparameterize(mu, logvar)
            reconstructed = self.decoder(z)
            reconstructed = self._apply_activations(reconstructed)
            
        return reconstructed.cpu().numpy()
    
    def plot_loss_history(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the loss history during training.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot total loss
        plt.subplot(3, 1, 1)
        plt.plot(self.loss_history['total_loss'])
        plt.title('Total VAE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot reconstruction loss
        plt.subplot(3, 1, 2)
        plt.plot(self.loss_history['reconstruction_loss'])
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot KL divergence
        plt.subplot(3, 1, 3)
        plt.plot(self.loss_history['kl_loss'])
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_latent_space(self, data: np.ndarray, labels=None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the latent space distribution.
        
        Args:
            data: Preprocessed data to encode
            labels: Optional labels for coloring points
            figsize: Figure size
        """
        if self.latent_dim > 2:
            print("Warning: Latent dimension is greater than 2. Plotting first 2 dimensions only.")
        
        # Set encoder to evaluation mode
        self.encoder.eval()
        
        # Convert data to tensor
        tensor_x = torch.tensor(data, dtype=torch.float32).to(self.device)
        
        # Encode data
        with torch.no_grad():
            mu, _ = self.encoder(tensor_x)
            
        # Get first two dimensions of latent space
        latent_data = mu.cpu().numpy()[:, :2]
        
        # Plot
        plt.figure(figsize=figsize)
        
        if labels is not None:
            # Color by labels
            scatter = plt.scatter(latent_data[:, 0], latent_data[:, 1], c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter)
        else:
            # Single color
            plt.scatter(latent_data[:, 0], latent_data[:, 1], alpha=0.6)
        
        plt.title('Latent Space Distribution')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True)
        plt.show()
    
    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        if self.encoder is None or self.decoder is None:
            raise ValueError("The model must be trained before saving")
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_hidden_dims': self.encoder_hidden_dims,
            'decoder_hidden_dims': self.decoder_hidden_dims,
            'latent_dim': self.latent_dim,
            'beta': self.beta,
            'loss_history': self.loss_history
        }, path)
    
    def load(self, path: str, input_dim: int, metadata: Any = None):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            input_dim: Dimension of the input tabular data
            metadata: Metadata containing information about the features
            
        Returns:
            Self, for method chaining
        """
        # Load the model
        checkpoint = torch.load(path)
        
        # Initialize hyperparameters
        self.latent_dim = checkpoint['latent_dim']
        self.encoder_hidden_dims = checkpoint['encoder_hidden_dims']
        self.decoder_hidden_dims = checkpoint['decoder_hidden_dims']
        self.beta = checkpoint['beta']
        
        # Initialize networks
        self._initialize_networks(input_dim)
        
        # Load weights
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Load metadata if provided
        if metadata is not None:
            self.metadata = metadata
            
        # Load loss history
        self.loss_history = checkpoint['loss_history']
        
        return self