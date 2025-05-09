import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
import time

class TabularGenerator(nn.Module):
    """
    Generator network for tabular data.
    
    This model takes a noise vector and generates synthetic tabular data.
    The architecture is designed to handle mixed data types (numerical and categorical).
    """
    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: List[int] = [256, 512, 256]):
        """
        Initialize the generator network.
        
        Args:
            noise_dim: Dimension of the input noise vector
            output_dim: Dimension of the output (number of features in the tabular data)
            hidden_dims: List of hidden layer dimensions
        """
        super(TabularGenerator, self).__init__()
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(noise_dim, hidden_dims[0]))
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
        Forward pass of the generator.
        
        Args:
            z: Input noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated tabular data of shape (batch_size, output_dim)
        """
        return self.model(z)

class TabularDiscriminator(nn.Module):
    """
    Discriminator network for tabular data.
    
    This model takes tabular data and predicts whether it's real or generated.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize the discriminator network.
        
        Args:
            input_dim: Dimension of the input tabular data
            hidden_dims: List of hidden layer dimensions
        """
        super(TabularDiscriminator, self).__init__()
        
        # Build the network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input tabular data of shape (batch_size, input_dim)
            
        Returns:
            Probability of the data being real (vs. generated) of shape (batch_size, 1)
        """
        return self.model(x)

class FeatureActivation:
    """
    Utility class to apply appropriate activations to different feature types.
    """
    @staticmethod
    def apply_activations(
        generated_data: torch.Tensor, 
        metadata: Any,
        sigmoid_temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Apply appropriate activations to the generated data based on feature types.
        
        Args:
            generated_data: Raw output from the generator
            metadata: Metadata containing information about feature types
            sigmoid_temperature: Temperature parameter for sigmoid activation
                                 Higher values make the sigmoid steeper
        
        Returns:
            Processed generated data with appropriate activations applied
        """
        result = generated_data.clone()
        
        # Get indices for different feature types
        num_features = len(metadata.numeric_cols)
        cat_features_start = num_features
        
        # Apply activations for numerical features
        if num_features > 0:
            # No activation needed for numerical features as they can take any range of values
            # They will be transformed later based on the original data distribution
            pass
        
        # Apply softmax for categorical features
        if metadata.categorical_cols:
            cat_start_idx = cat_features_start
            
            # Process each categorical feature
            for i, col in enumerate(metadata.categorical_cols):
                # Get number of categories for this feature
                if hasattr(metadata.categorical_encoder, 'categories_'):
                    num_categories = len(metadata.categorical_encoder.categories_[i])
                    
                    # Extract the logits for this categorical feature
                    cat_end_idx = cat_start_idx + num_categories
                    logits = generated_data[:, cat_start_idx:cat_end_idx]
                    
                    # Apply softmax with temperature
                    probabilities = torch.nn.functional.softmax(logits * sigmoid_temperature, dim=1)
                    
                    # Update the result tensor
                    result[:, cat_start_idx:cat_end_idx] = probabilities
                    
                    # Update start index for next categorical feature
                    cat_start_idx = cat_end_idx
        
        return result

class TabularGAN:
    """
    Implementation of a GAN for generating synthetic tabular data.
    """
    def __init__(
        self,
        noise_dim: int = 128,
        generator_hidden_dims: List[int] = [256, 512, 256],
        discriminator_hidden_dims: List[int] = [256, 128, 64],
        lr_generator: float = 0.0002,
        lr_discriminator: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        device: str = None
    ):
        """
        Initialize the TabularGAN.
        
        Args:
            noise_dim: Dimension of the input noise vector
            generator_hidden_dims: List of hidden layer dimensions for the generator
            discriminator_hidden_dims: List of hidden layer dimensions for the discriminator
            lr_generator: Learning rate for the generator optimizer
            lr_discriminator: Learning rate for the discriminator optimizer
            b1: Beta1 parameter for Adam optimizer
            b2: Beta2 parameter for Adam optimizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Determine the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.noise_dim = noise_dim
        self.generator_hidden_dims = generator_hidden_dims
        self.discriminator_hidden_dims = discriminator_hidden_dims
        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.b1 = b1
        self.b2 = b2
        
        # The input_dim and output_dim will be set during training
        self.generator = None
        self.discriminator = None
        self.metadata = None
        
        # Optimizer will be initialized during training
        self.optimizer_G = None
        self.optimizer_D = None
        
        # Training history
        self.loss_history = {
            'G_loss': [],
            'D_loss': [],
            'D_real': [],
            'D_fake': []
        }
    
    def _initialize_networks(self, input_dim: int):
        """
        Initialize the generator and discriminator networks.
        
        Args:
            input_dim: Dimension of the preprocessed tabular data
        """
        self.generator = TabularGenerator(
            noise_dim=self.noise_dim,
            output_dim=input_dim,
            hidden_dims=self.generator_hidden_dims
        ).to(self.device)
        
        self.discriminator = TabularDiscriminator(
            input_dim=input_dim,
            hidden_dims=self.discriminator_hidden_dims
        ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.lr_generator,
            betas=(self.b1, self.b2)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_discriminator,
            betas=(self.b1, self.b2)
        )
    
    def train(
        self,
        preprocessed_data: np.ndarray,
        metadata: Any,
        epochs: int = 300,
        batch_size: int = 64,
        d_steps: int = 1,
        g_steps: int = 1,
        early_stopping_patience: int = 30,
        verbose: bool = True,
        progress_interval: int = 10
    ):
        """
        Train the GAN model on preprocessed tabular data.
        
        Args:
            preprocessed_data: Preprocessed tabular data
            metadata: Metadata containing information about the features
            epochs: Number of training epochs
            batch_size: Batch size for training
            d_steps: Number of discriminator updates per iteration
            g_steps: Number of generator updates per iteration
            early_stopping_patience: Number of epochs with no improvement to wait before stopping
            verbose: Whether to print progress during training
            progress_interval: Interval (in epochs) for printing progress
            
        Returns:
            Self, for method chaining
        """
        self.metadata = metadata
        input_dim = preprocessed_data.shape[1]
        
        # Initialize networks if not already initialized or if input_dim has changed
        if self.generator is None or self.discriminator is None:
            self._initialize_networks(input_dim)
        
        # Create a DataLoader for batching
        tensor_x = torch.tensor(preprocessed_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Adversarial loss
        adversarial_loss = nn.BCELoss()
        
        # Training loop
        best_g_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            d_real_acc = []
            d_fake_acc = []
            
            for i, (real_data,) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Create labels for real and fake data
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                
                # Discriminator on real data
                real_pred = self.discriminator(real_data)
                d_real_loss = adversarial_loss(real_pred, real_labels)
                
                # Discriminator on fake data
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_data = self.generator(z)
                fake_data = FeatureActivation.apply_activations(fake_data, metadata)
                fake_pred = self.discriminator(fake_data.detach())
                d_fake_loss = adversarial_loss(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # Calculate discriminator accuracy
                d_real_acc.append(((real_pred > 0.5).float() == real_labels).float().mean().item())
                d_fake_acc.append(((fake_pred < 0.5).float() == (1 - fake_labels)).float().mean().item())
                
                d_loss.backward()
                self.optimizer_D.step()
                
                d_losses.append(d_loss.item())
                
                # ---------------------
                # Train Generator
                # ---------------------
                if i % d_steps == 0:
                    for _ in range(g_steps):
                        self.optimizer_G.zero_grad()
                        
                        # Generate fake data
                        z = torch.randn(batch_size, self.noise_dim, device=self.device)
                        fake_data = self.generator(z)
                        fake_data = FeatureActivation.apply_activations(fake_data, metadata)
                        
                        # Discriminator on fake data
                        fake_pred = self.discriminator(fake_data)
                        
                        # Generator loss - maximize log(D(G(z)))
                        g_loss = adversarial_loss(fake_pred, real_labels)
                        
                        g_loss.backward()
                        self.optimizer_G.step()
                        
                        g_losses.append(g_loss.item())
            
            # Calculate epoch losses
            epoch_g_loss = np.mean(g_losses)
            epoch_d_loss = np.mean(d_losses)
            epoch_d_real = np.mean(d_real_acc)
            epoch_d_fake = np.mean(d_fake_acc)
            
            # Store losses
            self.loss_history['G_loss'].append(epoch_g_loss)
            self.loss_history['D_loss'].append(epoch_d_loss)
            self.loss_history['D_real'].append(epoch_d_real)
            self.loss_history['D_fake'].append(epoch_d_fake)
            
            # Print progress
            if verbose and (epoch % progress_interval == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch}/{epochs}] [D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}] [D real: {epoch_d_real:.4f}] [D fake: {epoch_d_fake:.4f}] [Time: {elapsed:.2f}s]")
            
            # Early stopping based on generator loss
            if epoch_g_loss < best_g_loss:
                best_g_loss = epoch_g_loss
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
        Generate synthetic tabular data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated tabular data as a numpy array
        """
        # Check if the model has been trained
        if self.generator is None:
            raise ValueError("The model must be trained before generating data")
        
        # Set generator to evaluation mode
        self.generator.eval()
        
        # Generate data
        with torch.no_grad():
            # Create batches to avoid memory issues
            batch_size = 500
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            generated_data = []
            
            for i in range(num_batches):
                current_batch_size = min(batch_size, n_samples - i * batch_size)
                z = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                
                # Generate data
                batch_data = self.generator(z)
                
                # Apply appropriate activations
                batch_data = FeatureActivation.apply_activations(batch_data, self.metadata)
                
                # Move to CPU and convert to numpy
                generated_data.append(batch_data.cpu().numpy())
            
            # Concatenate batches
            generated_data = np.vstack(generated_data)
        
        return generated_data
    
    def plot_loss_history(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the loss history during training.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot generator and discriminator losses
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_history['G_loss'], label='Generator')
        plt.plot(self.loss_history['D_loss'], label='Discriminator')
        plt.title('GAN Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot discriminator accuracy on real and fake data
        plt.subplot(2, 1, 2)
        plt.plot(self.loss_history['D_real'], label='D(real)')
        plt.plot(self.loss_history['D_fake'], label='D(fake)')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("The model must be trained before saving")
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_hidden_dims': self.generator_hidden_dims,
            'discriminator_hidden_dims': self.discriminator_hidden_dims,
            'noise_dim': self.noise_dim,
            'loss_history': self.loss_history
        }, path)
    
    def load(self, path: str, input_dim: int, metadata: Any = None):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            input_dim: Dimension of the preprocessed tabular data
            metadata: Metadata containing information about the features
            
        Returns:
            Self, for method chaining
        """
        # Load the model
        checkpoint = torch.load(path)
        
        # Initialize networks
        self.noise_dim = checkpoint['noise_dim']
        self.generator_hidden_dims = checkpoint['generator_hidden_dims']
        self.discriminator_hidden_dims = checkpoint['discriminator_hidden_dims']
        
        self._initialize_networks(input_dim)
        
        # Load weights
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load metadata if provided
        if metadata is not None:
            self.metadata = metadata
            
        # Load loss history
        self.loss_history = checkpoint['loss_history']
        
        return self