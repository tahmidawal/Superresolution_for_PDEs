import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models import UNet, PDEDataset, init_weights

class EnhancedPDEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict, device: str = 'cuda'):
        """
        Enhanced dataset class for PDE solutions with different theta types.
        
        Args:
            data_dict: Dictionary containing the dataset
            device: Device to store tensors on
        """
        self.device = device
        
        # Convert numpy arrays to tensors and move to device
        self.u_coarse = torch.from_numpy(data_dict['u_coarse']).float().to(device)
        self.u_fine = torch.from_numpy(data_dict['u_fine']).float().to(device)
        self.f_fine = torch.from_numpy(data_dict['f_fine']).float().to(device)
        self.theta_fine = torch.from_numpy(data_dict['theta_fine']).float().to(device)
        
        # Store subdomain flag and theta type if available
        self.has_subdomain_flag = 'is_subdomain' in data_dict
        if self.has_subdomain_flag:
            self.is_subdomain = torch.from_numpy(data_dict['is_subdomain']).bool().to(device)
        
        self.has_theta_type = 'theta_type' in data_dict
        if self.has_theta_type:
            self.theta_type = data_dict['theta_type']
            
            # Count samples by theta type
            theta_types, counts = np.unique(self.theta_type, return_counts=True)
            print("Theta type distribution:")
            for t, c in zip(theta_types, counts):
                print(f"  {t}: {c} samples")
        
        # Compute normalization statistics for u
        self.u_mean = self.u_fine.mean()
        self.u_std = self.u_fine.std()
        
        # Compute normalization statistics for f
        self.f_mean = self.f_fine.mean()
        self.f_std = self.f_fine.std()
        
        # For theta, normalize by type if available
        if self.has_theta_type:
            # Initialize storage for theta statistics by type
            self.theta_mean_by_type = {}
            self.theta_std_by_type = {}
            
            # Calculate statistics for each theta type
            for theta_type in np.unique(self.theta_type):
                mask = self.theta_type == theta_type
                theta_subset = self.theta_fine[torch.from_numpy(mask).to(device)]
                
                if theta_type == 'constant':
                    # For constant theta, don't normalize
                    self.theta_mean_by_type[theta_type] = 0
                    self.theta_std_by_type[theta_type] = 1
                    print(f"Constant theta detected, skipping normalization for this type")
                else:
                    # For other types, compute statistics
                    self.theta_mean_by_type[theta_type] = theta_subset.mean()
                    self.theta_std_by_type[theta_type] = theta_subset.std()
                    print(f"Theta type '{theta_type}' statistics: mean={self.theta_mean_by_type[theta_type]:.4f}, std={self.theta_std_by_type[theta_type]:.4f}")
        else:
            # If no theta type information, check if theta is constant
            self.theta_is_constant = (self.theta_fine.std() < 1e-6)
            if self.theta_is_constant:
                self.theta_mean = 0
                self.theta_std = 1
                print("Detected constant theta field, skipping normalization")
            else:
                self.theta_mean = self.theta_fine.mean()
                self.theta_std = self.theta_fine.std()
                print(f"Theta statistics: mean={self.theta_mean:.4f}, std={self.theta_std:.4f}")
        
        # Normalize the data
        self.u_fine_norm = (self.u_fine - self.u_mean) / self.u_std
        self.u_coarse_norm = (self.u_coarse - self.u_mean) / self.u_std
        self.f_fine_norm = (self.f_fine - self.f_mean) / self.f_std
        
        # Normalize theta based on type if available
        if self.has_theta_type:
            self.theta_fine_norm = torch.zeros_like(self.theta_fine)
            for theta_type in np.unique(self.theta_type):
                mask = self.theta_type == theta_type
                mask_tensor = torch.from_numpy(mask).to(device)
                
                if theta_type == 'constant':
                    # For constant theta, just pass through
                    self.theta_fine_norm[mask_tensor] = self.theta_fine[mask_tensor]
                else:
                    # For other types, normalize
                    mean = self.theta_mean_by_type[theta_type]
                    std = self.theta_std_by_type[theta_type]
                    self.theta_fine_norm[mask_tensor] = (self.theta_fine[mask_tensor] - mean) / std
        else:
            # If no theta type information, normalize based on constancy
            if self.theta_is_constant:
                self.theta_fine_norm = self.theta_fine
            else:
                self.theta_fine_norm = (self.theta_fine - self.theta_mean) / self.theta_std
        
        # Upsample coarse solution to fine grid
        self.u_coarse_upsampled = nn.functional.interpolate(
            self.u_coarse_norm.unsqueeze(1),
            size=(40, 40),
            mode='bilinear',
            align_corners=True
        )
        
    def __len__(self) -> int:
        return len(self.u_fine)
    
    def __getitem__(self, idx: int) -> tuple:
        # Combine normalized inputs: [upsampled_coarse_solution, theta, f]
        x = torch.cat([
            self.u_coarse_upsampled[idx],
            self.theta_fine_norm[idx].unsqueeze(0),
            self.f_fine_norm[idx].unsqueeze(0)
        ], dim=0)
        
        # Target is the normalized fine solution
        y = self.u_fine_norm[idx].unsqueeze(0)
        
        return x, y
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the model output back to original scale."""
        return x * self.u_std + self.u_mean

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: str,
    save_dir: Path,
    writer: SummaryWriter,
    grad_clip: float = 1.0,
    early_stopping_patience: int = 20
) -> dict:
    """
    Train the model and save checkpoints.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save checkpoints
        writer: TensorBoard writer
        grad_clip: Maximum norm of gradients
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Dictionary containing training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'num_epochs': 0
    }
    
    # Early stopping counter
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            no_improvement_count = 0  # Reset counter
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"Saved new best model with val_loss: {val_loss:.6f}")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs (best: {history['best_val_loss']:.6f} at epoch {history['best_epoch']+1})")
        
        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Store the total number of epochs in history (outside the loop)
    history['num_epochs'] = len(history['train_loss'])
    
    return history

def plot_losses(history: dict, save_dir: Path):
    """
    Plot training and validation losses.
    
    Args:
        history: Dictionary containing loss history
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    # Highlight the best model
    best_epoch = history['best_epoch'] + 1  # Convert to 1-indexed
    best_val_loss = history['best_val_loss']
    plt.plot(best_epoch, best_val_loss, 'go', markersize=10, 
             label=f'Best Model (Epoch {best_epoch}, Loss: {best_val_loss:.6f})')
    
    # Add vertical line at early stopping point if stopped early
    if len(epochs) < history.get('num_epochs', float('inf')):
        plt.axvline(x=len(epochs), color='k', linestyle='--', 
                   label=f'Early Stopping (Epoch {len(epochs)})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate(f'Best: {best_val_loss:.6f}', 
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 5, best_val_loss * 1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150)
    plt.close()

def stratified_split_by_theta_type(data: dict, val_split: float = 0.2):
    """
    Split the dataset into training and validation sets, stratified by theta type.
    
    Args:
        data: Dictionary containing the dataset
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data)
    """
    # Get unique theta types and their counts
    theta_types, counts = np.unique(data['theta_type'], return_counts=True)
    print("Stratifying split by theta type:")
    
    train_indices = []
    val_indices = []
    
    # For each theta type, split indices
    for theta_type in theta_types:
        # Get indices for this theta type
        type_indices = np.where(data['theta_type'] == theta_type)[0]
        np.random.shuffle(type_indices)
        
        # Calculate split sizes
        val_size = int(len(type_indices) * val_split)
        
        # Split indices
        val_indices_for_type = type_indices[:val_size]
        train_indices_for_type = type_indices[val_size:]
        
        # Add to overall indices
        train_indices.extend(train_indices_for_type)
        val_indices.extend(val_indices_for_type)
        
        print(f"  {theta_type}: {len(train_indices_for_type)} train, {len(val_indices_for_type)} val")
    
    # Shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Create train and validation datasets
    train_data = {key: data[key][train_indices] for key in data.keys()}
    val_data = {key: data[key][val_indices] for key in data.keys()}
    
    return train_data, val_data

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 500,
        'learning_rate': 2e-4,
        'min_lr': 1e-6,
        'patience': 10,  # For learning rate scheduler
        'early_stopping_patience': 20,  # For early stopping
        'val_split': 0.2,
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,  # Number of workers for data loading
        'pin_memory': True,  # Faster data transfer to GPU
        'dataset_path': 'data/pde_dataset_v3.npz',  # Path to the mixed theta dataset
        'stratify_by_theta_type': True  # Stratify train/val split by theta type
    }
    
    # Create directories
    base_dir = Path('results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = base_dir / f'enhanced_run_v3_{timestamp}'
    save_dir.mkdir(parents=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    
    # Load data
    print(f"Loading dataset from {config['dataset_path']}...")
    data = dict(np.load(config['dataset_path']))
    
    # Print dataset statistics
    n_samples = len(data['u_fine'])
    print(f"Total samples: {n_samples}")
    
    # Split data into train and validation sets
    if config['stratify_by_theta_type'] and 'theta_type' in data:
        train_data, val_data = stratified_split_by_theta_type(data, config['val_split'])
    else:
        # Create a random permutation of indices
        indices = np.random.permutation(n_samples)
        
        # Split indices into train and validation
        val_size = int(n_samples * config['val_split'])
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create train and validation datasets
        train_data = {key: data[key][train_indices] for key in data.keys()}
        val_data = {key: data[key][val_indices] for key in data.keys()}
    
    print(f"Training samples: {len(train_data['u_fine'])}")
    print(f"Validation samples: {len(val_data['u_fine'])}")
    
    # Create PyTorch datasets with enhanced dataset class
    train_dataset = EnhancedPDEDataset(train_data, device=config['device'])
    val_dataset = EnhancedPDEDataset(val_data, device=config['device'])
    
    # Create data loaders with shuffling for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # No need to shuffle validation data
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Initialize model
    model = UNet().to(config['device'])
    model.apply(init_weights)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['patience'],
        min_lr=config['min_lr'],
        verbose=True
    )
    
    # Train model
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_dir=save_dir,
        writer=writer,
        grad_clip=config['grad_clip'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Plot and save training history
    plot_losses(history, save_dir)
    
    # Save final model state
    final_checkpoint = {
        'epoch': len(history['train_loss']) - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch']
    }
    torch.save(final_checkpoint, save_dir / 'final_model.pth')
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Best validation loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']+1})")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main() 