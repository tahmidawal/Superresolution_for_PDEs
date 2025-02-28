import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
from datetime import datetime

from models import UNet, init_weights
from enhanced_data_generation_v3 import EnhancedPoissonSolverV3

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

def load_model(model_path: Path, device: str = 'cuda'):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = UNet().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully (from epoch {checkpoint['epoch']+1})")
    return model

def generate_test_data(n_samples_per_type: int = 10):
    """
    Generate test data with different theta types.
    
    Args:
        n_samples_per_type: Number of samples to generate for each theta type
        
    Returns:
        Dictionary containing the test dataset
    """
    print("Generating test data...")
    solver = EnhancedPoissonSolverV3()
    
    # Generate dataset with different theta types
    dataset = solver.generate_mixed_theta_dataset(
        n_constant=n_samples_per_type,
        n_grf=n_samples_per_type,
        n_radial=n_samples_per_type,
        k_range=(0.5, 8.0)  # Use a wider k range for testing
    )
    
    print(f"Generated {len(dataset['u_fine'])} test samples")
    
    # Save the test dataset
    save_path = Path('data/test_dataset_v3.npz')
    np.savez(save_path, **dataset)
    print(f"Saved test dataset to {save_path}")
    
    # Plot theta examples
    solver.plot_theta_examples(save_dir=Path('results'))
    
    # Plot random samples
    solver.plot_random_samples(
        dataset, 
        n_samples=10, 
        save_path='results/test_samples_v3.png'
    )
    
    return dataset

def evaluate_by_theta_type(model, dataset, device: str = 'cuda'):
    """
    Evaluate the model performance by theta type.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics by theta type
    """
    print("Evaluating model by theta type...")
    model.eval()
    
    # Group samples by theta type
    theta_types = np.unique(dataset.theta_type)
    metrics_by_type = {theta_type: {'mae': [], 'rmse': [], 'max_error': []} for theta_type in theta_types}
    
    # Process each sample
    for idx in tqdm(range(len(dataset))):
        # Get data for this sample
        inputs, targets = dataset[idx]
        theta_type = dataset.theta_type[idx]
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(inputs.unsqueeze(0))
            
        # Denormalize
        outputs_denorm = dataset.denormalize(outputs)
        targets_denorm = dataset.denormalize(targets.unsqueeze(0))
        
        # Calculate metrics
        error = torch.abs(outputs_denorm - targets_denorm)
        mae = error.mean().item()
        rmse = torch.sqrt((error ** 2).mean()).item()
        max_error = error.max().item()
        
        # Store metrics by theta type
        metrics_by_type[theta_type]['mae'].append(mae)
        metrics_by_type[theta_type]['rmse'].append(rmse)
        metrics_by_type[theta_type]['max_error'].append(max_error)
    
    # Calculate average metrics by theta type
    avg_metrics_by_type = {}
    for theta_type in theta_types:
        avg_metrics_by_type[theta_type] = {
            'mae': np.mean(metrics_by_type[theta_type]['mae']),
            'rmse': np.mean(metrics_by_type[theta_type]['rmse']),
            'max_error': np.mean(metrics_by_type[theta_type]['max_error']),
            'num_samples': len(metrics_by_type[theta_type]['mae'])
        }
    
    return avg_metrics_by_type

def plot_metrics_by_theta_type(metrics_by_type, save_dir: Path):
    """
    Plot metrics by theta type.
    
    Args:
        metrics_by_type: Dictionary of metrics by theta type
        save_dir: Directory to save the plots
    """
    print("Plotting metrics by theta type...")
    
    # Extract data for plotting
    theta_types = list(metrics_by_type.keys())
    mae_values = [metrics_by_type[t]['mae'] for t in theta_types]
    rmse_values = [metrics_by_type[t]['rmse'] for t in theta_types]
    max_error_values = [metrics_by_type[t]['max_error'] for t in theta_types]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(theta_types))
    width = 0.25
    
    plt.bar(x - width, mae_values, width, label='MAE')
    plt.bar(x, rmse_values, width, label='RMSE')
    plt.bar(x + width, max_error_values, width, label='Max Error')
    
    plt.xlabel('Theta Type')
    plt.ylabel('Error')
    plt.title('Model Performance by Theta Type')
    plt.xticks(x, theta_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(mae_values):
        plt.text(i - width, v + 0.01, f'{v:.4f}', ha='center')
    for i, v in enumerate(rmse_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    for i, v in enumerate(max_error_values):
        plt.text(i + width, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_by_theta_type.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_predictions(model, dataset, save_dir: Path, n_samples: int = 5):
    """
    Plot sample predictions for each theta type.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        save_dir: Directory to save the plots
        n_samples: Number of samples to plot for each theta type
    """
    print("Plotting sample predictions...")
    model.eval()
    
    # Group samples by theta type
    theta_types = np.unique(dataset.theta_type)
    samples_by_type = {theta_type: [] for theta_type in theta_types}
    
    # Collect sample indices by theta type
    for idx in range(len(dataset)):
        theta_type = dataset.theta_type[idx]
        if len(samples_by_type[theta_type]) < n_samples:
            samples_by_type[theta_type].append(idx)
    
    # Create plots directory
    plots_dir = save_dir / 'sample_predictions'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each theta type
    for theta_type in theta_types:
        for i, idx in enumerate(samples_by_type[theta_type]):
            # Get data for this sample
            inputs, targets = dataset[idx]
            
            # Get model prediction
            with torch.no_grad():
                outputs = model(inputs.unsqueeze(0))
            
            # Denormalize
            outputs_denorm = dataset.denormalize(outputs)
            targets_denorm = dataset.denormalize(targets.unsqueeze(0))
            
            # Convert to numpy for plotting
            u_coarse = dataset.u_coarse[idx].cpu().numpy()
            u_fine = dataset.u_fine[idx].cpu().numpy()
            theta_fine = dataset.theta_fine[idx].cpu().numpy()
            f_fine = dataset.f_fine[idx].cpu().numpy()
            prediction = outputs_denorm.squeeze().cpu().numpy()
            
            # Calculate error
            error = np.abs(prediction - u_fine)
            mae = np.mean(error)
            
            # Create plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            plt.suptitle(f'Prediction for {theta_type} Theta - Sample {i+1}\nMAE: {mae:.6f}', fontsize=16)
            
            # First row
            im1 = axes[0, 0].imshow(u_coarse)
            axes[0, 0].set_title('Coarse Solution (20×20)')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(u_fine)
            axes[0, 1].set_title('Ground Truth (40×40)')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(prediction)
            axes[0, 2].set_title('Model Prediction (40×40)')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # Second row
            im4 = axes[1, 0].imshow(theta_fine)
            axes[1, 0].set_title('Theta Field')
            plt.colorbar(im4, ax=axes[1, 0])
            
            im5 = axes[1, 1].imshow(f_fine)
            axes[1, 1].set_title('Forcing Term')
            plt.colorbar(im5, ax=axes[1, 1])
            
            im6 = axes[1, 2].imshow(error)
            axes[1, 2].set_title(f'Error (MAE: {mae:.6f})')
            plt.colorbar(im6, ax=axes[1, 2])
            
            # Remove axis ticks
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{theta_type}_sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate enhanced model on different theta types')
    parser.add_argument('--model_path', type=str, help='Path to the model file', 
                       default='results/enhanced_run_v3_20250227_231745/best_model.pth')
    parser.add_argument('--device', type=str, help='Device to run evaluation on',
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_samples', type=int, help='Number of samples per theta type',
                       default=10)
    parser.add_argument('--generate_data', action='store_true', help='Generate new test data')
    args = parser.parse_args()

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'results/evaluation_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = args.device
    print(f"Using device: {device}")
    
    # Generate or load test data
    if args.generate_data:
        data = generate_test_data(n_samples_per_type=args.n_samples)
    else:
        # Try to load existing test data
        try:
            data = dict(np.load('data/test_dataset_v3.npz'))
            print(f"Loaded test dataset with {len(data['u_fine'])} samples")
        except FileNotFoundError:
            print("Test dataset not found, generating new data...")
            data = generate_test_data(n_samples_per_type=args.n_samples)
    
    # Create dataset
    dataset = EnhancedPDEDataset(data, device=device)
    
    # Load model
    model_path = Path(args.model_path)
    model = load_model(model_path, device)
    
    # Evaluate model by theta type
    metrics_by_type = evaluate_by_theta_type(model, dataset, device)
    
    # Print metrics
    print("\nEvaluation Results by Theta Type:")
    for theta_type, metrics in metrics_by_type.items():
        print(f"\n{theta_type} Theta ({metrics['num_samples']} samples):")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  Max Error: {metrics['max_error']:.6f}")
    
    # Save metrics
    with open(results_dir / 'metrics_by_theta_type.json', 'w') as f:
        json.dump(metrics_by_type, f, indent=4)
    
    # Plot metrics by theta type
    plot_metrics_by_theta_type(metrics_by_type, results_dir)
    
    # Plot sample predictions
    plot_sample_predictions(model, dataset, results_dir, n_samples=5)
    
    print(f"\nEvaluation complete. Results saved to {results_dir}")

if __name__ == '__main__':
    main() 