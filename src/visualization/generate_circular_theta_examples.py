import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import sys
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Add the src directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.enhanced_data_generation_v3 import EnhancedPoissonSolverV3
from src.models import UNet
from src.evaluate_enhanced_model import load_model

def generate_smooth_circular_theta(n: int, radius: float = 0.4, smoothness: float = 0.05) -> np.ndarray:
    """
    Generate a smooth circular theta field with specified radius.
    
    Args:
        n: Grid size
        radius: Radius of the circle (in normalized coordinates [0,1])
        smoothness: Parameter controlling the smoothness of the circle edge
        
    Returns:
        2D array containing the theta field
    """
    # Create grid coordinates
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center
    center_x, center_y = 0.5, 0.5
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create smooth transition using sigmoid function
    theta = 1.0 / (1.0 + np.exp((distance - radius) / smoothness))
    
    # Scale theta to range [0.1, 1.0]
    theta = 0.1 + 0.9 * theta
    
    return theta

def generate_and_save_examples(n_samples: int = 3, save_dir: str = 'results/circular_theta_examples'):
    """
    Generate and save examples with smooth circular theta fields.
    
    Args:
        n_samples: Number of samples to generate
        save_dir: Directory to save the examples
    """
    # Create save directory
    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    # Initialize solver
    solver = EnhancedPoissonSolverV3(n_coarse=40, n_fine=80, n_superfine=160)
    
    # Load the trained model
    model_path = Path('results/enhanced_run_v3_20250227_231745/best_model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    model.eval()
    
    # Generate samples
    for i in range(n_samples):
        # Generate theta field
        theta_fine = generate_smooth_circular_theta(solver.n_fine)
        theta_coarse = generate_smooth_circular_theta(solver.n_coarse)
        
        # Generate random wave numbers for forcing term
        k1 = np.random.uniform(0.5, 5.0)
        k2 = np.random.uniform(0.5, 5.0)
        
        # Generate forcing terms
        f_fine = solver.generate_forcing_term(k1, k2, 'fine')
        f_coarse = solver.generate_forcing_term(k1, k2, 'coarse')
        
        # Solve PDE
        u_fine = solver.solve_poisson(f_fine, theta_fine, 'fine')
        u_coarse = solver.solve_poisson(f_coarse, theta_coarse, 'coarse')
        
        # Prepare inputs for the model
        # Normalize data
        u_mean = u_fine.mean()
        u_std = u_fine.std()
        f_mean = f_fine.mean()
        f_std = f_fine.std()
        theta_mean = theta_fine.mean()
        theta_std = theta_fine.std()
        
        u_fine_norm = (u_fine - u_mean) / u_std
        u_coarse_norm = (u_coarse - u_mean) / u_std
        f_fine_norm = (f_fine - f_mean) / f_std
        theta_fine_norm = (theta_fine - theta_mean) / theta_std
        
        # Upsample coarse solution to fine grid
        u_coarse_upsampled = F.interpolate(
            torch.from_numpy(u_coarse_norm).float().unsqueeze(0).unsqueeze(0),
            size=(80, 80),
            mode='bilinear',
            align_corners=True
        )
        
        # Combine inputs
        model_input = torch.cat([
            u_coarse_upsampled,
            torch.from_numpy(theta_fine_norm).float().unsqueeze(0).unsqueeze(0),
            torch.from_numpy(f_fine_norm).float().unsqueeze(0).unsqueeze(0)
        ], dim=1).to(device)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(model_input).cpu().numpy()[0, 0]
        
        # Denormalize prediction
        prediction = prediction * u_std + u_mean
        
        # Calculate error
        error = np.abs(prediction - u_fine)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot coarse solution
        im1 = axes[0, 0].imshow(u_coarse)
        axes[0, 0].set_title('Coarse Solution (40x40)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot ground truth
        im2 = axes[0, 1].imshow(u_fine)
        axes[0, 1].set_title('Ground Truth (80x80)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot prediction
        im3 = axes[0, 2].imshow(prediction)
        axes[0, 2].set_title('ML Prediction (80x80)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Plot theta field
        im4 = axes[1, 0].imshow(theta_fine)
        axes[1, 0].set_title('Theta Field (Smooth Circle)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Plot forcing term
        im5 = axes[1, 1].imshow(f_fine)
        axes[1, 1].set_title('Forcing Term')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Plot error
        im6 = axes[1, 2].imshow(error)
        axes[1, 2].set_title(f'Error (MAE: {error.mean():.6f})')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Remove axis ticks
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path / f'circular_sample_{i+1}.png', dpi=150)
        plt.close()
        
        print(f"Generated and saved circular theta example {i+1}")
    
    print(f"All examples saved to {save_dir}")

if __name__ == '__main__':
    generate_and_save_examples(n_samples=3) 