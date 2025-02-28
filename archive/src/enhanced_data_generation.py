import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import random
from data_generation import PoissonSolver

class EnhancedPoissonSolver(PoissonSolver):
    def __init__(self, n_coarse: int = 20, n_fine: int = 40, n_superfine: int = 80):
        """
        Initialize the enhanced Poisson equation solver with super-fine grid.
        
        Args:
            n_coarse: Number of grid points in each dimension for coarse grid
            n_fine: Number of grid points in each dimension for fine grid
            n_superfine: Number of grid points in each dimension for super-fine grid
        """
        super().__init__(n_coarse, n_fine)
        self.n_superfine = n_superfine
        
        # Initialize super-fine grid
        self.x_superfine = np.linspace(0, 1, n_superfine)
        self.y_superfine = np.linspace(0, 1, n_superfine)
        
        # Create meshgrid
        self.X_superfine, self.Y_superfine = np.meshgrid(self.x_superfine, self.y_superfine)
        
        # Initialize Laplacian matrix
        self.L_superfine = self._create_laplacian(n_superfine)

    def generate_forcing_term_superfine(self, k1: float, k2: float) -> np.ndarray:
        """
        Generate the forcing term f(x,y) = sin(k₁ * 2πx) * sin(k₂ * 2πy) on super-fine grid.
        
        Args:
            k1: First wave number
            k2: Second wave number
            
        Returns:
            2D array containing the forcing term values
        """
        return np.sin(2 * np.pi * k1 * self.X_superfine) * np.sin(2 * np.pi * k2 * self.Y_superfine)

    def solve_poisson_superfine(self, f: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Solve the Poisson equation -∇·(θ∇u) = f with zero Dirichlet boundary conditions on super-fine grid.
        
        Args:
            f: Forcing term
            theta: Diffusion coefficient field
            
        Returns:
            Solution u
        """
        # Reshape inputs to 1D arrays
        f_flat = f.reshape(-1)
        theta_flat = theta.reshape(-1)
        
        # Modify Laplacian with theta
        L_theta = diags(theta_flat) @ self.L_superfine
        
        # Solve the system
        u_flat = spsolve(L_theta, f_flat)
        
        return u_flat.reshape((self.n_superfine, self.n_superfine))

    def extract_subdomain(self, field: np.ndarray, start_x: int, start_y: int, size: int) -> np.ndarray:
        """
        Extract a subdomain from a larger field.
        
        Args:
            field: The field to extract from
            start_x: Starting x index
            start_y: Starting y index
            size: Size of the subdomain
            
        Returns:
            Extracted subdomain
        """
        return field[start_y:start_y+size, start_x:start_x+size]

    def downsample(self, field: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Downsample a field by a given factor.
        
        Args:
            field: The field to downsample
            factor: Downsampling factor
            
        Returns:
            Downsampled field
        """
        return field[::factor, ::factor]

    def generate_subdomain_dataset(self, n_samples: int, k_range: Tuple[float, float] = (0.5, 12.0)) -> dict:
        """
        Generate a dataset of PDE solutions from subdomains of super-fine grid.
        
        Args:
            n_samples: Number of samples to generate
            k_range: Range for random wave numbers
            
        Returns:
            Dictionary containing the dataset
        """
        dataset = {
            'u_coarse': [],
            'u_fine': [],
            'f_coarse': [],
            'f_fine': [],
            'theta_coarse': [],
            'theta_fine': [],
            'k1': [],
            'k2': [],
            'is_subdomain': []  # Flag to indicate if sample is from subdomain
        }
        
        for _ in range(n_samples):
            # Generate random wave numbers
            k1 = np.random.uniform(*k_range)
            k2 = np.random.uniform(*k_range)
            
            # Use constant theta field (1.0) instead of random
            theta_superfine = np.ones((self.n_superfine, self.n_superfine))
            
            # Generate forcing term for super-fine grid
            f_superfine = self.generate_forcing_term_superfine(k1, k2)
            
            # Solve PDE on super-fine grid
            u_superfine = self.solve_poisson_superfine(f_superfine, theta_superfine)
            
            # Select random starting point for subdomain
            max_start = self.n_superfine - self.n_fine
            start_x = np.random.randint(0, max_start)
            start_y = np.random.randint(0, max_start)
            
            # Extract subdomains
            theta_fine = self.extract_subdomain(theta_superfine, start_x, start_y, self.n_fine)
            f_fine = self.extract_subdomain(f_superfine, start_x, start_y, self.n_fine)
            u_fine = self.extract_subdomain(u_superfine, start_x, start_y, self.n_fine)
            
            # Downsample to coarse grid
            theta_coarse = self.downsample(theta_fine)
            f_coarse = self.downsample(f_fine)
            u_coarse = self.downsample(u_fine)
            
            # Store results
            dataset['u_coarse'].append(u_coarse)
            dataset['u_fine'].append(u_fine)
            dataset['f_coarse'].append(f_coarse)
            dataset['f_fine'].append(f_fine)
            dataset['theta_coarse'].append(theta_coarse)
            dataset['theta_fine'].append(theta_fine)
            dataset['k1'].append(k1)
            dataset['k2'].append(k2)
            dataset['is_subdomain'].append(True)  # Mark as subdomain sample
        
        # Convert lists to arrays
        for key in dataset:
            dataset[key] = np.array(dataset[key])
            
        return dataset

    def combine_datasets(self, dataset1: Dict, dataset2: Dict) -> Dict:
        """
        Combine two datasets.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            
        Returns:
            Combined dataset
        """
        combined_dataset = {}
        
        # Add is_subdomain flag to dataset1 if it doesn't exist
        if 'is_subdomain' not in dataset1:
            dataset1['is_subdomain'] = np.zeros(len(dataset1['u_fine']), dtype=bool)
        
        # Combine datasets
        for key in dataset1:
            if key in dataset2:
                combined_dataset[key] = np.concatenate([dataset1[key], dataset2[key]])
            else:
                combined_dataset[key] = dataset1[key]
        
        return combined_dataset

    def plot_random_samples(self, dataset: Dict, n_samples: int = 20, save_path: str = 'results/random_samples.png'):
        """
        Plot random samples from the dataset.
        
        Args:
            dataset: Dataset to plot from
            n_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        # Create a figure with subplots
        n_cols = 4  # coarse u, fine u, theta, f
        fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 4, n_samples * 3))
        
        # Select random indices
        indices = random.sample(range(len(dataset['u_fine'])), n_samples)
        
        for i, idx in enumerate(indices):
            # Plot coarse solution
            im1 = axes[i, 0].imshow(dataset['u_coarse'][idx])
            axes[i, 0].set_title(f"Coarse u {idx}" + (" (subdomain)" if dataset['is_subdomain'][idx] else ""))
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot fine solution
            im2 = axes[i, 1].imshow(dataset['u_fine'][idx])
            axes[i, 1].set_title(f"Fine u {idx}")
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot theta field
            im3 = axes[i, 2].imshow(dataset['theta_fine'][idx])
            axes[i, 2].set_title(f"Theta {idx}")
            plt.colorbar(im3, ax=axes[i, 2])
            
            # Plot forcing term
            im4 = axes[i, 3].imshow(dataset['f_fine'][idx])
            axes[i, 3].set_title(f"Forcing {idx}")
            plt.colorbar(im4, ax=axes[i, 3])
            
            # Remove axis ticks
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Ensure the directory exists
        save_dir = Path(save_path).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    # Initialize the enhanced solver
    solver = EnhancedPoissonSolver(n_coarse=20, n_fine=40, n_superfine=80)
    
    # Load existing dataset if available
    existing_dataset_path = data_dir / 'pde_dataset.npz'
    if existing_dataset_path.exists():
        print("Loading existing dataset...")
        existing_dataset = dict(np.load(existing_dataset_path))
        print(f"Loaded {len(existing_dataset['u_fine'])} existing samples")
    else:
        print("No existing dataset found. Generating standard dataset...")
        existing_dataset = solver.generate_dataset(n_samples=1000, k_range=(0.5, 5.0))
        print(f"Generated {len(existing_dataset['u_fine'])} standard samples")
    
    # Generate subdomain dataset
    n_subdomain_samples = 1000
    print(f"Generating {n_subdomain_samples} subdomain samples...")
    subdomain_dataset = solver.generate_subdomain_dataset(
        n_samples=n_subdomain_samples, 
        k_range=(0.5, 12.0)  # Wider k_range for subdomains
    )
    print(f"Generated {len(subdomain_dataset['u_fine'])} subdomain samples")
    
    # Combine datasets
    combined_dataset = solver.combine_datasets(existing_dataset, subdomain_dataset)
    print(f"Combined dataset has {len(combined_dataset['u_fine'])} samples")
    
    # Save combined dataset
    solver.save_dataset(combined_dataset)
    print("Combined dataset saved successfully!")
    
    # Plot random samples
    solver.plot_random_samples(
        combined_dataset, 
        n_samples=20, 
        save_path=str(results_dir / 'random_samples.png')
    ) 