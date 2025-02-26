import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

def generate_poisson_problem_sin(grid_size: int, k1: float, k2: float, amplitude: float = 1.0, seed: int = None) -> Dict[str, Any]:
    """
    Generate a Poisson problem with sinusoidal source term.
    
    Args:
        grid_size: Number of grid points in each dimension
        k1: First wave number
        k2: Second wave number
        amplitude: Amplitude of the sinusoidal source
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing the source term, solution, and grid information
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Generate source term: f(x,y) = amplitude * sin(k₁ * 2πx) * sin(k₂ * 2πy)
    source = amplitude * np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    
    # Generate random diffusion coefficient (theta)
    theta = np.random.uniform(0.5, 2.0, size=(grid_size, grid_size))
    
    # Create Laplacian matrix
    h = 1.0 / (grid_size - 1)
    n2 = grid_size * grid_size
    
    # Create 1D Laplacian
    main_diag = -4 * np.ones(n2)
    off_diag = np.ones(n2-1)
    off_diag[np.arange(grid_size-1, n2-1, grid_size)] = 0  # Remove connections across boundary
    
    # Construct sparse matrix
    diagonals = [main_diag, off_diag, off_diag, np.ones(grid_size*(grid_size-1)), np.ones(grid_size*(grid_size-1))]
    offsets = [0, 1, -1, grid_size, -grid_size]
    L = diags(diagonals, offsets, shape=(n2, n2)) / (h * h)
    
    # Reshape inputs to 1D arrays
    source_flat = source.reshape(-1)
    theta_flat = theta.reshape(-1)
    
    # Modify Laplacian with theta
    L_theta = diags(theta_flat) @ L
    
    # Solve the system
    solution_flat = spsolve(L_theta, source_flat)
    solution = solution_flat.reshape((grid_size, grid_size))
    
    return {
        'source': source,
        'solution': solution,
        'theta': theta,
        'grid_size': grid_size,
        'k1': k1,
        'k2': k2
    }

if __name__ == "__main__":
    # Example usage
    problem = generate_poisson_problem_sin(grid_size=100, k1=5.0, k2=5.0)
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(problem['source'], cmap='viridis')
    plt.colorbar()
    plt.title('Source Term')
    
    plt.subplot(132)
    plt.imshow(problem['solution'], cmap='viridis')
    plt.colorbar()
    plt.title('Solution')
    
    plt.subplot(133)
    plt.imshow(problem['theta'], cmap='viridis')
    plt.colorbar()
    plt.title('Diffusion Coefficient')
    
    plt.tight_layout()
    plt.savefig('poisson_problem_example.png')
    plt.close() 