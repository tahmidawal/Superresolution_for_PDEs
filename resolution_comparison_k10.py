import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple

# Modified solve_multi_resolution function with fixed k=10
def solve_multi_resolution_k10(n_coarse: int = 40, resolutions: List[int] = [80, 160, 320, 640]):
    """
    Solve PDE at multiple resolutions, starting from n_coarse.
    Uses fixed wave numbers k1=k2=10 for high frequency test.
    Returns solutions at all resolutions for comparison.
    """
    print("\nGenerating multi-resolution test case with k=10...")
    
    # Create solvers for each resolution
    solvers = {}
    for n_fine in resolutions:
        n_coarse_for_solver = n_fine // 2
        solvers[n_fine] = PoissonSolver(n_coarse=n_coarse_for_solver, n_fine=n_fine)
    
    # Fixed k values at 10 for high frequency test
    k1 = 10.0
    k2 = 10.0
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate fields on finest grid (640x640)
    n_finest = max(resolutions)
    x = np.linspace(0, 1, n_finest)
    y = np.linspace(0, 1, n_finest)
    X, Y = np.meshgrid(x, y)
    f_finest = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    theta_finest = np.random.uniform(0.5, 2.0, size=(n_finest, n_finest))
    
    # Initialize data dictionary
    data = {
        'k1': k1,
        'k2': k2,
        'f': {},
        'theta': {},
        'u': {}
    }
    
    # Downsample and solve for each resolution
    print("\nSolving Poisson equation at all resolutions...")
    for res in [n_coarse] + resolutions:
        # Downsample f and theta
        if res == n_finest:
            data['f'][res] = f_finest
            data['theta'][res] = theta_finest
        else:
            step = n_finest // res
            data['f'][res] = f_finest[::step, ::step]
            data['theta'][res] = theta_finest[::step, ::step]
        
        # Solve PDE
        if res == n_coarse:
            solver = PoissonSolver(n_coarse=res//2, n_fine=res)
            data['u'][res] = solver.solve_poisson(
                data['f'][res], 
                data['theta'][res], 
                'fine'
            )
        else:
            data['u'][res] = solvers[res].solve_poisson(
                data['f'][res],
                data['theta'][res],
                'fine'
            )
        
        print(f"\nSolution statistics for {res}x{res}:")
        print(f"u_{res} - min: {data['u'][res].min():.6f}, max: {data['u'][res].max():.6f}")
    
    return data

# Reuse existing functions from resolution_comparison.py
from resolution_comparison import (
    upscale_subdomain, split_into_subdomains,
    stitch_subdomains, GlobalNormalization, ml_multi_level_upscale
)

def bilinear_multi_level_upscale(data: dict, target_resolution: int) -> np.ndarray:
    """
    Perform multi-level upscaling using bilinear interpolation from 40x40 to target resolution.
    Similar to ML multi-level approach but using bilinear interpolation instead.
    """
    current_res = 40
    current_solution = data['u'][current_res]
    
    while current_res < target_resolution:
        next_res = current_res * 2
        print(f"\nUpscaling {current_res}x{current_res} → {next_res}x{next_res}")
        
        # Upscale using bilinear interpolation
        current_solution = F.interpolate(
            torch.from_numpy(current_solution).float().unsqueeze(0).unsqueeze(0),
            size=(next_res, next_res),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        current_res = next_res
    
    return current_solution

def plot_enhanced_resolution_comparison_k10(data: dict, ml_solutions: dict, 
                                     bilinear_multi_solutions: dict,
                                     bilinear_direct_solutions: dict, 
                                     save_dir: Path):
    """
    Create enhanced comparison plots for each resolution, now including multi-level bilinear.
    Includes k=10 in the title for clarity.
    """
    resolutions = sorted([res for res in ml_solutions.keys()])
    
    # Create a figure with metrics for all resolutions
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    x = np.arange(len(resolutions))
    width = 0.25
    
    # Calculate MAE for each method and resolution
    ml_mae = [np.mean(np.abs(ml_solutions[res] - data['u'][res])) for res in resolutions]
    bl_multi_mae = [np.mean(np.abs(bilinear_multi_solutions[res] - data['u'][res])) for res in resolutions]
    bl_direct_mae = [np.mean(np.abs(bilinear_direct_solutions[res] - data['u'][res])) for res in resolutions]
    
    # Plot bars
    plt.bar(x - width, ml_mae, width, label='ML Multi-level')
    plt.bar(x, bl_multi_mae, width, label='Bilinear Multi-level')
    plt.bar(x + width, bl_direct_mae, width, label='Direct Bilinear')
    
    # Add value labels
    for i, v in enumerate(ml_mae):
        plt.text(i - width, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
    for i, v in enumerate(bl_multi_mae):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
    for i, v in enumerate(bl_direct_mae):
        plt.text(i + width, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    plt.xlabel('Resolution')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Upscaling Error Comparison (k₁=k₂=10.0)')
    plt.xticks(x, [f'{res}x{res}' for res in resolutions])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the comparison metrics
    plt.tight_layout()
    plt.savefig(save_dir / 'k10_comparison_metrics.png', dpi=300)
    plt.close()
    
    # Create detailed comparison plots for each resolution
    for res in resolutions:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Ground truth
        im0 = axes[0, 0].imshow(data['u'][res], cmap='viridis')
        axes[0, 0].set_title(f'Ground Truth ({res}x{res})')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # ML solution
        im1 = axes[0, 1].imshow(ml_solutions[res], cmap='viridis')
        axes[0, 1].set_title(f'ML Multi-level Solution')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # ML error
        ml_error = np.abs(ml_solutions[res] - data['u'][res])
        im2 = axes[0, 2].imshow(ml_error, cmap='hot')
        axes[0, 2].set_title(f'ML Error (MAE: {np.mean(ml_error):.6f})')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Bilinear multi-level solution
        im3 = axes[1, 0].imshow(bilinear_multi_solutions[res], cmap='viridis')
        axes[1, 0].set_title(f'Bilinear Multi-level Solution')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Bilinear multi-level error
        bl_multi_error = np.abs(bilinear_multi_solutions[res] - data['u'][res])
        im4 = axes[1, 1].imshow(bl_multi_error, cmap='hot')
        axes[1, 1].set_title(f'Bilinear Multi-level Error (MAE: {np.mean(bl_multi_error):.6f})')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Direct bilinear solution
        im5 = axes[1, 2].imshow(bilinear_direct_solutions[res], cmap='viridis')
        axes[1, 2].set_title(f'Direct Bilinear Solution')
        plt.colorbar(im5, ax=axes[1, 2])
        
        # Direct bilinear error
        bl_direct_error = np.abs(bilinear_direct_solutions[res] - data['u'][res])
        im6 = axes[2, 0].imshow(bl_direct_error, cmap='hot')
        axes[2, 0].set_title(f'Direct Bilinear Error (MAE: {np.mean(bl_direct_error):.6f})')
        plt.colorbar(im6, ax=axes[2, 0])
        
        # Error distribution plots
        axes[2, 1].hist(ml_error.flatten(), bins=50, alpha=0.7, label=f'ML (σ={np.std(ml_error):.4f})')
        axes[2, 1].hist(bl_multi_error.flatten(), bins=50, alpha=0.5, label=f'Bilinear Multi (σ={np.std(bl_multi_error):.4f})')
        axes[2, 1].set_title('Error Distribution')
        axes[2, 1].set_xlabel('Absolute Error')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        
        # RMSE comparison
        ml_rmse = np.sqrt(np.mean((ml_solutions[res] - data['u'][res])**2))
        bl_multi_rmse = np.sqrt(np.mean((bilinear_multi_solutions[res] - data['u'][res])**2))
        bl_direct_rmse = np.sqrt(np.mean((bilinear_direct_solutions[res] - data['u'][res])**2))
        
        methods = ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']
        rmse_values = [ml_rmse, bl_multi_rmse, bl_direct_rmse]
        
        axes[2, 2].bar(methods, rmse_values)
        axes[2, 2].set_title('RMSE Comparison')
        axes[2, 2].set_ylabel('Root Mean Square Error')
        for i, v in enumerate(rmse_values):
            axes[2, 2].text(i, v + 0.001, f'{v:.6f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'k10_comparison_{res}x{res}.png', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare upscaling methods with k=10')
    parser.add_argument('--model_path', type=str, default='models/unet_pde_upscaler.pth',
                        help='Path to the trained model')
    parser.add_argument('--results_dir', type=str, default='results/k10_comparison',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create results directory
    save_dir = Path(args.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    
    # Define resolutions to test
    resolutions = [80, 160, 320, 640]
    
    # Generate test data with k=10
    data = solve_multi_resolution_k10(n_coarse=40, resolutions=resolutions)
    
    # Initialize solution dictionaries
    ml_solutions = {}
    bilinear_multi_solutions = {}
    bilinear_direct_solutions = {}
    
    # Perform upscaling for each target resolution
    for res in resolutions:
        print(f"\n=== Testing upscaling to {res}x{res} with k=10 ===")
        
        # ML multi-level upscaling
        print("\nPerforming ML multi-level upscaling...")
        ml_solutions[res] = ml_multi_level_upscale(
            model, data, res, device
        )
        
        # Multi-level bilinear upscaling
        print("\nPerforming multi-level bilinear upscaling...")
        bilinear_multi_solutions[res] = bilinear_multi_level_upscale(
            data, res
        )
        
        # Direct bilinear upscaling
        print("\nPerforming direct bilinear upscaling...")
        bilinear_direct_solutions[res] = F.interpolate(
            torch.from_numpy(data['u'][40]).float().unsqueeze(0).unsqueeze(0),
            size=(res, res),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Calculate metrics
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ml_rmse = np.sqrt(np.mean((ml_solutions[res] - data['u'][res])**2))
        
        bl_multi_mae = np.mean(np.abs(bilinear_multi_solutions[res] - data['u'][res]))
        bl_multi_rmse = np.sqrt(np.mean((bilinear_multi_solutions[res] - data['u'][res])**2))
        
        bl_direct_mae = np.mean(np.abs(bilinear_direct_solutions[res] - data['u'][res]))
        bl_direct_rmse = np.sqrt(np.mean((bilinear_direct_solutions[res] - data['u'][res])**2))
        
        print(f"\nResults for {res}x{res} with k=10:")
        print(f"ML Multi-level - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
        print(f"Bilinear Multi-level - MAE: {bl_multi_mae:.6f}, RMSE: {bl_multi_rmse:.6f}")
        print(f"Direct Bilinear - MAE: {bl_direct_mae:.6f}, RMSE: {bl_direct_rmse:.6f}")
    
    # Create comparison plots
    plot_enhanced_resolution_comparison_k10(
        data, ml_solutions, bilinear_multi_solutions, bilinear_direct_solutions, save_dir
    )
    
    print(f"\nResults saved to {save_dir}")

if __name__ == "__main__":
    main() 