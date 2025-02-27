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

# Reuse existing functions
from resolution_comparison import (
    solve_multi_resolution, upscale_subdomain, split_into_subdomains,
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

def cubic_multi_level_upscale(data: dict, target_resolution: int) -> np.ndarray:
    """
    Perform multi-level upscaling using bicubic interpolation from 40x40 to target resolution.
    Similar to ML multi-level approach but using bicubic interpolation instead.
    """
    current_res = 40
    current_solution = data['u'][current_res]
    
    while current_res < target_resolution:
        next_res = current_res * 2
        print(f"\nUpscaling {current_res}x{current_res} → {next_res}x{next_res}")
        
        # Upscale using bicubic interpolation
        current_solution = F.interpolate(
            torch.from_numpy(current_solution).float().unsqueeze(0).unsqueeze(0),
            size=(next_res, next_res),
            mode='bicubic',
            align_corners=True
        ).squeeze().numpy()
        
        current_res = next_res
    
    return current_solution

def plot_enhanced_resolution_comparison(data: dict, ml_solutions: dict, 
                                     bilinear_multi_solutions: dict,
                                     bilinear_direct_solutions: dict,
                                     cubic_multi_solutions: dict,
                                     cubic_direct_solutions: dict,
                                     save_dir: Path):
    """
    Create enhanced comparison plots for each resolution, now including multi-level bilinear and cubic.
    """
    resolutions = sorted([res for res in ml_solutions.keys()])
    
    # Plot error metrics vs resolution
    plt.figure(figsize=(14, 10))
    plt.title('Error Metrics vs Resolution', fontsize=14)
    
    ml_maes = []
    ml_rmses = []
    bilinear_multi_maes = []
    bilinear_multi_rmses = []
    bilinear_direct_maes = []
    bilinear_direct_rmses = []
    cubic_multi_maes = []
    cubic_multi_rmses = []
    cubic_direct_maes = []
    cubic_direct_rmses = []
    
    for res in resolutions:
        # ML metrics
        ml_error = np.abs(ml_solutions[res] - data['u'][res])
        ml_mae = np.mean(ml_error)
        ml_rmse = np.sqrt(np.mean(ml_error**2))
        ml_maes.append(ml_mae)
        ml_rmses.append(ml_rmse)
        
        # Multi-level bilinear metrics
        bl_multi_error = np.abs(bilinear_multi_solutions[res] - data['u'][res])
        bl_multi_mae = np.mean(bl_multi_error)
        bl_multi_rmse = np.sqrt(np.mean(bl_multi_error**2))
        bilinear_multi_maes.append(bl_multi_mae)
        bilinear_multi_rmses.append(bl_multi_rmse)
        
        # Direct bilinear metrics
        bl_direct_error = np.abs(bilinear_direct_solutions[res] - data['u'][res])
        bl_direct_mae = np.mean(bl_direct_error)
        bl_direct_rmse = np.sqrt(np.mean(bl_direct_error**2))
        bilinear_direct_maes.append(bl_direct_mae)
        bilinear_direct_rmses.append(bl_direct_rmse)
        
        # Multi-level cubic metrics
        cubic_multi_error = np.abs(cubic_multi_solutions[res] - data['u'][res])
        cubic_multi_mae = np.mean(cubic_multi_error)
        cubic_multi_rmse = np.sqrt(np.mean(cubic_multi_error**2))
        cubic_multi_maes.append(cubic_multi_mae)
        cubic_multi_rmses.append(cubic_multi_rmse)
        
        # Direct cubic metrics
        cubic_direct_error = np.abs(cubic_direct_solutions[res] - data['u'][res])
        cubic_direct_mae = np.mean(cubic_direct_error)
        cubic_direct_rmse = np.sqrt(np.mean(cubic_direct_error**2))
        cubic_direct_maes.append(cubic_direct_mae)
        cubic_direct_rmses.append(cubic_direct_rmse)
    
    # Plot metrics
    plt.plot(resolutions, ml_maes, 'bo-', label='ML Multi-level MAE', linewidth=2)
    plt.plot(resolutions, ml_rmses, 'b^--', label='ML Multi-level RMSE', linewidth=2)
    plt.plot(resolutions, bilinear_multi_maes, 'go-', label='Bilinear Multi-level MAE', linewidth=2)
    plt.plot(resolutions, bilinear_multi_rmses, 'g^--', label='Bilinear Multi-level RMSE', linewidth=2)
    plt.plot(resolutions, bilinear_direct_maes, 'ro-', label='Direct Bilinear MAE', linewidth=2)
    plt.plot(resolutions, bilinear_direct_rmses, 'r^--', label='Direct Bilinear RMSE', linewidth=2)
    plt.plot(resolutions, cubic_multi_maes, 'mo-', label='Cubic Multi-level MAE', linewidth=2)
    plt.plot(resolutions, cubic_multi_rmses, 'm^--', label='Cubic Multi-level RMSE', linewidth=2)
    plt.plot(resolutions, cubic_direct_maes, 'co-', label='Direct Cubic MAE', linewidth=2)
    plt.plot(resolutions, cubic_direct_rmses, 'c^--', label='Direct Cubic RMSE', linewidth=2)
    
    plt.xlabel('Resolution', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions])
    
    # Add value labels for MAE (only for ML, best bilinear and best cubic to avoid clutter)
    for i, res in enumerate(resolutions):
        plt.text(res, ml_maes[i], f'{ml_maes[i]:.6f}', 
                verticalalignment='bottom', horizontalalignment='right')
        
        # Add only the best bilinear method
        best_bilinear_mae = min(bilinear_multi_maes[i], bilinear_direct_maes[i])
        if best_bilinear_mae == bilinear_multi_maes[i]:
            plt.text(res, bilinear_multi_maes[i], f'{bilinear_multi_maes[i]:.6f}',
                    verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.text(res, bilinear_direct_maes[i], f'{bilinear_direct_maes[i]:.6f}',
                    verticalalignment='bottom', horizontalalignment='right')
        
        # Add only the best cubic method
        best_cubic_mae = min(cubic_multi_maes[i], cubic_direct_maes[i])
        if best_cubic_mae == cubic_multi_maes[i]:
            plt.text(res, cubic_multi_maes[i], f'{cubic_multi_maes[i]:.6f}',
                    verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.text(res, cubic_direct_maes[i], f'{cubic_direct_maes[i]:.6f}',
                    verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'resolution_comparison_metrics_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison plots for each resolution
    for res in resolutions:
        fig = plt.figure(figsize=(20, 20))
        plt.suptitle(f'Solution Comparison at {res}x{res}', fontsize=16)
        
        # Use consistent normalization
        vmin = min(
            data['u'][res].min(), 
            ml_solutions[res].min(), 
            bilinear_multi_solutions[res].min(),
            bilinear_direct_solutions[res].min(),
            cubic_multi_solutions[res].min(),
            cubic_direct_solutions[res].min()
        )
        vmax = max(
            data['u'][res].max(), 
            ml_solutions[res].max(),
            bilinear_multi_solutions[res].max(),
            bilinear_direct_solutions[res].max(),
            cubic_multi_solutions[res].max(),
            cubic_direct_solutions[res].max()
        )
        
        # Ground truth
        ax1 = plt.subplot(3, 3, 1)
        im1 = ax1.imshow(data['u'][res], vmin=vmin, vmax=vmax)
        ax1.set_title(f'Ground Truth ({res}x{res})')
        plt.colorbar(im1, ax=ax1)
        
        # ML solution
        ax2 = plt.subplot(3, 3, 2)
        im2 = ax2.imshow(ml_solutions[res], vmin=vmin, vmax=vmax)
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ax2.set_title(f'ML Multi-level\nMAE: {ml_mae:.6f}')
        plt.colorbar(im2, ax=ax2)
        
        # Multi-level bilinear solution
        ax3 = plt.subplot(3, 3, 3)
        im3 = ax3.imshow(bilinear_multi_solutions[res], vmin=vmin, vmax=vmax)
        bl_multi_mae = np.mean(np.abs(bilinear_multi_solutions[res] - data['u'][res]))
        ax3.set_title(f'Bilinear Multi-level\nMAE: {bl_multi_mae:.6f}')
        plt.colorbar(im3, ax=ax3)
        
        # Direct bilinear solution
        ax4 = plt.subplot(3, 3, 4)
        im4 = ax4.imshow(bilinear_direct_solutions[res], vmin=vmin, vmax=vmax)
        bl_direct_mae = np.mean(np.abs(bilinear_direct_solutions[res] - data['u'][res]))
        ax4.set_title(f'Direct Bilinear\nMAE: {bl_direct_mae:.6f}')
        plt.colorbar(im4, ax=ax4)
        
        # Multi-level cubic solution
        ax5 = plt.subplot(3, 3, 5)
        im5 = ax5.imshow(cubic_multi_solutions[res], vmin=vmin, vmax=vmax)
        cubic_multi_mae = np.mean(np.abs(cubic_multi_solutions[res] - data['u'][res]))
        ax5.set_title(f'Cubic Multi-level\nMAE: {cubic_multi_mae:.6f}')
        plt.colorbar(im5, ax=ax5)
        
        # Direct cubic solution
        ax6 = plt.subplot(3, 3, 6)
        im6 = ax6.imshow(cubic_direct_solutions[res], vmin=vmin, vmax=vmax)
        cubic_direct_mae = np.mean(np.abs(cubic_direct_solutions[res] - data['u'][res]))
        ax6.set_title(f'Direct Cubic\nMAE: {cubic_direct_mae:.6f}')
        plt.colorbar(im6, ax=ax6)
        
        # Error plots - ML vs best traditional method
        ax7 = plt.subplot(3, 3, 7)
        error_ml = np.abs(ml_solutions[res] - data['u'][res])
        im7 = ax7.imshow(error_ml)
        ax7.set_title('ML Error')
        plt.colorbar(im7, ax=ax7)
        
        # Best bilinear error
        ax8 = plt.subplot(3, 3, 8)
        best_bilinear_error = np.minimum(
            np.abs(bilinear_multi_solutions[res] - data['u'][res]),
            np.abs(bilinear_direct_solutions[res] - data['u'][res])
        )
        im8 = ax8.imshow(best_bilinear_error)
        ax8.set_title('Best Bilinear Error')
        plt.colorbar(im8, ax=ax8)
        
        # Best cubic error
        ax9 = plt.subplot(3, 3, 9)
        best_cubic_error = np.minimum(
            np.abs(cubic_multi_solutions[res] - data['u'][res]),
            np.abs(cubic_direct_solutions[res] - data['u'][res])
        )
        im9 = ax9.imshow(best_cubic_error)
        ax9.set_title('Best Cubic Error')
        plt.colorbar(im9, ax=ax9)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'comparison_enhanced_{res}x{res}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create error distribution plot
        plt.figure(figsize=(14, 10))
        plt.title(f'Error Distribution at {res}x{res}', fontsize=14)
        
        sns.kdeplot(data=error_ml.flatten(), 
                   label=f'ML Multi-level (MAE: {ml_mae:.6f})',
                   fill=True, alpha=0.3)
        sns.kdeplot(data=np.abs(bilinear_multi_solutions[res] - data['u'][res]).flatten(),
                   label=f'Bilinear Multi-level (MAE: {bl_multi_mae:.6f})',
                   fill=True, alpha=0.3)
        sns.kdeplot(data=np.abs(bilinear_direct_solutions[res] - data['u'][res]).flatten(),
                   label=f'Direct Bilinear (MAE: {bl_direct_mae:.6f})',
                   fill=True, alpha=0.3)
        sns.kdeplot(data=np.abs(cubic_multi_solutions[res] - data['u'][res]).flatten(),
                   label=f'Cubic Multi-level (MAE: {cubic_multi_mae:.6f})',
                   fill=True, alpha=0.3)
        sns.kdeplot(data=np.abs(cubic_direct_solutions[res] - data['u'][res]).flatten(),
                   label=f'Direct Cubic (MAE: {cubic_direct_mae:.6f})',
                   fill=True, alpha=0.3)
        
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add statistical information
        ml_std = np.std(error_ml)
        bl_multi_std = np.std(np.abs(bilinear_multi_solutions[res] - data['u'][res]))
        bl_direct_std = np.std(np.abs(bilinear_direct_solutions[res] - data['u'][res]))
        cubic_multi_std = np.std(np.abs(cubic_multi_solutions[res] - data['u'][res]))
        cubic_direct_std = np.std(np.abs(cubic_direct_solutions[res] - data['u'][res]))
        
        plt.text(0.98, 0.95,
                f'ML Std: {ml_std:.6f}\n' +
                f'Bilinear Multi-level Std: {bl_multi_std:.6f}\n' +
                f'Direct Bilinear Std: {bl_direct_std:.6f}\n' +
                f'Cubic Multi-level Std: {cubic_multi_std:.6f}\n' +
                f'Direct Cubic Std: {cubic_direct_std:.6f}',
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / f'error_distribution_enhanced_{res}x{res}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced multi-resolution upscaling comparison')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    model.eval()
    
    # Create results directory
    results_dir = model_path.parent / 'resolution_comparison_enhanced_results'
    results_dir.mkdir(exist_ok=True)
    
    # Define resolutions to test
    resolutions = [80, 160, 320, 640]
    
    # Generate test data
    data = solve_multi_resolution(n_coarse=40, resolutions=resolutions)
    
    # Initialize solution dictionaries
    ml_solutions = {}
    bilinear_multi_solutions = {}
    bilinear_direct_solutions = {}
    cubic_multi_solutions = {}
    cubic_direct_solutions = {}
    
    # Perform upscaling for each target resolution
    for res in resolutions:
        print(f"\n=== Testing upscaling to {res}x{res} ===")
        
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
        
        # Multi-level cubic upscaling
        print("\nPerforming multi-level cubic upscaling...")
        cubic_multi_solutions[res] = cubic_multi_level_upscale(
            data, res
        )
        
        # Direct cubic upscaling
        print("\nPerforming direct cubic upscaling...")
        cubic_direct_solutions[res] = F.interpolate(
            torch.from_numpy(data['u'][40]).float().unsqueeze(0).unsqueeze(0),
            size=(res, res),
            mode='bicubic',
            align_corners=True
        ).squeeze().numpy()
        
        # Calculate metrics
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ml_rmse = np.sqrt(np.mean((ml_solutions[res] - data['u'][res])**2))
        
        bl_multi_mae = np.mean(np.abs(bilinear_multi_solutions[res] - data['u'][res]))
        bl_multi_rmse = np.sqrt(np.mean((bilinear_multi_solutions[res] - data['u'][res])**2))
        
        bl_direct_mae = np.mean(np.abs(bilinear_direct_solutions[res] - data['u'][res]))
        bl_direct_rmse = np.sqrt(np.mean((bilinear_direct_solutions[res] - data['u'][res])**2))
        
        cubic_multi_mae = np.mean(np.abs(cubic_multi_solutions[res] - data['u'][res]))
        cubic_multi_rmse = np.sqrt(np.mean((cubic_multi_solutions[res] - data['u'][res])**2))
        
        cubic_direct_mae = np.mean(np.abs(cubic_direct_solutions[res] - data['u'][res]))
        cubic_direct_rmse = np.sqrt(np.mean((cubic_direct_solutions[res] - data['u'][res])**2))
        
        print(f"\nResults for {res}x{res}:")
        print(f"ML Multi-level - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
        print(f"Bilinear Multi-level - MAE: {bl_multi_mae:.6f}, RMSE: {bl_multi_rmse:.6f}")
        print(f"Direct Bilinear - MAE: {bl_direct_mae:.6f}, RMSE: {bl_direct_rmse:.6f}")
        print(f"Cubic Multi-level - MAE: {cubic_multi_mae:.6f}, RMSE: {cubic_multi_rmse:.6f}")
        print(f"Direct Cubic - MAE: {cubic_direct_mae:.6f}, RMSE: {cubic_direct_rmse:.6f}")
    
    # Create comparison plots
    plot_enhanced_resolution_comparison(
        data, ml_solutions, bilinear_multi_solutions, 
        bilinear_direct_solutions, cubic_multi_solutions,
        cubic_direct_solutions, results_dir
    )

if __name__ == '__main__':
    main() 