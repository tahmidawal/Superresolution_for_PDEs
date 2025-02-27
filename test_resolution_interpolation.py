import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to Python path
sys.path.append('src')
from resolution_comparison_enhanced import bilinear_multi_level_upscale, cubic_multi_level_upscale

def create_test_grid(size=40):
    """Create a simple test grid with a known pattern."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple pattern: a Gaussian bump
    sigma = 0.1
    x0, y0 = 0.5, 0.5  # center
    Z = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    return Z

def create_test_data():
    """Create a simple test dataset for interpolation testing."""
    # Create a 40x40 grid
    u_40 = create_test_grid(40)
    
    # Create an 80x80 grid (higher resolution version of the same pattern)
    u_80 = create_test_grid(80)
    
    # Create data dictionary in the format expected by the interpolation functions
    data = {
        'u': {40: u_40, 80: u_80},
        'f': {40: np.zeros((40, 40)), 80: np.zeros((80, 80))},  # Dummy f values
        'theta': {40: np.ones((40, 40)), 80: np.ones((80, 80))}  # Dummy theta values
    }
    
    return data

def direct_interpolate(array, target_size, mode='bilinear'):
    """Directly interpolate array to target size."""
    return F.interpolate(
        torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0),
        size=(target_size, target_size),
        mode=mode,
        align_corners=True
    ).squeeze().numpy()

def compare_interpolation_methods(data, target_resolution):
    """Compare different interpolation methods."""
    # Get the coarse solution
    u_coarse = data['u'][40]
    
    # Ground truth at target resolution
    u_true = data['u'][target_resolution] if target_resolution in data['u'] else None
    
    # Direct bilinear interpolation
    direct_bilinear = direct_interpolate(u_coarse, target_resolution, mode='bilinear')
    
    # Direct cubic interpolation
    direct_cubic = direct_interpolate(u_coarse, target_resolution, mode='bicubic')
    
    # Multi-level bilinear interpolation using the actual function
    multi_bilinear = bilinear_multi_level_upscale(data, target_resolution)
    
    # Multi-level cubic interpolation using the actual function
    multi_cubic = cubic_multi_level_upscale(data, target_resolution)
    
    # Calculate differences
    bilinear_diff = np.abs(direct_bilinear - multi_bilinear)
    cubic_diff = np.abs(direct_cubic - multi_cubic)
    cross_diff = np.abs(direct_bilinear - direct_cubic)
    
    print(f"\nResults for {target_resolution}x{target_resolution}:")
    print(f"Direct Bilinear - min: {direct_bilinear.min():.6f}, max: {direct_bilinear.max():.6f}")
    print(f"Multi Bilinear - min: {multi_bilinear.min():.6f}, max: {multi_bilinear.max():.6f}")
    print(f"Direct Cubic - min: {direct_cubic.min():.6f}, max: {direct_cubic.max():.6f}")
    print(f"Multi Cubic - min: {multi_cubic.min():.6f}, max: {multi_cubic.max():.6f}")
    
    print(f"\nMax difference between direct and multi-level bilinear: {bilinear_diff.max():.6f}")
    print(f"Max difference between direct and multi-level cubic: {cubic_diff.max():.6f}")
    print(f"Max difference between direct bilinear and direct cubic: {cross_diff.max():.6f}")
    
    # If we have ground truth, calculate errors
    if u_true is not None:
        direct_bilinear_error = np.abs(direct_bilinear - u_true)
        multi_bilinear_error = np.abs(multi_bilinear - u_true)
        direct_cubic_error = np.abs(direct_cubic - u_true)
        multi_cubic_error = np.abs(multi_cubic - u_true)
        
        print(f"\nErrors compared to ground truth:")
        print(f"Direct Bilinear - MAE: {direct_bilinear_error.mean():.6f}, Max: {direct_bilinear_error.max():.6f}")
        print(f"Multi Bilinear - MAE: {multi_bilinear_error.mean():.6f}, Max: {multi_bilinear_error.max():.6f}")
        print(f"Direct Cubic - MAE: {direct_cubic_error.mean():.6f}, Max: {direct_cubic_error.max():.6f}")
        print(f"Multi Cubic - MAE: {multi_cubic_error.mean():.6f}, Max: {multi_cubic_error.max():.6f}")
    
    # Plot results
    plot_results(u_coarse, direct_bilinear, multi_bilinear, direct_cubic, multi_cubic, u_true, target_resolution)
    
    return {
        'direct_bilinear': direct_bilinear,
        'multi_bilinear': multi_bilinear,
        'direct_cubic': direct_cubic,
        'multi_cubic': multi_cubic,
        'ground_truth': u_true
    }

def plot_results(u_coarse, direct_bilinear, multi_bilinear, direct_cubic, multi_cubic, u_true, target_resolution):
    """Plot the results of different interpolation methods."""
    n_plots = 6 if u_true is not None else 5
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Determine common color scale
    vmin = min(u_coarse.min(), direct_bilinear.min(), multi_bilinear.min(), 
              direct_cubic.min(), multi_cubic.min())
    vmax = max(u_coarse.max(), direct_bilinear.max(), multi_bilinear.max(),
              direct_cubic.max(), multi_cubic.max())
    
    if u_true is not None:
        vmin = min(vmin, u_true.min())
        vmax = max(vmax, u_true.max())
    
    # Original coarse
    im0 = axes[0].imshow(u_coarse, origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Original Coarse (40x40)')
    plt.colorbar(im0, ax=axes[0])
    
    # Direct bilinear
    im1 = axes[1].imshow(direct_bilinear, origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Direct Bilinear ({target_resolution}x{target_resolution})')
    plt.colorbar(im1, ax=axes[1])
    
    # Multi-level bilinear
    im2 = axes[2].imshow(multi_bilinear, origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Multi-level Bilinear ({target_resolution}x{target_resolution})')
    plt.colorbar(im2, ax=axes[2])
    
    # Direct cubic
    im3 = axes[3].imshow(direct_cubic, origin='lower', vmin=vmin, vmax=vmax)
    axes[3].set_title(f'Direct Cubic ({target_resolution}x{target_resolution})')
    plt.colorbar(im3, ax=axes[3])
    
    # Multi-level cubic
    im4 = axes[4].imshow(multi_cubic, origin='lower', vmin=vmin, vmax=vmax)
    axes[4].set_title(f'Multi-level Cubic ({target_resolution}x{target_resolution})')
    plt.colorbar(im4, ax=axes[4])
    
    # Ground truth or difference plot
    if u_true is not None:
        im5 = axes[5].imshow(u_true, origin='lower', vmin=vmin, vmax=vmax)
        axes[5].set_title(f'Ground Truth ({target_resolution}x{target_resolution})')
        plt.colorbar(im5, ax=axes[5])
    else:
        # Show difference between multi-level methods
        diff = np.abs(multi_bilinear - multi_cubic)
        im5 = axes[5].imshow(diff, origin='lower')
        axes[5].set_title(f'|Multi Bilinear - Multi Cubic|\nMax diff: {diff.max():.6f}')
        plt.colorbar(im5, ax=axes[5])
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = Path('results/resolution_interpolation_test')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(results_dir / f'interpolation_comparison_{target_resolution}.png', dpi=150)
    print(f"Plot saved to {results_dir / f'interpolation_comparison_{target_resolution}.png'}")
    
    # Also create error plots if ground truth is available
    if u_true is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Calculate errors
        errors = [
            np.abs(direct_bilinear - u_true),
            np.abs(multi_bilinear - u_true),
            np.abs(direct_cubic - u_true),
            np.abs(multi_cubic - u_true)
        ]
        
        titles = [
            f'Direct Bilinear Error\nMAE: {errors[0].mean():.6f}',
            f'Multi-level Bilinear Error\nMAE: {errors[1].mean():.6f}',
            f'Direct Cubic Error\nMAE: {errors[2].mean():.6f}',
            f'Multi-level Cubic Error\nMAE: {errors[3].mean():.6f}'
        ]
        
        # Common color scale for errors
        error_vmax = max([e.max() for e in errors])
        
        for i, (error, title) in enumerate(zip(errors, titles)):
            im = axes[i].imshow(error, origin='lower', vmax=error_vmax)
            axes[i].set_title(title)
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(results_dir / f'error_comparison_{target_resolution}.png', dpi=150)
        print(f"Error plot saved to {results_dir / f'error_comparison_{target_resolution}.png'}")

def main():
    # Create test data
    print("Creating test data...")
    data = create_test_data()
    
    # Test interpolation to 80x80 (we have ground truth)
    print("\nTesting interpolation to 80x80...")
    results_80 = compare_interpolation_methods(data, 80)
    
    # Test interpolation to 160x160 (no ground truth)
    print("\nTesting interpolation to 160x160...")
    results_160 = compare_interpolation_methods(data, 160)

if __name__ == "__main__":
    main() 