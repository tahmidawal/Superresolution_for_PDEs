import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_grid(size=20):
    """Create a simple test grid with a known pattern."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple pattern: a Gaussian bump
    sigma = 0.1
    x0, y0 = 0.5, 0.5  # center
    Z = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    return Z

def interpolate_grid(grid, target_size, mode='bilinear'):
    """Interpolate grid to target size using specified mode."""
    grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    
    interpolated = F.interpolate(
        grid_tensor,
        size=(target_size, target_size),
        mode=mode,
        align_corners=True
    ).squeeze().numpy()
    
    return interpolated

def multi_level_interpolate(grid, target_size, mode='bilinear'):
    """Perform multi-level interpolation, doubling resolution at each step."""
    current_grid = grid
    current_size = grid.shape[0]
    
    while current_size < target_size:
        next_size = current_size * 2
        current_grid = interpolate_grid(current_grid, next_size, mode)
        current_size = next_size
        
    return current_grid

def plot_results(original, direct_bilinear, multi_bilinear, direct_cubic, multi_cubic):
    """Plot the original and interpolated grids for comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    im0 = axes[0, 0].imshow(original, origin='lower')
    axes[0, 0].set_title(f'Original ({original.shape[0]}x{original.shape[0]})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Direct bilinear
    im1 = axes[0, 1].imshow(direct_bilinear, origin='lower')
    axes[0, 1].set_title(f'Direct Bilinear ({direct_bilinear.shape[0]}x{direct_bilinear.shape[0]})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Multi-level bilinear
    im2 = axes[0, 2].imshow(multi_bilinear, origin='lower')
    axes[0, 2].set_title(f'Multi-level Bilinear ({multi_bilinear.shape[0]}x{multi_bilinear.shape[0]})')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Direct cubic
    im3 = axes[1, 0].imshow(direct_cubic, origin='lower')
    axes[1, 0].set_title(f'Direct Cubic ({direct_cubic.shape[0]}x{direct_cubic.shape[0]})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Multi-level cubic
    im4 = axes[1, 1].imshow(multi_cubic, origin='lower')
    axes[1, 1].set_title(f'Multi-level Cubic ({multi_cubic.shape[0]}x{multi_cubic.shape[0]})')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Difference between multi-level bilinear and cubic
    diff = np.abs(multi_bilinear - multi_cubic)
    im5 = axes[1, 2].imshow(diff, origin='lower')
    axes[1, 2].set_title(f'|Multi Bilinear - Multi Cubic|\nMax diff: {diff.max():.6f}')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = Path('results/interpolation_test')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(results_dir / 'interpolation_comparison.png', dpi=150)
    print(f"Plot saved to {results_dir / 'interpolation_comparison.png'}")
    
    # Also save a plot of the cross-section through the center
    plt.figure(figsize=(10, 6))
    center = original.shape[0] // 2
    target_center = direct_bilinear.shape[0] // 2
    
    # For the original, we need to sample at the right locations
    x_original = np.linspace(0, 1, original.shape[0])
    x_target = np.linspace(0, 1, direct_bilinear.shape[0])
    
    plt.plot(x_target, direct_bilinear[target_center, :], 'r-', label='Direct Bilinear')
    plt.plot(x_target, multi_bilinear[target_center, :], 'g--', label='Multi-level Bilinear')
    plt.plot(x_target, direct_cubic[target_center, :], 'b-', label='Direct Cubic')
    plt.plot(x_target, multi_cubic[target_center, :], 'm--', label='Multi-level Cubic')
    plt.plot(x_original, original[center, :], 'ko-', label='Original')
    
    plt.title('Cross-section Comparison (Horizontal Center Line)')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(results_dir / 'cross_section_comparison.png', dpi=150)
    print(f"Cross-section plot saved to {results_dir / 'cross_section_comparison.png'}")

def main():
    # Create a test grid
    original_size = 20
    target_size = 80
    
    print(f"Creating test grid of size {original_size}x{original_size}")
    test_grid = create_test_grid(original_size)
    
    # Perform direct interpolation
    print(f"Performing direct bilinear interpolation to {target_size}x{target_size}")
    direct_bilinear = interpolate_grid(test_grid, target_size, mode='bilinear')
    
    print(f"Performing direct cubic interpolation to {target_size}x{target_size}")
    direct_cubic = interpolate_grid(test_grid, target_size, mode='bicubic')
    
    # Perform multi-level interpolation
    print(f"Performing multi-level bilinear interpolation to {target_size}x{target_size}")
    multi_bilinear = multi_level_interpolate(test_grid, target_size, mode='bilinear')
    
    print(f"Performing multi-level cubic interpolation to {target_size}x{target_size}")
    multi_cubic = multi_level_interpolate(test_grid, target_size, mode='bicubic')
    
    # Compare results
    print("\nComparing interpolation methods:")
    print(f"Direct Bilinear - min: {direct_bilinear.min():.6f}, max: {direct_bilinear.max():.6f}")
    print(f"Multi Bilinear - min: {multi_bilinear.min():.6f}, max: {multi_bilinear.max():.6f}")
    print(f"Direct Cubic - min: {direct_cubic.min():.6f}, max: {direct_cubic.max():.6f}")
    print(f"Multi Cubic - min: {multi_cubic.min():.6f}, max: {multi_cubic.max():.6f}")
    
    # Calculate differences
    bilinear_diff = np.abs(direct_bilinear - multi_bilinear)
    cubic_diff = np.abs(direct_cubic - multi_cubic)
    cross_diff = np.abs(direct_bilinear - direct_cubic)
    
    print(f"\nMax difference between direct and multi-level bilinear: {bilinear_diff.max():.6f}")
    print(f"Max difference between direct and multi-level cubic: {cubic_diff.max():.6f}")
    print(f"Max difference between direct bilinear and direct cubic: {cross_diff.max():.6f}")
    
    # Plot results
    plot_results(test_grid, direct_bilinear, multi_bilinear, direct_cubic, multi_cubic)

if __name__ == "__main__":
    main() 