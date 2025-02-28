import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import load_model
import matplotlib.pyplot as plt
import argparse

def solve_large_scale(n_coarse: int = 80, n_fine: int = 160):
    """
    Solve PDE on large scale grids using higher wave numbers for testing.
    Returns both coarse and fine solutions for comparison.
    """
    print("\nGenerating large-scale test case (high frequency)...")
    solver = PoissonSolver(n_coarse=n_coarse, n_fine=n_fine)
    
    # Generate random k values from higher range (5.0, 8.0)
    k1 = np.random.uniform(5.0, 8.0)
    k2 = np.random.uniform(5.0, 8.0)
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate source term (f) on fine grid
    x = np.linspace(0, 1, n_fine)
    y = np.linspace(0, 1, n_fine)
    X, Y = np.meshgrid(x, y)
    f_fine = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    
    # Generate theta field on fine grid (random between 0.5 and 2.0, as in training)
    theta_fine = np.random.uniform(0.5, 2.0, size=(n_fine, n_fine))
    
    # Print stats for debugging
    print(f"\nInput field statistics:")
    print(f"f_fine - min: {f_fine.min():.6f}, max: {f_fine.max():.6f}")
    print(f"theta_fine - min: {theta_fine.min():.6f}, max: {theta_fine.max():.6f}")
    
    # Downsample f and theta to coarse grid
    f_coarse = f_fine[::2, ::2]
    theta_coarse = theta_fine[::2, ::2]
    
    # Solve on both grids
    print("\nSolving Poisson equation on both grids...")
    u_fine = solver.solve_poisson(f_fine, theta_fine, 'fine')
    u_coarse = solver.solve_poisson(f_coarse, theta_coarse, 'coarse')
    
    print(f"Solution statistics:")
    print(f"u_fine - min: {u_fine.min():.6f}, max: {u_fine.max():.6f}")
    print(f"u_coarse - min: {u_coarse.min():.6f}, max: {u_coarse.max():.6f}")
    
    return {
        'u_fine': u_fine,
        'u_coarse': u_coarse,
        'f_fine': f_fine,
        'f_coarse': f_coarse,
        'theta_fine': theta_fine,
        'theta_coarse': theta_coarse,
        'k1': k1,
        'k2': k2
    }

def split_into_subdomains(array: np.ndarray, subdomain_size: int) -> list:
    """
    Split a 2D array into subdomains.
    Returns a 2D list of subdomains.
    """
    n_subdomains = array.shape[0] // subdomain_size
    subdomains = []
    
    for i in range(n_subdomains):
        row = []
        for j in range(n_subdomains):
            start_i = i * subdomain_size
            start_j = j * subdomain_size
            end_i = start_i + subdomain_size
            end_j = start_j + subdomain_size
            subdomain = array[start_i:end_i, start_j:end_j]
            row.append(subdomain)
        subdomains.append(row)
    
    return subdomains

def stitch_subdomains(subdomains: list) -> np.ndarray:
    """
    Stitch subdomains back together into a single array.
    """
    n_rows = len(subdomains)
    n_cols = len(subdomains[0])
    subdomain_size = subdomains[0][0].shape[0]
    full_size = n_rows * subdomain_size
    
    result = np.zeros((full_size, full_size))
    
    for i in range(n_rows):
        for j in range(n_cols):
            start_i = i * subdomain_size
            start_j = j * subdomain_size
            end_i = start_i + subdomain_size
            end_j = start_j + subdomain_size
            result[start_i:end_i, start_j:end_j] = subdomains[i][j]
    
    return result

class GlobalNormalization:
    def __init__(self, data: dict):
        """
        Compute global normalization statistics from the full dataset.
        """
        # Convert to tensors for computation
        u_fine = torch.from_numpy(data['u_fine']).float()
        u_coarse = torch.from_numpy(data['u_coarse']).float()
        f_fine = torch.from_numpy(data['f_fine']).float()
        theta_fine = torch.from_numpy(data['theta_fine']).float()
        
        # Compute global statistics
        self.u_mean = u_fine.mean()
        self.u_std = u_fine.std()
        self.f_mean = f_fine.mean()
        self.f_std = f_fine.std()
        
        # For theta, if it's constant, don't normalize
        self.theta_is_constant = (theta_fine.std() < 1e-6)
        if self.theta_is_constant:
            self.theta_mean = 0
            self.theta_std = 1
            print("Detected constant theta field, skipping normalization")
        else:
            self.theta_mean = theta_fine.mean()
            self.theta_std = theta_fine.std()
        
        print("\nGlobal normalization statistics:")
        print(f"u_mean: {self.u_mean:.6f}, u_std: {self.u_std:.6f}")
        print(f"f_mean: {self.f_mean:.6f}, f_std: {self.f_std:.6f}")
        print(f"theta_mean: {self.theta_mean:.6f}, theta_std: {self.theta_std:.6f}")

def upscale_subdomain(model: torch.nn.Module, u_coarse: np.ndarray, 
                     f_fine: np.ndarray, theta_fine: np.ndarray,
                     global_norm: GlobalNormalization,
                     device: str) -> np.ndarray:
    """
    Apply the ML model to upscale a single subdomain using global normalization.
    """
    # Convert inputs to tensors
    u_coarse = torch.from_numpy(u_coarse).float().to(device)
    f_fine = torch.from_numpy(f_fine).float().to(device)
    theta_fine = torch.from_numpy(theta_fine).float().to(device)
    
    # Normalize using global statistics
    u_coarse_norm = (u_coarse - global_norm.u_mean) / global_norm.u_std
    f_fine_norm = (f_fine - global_norm.f_mean) / global_norm.f_std
    
    if global_norm.theta_is_constant:
        theta_fine_norm = theta_fine
    else:
        theta_fine_norm = (theta_fine - global_norm.theta_mean) / global_norm.theta_std
    
    # Upsample coarse solution
    u_coarse_upsampled = F.interpolate(
        u_coarse_norm.unsqueeze(0).unsqueeze(0),
        size=(40, 40),
        mode='bilinear',
        align_corners=True
    )
    
    # Combine inputs
    inputs = torch.cat([
        u_coarse_upsampled.squeeze(0),
        theta_fine_norm.unsqueeze(0),
        f_fine_norm.unsqueeze(0)
    ], dim=0)
    
    # Print input stats for debugging
    print(f"Input stats - min: {inputs.min().item():.6f}, max: {inputs.max().item():.6f}")
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(inputs.unsqueeze(0))
        print(f"Prediction stats (before denorm) - min: {prediction.min().item():.6f}, max: {prediction.max().item():.6f}")
        
        # Denormalize
        prediction = prediction * global_norm.u_std + global_norm.u_mean
        print(f"Prediction stats (after denorm) - min: {prediction.min().item():.6f}, max: {prediction.max().item():.6f}")
        
        prediction = prediction.squeeze().cpu().numpy()
    
    return prediction

def plot_comparison(u_coarse: np.ndarray, u_fine: np.ndarray, 
                   u_bilinear: np.ndarray, u_ml: np.ndarray,
                   theta_fine: np.ndarray, save_dir: Path):
    """
    Create detailed comparison plots.
    """
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('Large-scale Solution Comparison (160×160)\nTraining-like Parameters', fontsize=16)
    
    # Use consistent normalization
    vmin = min(u_coarse.min(), u_fine.min(), u_bilinear.min(), u_ml.min())
    vmax = max(u_coarse.max(), u_fine.max(), u_bilinear.max(), u_ml.max())
    
    # Solutions
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(u_coarse, vmin=vmin, vmax=vmax)
    ax1.set_title('Coarse Solution (80×80)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(u_fine, vmin=vmin, vmax=vmax)
    ax2.set_title('Ground Truth (160×160)')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(233)
    im3 = ax3.imshow(theta_fine, vmin=0.5, vmax=2.0)
    ax3.set_title('θ (Diffusion Coefficient)')
    plt.colorbar(im3, ax=ax3)
    
    # Upscaled solutions and errors
    ax4 = plt.subplot(234)
    im4 = ax4.imshow(u_bilinear, vmin=vmin, vmax=vmax)
    bilinear_mae = np.mean(np.abs(u_bilinear - u_fine))
    ax4.set_title(f'Bilinear (MAE: {bilinear_mae:.6f})')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(235)
    im5 = ax5.imshow(u_ml, vmin=vmin, vmax=vmax)
    ml_mae = np.mean(np.abs(u_ml - u_fine))
    ax5.set_title(f'ML Subdomain (MAE: {ml_mae:.6f})')
    plt.colorbar(im5, ax=ax5)
    
    # Error difference
    ax6 = plt.subplot(236)
    error_diff = np.abs(u_ml - u_fine) - np.abs(u_bilinear - u_fine)
    im6 = ax6.imshow(error_diff, cmap='RdBu')
    ax6.set_title('Error Difference\n(Blue: ML better, Red: Bilinear better)')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'large_scale_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.title('Error Distribution')
    
    ml_errors = np.abs(u_ml - u_fine).flatten()
    bilinear_errors = np.abs(u_bilinear - u_fine).flatten()
    
    plt.hist(ml_errors, bins=50, alpha=0.5, label=f'ML (MAE: {ml_mae:.6f})', density=True)
    plt.hist(bilinear_errors, bins=50, alpha=0.5, label=f'Bilinear (MAE: {bilinear_mae:.6f})', density=True)
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'large_scale_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cross-section plots
    plt.figure(figsize=(15, 5))
    plt.suptitle('Solution Cross-sections at y=80', fontsize=12)
    
    mid_idx = u_fine.shape[0] // 2
    x = np.linspace(0, 1, u_fine.shape[1])
    
    plt.plot(x, u_fine[mid_idx, :], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(x, u_bilinear[mid_idx, :], 'r--', label='Bilinear', linewidth=2)
    plt.plot(x, u_ml[mid_idx, :], 'b--', label='ML Model', linewidth=2)
    
    plt.xlabel('x coordinate')
    plt.ylabel('Solution value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'cross_section_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def solve_multi_level(n_coarse: int = 40, n_mid: int = 80, n_fine: int = 160):
    """
    Solve PDE starting from a very coarse grid (40x40) and upscale in two stages.
    Returns solutions at all resolutions for comparison.
    """
    print("\nGenerating multi-level test case...")
    
    # Create solvers for each resolution
    solver_fine = PoissonSolver(n_coarse=n_mid, n_fine=n_fine)    # 80x160
    solver_mid = PoissonSolver(n_coarse=n_coarse, n_fine=n_mid)   # 40x80
    
    # Generate random k values from higher range (5.0, 8.0)
    k1 = np.random.uniform(5.0, 8.0)
    k2 = np.random.uniform(5.0, 8.0)
    print(f"Using wave numbers: k₁={k1:.2f}, k₂={k2:.2f}")
    
    # Generate source term (f) on finest grid
    x = np.linspace(0, 1, n_fine)
    y = np.linspace(0, 1, n_fine)
    X, Y = np.meshgrid(x, y)
    f_fine = np.sin(k1 * 2 * np.pi * X) * np.sin(k2 * 2 * np.pi * Y)
    
    # Downsample f to mid and coarse grids
    f_mid = f_fine[::2, ::2]  # 80x80
    f_coarse = f_mid[::2, ::2]  # 40x40
    
    # Generate theta field on fine grid
    theta_fine = np.random.uniform(0.5, 2.0, size=(n_fine, n_fine))
    theta_mid = theta_fine[::2, ::2]
    theta_coarse = theta_mid[::2, ::2]
    
    # Solve on all grids
    print("\nSolving Poisson equation on all grids...")
    u_fine = solver_fine.solve_poisson(f_fine, theta_fine, 'fine')
    u_mid = solver_mid.solve_poisson(f_mid, theta_mid, 'fine')
    u_coarse = solver_mid.solve_poisson(f_coarse, theta_coarse, 'coarse')
    
    print(f"\nSolution statistics:")
    print(f"u_fine (160x160) - min: {u_fine.min():.6f}, max: {u_fine.max():.6f}")
    print(f"u_mid (80x80) - min: {u_mid.min():.6f}, max: {u_mid.max():.6f}")
    print(f"u_coarse (40x40) - min: {u_coarse.min():.6f}, max: {u_coarse.max():.6f}")
    
    return {
        'u_fine': u_fine,
        'u_mid': u_mid,
        'u_coarse': u_coarse,
        'f_fine': f_fine,
        'f_mid': f_mid,
        'f_coarse': f_coarse,
        'theta_fine': theta_fine,
        'theta_mid': theta_mid,
        'theta_coarse': theta_coarse,
        'k1': k1,
        'k2': k2
    }

def plot_multi_level_comparison(data: dict, ml_two_stage: np.ndarray, 
                              bilinear_two_stage: np.ndarray,
                              bilinear_direct: np.ndarray,
                              save_dir: Path):
    """
    Create detailed comparison plots for multi-level upscaling results.
    """
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle('Multi-level Upscaling Comparison (40×40 → 160×160)', fontsize=16)
    
    # Use consistent normalization
    solutions = [data['u_fine'], ml_two_stage, bilinear_two_stage, bilinear_direct]
    vmin = min(s.min() for s in solutions)
    vmax = max(s.max() for s in solutions)
    
    # Original solutions at different scales
    ax1 = plt.subplot(331)
    im1 = ax1.imshow(data['u_coarse'], vmin=vmin, vmax=vmax)
    ax1.set_title('Original (40×40)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(332)
    im2 = ax2.imshow(data['u_mid'], vmin=vmin, vmax=vmax)
    ax2.set_title('Ground Truth (80×80)')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(333)
    im3 = ax3.imshow(data['u_fine'], vmin=vmin, vmax=vmax)
    ax3.set_title('Ground Truth (160×160)')
    plt.colorbar(im3, ax=ax3)
    
    # Upscaled solutions
    ax4 = plt.subplot(334)
    im4 = ax4.imshow(ml_two_stage, vmin=vmin, vmax=vmax)
    ml_mae = np.mean(np.abs(ml_two_stage - data['u_fine']))
    ax4.set_title(f'ML Two-Stage\nMAE: {ml_mae:.6f}')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(335)
    im5 = ax5.imshow(bilinear_two_stage, vmin=vmin, vmax=vmax)
    bilinear_two_mae = np.mean(np.abs(bilinear_two_stage - data['u_fine']))
    ax5.set_title(f'Bilinear Two-Stage\nMAE: {bilinear_two_mae:.6f}')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = plt.subplot(336)
    im6 = ax6.imshow(bilinear_direct, vmin=vmin, vmax=vmax)
    bilinear_direct_mae = np.mean(np.abs(bilinear_direct - data['u_fine']))
    ax6.set_title(f'Bilinear Direct\nMAE: {bilinear_direct_mae:.6f}')
    plt.colorbar(im6, ax=ax6)
    
    # Error plots
    ax7 = plt.subplot(337)
    error_ml = np.abs(ml_two_stage - data['u_fine'])
    im7 = ax7.imshow(error_ml)
    ax7.set_title('ML Two-Stage Error')
    plt.colorbar(im7, ax=ax7)
    
    ax8 = plt.subplot(338)
    error_bilinear_two = np.abs(bilinear_two_stage - data['u_fine'])
    im8 = ax8.imshow(error_bilinear_two)
    ax8.set_title('Bilinear Two-Stage Error')
    plt.colorbar(im8, ax=ax8)
    
    ax9 = plt.subplot(339)
    error_bilinear_direct = np.abs(bilinear_direct - data['u_fine'])
    im9 = ax9.imshow(error_bilinear_direct)
    ax9.set_title('Bilinear Direct Error')
    plt.colorbar(im9, ax=ax9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multi_level_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distributions
    plt.figure(figsize=(12, 6))
    plt.title('Error Distribution Comparison')
    
    plt.hist(error_ml.flatten(), bins=50, alpha=0.5, 
             label=f'ML Two-Stage (MAE: {ml_mae:.6f})', density=True)
    plt.hist(error_bilinear_two.flatten(), bins=50, alpha=0.5,
             label=f'Bilinear Two-Stage (MAE: {bilinear_two_mae:.6f})', density=True)
    plt.hist(error_bilinear_direct.flatten(), bins=50, alpha=0.5,
             label=f'Bilinear Direct (MAE: {bilinear_direct_mae:.6f})', density=True)
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'multi_level_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cross-section comparison
    plt.figure(figsize=(15, 5))
    plt.suptitle('Solution Cross-sections at y=80', fontsize=12)
    
    mid_idx = data['u_fine'].shape[0] // 2
    x = np.linspace(0, 1, data['u_fine'].shape[1])
    
    plt.plot(x, data['u_fine'][mid_idx, :], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(x, ml_two_stage[mid_idx, :], 'b--', label='ML Two-Stage', linewidth=2)
    plt.plot(x, bilinear_two_stage[mid_idx, :], 'r--', label='Bilinear Two-Stage', linewidth=2)
    plt.plot(x, bilinear_direct[mid_idx, :], 'g--', label='Bilinear Direct', linewidth=2)
    
    plt.xlabel('x coordinate')
    plt.ylabel('Solution value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'multi_level_cross_section.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_multi_level_upscaling(model: torch.nn.Module, device: str, save_dir: Path):
    """
    Test multi-level upscaling approaches:
    1. ML Two-Stage: 40x40 → 80x80 → 160x160
    2. Bilinear Two-Stage: 40x40 → 80x80 → 160x160
    3. Direct Bilinear: 40x40 → 160x160
    """
    print("\n=== Testing Multi-level Upscaling ===")
    
    # Generate test case
    data = solve_multi_level()
    
    # Compute global normalization statistics for both stages
    global_norm_stage1 = GlobalNormalization({
        'u_fine': data['u_mid'],
        'u_coarse': data['u_coarse'],
        'f_fine': data['f_mid'],
        'theta_fine': data['theta_mid']
    })
    
    global_norm_stage2 = GlobalNormalization({
        'u_fine': data['u_fine'],
        'u_coarse': data['u_mid'],
        'f_fine': data['f_fine'],
        'theta_fine': data['theta_fine']
    })
    
    # ML Two-Stage Upscaling
    print("\nPerforming ML Two-Stage upscaling...")
    
    # Stage 1: 40x40 → 80x80
    coarse_subdomains = split_into_subdomains(data['u_coarse'], 20)
    f_mid_subdomains = split_into_subdomains(data['f_mid'], 40)
    theta_mid_subdomains = split_into_subdomains(data['theta_mid'], 40)
    
    stage1_subdomains = []
    for i in range(2):
        row = []
        for j in range(2):
            print(f"Stage 1 - Processing subdomain ({i+1}, {j+1})")
            prediction = upscale_subdomain(
                model,
                coarse_subdomains[i][j],
                f_mid_subdomains[i][j],
                theta_mid_subdomains[i][j],
                global_norm_stage1,
                device
            )
            row.append(prediction)
        stage1_subdomains.append(row)
    
    ml_stage1 = stitch_subdomains(stage1_subdomains)  # 80x80
    
    # Stage 2: 80x80 → 160x160
    mid_subdomains = split_into_subdomains(ml_stage1, 20)
    f_fine_subdomains = split_into_subdomains(data['f_fine'], 40)
    theta_fine_subdomains = split_into_subdomains(data['theta_fine'], 40)
    
    stage2_subdomains = []
    for i in range(4):
        row = []
        for j in range(4):
            print(f"Stage 2 - Processing subdomain ({i+1}, {j+1})")
            prediction = upscale_subdomain(
                model,
                mid_subdomains[i][j],
                f_fine_subdomains[i][j],
                theta_fine_subdomains[i][j],
                global_norm_stage2,
                device
            )
            row.append(prediction)
        stage2_subdomains.append(row)
    
    ml_two_stage = stitch_subdomains(stage2_subdomains)  # 160x160
    
    # Bilinear Two-Stage Upscaling
    print("\nPerforming Bilinear Two-Stage upscaling...")
    bilinear_stage1 = F.interpolate(
        torch.from_numpy(data['u_coarse']).float().unsqueeze(0).unsqueeze(0),
        size=(80, 80),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    bilinear_two_stage = F.interpolate(
        torch.from_numpy(bilinear_stage1).float().unsqueeze(0).unsqueeze(0),
        size=(160, 160),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Direct Bilinear Upscaling
    print("\nPerforming Direct Bilinear upscaling...")
    bilinear_direct = F.interpolate(
        torch.from_numpy(data['u_coarse']).float().unsqueeze(0).unsqueeze(0),
        size=(160, 160),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    # Calculate metrics
    ml_mae = np.mean(np.abs(ml_two_stage - data['u_fine']))
    ml_rmse = np.sqrt(np.mean((ml_two_stage - data['u_fine'])**2))
    
    bilinear_two_mae = np.mean(np.abs(bilinear_two_stage - data['u_fine']))
    bilinear_two_rmse = np.sqrt(np.mean((bilinear_two_stage - data['u_fine'])**2))
    
    bilinear_direct_mae = np.mean(np.abs(bilinear_direct - data['u_fine']))
    bilinear_direct_rmse = np.sqrt(np.mean((bilinear_direct - data['u_fine'])**2))
    
    print("\nResults:")
    print(f"ML Two-Stage - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
    print(f"Bilinear Two-Stage - MAE: {bilinear_two_mae:.6f}, RMSE: {bilinear_two_rmse:.6f}")
    print(f"Bilinear Direct - MAE: {bilinear_direct_mae:.6f}, RMSE: {bilinear_direct_rmse:.6f}")
    
    # Create comparison plots
    plot_multi_level_comparison(
        data,
        ml_two_stage,
        bilinear_two_stage,
        bilinear_direct,
        save_dir
    )

def main():
    parser = argparse.ArgumentParser(description='Large-scale subdomain-based upscaling')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
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
    results_dir = model_path.parent / 'large_scale_results'
    results_dir.mkdir(exist_ok=True)
    
    # Test original large-scale upscaling
    print("\n=== Testing Original Large-scale Upscaling ===")
    data = solve_large_scale()
    global_norm = GlobalNormalization(data)
    
    coarse_subdomains = split_into_subdomains(data['u_coarse'], 20)
    f_fine_subdomains = split_into_subdomains(data['f_fine'], 40)
    theta_fine_subdomains = split_into_subdomains(data['theta_fine'], 40)
    
    upscaled_subdomains = []
    print("\nProcessing subdomains...")
    for i in range(4):
        row = []
        for j in range(4):
            print(f"Processing subdomain ({i+1}, {j+1})")
            prediction = upscale_subdomain(
                model,
                coarse_subdomains[i][j],
                f_fine_subdomains[i][j],
                theta_fine_subdomains[i][j],
                global_norm,
                device
            )
            row.append(prediction)
        upscaled_subdomains.append(row)
    
    ml_solution = stitch_subdomains(upscaled_subdomains)
    
    bilinear_solution = F.interpolate(
        torch.from_numpy(data['u_coarse']).float().unsqueeze(0).unsqueeze(0),
        size=(160, 160),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()
    
    ml_mae = np.mean(np.abs(ml_solution - data['u_fine']))
    ml_rmse = np.sqrt(np.mean((ml_solution - data['u_fine'])**2))
    bilinear_mae = np.mean(np.abs(bilinear_solution - data['u_fine']))
    bilinear_rmse = np.sqrt(np.mean((bilinear_solution - data['u_fine'])**2))
    
    print("\nOriginal Large-scale Results:")
    print(f"ML Model (Subdomain) - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
    print(f"Bilinear - MAE: {bilinear_mae:.6f}, RMSE: {bilinear_rmse:.6f}")
    
    plot_comparison(
        data['u_coarse'],
        data['u_fine'],
        bilinear_solution,
        ml_solution,
        data['theta_fine'],
        results_dir
    )
    
    # Test multi-level upscaling
    test_multi_level_upscaling(model, device, results_dir)

if __name__ == '__main__':
    main() 