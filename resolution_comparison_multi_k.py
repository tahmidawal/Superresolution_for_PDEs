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
import pandas as pd
from typing import Dict, List, Tuple
import time
import json

# Reuse existing functions from resolution_comparison.py
from resolution_comparison import (
    upscale_subdomain, split_into_subdomains,
    stitch_subdomains, GlobalNormalization, ml_multi_level_upscale
)

def solve_multi_resolution_fixed_k(n_coarse: int = 40, resolutions: List[int] = [80, 160, 320, 640], k1: float = 10.0, k2: float = 10.0):
    """
    Solve PDE at multiple resolutions, starting from n_coarse.
    Uses fixed wave numbers k1 and k2 for the test.
    Returns solutions at all resolutions for comparison.
    """
    print(f"\nGenerating multi-resolution test case with k₁={k1:.2f}, k₂={k2:.2f}...")
    
    # Create solvers for each resolution
    solvers = {}
    for n_fine in resolutions:
        n_coarse_for_solver = n_fine // 2
        solvers[n_fine] = PoissonSolver(n_coarse=n_coarse_for_solver, n_fine=n_fine)
    
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

def run_single_sample(model: torch.nn.Module, device: str, k1: float, k2: float, 
                     resolutions: List[int], sample_idx: int) -> Dict:
    """
    Run a single sample with specified k values and return metrics.
    """
    print(f"\n=== Running Sample {sample_idx + 1} with k₁={k1:.2f}, k₂={k2:.2f} ===")
    
    # Generate test data with specified k values
    data = solve_multi_resolution_fixed_k(n_coarse=40, resolutions=resolutions, k1=k1, k2=k2)
    
    # Initialize solution dictionaries
    ml_solutions = {}
    bilinear_multi_solutions = {}
    bilinear_direct_solutions = {}
    
    # Store metrics for this sample
    metrics = {
        'sample_idx': sample_idx,
        'k1': k1,
        'k2': k2,
        'resolutions': resolutions,
        'ml_mae': [],
        'ml_rmse': [],
        'bilinear_multi_mae': [],
        'bilinear_multi_rmse': [],
        'bilinear_direct_mae': [],
        'bilinear_direct_rmse': []
    }
    
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
        
        # Calculate metrics
        ml_mae = np.mean(np.abs(ml_solutions[res] - data['u'][res]))
        ml_rmse = np.sqrt(np.mean((ml_solutions[res] - data['u'][res])**2))
        
        bl_multi_mae = np.mean(np.abs(bilinear_multi_solutions[res] - data['u'][res]))
        bl_multi_rmse = np.sqrt(np.mean((bilinear_multi_solutions[res] - data['u'][res])**2))
        
        bl_direct_mae = np.mean(np.abs(bilinear_direct_solutions[res] - data['u'][res]))
        bl_direct_rmse = np.sqrt(np.mean((bilinear_direct_solutions[res] - data['u'][res])**2))
        
        # Store metrics
        metrics['ml_mae'].append(ml_mae)
        metrics['ml_rmse'].append(ml_rmse)
        metrics['bilinear_multi_mae'].append(bl_multi_mae)
        metrics['bilinear_multi_rmse'].append(bl_multi_rmse)
        metrics['bilinear_direct_mae'].append(bl_direct_mae)
        metrics['bilinear_direct_rmse'].append(bl_direct_rmse)
        
        print(f"\nResults for {res}x{res}:")
        print(f"ML Multi-level - MAE: {ml_mae:.6f}, RMSE: {ml_rmse:.6f}")
        print(f"Bilinear Multi-level - MAE: {bl_multi_mae:.6f}, RMSE: {bl_multi_rmse:.6f}")
        print(f"Direct Bilinear - MAE: {bl_direct_mae:.6f}, RMSE: {bl_direct_rmse:.6f}")
    
    return metrics

def plot_combined_metrics(all_metrics: List[Dict], save_dir: Path, k_values: List[float]):
    """
    Create combined plots showing metrics from all samples grouped by k value.
    """
    # Convert metrics to DataFrame for easier analysis
    df_list = []
    
    for metrics in all_metrics:
        k1 = metrics['k1']
        k2 = metrics['k2']
        k_label = f"k₁={k1:.1f}, k₂={k2:.1f}"
        
        for i, res in enumerate(metrics['resolutions']):
            df_list.extend([
                {
                    'Resolution': res,
                    'Method': 'ML Multi-level',
                    'MAE': metrics['ml_mae'][i],
                    'RMSE': metrics['ml_rmse'][i],
                    'k1': k1,
                    'k2': k2,
                    'k_label': k_label,
                    'sample_idx': metrics['sample_idx']
                },
                {
                    'Resolution': res,
                    'Method': 'Bilinear Multi-level',
                    'MAE': metrics['bilinear_multi_mae'][i],
                    'RMSE': metrics['bilinear_multi_rmse'][i],
                    'k1': k1,
                    'k2': k2,
                    'k_label': k_label,
                    'sample_idx': metrics['sample_idx']
                },
                {
                    'Resolution': res,
                    'Method': 'Direct Bilinear',
                    'MAE': metrics['bilinear_direct_mae'][i],
                    'RMSE': metrics['bilinear_direct_rmse'][i],
                    'k1': k1,
                    'k2': k2,
                    'k_label': k_label,
                    'sample_idx': metrics['sample_idx']
                }
            ])
    
    df = pd.DataFrame(df_list)
    
    # 1. Plot MAE by resolution for each k value and method
    plt.figure(figsize=(15, 10))
    plt.title('Mean Absolute Error by Resolution and k Value', fontsize=16)
    
    # Group by k value and method
    for k in k_values:
        k_df = df[(df['k1'] == k) & (df['k2'] == k)]
        
        for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
            method_df = k_df[k_df['Method'] == method]
            
            # Calculate mean and std for each resolution
            mean_mae = method_df.groupby('Resolution')['MAE'].mean()
            std_mae = method_df.groupby('Resolution')['MAE'].std()
            
            # Plot with error bars
            label = f"{method} (k={k})"
            linestyle = '-' if method == 'ML Multi-level' else '--' if method == 'Bilinear Multi-level' else '-.'
            marker = 'o' if method == 'ML Multi-level' else 's' if method == 'Bilinear Multi-level' else '^'
            
            plt.errorbar(mean_mae.index, mean_mae.values, yerr=std_mae.values,
                        label=label, linestyle=linestyle, marker=marker, capsize=5)
    
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(df['Resolution'].unique(), [f'{r}x{r}' for r in df['Resolution'].unique()])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_mae_by_resolution.png', dpi=300)
    plt.close()
    
    # 2. Plot MAE by k value for each resolution and method
    plt.figure(figsize=(15, 10))
    plt.title('Mean Absolute Error by k Value and Resolution', fontsize=16)
    
    # Group by resolution and method
    for res in df['Resolution'].unique():
        res_df = df[df['Resolution'] == res]
        
        for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
            method_df = res_df[res_df['Method'] == method]
            
            # Calculate mean and std for each k value
            mean_mae = method_df.groupby('k1')['MAE'].mean()
            std_mae = method_df.groupby('k1')['MAE'].std()
            
            # Plot with error bars
            label = f"{method} ({res}x{res})"
            linestyle = '-' if method == 'ML Multi-level' else '--' if method == 'Bilinear Multi-level' else '-.'
            marker = 'o' if method == 'ML Multi-level' else 's' if method == 'Bilinear Multi-level' else '^'
            
            plt.errorbar(mean_mae.index, mean_mae.values, yerr=std_mae.values,
                        label=label, linestyle=linestyle, marker=marker, capsize=5)
    
    plt.xlabel('k Value', fontsize=14)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_mae_by_k_value.png', dpi=300)
    plt.close()
    
    # 3. Create a heatmap of ML advantage over bilinear methods
    plt.figure(figsize=(12, 8))
    plt.title('ML Advantage Over Bilinear Methods (Ratio of MAEs)', fontsize=16)
    
    # Calculate ML advantage ratio for each resolution and k value
    advantage_data = []
    
    for k in k_values:
        for res in df['Resolution'].unique():
            k_res_df = df[(df['k1'] == k) & (df['k2'] == k) & (df['Resolution'] == res)]
            
            ml_mae = k_res_df[k_res_df['Method'] == 'ML Multi-level']['MAE'].mean()
            bl_mae = k_res_df[k_res_df['Method'] == 'Bilinear Multi-level']['MAE'].mean()
            
            # Calculate advantage ratio (bilinear MAE / ML MAE)
            # Higher values mean ML is better
            advantage_ratio = bl_mae / ml_mae
            
            advantage_data.append({
                'k': k,
                'Resolution': res,
                'Advantage Ratio': advantage_ratio
            })
    
    advantage_df = pd.DataFrame(advantage_data)
    advantage_pivot = advantage_df.pivot(index='k', columns='Resolution', values='Advantage Ratio')
    
    # Create heatmap
    sns.heatmap(advantage_pivot, annot=True, fmt='.2f', cmap='viridis',
               cbar_kws={'label': 'Bilinear MAE / ML MAE'})
    
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('k Value', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'ml_advantage_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Create a summary table of all metrics
    summary = df.groupby(['k1', 'Resolution', 'Method']).agg({
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    summary.to_csv(save_dir / 'metrics_summary.csv')
    
    # Create a more readable text summary
    with open(save_dir / 'metrics_summary.txt', 'w') as f:
        f.write("Statistical Summary of Multiple Samples\n")
        f.write("=" * 50 + "\n\n")
        
        for k in k_values:
            f.write(f"\nResults for k₁=k₂={k:.1f}:\n")
            f.write("=" * 30 + "\n")
            
            for res in df['Resolution'].unique():
                f.write(f"\n  Resolution {res}x{res}:\n")
                f.write("  " + "-" * 25 + "\n")
                
                for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
                    method_data = df[(df['k1'] == k) & (df['k2'] == k) & 
                                    (df['Resolution'] == res) & (df['Method'] == method)]
                    
                    f.write(f"\n  {method}:\n")
                    f.write(f"    MAE:  mean = {method_data['MAE'].mean():.6f} ± {method_data['MAE'].std():.6f}\n")
                    f.write(f"          min = {method_data['MAE'].min():.6f}, max = {method_data['MAE'].max():.6f}\n")
                    f.write(f"    RMSE: mean = {method_data['RMSE'].mean():.6f} ± {method_data['RMSE'].std():.6f}\n")
                    f.write(f"          min = {method_data['RMSE'].min():.6f}, max = {method_data['RMSE'].max():.6f}\n")
    
    # 5. Save raw data as JSON for further analysis
    with open(save_dir / 'raw_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def main():
    parser = argparse.ArgumentParser(description='Run multiple samples with different k values')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples per k value')
    parser.add_argument('--results_dir', type=str, default='results/multi_k_comparison',
                       help='Directory to save results')
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
    save_dir = Path(args.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define resolutions to test
    resolutions = [80, 160, 320, 640]
    
    # Define k values to test
    k_values = [9.0, 10.0, 11.0]
    
    # Run samples for each k value
    all_metrics = []
    sample_idx = 0
    
    for k in k_values:
        for i in range(args.n_samples):
            metrics = run_single_sample(
                model=model,
                device=device,
                k1=k,
                k2=k,
                resolutions=resolutions,
                sample_idx=sample_idx
            )
            all_metrics.append(metrics)
            sample_idx += 1
    
    # Plot combined metrics
    plot_combined_metrics(all_metrics, save_dir, k_values)
    
    print(f"\nResults saved to {save_dir}")

if __name__ == "__main__":
    main() 