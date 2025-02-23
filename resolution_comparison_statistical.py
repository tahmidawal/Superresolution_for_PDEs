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
import pandas as pd
from datetime import datetime

# Reuse existing functions
from resolution_comparison_enhanced import (
    solve_multi_resolution, bilinear_multi_level_upscale,
    plot_enhanced_resolution_comparison, ml_multi_level_upscale,
    split_into_subdomains,
    stitch_subdomains, 
    GlobalNormalization
)

def solve_multi_resolution(n_coarse: int = 40, resolutions: List[int] = [80, 160, 320, 640]):
    """
    Solve PDE at multiple resolutions, starting from n_coarse.
    Returns solutions at all resolutions for comparison.
    """
    print("\nGenerating multi-resolution test case...")
    
    # Create solvers for each resolution
    solvers = {}
    for n_fine in resolutions:
        n_coarse_for_solver = n_fine // 2
        solvers[n_fine] = PoissonSolver(n_coarse=n_coarse_for_solver, n_fine=n_fine)
    
    # Generate random k values with higher frequencies for more challenging test
    k1 = np.random.uniform(8.0, 12.0)  # Increased from (5.0, 8.0)
    k2 = np.random.uniform(8.0, 12.0)  # Increased from (5.0, 8.0)
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

def run_single_example(model: torch.nn.Module, device: str, 
                      example_idx: int, save_dir: Path,
                      resolutions: List[int]) -> Dict:
    """
    Run a single example and return the metrics.
    """
    print(f"\n=== Running Example {example_idx + 1}/10 ===")
    
    # Create directory for this example
    example_dir = save_dir / f'example_{example_idx + 1}'
    example_dir.mkdir(exist_ok=True)
    
    # Generate test data
    data = solve_multi_resolution(n_coarse=40, resolutions=resolutions)
    
    # Initialize solution dictionaries
    ml_solutions = {}
    bilinear_multi_solutions = {}
    bilinear_direct_solutions = {}
    
    # Store metrics for this example
    metrics = {
        'example': example_idx + 1,
        'k1': data['k1'],
        'k2': data['k2'],
        'resolutions': [],
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
        metrics['resolutions'].append(res)
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
    
    # Create comparison plots for this example
    plot_enhanced_resolution_comparison(
        data, ml_solutions, bilinear_multi_solutions, 
        bilinear_direct_solutions, example_dir
    )
    
    return metrics

def plot_combined_metrics(all_metrics: List[Dict], save_dir: Path):
    """
    Create a combined plot showing metrics from all examples in one figure.
    """
    plt.figure(figsize=(15, 10))
    plt.title('Error Metrics Across All Examples', fontsize=14)
    
    # Plot each example with different line styles but same colors per method
    colors = {'ML Multi-level': 'blue', 'Bilinear Multi-level': 'green', 'Direct Bilinear': 'red'}
    
    for example_idx, metrics in enumerate(all_metrics):
        resolutions = metrics['resolutions']
        
        # ML metrics
        plt.plot(resolutions, metrics['ml_mae'], 
                color=colors['ML Multi-level'], alpha=0.3,
                linestyle='-', marker='o',
                label=f'ML Multi-level (Ex {example_idx+1})' if example_idx == 0 else None)
        
        # Bilinear multi-level metrics
        plt.plot(resolutions, metrics['bilinear_multi_mae'],
                color=colors['Bilinear Multi-level'], alpha=0.3,
                linestyle='-', marker='s',
                label=f'Bilinear Multi-level (Ex {example_idx+1})' if example_idx == 0 else None)
        
        # Direct bilinear metrics
        plt.plot(resolutions, metrics['bilinear_direct_mae'],
                color=colors['Direct Bilinear'], alpha=0.3,
                linestyle='-', marker='^',
                label=f'Direct Bilinear (Ex {example_idx+1})' if example_idx == 0 else None)
    
    # Plot mean values with thicker lines
    df_list = []
    for metrics in all_metrics:
        for i, res in enumerate(metrics['resolutions']):
            df_list.extend([
                {
                    'Resolution': res,
                    'Method': 'ML Multi-level',
                    'MAE': metrics['ml_mae'][i]
                },
                {
                    'Resolution': res,
                    'Method': 'Bilinear Multi-level',
                    'MAE': metrics['bilinear_multi_mae'][i]
                },
                {
                    'Resolution': res,
                    'Method': 'Direct Bilinear',
                    'MAE': metrics['bilinear_direct_mae'][i]
                }
            ])
    
    df = pd.DataFrame(df_list)
    
    for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
        method_data = df[df['Method'] == method]
        mean_mae = method_data.groupby('Resolution')['MAE'].mean()
        std_mae = method_data.groupby('Resolution')['MAE'].std()
        
        plt.plot(mean_mae.index, mean_mae.values,
                color=colors[method], linewidth=3, linestyle='-',
                label=f'{method} (Mean)')
        
        # Add error bands
        plt.fill_between(mean_mae.index,
                        mean_mae.values - std_mae.values,
                        mean_mae.values + std_mae.values,
                        color=colors[method], alpha=0.2)
    
    plt.xlabel('Resolution', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions])
    
    # Add value labels for mean values
    for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
        method_data = df[df['Method'] == method]
        mean_mae = method_data.groupby('Resolution')['MAE'].mean()
        for res, val in mean_mae.items():
            plt.text(res, val, f'{val:.6f}',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    color=colors[method])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_analysis(all_metrics: List[Dict], save_dir: Path):
    """
    Create statistical analysis plots from all examples.
    """
    # Create combined metrics plot
    plot_combined_metrics(all_metrics, save_dir)
    
    # Convert metrics to DataFrame for easier analysis
    df_list = []
    for metrics in all_metrics:
        for i, res in enumerate(metrics['resolutions']):
            df_list.append({
                'Example': metrics['example'],
                'Resolution': res,
                'k1': metrics['k1'],
                'k2': metrics['k2'],
                'Method': 'ML Multi-level',
                'MAE': metrics['ml_mae'][i],
                'RMSE': metrics['ml_rmse'][i]
            })
            df_list.append({
                'Example': metrics['example'],
                'Resolution': res,
                'k1': metrics['k1'],
                'k2': metrics['k2'],
                'Method': 'Bilinear Multi-level',
                'MAE': metrics['bilinear_multi_mae'][i],
                'RMSE': metrics['bilinear_multi_rmse'][i]
            })
            df_list.append({
                'Example': metrics['example'],
                'Resolution': res,
                'k1': metrics['k1'],
                'k2': metrics['k2'],
                'Method': 'Direct Bilinear',
                'MAE': metrics['bilinear_direct_mae'][i],
                'RMSE': metrics['bilinear_direct_rmse'][i]
            })
    
    df = pd.DataFrame(df_list)
    
    # 1. Box plots for each resolution and method
    plt.figure(figsize=(15, 10))
    plt.title('MAE Distribution Across Examples', fontsize=14)
    
    sns.boxplot(data=df, x='Resolution', y='MAE', hue='Method')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mae_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean and standard deviation plot
    plt.figure(figsize=(15, 10))
    plt.title('Mean MAE with Standard Deviation', fontsize=14)
    
    for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
        method_data = df[df['Method'] == method]
        mean_mae = method_data.groupby('Resolution')['MAE'].mean()
        std_mae = method_data.groupby('Resolution')['MAE'].std()
        
        plt.errorbar(mean_mae.index, mean_mae.values, yerr=std_mae.values,
                    label=method, marker='o', capsize=5)
    
    plt.xlabel('Resolution', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mae_mean_std.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance relative to wave numbers
    plt.figure(figsize=(15, 10))
    plt.title('MAE vs Total Wave Number (k₁ + k₂)', fontsize=14)
    
    for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
        method_data = df[df['Method'] == method]
        plt.scatter(method_data['k1'] + method_data['k2'], method_data['MAE'],
                   label=method, alpha=0.6)
    
    plt.xlabel('Total Wave Number (k₁ + k₂)', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'mae_vs_wave_numbers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistical summary
    summary = df.groupby(['Resolution', 'Method']).agg({
        'MAE': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    summary.to_csv(save_dir / 'statistical_summary.csv')
    
    # Create a more readable text summary
    with open(save_dir / 'statistical_summary.txt', 'w') as f:
        f.write("Statistical Summary of 10 Examples\n")
        f.write("=" * 50 + "\n\n")
        
        for res in df['Resolution'].unique():
            f.write(f"\nResults for {res}x{res} Resolution:\n")
            f.write("-" * 40 + "\n")
            
            for method in ['ML Multi-level', 'Bilinear Multi-level', 'Direct Bilinear']:
                method_data = df[(df['Resolution'] == res) & (df['Method'] == method)]
                
                f.write(f"\n{method}:\n")
                f.write(f"  MAE:  mean = {method_data['MAE'].mean():.6f} ± {method_data['MAE'].std():.6f}\n")
                f.write(f"        min = {method_data['MAE'].min():.6f}, max = {method_data['MAE'].max():.6f}\n")
                f.write(f"  RMSE: mean = {method_data['RMSE'].mean():.6f} ± {method_data['RMSE'].std():.6f}\n")
                f.write(f"        min = {method_data['RMSE'].min():.6f}, max = {method_data['RMSE'].max():.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of upscaling methods')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file')
    parser.add_argument('--n_examples', type=int, default=10,
                       help='Number of examples to run')
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
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = model_path.parent / f'resolution_comparison_statistical_{timestamp}'
    results_dir.mkdir(exist_ok=True)
    
    # Define resolutions to test
    resolutions = [80, 160, 320, 640]
    
    # Run multiple examples
    all_metrics = []
    for i in range(args.n_examples):
        metrics = run_single_example(
            model, device, i, results_dir, resolutions
        )
        all_metrics.append(metrics)
    
    # Perform statistical analysis
    plot_statistical_analysis(all_metrics, results_dir)
    
    print(f"\nResults saved in: {results_dir}")

if __name__ == '__main__':
    main() 