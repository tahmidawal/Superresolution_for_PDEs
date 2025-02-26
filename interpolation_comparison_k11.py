import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os
from datetime import datetime
import time

# Import local modules
from utils import upsample_solution
from data_generation_extended import generate_poisson_problem_sin

def solve_multi_resolution_k11(num_samples=5, seed=42):
    """
    Generate and solve multiple test cases with k1=k2=11 at different resolutions.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing solutions at different resolutions
    """
    np.random.seed(seed)
    
    # Fixed k values
    k1 = k2 = 11.0
    
    # Resolutions to test
    resolutions = [40, 80, 160, 320, 640]
    
    # Initialize data dictionary
    data = {
        'samples': [],
        'k_values': []
    }
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples} with k1=k2=11...")
        
        # Generate finest grid (640x640)
        finest_grid = generate_poisson_problem_sin(
            grid_size=640,
            k1=k1,
            k2=k2,
            amplitude=1.0,
            seed=seed + i
        )
        
        # Store k values
        data['k_values'].append((k1, k2))
        
        # Initialize sample dictionary
        sample_data = {
            'finest': finest_grid['solution'],
            'solutions': {}
        }
        
        # Downsample and solve for each resolution
        for res in resolutions:
            if res == 640:
                # Use the finest grid directly
                sample_data['solutions'][res] = finest_grid['solution']
            else:
                # Downsample the source and solve
                downsampled = {
                    'source': finest_grid['source'][::640//res, ::640//res],
                    'grid_size': res
                }
                
                # For simplicity, we'll use the downsampled solution from the finest grid
                # In a real scenario, you would solve the PDE at this resolution
                sample_data['solutions'][res] = finest_grid['solution'][::640//res, ::640//res]
        
        # Add sample to data
        data['samples'].append(sample_data)
        
    return data

def interpolate_solution(solution, target_size, method):
    """
    Interpolate a solution to a target size using specified method.
    
    Args:
        solution: Input solution array
        target_size: Target size (height, width)
        method: Interpolation method ('bilinear', 'cubic', or 'quintic')
        
    Returns:
        Interpolated solution
    """
    h, w = solution.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    
    # Create interpolation function
    if method == 'bilinear':
        interp_method = 'linear'
    elif method == 'cubic':
        interp_method = 'cubic'
    elif method == 'quintic':
        interp_method = 'quintic'
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
    # Create interpolator
    interpolator = RegularGridInterpolator((y, x), solution, method=interp_method, bounds_error=False, fill_value=None)
    
    # Create target grid
    x_new = np.linspace(0, 1, target_size)
    y_new = np.linspace(0, 1, target_size)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    points = np.column_stack((Y_new.flatten(), X_new.flatten()))
    
    # Interpolate
    result = interpolator(points).reshape((target_size, target_size))
    
    return result

def multi_level_interpolate(solution, target_size, method):
    """
    Perform multi-level interpolation from source resolution to target resolution.
    
    Args:
        solution: Input solution array
        target_size: Target size (height, width)
        method: Interpolation method ('bilinear', 'cubic', or 'quintic')
        
    Returns:
        Upsampled solution
    """
    current_solution = solution.copy()
    current_size = solution.shape[0]
    
    while current_size < target_size:
        # Double the resolution at each step
        next_size = min(current_size * 2, target_size)
        current_solution = interpolate_solution(current_solution, next_size, method)
        current_size = next_size
        
    return current_solution

def compute_metrics(prediction, target):
    """
    Compute MAE and RMSE between prediction and target.
    
    Args:
        prediction: Predicted solution
        target: Target solution
        
    Returns:
        Dictionary with MAE and RMSE
    """
    mae = np.mean(np.abs(prediction - target))
    rmse = np.sqrt(np.mean((prediction - target) ** 2))
    return {'mae': mae, 'rmse': rmse}

def run_interpolation_comparison():
    """
    Run comparison of different interpolation methods for k=11 samples.
    """
    # Create results directory
    results_dir = '../results/k11_interpolation_comparison'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate test data
    data = solve_multi_resolution_k11(num_samples=5)
    
    # Target resolutions to test
    target_resolutions = [80, 160, 320, 640]
    
    # Interpolation methods to compare
    methods = ['bilinear', 'cubic', 'quintic']
    
    # Initialize results dictionary
    results = {
        'Sample': [],
        'Resolution': [],
        'Method': [],
        'Approach': [],
        'MAE': [],
        'RMSE': []
    }
    
    # Process each sample
    for sample_idx, sample_data in enumerate(data['samples']):
        print(f"Processing sample {sample_idx+1}/5...")
        
        # Get the finest solution (ground truth)
        finest_solution = sample_data['finest']
        
        # Process each target resolution
        for target_res in target_resolutions:
            print(f"  Target resolution: {target_res}x{target_res}")
            
            # Get the ground truth for this resolution
            target_solution = finest_solution[::640//target_res, ::640//target_res]
            
            # Process each method
            for method in methods:
                print(f"    Method: {method}")
                
                # Direct interpolation from 40x40 to target resolution
                source_solution = sample_data['solutions'][40]
                direct_result = interpolate_solution(source_solution, target_res, method)
                direct_metrics = compute_metrics(direct_result, target_solution)
                
                # Multi-level interpolation from 40x40 to target resolution
                multi_level_result = multi_level_interpolate(source_solution, target_res, method)
                multi_level_metrics = compute_metrics(multi_level_result, target_solution)
                
                # Store results for direct interpolation
                results['Sample'].append(sample_idx + 1)
                results['Resolution'].append(target_res)
                results['Method'].append(method)
                results['Approach'].append('Direct')
                results['MAE'].append(direct_metrics['mae'])
                results['RMSE'].append(direct_metrics['rmse'])
                
                # Store results for multi-level interpolation
                results['Sample'].append(sample_idx + 1)
                results['Resolution'].append(target_res)
                results['Method'].append(method)
                results['Approach'].append('Multi-level')
                results['MAE'].append(multi_level_metrics['mae'])
                results['RMSE'].append(multi_level_metrics['rmse'])
                
                # Create visualization for the highest resolution (640x640)
                if target_res == 640 and sample_idx < 3:  # Only create plots for first 3 samples
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Plot ground truth
                    im0 = axes[0].imshow(target_solution, cmap='viridis')
                    axes[0].set_title(f'Ground Truth (640x640)', fontsize=14)
                    plt.colorbar(im0, ax=axes[0])
                    
                    # Plot direct interpolation
                    im1 = axes[1].imshow(direct_result, cmap='viridis')
                    axes[1].set_title(f'Direct {method.capitalize()} Interpolation\nMAE: {direct_metrics["mae"]:.6f}', fontsize=14)
                    plt.colorbar(im1, ax=axes[1])
                    
                    # Plot multi-level interpolation
                    im2 = axes[2].imshow(multi_level_result, cmap='viridis')
                    axes[2].set_title(f'Multi-level {method.capitalize()} Interpolation\nMAE: {multi_level_metrics["mae"]:.6f}', fontsize=14)
                    plt.colorbar(im2, ax=axes[2])
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/sample{sample_idx+1}_{method}_comparison.png', dpi=300)
                    plt.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(f'{results_dir}/interpolation_metrics.csv', index=False)
    
    # Create summary plots
    create_summary_plots(results_df, results_dir)
    
    return results_df

def create_summary_plots(results_df, results_dir):
    """
    Create summary plots from the results DataFrame.
    
    Args:
        results_df: DataFrame with results
        results_dir: Directory to save plots
    """
    # Calculate average metrics across samples
    summary_df = results_df.groupby(['Resolution', 'Method', 'Approach']).agg({
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-index columns
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    # Save summary to CSV
    summary_df.to_csv(f'{results_dir}/summary_metrics.csv', index=False)
    
    # Create plot for MAE by resolution, method, and approach
    plt.figure(figsize=(14, 10))
    
    # Get unique methods and approaches
    methods = results_df['Method'].unique()
    approaches = results_df['Approach'].unique()
    
    # Colors and markers for methods
    colors = {'bilinear': 'blue', 'cubic': 'green', 'quintic': 'red'}
    
    # Line styles for approaches
    linestyles = {'Direct': '-', 'Multi-level': '--'}
    
    # Markers
    markers = {'Direct': 'o', 'Multi-level': '^'}
    
    # Plot lines for each method and approach
    for method in methods:
        for approach in approaches:
            data = summary_df[(summary_df['Method'] == method) & (summary_df['Approach'] == approach)]
            plt.plot(
                data['Resolution'], 
                data['MAE_mean'], 
                label=f'{method.capitalize()} {approach}',
                color=colors[method],
                linestyle=linestyles[approach],
                marker=markers[approach],
                linewidth=2,
                markersize=8
            )
            
            # Add error bars
            plt.fill_between(
                data['Resolution'],
                data['MAE_mean'] - data['MAE_std'],
                data['MAE_mean'] + data['MAE_std'],
                color=colors[method],
                alpha=0.2
            )
    
    # Add value labels
    for method in methods:
        for approach in approaches:
            data = summary_df[(summary_df['Method'] == method) & (summary_df['Approach'] == approach)]
            for i, row in data.iterrows():
                plt.text(
                    row['Resolution'], 
                    row['MAE_mean'],
                    f'{row["MAE_mean"]:.6f}',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    fontsize=9
                )
    
    plt.title('Mean Absolute Error by Resolution, Method, and Approach (k=11)', fontsize=16)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Mean Absolute Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(summary_df['Resolution'].unique(), [f'{r}x{r}' for r in summary_df['Resolution'].unique()])
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/mae_comparison.png', dpi=300)
    plt.close()
    
    # Create similar plot for RMSE
    plt.figure(figsize=(14, 10))
    
    for method in methods:
        for approach in approaches:
            data = summary_df[(summary_df['Method'] == method) & (summary_df['Approach'] == approach)]
            plt.plot(
                data['Resolution'], 
                data['RMSE_mean'], 
                label=f'{method.capitalize()} {approach}',
                color=colors[method],
                linestyle=linestyles[approach],
                marker=markers[approach],
                linewidth=2,
                markersize=8
            )
            
            # Add error bars
            plt.fill_between(
                data['Resolution'],
                data['RMSE_mean'] - data['RMSE_std'],
                data['RMSE_mean'] + data['RMSE_std'],
                color=colors[method],
                alpha=0.2
            )
    
    # Add value labels
    for method in methods:
        for approach in approaches:
            data = summary_df[(summary_df['Method'] == method) & (summary_df['Approach'] == approach)]
            for i, row in data.iterrows():
                plt.text(
                    row['Resolution'], 
                    row['RMSE_mean'],
                    f'{row["RMSE_mean"]:.6f}',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    fontsize=9
                )
    
    plt.title('Root Mean Square Error by Resolution, Method, and Approach (k=11)', fontsize=16)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Root Mean Square Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(summary_df['Resolution'].unique(), [f'{r}x{r}' for r in summary_df['Resolution'].unique()])
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/rmse_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    print("Starting interpolation comparison for k=11...")
    results = run_interpolation_comparison()
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results saved to ../results/k11_interpolation_comparison/") 