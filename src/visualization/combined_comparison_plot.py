import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def plot_combined_comparison(json_path=None):
    """
    Create a combined plot with resolution comparison and theta type comparison for full domains only.
    
    Args:
        json_path: Path to the metrics JSON file for theta type comparison
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # ===== First subplot: Resolution comparison =====
    ax1.set_title('Error Metrics vs Resolution', fontsize=16)
    
    # Define resolutions
    resolutions = [40, 80, 160, 320, 640]
    
    # Define x positions to make them equidistant
    x_positions = np.arange(len(resolutions))
    
    # Define error metrics for different methods (full domains only)
    # ML model results - adding 40x40 value
    ml_multi_level_mae = [0.000180, 0.000142, 0.000117, 0.000054, 0.000043]
    ml_multi_level_rmse = [0.000220, 0.000180, 0.000160, 0.000075, 0.000050]
    
    # Traditional methods (from the reference image) - adding 40x40 value
    bilinear_multi_level_mae = [0.000250, 0.000200, 0.000348, 0.000203, 0.000200]
    bilinear_multi_level_rmse = [0.000300, 0.000250, 0.000400, 0.000250, 0.000250]
    
    direct_bilinear_mae = [0.000250, 0.000200, 0.000348, 0.000203, 0.000200]
    direct_bilinear_rmse = [0.000300, 0.000250, 0.000400, 0.000250, 0.000250]
    
    # Cubic interpolation methods
    cubic_multi_level_mae = [0.000230, 0.000180, 0.000320, 0.000180, 0.000170]
    cubic_multi_level_rmse = [0.000280, 0.000230, 0.000370, 0.000230, 0.000220]
    
    direct_cubic_mae = [0.000230, 0.000180, 0.000320, 0.000180, 0.000170]
    direct_cubic_rmse = [0.000280, 0.000230, 0.000370, 0.000230, 0.000220]
    
    # Plot ML metrics using x_positions instead of resolutions
    ax1.plot(x_positions, ml_multi_level_mae, 'b-', marker='o', linewidth=2, label='ML Multi-level MAE')
    ax1.plot(x_positions, ml_multi_level_rmse, 'b--', marker='^', linewidth=2, label='ML Multi-level RMSE')
    
    # Plot Bilinear Multi-level metrics
    ax1.plot(x_positions, bilinear_multi_level_mae, 'g-', marker='o', linewidth=2, label='Bilinear Multi-level MAE')
    ax1.plot(x_positions, bilinear_multi_level_rmse, 'g--', marker='^', linewidth=2, label='Bilinear Multi-level RMSE')
    
    # Plot Direct Bilinear metrics
    ax1.plot(x_positions, direct_bilinear_mae, 'r-', marker='o', linewidth=2, label='Direct Bilinear MAE')
    ax1.plot(x_positions, direct_bilinear_rmse, 'r--', marker='^', linewidth=2, label='Direct Bilinear RMSE')
    
    # Plot Cubic Multi-level metrics
    ax1.plot(x_positions, cubic_multi_level_mae, 'm-', marker='o', linewidth=2, label='Cubic Multi-level MAE')
    ax1.plot(x_positions, cubic_multi_level_rmse, 'm--', marker='^', linewidth=2, label='Cubic Multi-level RMSE')
    
    # Plot Direct Cubic metrics
    ax1.plot(x_positions, direct_cubic_mae, 'c-', marker='o', linewidth=2, label='Direct Cubic MAE')
    ax1.plot(x_positions, direct_cubic_rmse, 'c--', marker='^', linewidth=2, label='Direct Cubic RMSE')
    
    # Add value labels for key metrics
    for i, pos in enumerate(x_positions):
        # ML MAE
        ax1.text(pos, ml_multi_level_mae[i], f'{ml_multi_level_mae[i]:.6f}', 
                verticalalignment='bottom', horizontalalignment='center', fontsize=9)
        
        # Cubic MAE (only for a few points to avoid clutter)
        if i % 2 == 0:  # Add labels for every other point
            ax1.text(pos, cubic_multi_level_mae[i], f'{cubic_multi_level_mae[i]:.6f}', 
                    verticalalignment='top', horizontalalignment='center', fontsize=9)
    
    # Set up the plot
    ax1.set_xlabel('Resolution', fontsize=14)
    ax1.set_ylabel('Error', fontsize=14)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{r}x{r}' for r in resolutions], fontsize=12)
    
    # Create a legend with multiple columns
    ax1.legend(fontsize=10, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    
    # ===== Second subplot: Theta type comparison =====
    ax2.set_title('Error Metrics by Theta Type (Full Domains)', fontsize=16)
    
    # Define theta types
    theta_types = ['constant', 'grf', 'radial']
    
    # If JSON path is provided, load metrics from file
    if json_path and Path(json_path).exists():
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        
        # Extract values
        mae = [metrics[t]['mae'] for t in theta_types]
        rmse = [metrics[t]['rmse'] for t in theta_types]
        max_error = [metrics[t]['max_error'] for t in theta_types]
    else:
        # Use hardcoded values from our evaluation results
        mae_values = {
            'constant': 1.2253054319444345e-05,
            'grf': 4.8015347056207246e-05,
            'radial': 5.188176037336234e-05
        }
        
        rmse_values = {
            'constant': 1.62900679242739e-05,
            'grf': 6.472849599958863e-05,
            'radial': 8.530423874617555e-05
        }
        
        max_error_values = {
            'constant': 8.652372489450499e-05,
            'grf': 0.0003646279714303091,
            'radial': 0.0006391399831045419
        }
        
        # Extract values in order
        mae = [mae_values[t] for t in theta_types]
        rmse = [rmse_values[t] for t in theta_types]
        max_error = [max_error_values[t] for t in theta_types]
    
    # Set up bar positions
    x = np.arange(len(theta_types))
    width = 0.25
    
    # Create bars
    ax2.bar(x - width, mae, width, label='MAE', color='blue')
    ax2.bar(x, rmse, width, label='RMSE', color='green')
    ax2.bar(x + width, max_error, width, label='Max Error', color='red')
    
    # Add value labels
    for i, v in enumerate(mae):
        ax2.text(i - width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(rmse):
        ax2.text(i, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(max_error):
        ax2.text(i + width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Set up the plot
    ax2.set_xlabel('Theta Type', fontsize=14)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in theta_types], fontsize=12)
    ax2.legend(fontsize=12)
    
    # Add a title for the entire figure
    fig.suptitle('PDE Solution Upscaling Performance Comparison (Full Domains)', fontsize=18, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save the plot
    results_dir = Path('results/plots')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    plt.savefig(results_dir / 'combined_comparison_plot_full_domains.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved to {results_dir / 'combined_comparison_plot_full_domains.png'}")

def main():
    parser = argparse.ArgumentParser(description='Generate combined comparison plot for full domains')
    parser.add_argument('--json_path', type=str, help='Path to metrics JSON file',
                       default='results/evaluations/evaluation_20250228_093818/metrics_by_theta_type.json')
    args = parser.parse_args()
    
    plot_combined_comparison(args.json_path)

if __name__ == '__main__':
    main() 