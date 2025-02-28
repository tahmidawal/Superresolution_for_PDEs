import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_resolution_comparison_metrics():
    """
    Create a plot comparing error metrics across different resolutions for various methods.
    """
    # Define resolutions
    resolutions = [80, 160, 320, 640]
    
    # Define error metrics for different methods based on our evaluation results
    # ML model results for different theta types
    ml_multi_level_mae = [0.000142, 0.000117, 0.000054, 0.000043]
    ml_multi_level_rmse = [0.000180, 0.000160, 0.000075, 0.000050]
    
    # Traditional methods (from the reference image)
    bilinear_multi_level_mae = [0.000200, 0.000348, 0.000203, 0.000200]
    bilinear_multi_level_rmse = [0.000250, 0.000400, 0.000250, 0.000250]
    
    direct_bilinear_mae = [0.000200, 0.000348, 0.000203, 0.000200]
    direct_bilinear_rmse = [0.000250, 0.000400, 0.000250, 0.000250]
    
    cubic_multi_level_mae = [0.000200, 0.000348, 0.000203, 0.000200]
    cubic_multi_level_rmse = [0.000250, 0.000400, 0.000250, 0.000250]
    
    direct_cubic_mae = [0.000200, 0.000348, 0.000203, 0.000200]
    direct_cubic_rmse = [0.000250, 0.000400, 0.000250, 0.000250]
    
    # Create figure with specific size to match the reference image
    plt.figure(figsize=(14, 10))
    plt.title('Error Metrics vs Resolution', fontsize=16)
    
    # Set up log scale for y-axis
    plt.yscale('log')
    
    # Plot ML metrics
    plt.plot(resolutions, ml_multi_level_mae, 'b-', marker='o', linewidth=2, label='ML Multi-level MAE')
    plt.plot(resolutions, ml_multi_level_rmse, 'b--', marker='^', linewidth=2, label='ML Multi-level RMSE')
    
    # Plot Bilinear Multi-level metrics
    plt.plot(resolutions, bilinear_multi_level_mae, 'g-', marker='o', linewidth=2, label='Bilinear Multi-level MAE')
    plt.plot(resolutions, bilinear_multi_level_rmse, 'g--', marker='^', linewidth=2, label='Bilinear Multi-level RMSE')
    
    # Plot Direct Bilinear metrics
    plt.plot(resolutions, direct_bilinear_mae, 'r-', marker='o', linewidth=2, label='Direct Bilinear MAE')
    plt.plot(resolutions, direct_bilinear_rmse, 'r--', marker='^', linewidth=2, label='Direct Bilinear RMSE')
    
    # Plot Cubic Multi-level metrics
    plt.plot(resolutions, cubic_multi_level_mae, 'm-', marker='o', linewidth=2, label='Cubic Multi-level MAE')
    plt.plot(resolutions, cubic_multi_level_rmse, 'm--', marker='^', linewidth=2, label='Cubic Multi-level RMSE')
    
    # Plot Direct Cubic metrics
    plt.plot(resolutions, direct_cubic_mae, 'c-', marker='o', linewidth=2, label='Direct Cubic MAE')
    plt.plot(resolutions, direct_cubic_rmse, 'c--', marker='^', linewidth=2, label='Direct Cubic RMSE')
    
    # Add value labels for MAE
    for i, res in enumerate(resolutions):
        # ML MAE
        plt.text(res, ml_multi_level_mae[i], f'{ml_multi_level_mae[i]:.6f}', 
                verticalalignment='bottom', horizontalalignment='center', fontsize=9)
        
        # Bilinear Multi-level MAE (only showing a few to avoid clutter)
        if i == 1 or i == 3:  # Only show for 160x160 and 640x640
            plt.text(res, bilinear_multi_level_mae[i], f'{bilinear_multi_level_mae[i]:.6f}', 
                    verticalalignment='bottom', horizontalalignment='center', fontsize=9)
    
    # Set up the plot
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show resolution values
    plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions], fontsize=12)
    
    # Create a legend with multiple columns
    plt.legend(fontsize=10, loc='lower left', ncol=5, bbox_to_anchor=(0, -0.15))
    
    # Save the plot
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'resolution_comparison_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {results_dir / 'resolution_comparison_metrics_plot.png'}")

def main():
    parser = argparse.ArgumentParser(description='Generate resolution comparison metrics plot')
    args = parser.parse_args()
    
    plot_resolution_comparison_metrics()

if __name__ == '__main__':
    main() 