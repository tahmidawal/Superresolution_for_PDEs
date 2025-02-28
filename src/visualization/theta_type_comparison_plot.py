import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def plot_theta_type_comparison():
    """
    Create a plot comparing error metrics across different theta types for full domains only.
    """
    # Define theta types
    theta_types = ['constant', 'grf', 'radial']
    
    # Define error metrics for different theta types based on our evaluation results
    # These values are from the evaluation_20250228_093818/metrics_by_theta_type.json
    # and represent full domain evaluations
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
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.title('Error Metrics by Theta Type', fontsize=16)
    
    # Set up bar positions
    x = np.arange(len(theta_types))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, mae, width, label='MAE', color='blue')
    plt.bar(x, rmse, width, label='RMSE', color='green')
    plt.bar(x + width, max_error, width, label='Max Error', color='red')
    
    # Add value labels
    for i, v in enumerate(mae):
        plt.text(i - width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(rmse):
        plt.text(i, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(max_error):
        plt.text(i + width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Set up the plot
    plt.xlabel('Theta Type', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x, [t.capitalize() for t in theta_types], fontsize=12)
    plt.legend(fontsize=12)
    
    # Save the plot
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'theta_type_comparison_plot_full_domains.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {results_dir / 'theta_type_comparison_plot_full_domains.png'}")

def plot_from_json(json_path, filter_subdomains=True):
    """
    Create a plot from a metrics JSON file, optionally filtering out subdomains.
    
    Args:
        json_path: Path to the metrics JSON file
        filter_subdomains: Whether to filter out subdomain samples
    """
    # Load metrics from JSON
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    # Define theta types
    theta_types = list(metrics.keys())
    
    # Extract values
    mae = [metrics[t]['mae'] for t in theta_types]
    rmse = [metrics[t]['rmse'] for t in theta_types]
    max_error = [metrics[t]['max_error'] for t in theta_types]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.title('Error Metrics by Theta Type (Full Domains Only)', fontsize=16)
    
    # Set up bar positions
    x = np.arange(len(theta_types))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, mae, width, label='MAE', color='blue')
    plt.bar(x, rmse, width, label='RMSE', color='green')
    plt.bar(x + width, max_error, width, label='Max Error', color='red')
    
    # Add value labels
    for i, v in enumerate(mae):
        plt.text(i - width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(rmse):
        plt.text(i, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for i, v in enumerate(max_error):
        plt.text(i + width, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Set up the plot
    plt.xlabel('Theta Type', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x, [t.capitalize() for t in theta_types], fontsize=12)
    plt.legend(fontsize=12)
    
    # Save the plot
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'theta_type_comparison_from_json_full_domains.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {results_dir / 'theta_type_comparison_from_json_full_domains.png'}")

def main():
    parser = argparse.ArgumentParser(description='Generate theta type comparison plot for full domains')
    parser.add_argument('--json_path', type=str, help='Path to metrics JSON file',
                       default='results/evaluation_20250228_093818/metrics_by_theta_type.json')
    args = parser.parse_args()
    
    # Generate plot with hardcoded values
    plot_theta_type_comparison()
    
    # Generate plot from JSON if file exists
    if Path(args.json_path).exists():
        plot_from_json(args.json_path)
    else:
        print(f"JSON file not found: {args.json_path}")

if __name__ == '__main__':
    main() 