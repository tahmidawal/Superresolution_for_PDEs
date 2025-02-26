import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_results(results_file):
    """
    Generate a readable summary of the interpolation comparison results.
    
    Args:
        results_file: Path to the summary metrics CSV file
    """
    print(f"Loading data from {results_file}")
    
    # Load the summary metrics
    df = pd.read_csv(results_file)
    
    print('\nInterpolation Method Comparison for k=11 (5 samples)\n')
    print('Mean Absolute Error (MAE):\n')
    
    # Create a pivot table for MAE
    mae_pivot = df.pivot_table(
        index=['Resolution', 'Approach'],
        columns='Method',
        values='MAE_mean'
    )
    
    # Format the values
    formatted_mae = mae_pivot.applymap(lambda x: f'{x:.8f}')
    print(formatted_mae)
    
    print('\nRoot Mean Square Error (RMSE):\n')
    
    # Create a pivot table for RMSE
    rmse_pivot = df.pivot_table(
        index=['Resolution', 'Approach'],
        columns='Method',
        values='RMSE_mean'
    )
    
    # Format the values
    formatted_rmse = rmse_pivot.applymap(lambda x: f'{x:.8f}')
    print(formatted_rmse)
    
    # Calculate the best method for each resolution and approach
    print('\nBest Method by Resolution and Approach (based on MAE):\n')
    best_methods = df.loc[df.groupby(['Resolution', 'Approach'])['MAE_mean'].idxmin()]
    best_methods = best_methods[['Resolution', 'Approach', 'Method', 'MAE_mean']]
    best_methods['MAE_mean'] = best_methods['MAE_mean'].apply(lambda x: f'{x:.8f}')
    print(best_methods)
    
    # Create a more visually appealing plot
    print("\nCreating enhanced MAE comparison plot...")
    plt.figure(figsize=(14, 10))
    
    # Set up the plot style
    sns.set_style("whitegrid")
    
    # Colors for methods
    colors = {'bilinear': 'blue', 'cubic': 'green', 'quintic': 'red'}
    
    # Line styles for approaches
    linestyles = {'Direct': '-', 'Multi-level': '--'}
    
    # Markers
    markers = {'Direct': 'o', 'Multi-level': '^'}
    
    # Plot lines for each method and approach
    for method in df['Method'].unique():
        for approach in df['Approach'].unique():
            data = df[(df['Method'] == method) & (df['Approach'] == approach)]
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
    
    # Add value labels
    for method in df['Method'].unique():
        for approach in df['Approach'].unique():
            data = df[(df['Method'] == method) & (df['Approach'] == approach)]
            for _, row in data.iterrows():
                plt.text(
                    row['Resolution'], 
                    row['MAE_mean'] * 1.02,  # Slight offset for readability
                    f'{row["MAE_mean"]:.8f}',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    fontsize=7,
                    rotation=45
                )
    
    plt.title('Mean Absolute Error by Resolution, Method, and Approach (k=11)', fontsize=16)
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Mean Absolute Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    plt.xticks(df['Resolution'].unique(), [f'{r}x{r}' for r in df['Resolution'].unique()])
    
    plt.tight_layout()
    output_path = '../results/k11_interpolation_comparison/enhanced_mae_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nEnhanced plot saved to '{output_path}'")

if __name__ == "__main__":
    print("Starting summary generation...")
    summarize_results('../results/k11_interpolation_comparison/summary_metrics.csv')
    print("Summary generation complete.") 