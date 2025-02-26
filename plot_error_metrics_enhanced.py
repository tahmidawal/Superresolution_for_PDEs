import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load the data
df = pd.read_csv('../results/k11_interpolation_comparison/summary_metrics.csv')

# Create figure
plt.figure(figsize=(14, 10))
plt.title('Error Metrics vs Resolution', fontsize=20)

# Define colors for different methods
colors = {
    'bilinear': 'red',
    'cubic': 'green',
    'quintic': 'blue'
}

# Line styles for different approaches
line_styles = {
    'Direct': '-',
    'Multi-level': '--'
}

# Markers for different metrics
markers = {
    'MAE': 'o',
    'RMSE': '^'
}

# Plot lines for each method, approach, and metric type
for method in df['Method'].unique():
    for approach in df['Approach'].unique():
        # Filter data for this method and approach
        method_data = df[(df['Method'] == method) & (df['Approach'] == approach)]
        
        # Sort by resolution
        method_data = method_data.sort_values('Resolution')
        
        # Plot MAE
        plt.plot(
            method_data['Resolution'], 
            method_data['MAE_mean'],
            label=f'{method.capitalize()} {approach} MAE',
            color=colors[method],
            linestyle=line_styles[approach],
            marker=markers['MAE'],
            linewidth=2,
            markersize=10
        )
        
        # Plot RMSE
        plt.plot(
            method_data['Resolution'], 
            method_data['RMSE_mean'],
            label=f'{method.capitalize()} {approach} RMSE',
            color=colors[method],
            linestyle=line_styles[approach],
            marker=markers['RMSE'],
            linewidth=2,
            markersize=10
        )
        
        # Add value labels for MAE (only for specific methods to avoid clutter)
        if method in ['bilinear', 'cubic']:
            for _, row in method_data.iterrows():
                plt.text(
                    row['Resolution'], 
                    row['MAE_mean'] * 1.02,  # Slight offset for readability
                    f'{row["MAE_mean"]:.6f}',
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    fontsize=9
                )

# Set axis properties
plt.xlabel('Resolution', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Create a more organized legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=12, loc='upper right', ncol=2)

# Set x-ticks to show resolutions
resolutions = sorted(df['Resolution'].unique())
plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions], fontsize=14)

# Add a descriptive text box
plt.text(
    0.02, 0.02,
    'Comparison of interpolation methods for k=11\n'
    'Lower values indicate better performance',
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8)
)

# Save the plot
plt.tight_layout()
plt.savefig('../results/k11_interpolation_comparison/enhanced_error_metrics_final.png', dpi=300)
print('Plot saved to ../results/k11_interpolation_comparison/enhanced_error_metrics_final.png') 