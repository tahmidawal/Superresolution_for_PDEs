import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load the data
df = pd.read_csv('../results/k11_interpolation_comparison/summary_metrics.csv')

# Load ML data from multi_k_comparison
try:
    ml_df = pd.read_csv('../results/multi_k_comparison/metrics_summary.csv', skiprows=2)
    # Filter for k=11 and ML Multi-level method
    ml_data = ml_df[(ml_df['k1'] == 11.0) & (ml_df['Method'] == 'ML Multi-level')]
    has_ml_data = True
    print("ML data loaded successfully")
except Exception as e:
    print(f"Could not load ML data: {e}")
    has_ml_data = False

# Create figure
plt.figure(figsize=(14, 10))
plt.title('Error Metrics vs Resolution (k=11)', fontsize=20)

# Define methods to include in the plot
methods_to_plot = ['bilinear', 'cubic', 'quintic']

# Define colors for different methods
colors = {
    'bilinear': 'red',
    'cubic': 'green',
    'quintic': 'blue',
    'ML': 'purple'
}

# Define line styles for different metrics
line_styles = {
    'MAE': '-',
    'RMSE': '--'
}

# Define markers
markers = {
    'Direct': 'o',
    'Multi-level': '^'
}

# Define line widths
line_widths = {
    'bilinear': 2,
    'cubic': 2,
    'quintic': 2,
    'ML': 3  # Make ML lines thicker to stand out
}

# Filter data for Multi-level approach only to match the example
filtered_df = df[df['Approach'] == 'Multi-level']

# Add ML method first if data is available (so it appears at the bottom of the legend)
if has_ml_data:
    # Sort by resolution
    ml_data = ml_data.sort_values('Resolution')
    
    # Plot MAE for ML
    plt.plot(
        ml_data['Resolution'], 
        ml_data.iloc[:, 3].values,  # MAE mean column
        label='ML Multi-level MAE',
        color=colors['ML'],
        linestyle=line_styles['MAE'],
        marker=markers['Multi-level'],
        linewidth=line_widths['ML'],
        markersize=10
    )
    
    # Plot RMSE for ML
    plt.plot(
        ml_data['Resolution'], 
        ml_data.iloc[:, 7].values,  # RMSE mean column
        label='ML Multi-level RMSE',
        color=colors['ML'],
        linestyle=line_styles['RMSE'],
        marker=markers['Multi-level'],
        linewidth=line_widths['ML'],
        markersize=10
    )
    
    # Add value labels for MAE
    for i, row in ml_data.iterrows():
        plt.text(
            row['Resolution'], 
            row.iloc[3] * 1.02,  # MAE mean value with slight offset
            f'{row.iloc[3]:.6f}',
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=9,
            fontweight='bold'  # Make ML labels bold
        )

# Plot lines for each method and metric type
for method in methods_to_plot:
    method_data = filtered_df[filtered_df['Method'] == method]
    
    # Sort by resolution
    method_data = method_data.sort_values('Resolution')
    
    # Plot MAE
    plt.plot(
        method_data['Resolution'], 
        method_data['MAE_mean'],
        label=f'{method.capitalize()} Multi-level MAE',
        color=colors[method],
        linestyle=line_styles['MAE'],
        marker=markers['Multi-level'],
        linewidth=line_widths[method],
        markersize=10
    )
    
    # Plot RMSE
    plt.plot(
        method_data['Resolution'], 
        method_data['RMSE_mean'],
        label=f'{method.capitalize()} Multi-level RMSE',
        color=colors[method],
        linestyle=line_styles['RMSE'],
        marker=markers['Multi-level'],
        linewidth=line_widths[method],
        markersize=10
    )
    
    # Add value labels for MAE
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
# Reverse the order so ML appears at the top
handles = handles[::-1]
labels = labels[::-1]
plt.legend(handles, labels, fontsize=12, loc='upper right', ncol=2)

# Set x-ticks to show resolutions
resolutions = sorted(filtered_df['Resolution'].unique())
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
plt.savefig('../results/k11_interpolation_comparison/final_error_metrics_with_ml.png', dpi=300)
print('Plot saved to ../results/k11_interpolation_comparison/final_error_metrics_with_ml.png') 