import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('../results/multi_k_comparison/metrics_summary.csv', skiprows=2)

# Print column names for debugging
print("Column names:", df.columns.tolist())

# Create figure
plt.figure(figsize=(12, 8))
plt.title('Error Metrics vs Resolution', fontsize=16)

# Filter for k=11 data
k11_data = df[df['k1'] == 11.0]

# Print first row for debugging
print("First row of k11_data:")
print(k11_data.iloc[0])

# Plot lines for each method
resolutions = sorted(k11_data['Resolution'].unique())

# ML Multi-level
ml_data = k11_data[k11_data['Method'] == 'ML Multi-level']
ml_mae = ml_data.iloc[:, 3].values  # mean column for MAE
ml_rmse = ml_data.iloc[:, 7].values  # mean column for RMSE

# Bilinear Multi-level
bl_multi_data = k11_data[k11_data['Method'] == 'Bilinear Multi-level']
bl_multi_mae = bl_multi_data.iloc[:, 3].values
bl_multi_rmse = bl_multi_data.iloc[:, 7].values

# Direct Bilinear
bl_direct_data = k11_data[k11_data['Method'] == 'Direct Bilinear']
bl_direct_mae = bl_direct_data.iloc[:, 3].values
bl_direct_rmse = bl_direct_data.iloc[:, 7].values

# Plot with log scale for y-axis
plt.plot(resolutions, ml_mae, 'bo-', label='ML Multi-level MAE', linewidth=2)
plt.plot(resolutions, ml_rmse, 'b^--', label='ML Multi-level RMSE', linewidth=2)
plt.plot(resolutions, bl_multi_mae, 'go-', label='Bilinear Multi-level MAE', linewidth=2)
plt.plot(resolutions, bl_multi_rmse, 'g^--', label='Bilinear Multi-level RMSE', linewidth=2)
plt.plot(resolutions, bl_direct_mae, 'ro-', label='Direct Bilinear MAE', linewidth=2)
plt.plot(resolutions, bl_direct_rmse, 'r^--', label='Direct Bilinear RMSE', linewidth=2)

# Add value labels
for i, res in enumerate(resolutions):
    plt.text(res, ml_mae[i], f'{ml_mae[i]:.6f}', 
            verticalalignment='bottom', horizontalalignment='center', fontsize=9)
    plt.text(res, bl_multi_mae[i], f'{bl_multi_mae[i]:.6f}',
            verticalalignment='bottom', horizontalalignment='center', fontsize=9)

plt.xlabel('Resolution', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(resolutions, [f'{r}x{r}' for r in resolutions])

# Save the plot
plt.tight_layout()
plt.savefig('../results/multi_k_comparison/k11_error_metrics.png', dpi=300)
print('Plot saved to ../results/multi_k_comparison/k11_error_metrics.png') 