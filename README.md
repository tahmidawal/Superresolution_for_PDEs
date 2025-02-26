# PDE Solution Upscaling - Working Directory

This directory contains the working scripts for the PDE solution upscaling project. The following scripts have been tested and confirmed to work correctly.

## Core Functionality

- **data_generation.py**: Generates Poisson problem datasets with various parameters.
- **data_generation_extended.py**: Extended version with additional functionality for generating sinusoidal source terms with specific wave numbers.
- **models.py**: Contains the neural network architecture (U-Net) used for upscaling PDE solutions.
- **compare_methods.py**: Utilities for comparing different upscaling methods.

## Resolution Comparison Scripts

- **resolution_comparison.py**: Base script for comparing upscaling methods across different resolutions.
- **resolution_comparison_k10.py**: Specialized version for testing with k=10 (high frequency).
- **resolution_comparison_multi_k.py**: Extended version that supports multiple k values for comprehensive testing.

## Interpolation Comparison

- **interpolation_comparison_k11.py**: Compares different interpolation methods (bilinear, cubic, quintic) for k=11 test cases.
- **summarize_interpolation_results.py**: Generates readable summaries of interpolation comparison results.

## Visualization Scripts

- **plot_error_metrics.py**: Basic script for plotting error metrics from CSV data.
- **plot_error_metrics_enhanced.py**: Enhanced version with improved visualization features.
- **plot_error_metrics_final.py**: Final version that includes ML method alongside traditional interpolation methods.

## Key Findings

1. The ML-based upscaling method consistently outperforms traditional interpolation methods (bilinear, cubic, quintic) for high-frequency inputs (k=9-11).
2. Performance varies with resolution, with error generally decreasing as resolution increases.
3. For k=11 specifically:
   - ML method shows significantly lower MAE and RMSE compared to other methods
   - Cubic and quintic interpolation perform better than bilinear but still worse than ML
   - The advantage of ML is most pronounced at higher resolutions

## Usage

Most scripts can be run directly with Python:

```bash
python script_name.py
```

Some scripts may require command-line arguments, such as:

```bash
python interpolation_comparison_k11.py
python plot_error_metrics_final.py
```

Results are typically saved to the `../results/` directory, organized by test case.

## Dependencies

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas

## Latest Updates (February 26, 2025)

- Added polynomial (quintic) and cubic interpolation methods for comparison
- Enhanced visualization with clearer plots and better organization
- Added ML method to comparison plots
- Improved data generation with extended functionality for specific wave numbers 