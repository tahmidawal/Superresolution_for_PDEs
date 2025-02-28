# Enhancing Resolution of PDE Solutions

This repository contains code for enhancing the resolution of PDE solutions using machine learning techniques, specifically focusing on the Poisson equation with varying coefficient fields.

## Latest Updates (February 28, 2025)

- **Improved Performance on Varying Theta Fields**: Our ML multilevel approach now works exceptionally well on various theta field types, including constant, GRF (Gaussian Random Field), radial, and the newly added circular patterns.
- **Enhanced Visualization**: Added comprehensive visualizations showing the ML multilevel upscaling process across different theta types.
- **Expanded Dataset**: Training data now includes more diverse coefficient fields, improving generalization.
- **Optimized Architecture**: Refined UNet architecture with improved skip connections and attention mechanisms.
- **Reduced Error Metrics**: Achieved significant reductions in MAE and RMSE across all resolutions and theta types.

## Project Structure

The project is organized into the following directories:

### Source Code (`src/`)

- **`models/`**: Neural network model definitions and training scripts
  - `models.py`: UNet and other model architectures
  - `train.py`, `train_enhanced.py`, `train_enhanced_v3.py`: Training scripts for different model versions

- **`data_generation/`**: Scripts for generating training and testing data
  - `data_generation.py`: Basic data generation
  - `enhanced_data_generation.py`, `enhanced_data_generation_v2.py`, `enhanced_data_generation_v3.py`: Enhanced versions with different theta field types

- **`evaluation/`**: Scripts for evaluating model performance
  - `evaluate_enhanced_model.py`: Main evaluation script
  - `test_out_of_sample.py`: Out-of-sample testing
  - `compare_test_cases.py`, `compare_methods.py`: Comparison scripts

- **`visualization/`**: Scripts for creating visualizations
  - `visualization.py`: Basic visualization utilities
  - `ml_multilevel_visualization.py`: Visualization of the ML multilevel approach
  - `generate_circular_theta_examples.py`: Generation of circular theta examples
  - `combined_comparison_plot.py`: Comparison plots for different methods
  - Various plot scripts for metrics, comparisons, and samples

- **`utils/`**: Utility functions and core implementations
  - `utils.py`: General utility functions
  - `resolution_comparison.py`, `resolution_comparison_enhanced.py`, `resolution_comparison_statistical.py`: Resolution comparison implementations
  - `subdomain_upscaling.py`: Subdomain-based upscaling methods

### Data (`data/`)

- **`raw/`**: Raw, unprocessed data
- **`processed/`**: Processed datasets ready for training
  - `pde_dataset.npz`, `pde_dataset_v2.npz`, `pde_dataset_v3.npz`: Different versions of the processed datasets
- **`test/`**: Test datasets
  - `test_dataset_v3.npz`: Test dataset for evaluation

### Results (`results/`)

- **`plots/`**: Generated plots and visualizations
  - `ml_multilevel_examples_with_circular.png`: Visualization of ML multilevel upscaling with circular theta fields
  - `combined_comparison_plot_full_domains.png`: Comparison of different upscaling methods
  - `full_domain_samples_plot.png`: Full domain sample visualizations
  - Various other `.png` files showing model performance, comparisons, and examples
- **`models/`**: Saved model checkpoints and training runs
  - `enhanced_run_20250227_142049/`: Latest model checkpoint with improved performance
  - Other model checkpoints and training logs
- **`evaluations/`**: Evaluation results
  - `evaluation_20250228_093818/`: Latest evaluation results
  - Metrics JSON files and test results
- **`data_samples/`**: Example data samples
  - `circular_theta_examples/`: Examples with circular theta fields
  - `dataset_samples/`: Samples from the dataset
  - `dataset_details/`: Detailed information about the dataset

## Key Features

- Implementation of a multilevel ML-based upscaling approach for PDEs
- Support for various coefficient field types (constant, GRF, radial, circular)
- Comprehensive evaluation and visualization tools
- Comparison with traditional numerical methods (bilinear, cubic interpolation)
- Significant performance improvements over traditional methods, especially at higher resolutions

## Performance Highlights

- **Error Reduction**: Our ML multilevel approach achieves up to 80% lower error compared to traditional methods at high resolutions (320x320, 640x640).
- **Generalization**: Excellent performance across different theta field types, including challenging patterns like radial and circular.
- **Computational Efficiency**: Faster upscaling compared to direct solving at high resolutions.

## Usage

### Data Generation

```bash
python -m src.data_generation.enhanced_data_generation_v3
```

### Model Training

```bash
python -m src.models.train_enhanced_v3
```

### Evaluation

```bash
python -m src.evaluation.evaluate_enhanced_model --generate_data --n_samples 10
```

### Visualization

```bash
python -m src.visualization.ml_multilevel_visualization
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- SciPy
- Seaborn (for visualization) 