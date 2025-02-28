import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import random
from matplotlib.gridspec import GridSpec
from PIL import Image
import matplotlib.image as mpimg

def plot_full_domain_samples(samples_dir=None):
    """
    Create a plot showing multiple full domain solution examples for each theta type.
    
    Args:
        samples_dir: Directory containing sample prediction images
    """
    # Set default path if not provided
    if samples_dir is None:
        samples_dir = 'results/evaluation_20250228_093818/sample_predictions'
    
    # Ensure path is a Path object
    samples_dir = Path(samples_dir)
    
    # Check if path exists
    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        return
    
    # Define theta types
    theta_types = ['constant', 'grf', 'radial']
    
    # Get sample images for each theta type
    sample_images = {}
    for theta_type in theta_types:
        sample_images[theta_type] = list(samples_dir.glob(f"{theta_type}_sample_*.png"))
        # Sort to ensure consistent order
        sample_images[theta_type].sort()
    
    # Select samples from each theta type
    n_samples_per_type = 3  # Show 3 samples per type
    selected_samples = {}
    for theta_type in theta_types:
        if len(sample_images[theta_type]) >= n_samples_per_type:
            selected_samples[theta_type] = sample_images[theta_type][:n_samples_per_type]
        else:
            print(f"Warning: Not enough samples for {theta_type}, using all available")
            selected_samples[theta_type] = sample_images[theta_type]
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 18))
    
    # Create a grid of subplots - 3 rows (one for each theta type) and n_samples_per_type columns
    for row, theta_type in enumerate(theta_types):
        for col, sample_path in enumerate(selected_samples[theta_type]):
            # Create subplot
            ax = plt.subplot2grid((3, n_samples_per_type), (row, col))
            
            # Display the complete sample image
            img = mpimg.imread(sample_path)
            ax.imshow(img)
            
            # Add title only for the first column
            if col == 0:
                ax.set_title(f"{theta_type.capitalize()} Samples (Full Domain)", fontsize=14, loc='left')
            else:
                ax.set_title(f"Sample {col+1}", fontsize=12)
            
            ax.axis('off')
    
    # Add a title for the entire figure
    fig.suptitle('PDE Solution Upscaling: Full Domain Examples', fontsize=20, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save the plot
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    plt.savefig(results_dir / 'full_domain_samples_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Full domain samples plot saved to {results_dir / 'full_domain_samples_plot.png'}")

def main():
    parser = argparse.ArgumentParser(description='Generate full domain samples plot')
    parser.add_argument('--samples_dir', type=str, help='Directory containing sample prediction images',
                       default='results/evaluation_20250228_093818/sample_predictions')
    args = parser.parse_args()
    
    plot_full_domain_samples(args.samples_dir)

if __name__ == '__main__':
    main() 