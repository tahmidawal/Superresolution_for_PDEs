import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import json
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.patches as patches

def plot_ml_multilevel_process(samples_dir=None, circular_samples_dir=None, results_dir=None):
    """
    Create a simplified visualization showing examples of ML multilevel upscaling,
    including examples with smooth circular theta fields.
    
    Args:
        samples_dir: Directory containing sample prediction images
        circular_samples_dir: Directory containing circular theta sample images
        results_dir: Directory to save the output visualization
    """
    # Set default paths if not provided
    if samples_dir is None:
        samples_dir = 'results/evaluations/evaluation_20250228_093818/sample_predictions'
    
    if circular_samples_dir is None:
        circular_samples_dir = 'results/data_samples/circular_theta_examples'
    
    if results_dir is None:
        results_dir = 'results/plots'
    
    # Ensure paths are Path objects
    samples_dir = Path(samples_dir)
    circular_samples_dir = Path(circular_samples_dir)
    results_dir = Path(results_dir)
    
    # Check if paths exist
    if not samples_dir.exists():
        print(f"Error: Samples directory not found at {samples_dir}")
        return
    
    if not circular_samples_dir.exists():
        print(f"Error: Circular samples directory not found at {circular_samples_dir}")
        return
    
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    # Define theta types
    theta_types = ['constant', 'grf', 'radial']
    
    # Get sample images for each theta type
    sample_images = {}
    for theta_type in theta_types:
        sample_images[theta_type] = list(samples_dir.glob(f"{theta_type}_sample_*.png"))
        # Sort to ensure consistent order
        sample_images[theta_type].sort()
    
    # Get circular theta sample images
    circular_samples = list(circular_samples_dir.glob("circular_sample_*.png"))
    circular_samples.sort()
    
    # Select three samples from each theta type
    selected_samples = {}
    for theta_type in theta_types:
        if len(sample_images[theta_type]) >= 3:
            selected_samples[theta_type] = sample_images[theta_type][:3]
        else:
            print(f"Warning: Not enough samples found for {theta_type}, using all available")
            selected_samples[theta_type] = sample_images[theta_type]
    
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    
    # Add a title for the entire figure
    fig.suptitle('ML Multilevel Upscaling Examples', fontsize=24, y=0.98)
    
    # Create a grid layout - 4 rows (one for each theta type + circular) and 3 columns (for the samples)
    gs = GridSpec(4, 3, figure=fig, wspace=0.05, hspace=0.1)
    
    # Process each theta type and sample
    for row, theta_type in enumerate(theta_types):
        if theta_type not in selected_samples:
            continue
        
        # Process each sample for this theta type
        for col, sample_path in enumerate(selected_samples[theta_type]):
            # Create a subplot for this sample
            ax = fig.add_subplot(gs[row, col])
            
            # Load the sample image
            img = mpimg.imread(sample_path)
            ax.imshow(img)
            
            # Add small title in the corner
            if col == 0:
                ax.text(10, 10, f"{theta_type.capitalize()}", 
                      color='white', fontsize=12, ha='left', va='top',
                      bbox=dict(facecolor='black', alpha=0.7))
            
            # Add sample number
            ax.text(img.shape[1]-10, 10, f"#{col+1}", 
                  color='white', fontsize=12, ha='right', va='top',
                  bbox=dict(facecolor='black', alpha=0.7))
            
            # Remove axis
            ax.axis('off')
    
    # Add circular theta samples in the fourth row
    for col, sample_path in enumerate(circular_samples[:3]):
        # Create a subplot for this sample
        ax = fig.add_subplot(gs[3, col])
        
        # Load the sample image
        img = mpimg.imread(sample_path)
        ax.imshow(img)
        
        # Add small title in the corner
        if col == 0:
            ax.text(10, 10, "Circular", 
                  color='white', fontsize=12, ha='left', va='top',
                  bbox=dict(facecolor='black', alpha=0.7))
        
        # Add sample number
        ax.text(img.shape[1]-10, 10, f"#{col+1}", 
              color='white', fontsize=12, ha='right', va='top',
              bbox=dict(facecolor='black', alpha=0.7))
        
        # Remove axis
        ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save the plot
    plt.savefig(results_dir / 'ml_multilevel_examples_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simplified visualization with circular theta examples saved to {results_dir / 'ml_multilevel_examples_simple.png'}")

def main():
    parser = argparse.ArgumentParser(description='Generate ML multilevel examples visualization with circular theta examples')
    parser.add_argument('--samples_dir', type=str, 
                       help='Directory containing sample prediction images',
                       default='results/evaluations/evaluation_20250228_093818/sample_predictions')
    parser.add_argument('--circular_samples_dir', type=str,
                       help='Directory containing circular theta sample images',
                       default='results/data_samples/circular_theta_examples')
    parser.add_argument('--results_dir', type=str, 
                       help='Directory to save the output visualization',
                       default='results/plots')
    args = parser.parse_args()
    
    plot_ml_multilevel_process(args.samples_dir, args.circular_samples_dir, args.results_dir)

if __name__ == '__main__':
    main() 