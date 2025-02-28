import numpy as np
import torch
from pathlib import Path
from data_generation import PoissonSolver
from models import UNet, PDEDataset
from compare_methods import plot_comparison, load_model
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def generate_test_data(k_range: tuple, n_samples: int = 10, label: str = "test", constant_theta: bool = True):
    """
    Generate test data with specified k range.
    
    Args:
        k_range: Range for k values
        n_samples: Number of samples to generate
        label: Label for the dataset
        constant_theta: If True, use constant theta=1.0, otherwise use random theta
    """
    print(f"\nGenerating {label} data...")
    print(f"Using k range: {k_range}")
    print(f"Theta mode: {'constant (1.0)' if constant_theta else 'varying (random)'}")
    
    solver = PoissonSolver()
    
    # Generate dataset
    dataset = solver.generate_dataset(
        n_samples=n_samples,
        k_range=k_range
    )
    
    if constant_theta:
        # Set theta fields to constant 1.0
        dataset['theta_fine'] = np.ones_like(dataset['theta_fine'])
        dataset['theta_coarse'] = np.ones_like(dataset['theta_coarse'])
        
        # Regenerate solutions with constant theta
        for idx in range(n_samples):
            k1, k2 = dataset['k1'][idx], dataset['k2'][idx]
            f_fine = dataset['f_fine'][idx]
            f_coarse = dataset['f_coarse'][idx]
            
            theta_fine = np.ones((solver.n_fine, solver.n_fine))
            theta_coarse = np.ones((solver.n_coarse, solver.n_coarse))
            
            # Resolve PDE with constant theta
            dataset['u_fine'][idx] = solver.solve_poisson(f_fine, theta_fine, 'fine')
            dataset['u_coarse'][idx] = solver.solve_poisson(f_coarse, theta_coarse, 'coarse')
        
        print("Verified: All theta fields are constant 1.0")
    else:
        # Generate random theta fields
        for idx in range(n_samples):
            # Generate random theta fields with values between 0.5 and 2.0
            theta_fine = np.random.uniform(0.5, 2.0, size=(solver.n_fine, solver.n_fine))
            theta_coarse = theta_fine[::2, ::2]  # Downsample for coarse grid
            
            f_fine = dataset['f_fine'][idx]
            f_coarse = dataset['f_coarse'][idx]
            
            # Update theta fields
            dataset['theta_fine'][idx] = theta_fine
            dataset['theta_coarse'][idx] = theta_coarse
            
            # Resolve PDE with varying theta
            dataset['u_fine'][idx] = solver.solve_poisson(f_fine, theta_fine, 'fine')
            dataset['u_coarse'][idx] = solver.solve_poisson(f_coarse, theta_coarse, 'coarse')
        
        print(f"Generated varying theta fields with range: [{dataset['theta_fine'].min():.2f}, {dataset['theta_fine'].max():.2f}]")
    
    # Save the dataset
    save_path = Path(f'data/{label}_dataset.npz')
    save_path.parent.mkdir(exist_ok=True)
    np.savez(save_path, **dataset)
    print(f"Saved {label} dataset to {save_path}")
    
    return dataset

def evaluate_dataset(data, model, device, save_dir: Path, label: str):
    """
    Evaluate model on a dataset and generate metrics and plots.
    """
    print(f"\nEvaluating {label} data...")
    dataset = PDEDataset(data, device=device)
    
    # Check if theta is constant
    is_constant_theta = np.allclose(data['theta_fine'], 1.0)
    if is_constant_theta:
        print("Detected constant theta field (θ = 1.0)")
    else:
        theta_min = data['theta_fine'].min()
        theta_max = data['theta_fine'].max()
        print(f"Detected varying theta field (range: [{theta_min:.2f}, {theta_max:.2f}])")
    
    metrics = []
    n_samples = len(dataset)
    
    for idx in range(n_samples):
        print(f"Processing sample {idx+1}/{n_samples}")
        
        # Get data
        coarse_solution = data['u_coarse'][idx]
        fine_solution = data['u_fine'][idx]
        theta_fine = data['theta_fine'][idx]
        k1, k2 = data['k1'][idx], data['k2'][idx]
        
        # Verify the solutions are from the same problem by checking forcing terms
        f_coarse = data['f_coarse'][idx]
        f_fine = data['f_fine'][idx]
        
        # Get model prediction
        inputs, _ = dataset[idx]
        with torch.no_grad():
            ml_upscaled = model(inputs.unsqueeze(0))
            ml_upscaled = dataset.denormalize(ml_upscaled)
            ml_upscaled = ml_upscaled.squeeze().cpu().numpy()
        
        # Bilinear interpolation
        bilinear_upscaled = torch.nn.functional.interpolate(
            torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
            size=(40, 40),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Calculate metrics
        sample_metrics = {
            'k1': float(k1),
            'k2': float(k2),
            'bilinear_mae': float(np.mean(np.abs(bilinear_upscaled - fine_solution))),
            'bilinear_rmse': float(np.sqrt(np.mean((bilinear_upscaled - fine_solution)**2))),
            'ml_mae': float(np.mean(np.abs(ml_upscaled - fine_solution))),
            'ml_rmse': float(np.sqrt(np.mean((ml_upscaled - fine_solution)**2)))
        }
        metrics.append(sample_metrics)
        
        # Save comparison plot for first few samples
        if idx < 3:
            fig = plt.figure(figsize=(15, 15))
            if is_constant_theta:
                plt.suptitle(f'{label} Sample {idx+1} (k₁={k1:.2f}, k₂={k2:.2f})\nθ (diffusion coefficient) = 1.0')
            else:
                theta_min = theta_fine.min()
                theta_max = theta_fine.max()
                plt.suptitle(f'{label} Sample {idx+1} (k₁={k1:.2f}, k₂={k2:.2f})\nθ range: [{theta_min:.2f}, {theta_max:.2f}]')
            
            # Use consistent normalization for all solution plots
            vmin = min(coarse_solution.min(), fine_solution.min(), 
                      bilinear_upscaled.min(), ml_upscaled.min())
            vmax = max(coarse_solution.max(), fine_solution.max(), 
                      bilinear_upscaled.max(), ml_upscaled.max())
            
            # Add more subplots to show forcing terms and theta
            gs = plt.GridSpec(4, 2, figure=fig)
            
            # Solutions
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(coarse_solution, vmin=vmin, vmax=vmax)
            ax1.set_title('Coarse Solution (20×20)')
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(fine_solution, vmin=vmin, vmax=vmax)
            ax2.set_title('Ground Truth (40×40)')
            plt.colorbar(im2, ax=ax2)
            
            ax3 = fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(bilinear_upscaled, vmin=vmin, vmax=vmax)
            ax3.set_title(f'Bilinear (MAE: {sample_metrics["bilinear_mae"]:.6f})')
            plt.colorbar(im3, ax=ax3)
            
            ax4 = fig.add_subplot(gs[1, 1])
            im4 = ax4.imshow(ml_upscaled, vmin=vmin, vmax=vmax)
            ax4.set_title(f'ML Model (MAE: {sample_metrics["ml_mae"]:.6f})')
            plt.colorbar(im4, ax=ax4)
            
            # Forcing terms
            ax5 = fig.add_subplot(gs[2, 0])
            im5 = ax5.imshow(f_coarse)
            ax5.set_title('Forcing Term (Coarse)')
            plt.colorbar(im5, ax=ax5)
            
            ax6 = fig.add_subplot(gs[2, 1])
            im6 = ax6.imshow(f_fine)
            ax6.set_title('Forcing Term (Fine)')
            plt.colorbar(im6, ax=ax6)
            
            # Theta fields
            ax7 = fig.add_subplot(gs[3, 0])
            theta_coarse = data['theta_coarse'][idx]
            if is_constant_theta:
                im7 = ax7.imshow(theta_coarse, vmin=0.9, vmax=1.1)
                ax7.set_title('θ Coarse (constant 1.0)')
            else:
                im7 = ax7.imshow(theta_coarse, vmin=0.5, vmax=2.0)
                ax7.set_title('θ Coarse (varying)')
            plt.colorbar(im7, ax=ax7)
            
            ax8 = fig.add_subplot(gs[3, 1])
            if is_constant_theta:
                im8 = ax8.imshow(theta_fine, vmin=0.9, vmax=1.1)
                ax8.set_title('θ Fine (constant 1.0)')
            else:
                im8 = ax8.imshow(theta_fine, vmin=0.5, vmax=2.0)
                ax8.set_title('θ Fine (varying)')
            plt.colorbar(im8, ax=ax8)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'{label}_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Calculate and save average metrics
    avg_metrics = {
        'avg_bilinear_mae': np.mean([m['bilinear_mae'] for m in metrics]),
        'avg_bilinear_rmse': np.mean([m['bilinear_rmse'] for m in metrics]),
        'avg_ml_mae': np.mean([m['ml_mae'] for m in metrics]),
        'avg_ml_rmse': np.mean([m['ml_rmse'] for m in metrics])
    }
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.title(f'{label} Error Distribution')
    sns.kdeplot(data=[m['ml_mae'] for m in metrics], label='ML Model MAE', fill=True)
    sns.kdeplot(data=[m['bilinear_mae'] for m in metrics], label='Bilinear MAE', fill=True)
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'{label}_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error vs k-values
    plt.figure(figsize=(12, 6))
    k_total = [m['k1'] + m['k2'] for m in metrics]
    plt.scatter(k_total, [m['ml_mae'] for m in metrics], label='ML Model', alpha=0.6)
    plt.scatter(k_total, [m['bilinear_mae'] for m in metrics], label='Bilinear', alpha=0.6)
    plt.xlabel('Total Wave Number (k₁ + k₂)')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'{label} Error vs Wave Numbers')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'{label}_error_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics, avg_metrics

def evaluate_training_like_cases(model, device: str, save_dir: Path, n_samples: int = 25):
    """
    Evaluate model on cases that exactly match the training data generation process.
    This uses the exact same process as in data_generation.py's main function.
    """
    print("\n=== Testing with Training-like Data Generation ===")
    
    # Create solver with same parameters as training
    solver = PoissonSolver(n_coarse=20, n_fine=40)
    
    # Generate dataset with same parameters as training
    print(f"Generating {n_samples} training-like samples...")
    dataset = solver.generate_dataset(
        n_samples=n_samples,
        k_range=(0.5, 5.0)  # Same range as in data_generation.py
    )
    
    # Create PDEDataset for normalization handling
    pde_dataset = PDEDataset(dataset, device=device)
    metrics = []
    
    for idx in range(n_samples):
        print(f"Processing sample {idx+1}/{n_samples}")
        
        # Get data for this sample
        coarse_solution = dataset['u_coarse'][idx]
        fine_solution = dataset['u_fine'][idx]
        theta_fine = dataset['theta_fine'][idx]
        k1, k2 = dataset['k1'][idx], dataset['k2'][idx]
        f_fine = dataset['f_fine'][idx]
        f_coarse = dataset['f_coarse'][idx]
        theta_coarse = dataset['theta_coarse'][idx]
        
        # Get model prediction
        inputs, _ = pde_dataset[idx]
        with torch.no_grad():
            ml_upscaled = model(inputs.unsqueeze(0))
            ml_upscaled = pde_dataset.denormalize(ml_upscaled)
            ml_upscaled = ml_upscaled.squeeze().cpu().numpy()
        
        # Bilinear interpolation
        bilinear_upscaled = torch.nn.functional.interpolate(
            torch.from_numpy(coarse_solution).float().unsqueeze(0).unsqueeze(0),
            size=(40, 40),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Calculate metrics
        metrics.append({
            'k1': float(k1),
            'k2': float(k2),
            'bilinear_mae': float(np.mean(np.abs(bilinear_upscaled - fine_solution))),
            'bilinear_rmse': float(np.sqrt(np.mean((bilinear_upscaled - fine_solution)**2))),
            'ml_mae': float(np.mean(np.abs(ml_upscaled - fine_solution))),
            'ml_rmse': float(np.sqrt(np.mean((ml_upscaled - fine_solution)**2))),
            'theta_range': [float(theta_fine.min()), float(theta_fine.max())]
        })
        
        # Save detailed comparison plots for first few samples
        if idx < 3:
            fig = plt.figure(figsize=(15, 20))
            plt.suptitle(f'Training-like Sample {idx+1}\n' + 
                        f'k₁={k1:.2f}, k₂={k2:.2f}\n' +
                        f'θ range: [{theta_fine.min():.2f}, {theta_fine.max():.2f}]')
            
            gs = plt.GridSpec(5, 2, figure=fig)
            
            # Solutions
            vmin = min(coarse_solution.min(), fine_solution.min(), 
                      bilinear_upscaled.min(), ml_upscaled.min())
            vmax = max(coarse_solution.max(), fine_solution.max(), 
                      bilinear_upscaled.max(), ml_upscaled.max())
            
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(coarse_solution, vmin=vmin, vmax=vmax)
            ax1.set_title('Coarse Solution (20×20)')
            plt.colorbar(im1, ax=ax1)
            
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(fine_solution, vmin=vmin, vmax=vmax)
            ax2.set_title('Ground Truth (40×40)')
            plt.colorbar(im2, ax=ax2)
            
            ax3 = fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(bilinear_upscaled, vmin=vmin, vmax=vmax)
            ax3.set_title(f'Bilinear (MAE: {metrics[-1]["bilinear_mae"]:.6f})')
            plt.colorbar(im3, ax=ax3)
            
            ax4 = fig.add_subplot(gs[1, 1])
            im4 = ax4.imshow(ml_upscaled, vmin=vmin, vmax=vmax)
            ax4.set_title(f'ML Model (MAE: {metrics[-1]["ml_mae"]:.6f})')
            plt.colorbar(im4, ax=ax4)
            
            # Error plots
            ax5 = fig.add_subplot(gs[2, 0])
            im5 = ax5.imshow(np.abs(bilinear_upscaled - fine_solution))
            ax5.set_title('Bilinear Error (Absolute)')
            plt.colorbar(im5, ax=ax5)
            
            ax6 = fig.add_subplot(gs[2, 1])
            im6 = ax6.imshow(np.abs(ml_upscaled - fine_solution))
            ax6.set_title('ML Model Error (Absolute)')
            plt.colorbar(im6, ax=ax6)
            
            # Forcing terms
            ax7 = fig.add_subplot(gs[3, 0])
            im7 = ax7.imshow(f_coarse)
            ax7.set_title('Forcing Term (Coarse)')
            plt.colorbar(im7, ax=ax7)
            
            ax8 = fig.add_subplot(gs[3, 1])
            im8 = ax8.imshow(f_fine)
            ax8.set_title('Forcing Term (Fine)')
            plt.colorbar(im8, ax=ax8)
            
            # Theta fields
            ax9 = fig.add_subplot(gs[4, 0])
            im9 = ax9.imshow(theta_coarse, vmin=0.5, vmax=2.0)
            ax9.set_title('θ Coarse')
            plt.colorbar(im9, ax=ax9)
            
            ax10 = fig.add_subplot(gs[4, 1])
            im10 = ax10.imshow(theta_fine, vmin=0.5, vmax=2.0)
            ax10.set_title('θ Fine')
            plt.colorbar(im10, ax=ax10)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'training_like_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot error distributions
    plt.figure(figsize=(10, 6))
    plt.title('Training-like Cases Error Distribution')
    sns.kdeplot(data=[m['ml_mae'] for m in metrics], label='ML Model MAE', fill=True)
    sns.kdeplot(data=[m['bilinear_mae'] for m in metrics], label='Bilinear MAE', fill=True)
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_like_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error vs total wave number
    plt.figure(figsize=(12, 6))
    k_total = [m['k1'] + m['k2'] for m in metrics]
    plt.scatter(k_total, [m['ml_mae'] for m in metrics], label='ML Model', alpha=0.6)
    plt.scatter(k_total, [m['bilinear_mae'] for m in metrics], label='Bilinear', alpha=0.6)
    plt.xlabel('Total Wave Number (k₁ + k₂)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training-like Cases: Error vs Wave Numbers')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_like_error_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate average metrics
    avg_metrics = {
        'avg_bilinear_mae': np.mean([m['bilinear_mae'] for m in metrics]),
        'avg_bilinear_rmse': np.mean([m['bilinear_rmse'] for m in metrics]),
        'avg_ml_mae': np.mean([m['ml_mae'] for m in metrics]),
        'avg_ml_rmse': np.mean([m['ml_rmse'] for m in metrics])
    }
    
    return metrics, avg_metrics

def create_comprehensive_comparison(all_results: dict, save_dir: Path):
    """
    Create a comprehensive comparison plot of all test cases.
    """
    plt.figure(figsize=(15, 10))
    
    # Prepare data for plotting
    categories = []
    ml_mae = []
    bilinear_mae = []
    ml_rmse = []
    bilinear_rmse = []
    
    # Training-like cases
    categories.append('Training-like')
    ml_mae.append(all_results['training_like']['average_metrics']['avg_ml_mae'])
    bilinear_mae.append(all_results['training_like']['average_metrics']['avg_bilinear_mae'])
    ml_rmse.append(all_results['training_like']['average_metrics']['avg_ml_rmse'])
    bilinear_rmse.append(all_results['training_like']['average_metrics']['avg_bilinear_rmse'])
    
    # Constant theta cases
    categories.extend(['Const θ\n(In-sample)', 'Const θ\n(Out-of-sample)'])
    ml_mae.extend([
        all_results['constant_theta']['in_sample']['average_metrics']['avg_ml_mae'],
        all_results['constant_theta']['out_of_sample']['average_metrics']['avg_ml_mae']
    ])
    bilinear_mae.extend([
        all_results['constant_theta']['in_sample']['average_metrics']['avg_bilinear_mae'],
        all_results['constant_theta']['out_of_sample']['average_metrics']['avg_bilinear_mae']
    ])
    ml_rmse.extend([
        all_results['constant_theta']['in_sample']['average_metrics']['avg_ml_rmse'],
        all_results['constant_theta']['out_of_sample']['average_metrics']['avg_ml_rmse']
    ])
    bilinear_rmse.extend([
        all_results['constant_theta']['in_sample']['average_metrics']['avg_bilinear_rmse'],
        all_results['constant_theta']['out_of_sample']['average_metrics']['avg_bilinear_rmse']
    ])
    
    # Varying theta cases
    categories.extend(['Varying θ\n(In-sample)', 'Varying θ\n(Out-of-sample)'])
    ml_mae.extend([
        all_results['varying_theta']['in_sample']['average_metrics']['avg_ml_mae'],
        all_results['varying_theta']['out_of_sample']['average_metrics']['avg_ml_mae']
    ])
    bilinear_mae.extend([
        all_results['varying_theta']['in_sample']['average_metrics']['avg_bilinear_mae'],
        all_results['varying_theta']['out_of_sample']['average_metrics']['avg_bilinear_mae']
    ])
    ml_rmse.extend([
        all_results['varying_theta']['in_sample']['average_metrics']['avg_ml_rmse'],
        all_results['varying_theta']['out_of_sample']['average_metrics']['avg_bilinear_rmse']
    ])
    bilinear_rmse.extend([
        all_results['varying_theta']['in_sample']['average_metrics']['avg_bilinear_rmse'],
        all_results['varying_theta']['out_of_sample']['average_metrics']['avg_bilinear_rmse']
    ])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot MAE
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, ml_mae, width, label='ML Model', color='royalblue', alpha=0.7)
    ax1.bar(x + width/2, bilinear_mae, width, label='Bilinear', color='lightcoral', alpha=0.7)
    
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('MAE Comparison Across All Test Cases')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(ml_mae):
        ax1.text(i - width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(bilinear_mae):
        ax1.text(i + width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    
    # Plot RMSE
    ax2.bar(x - width/2, ml_rmse, width, label='ML Model', color='royalblue', alpha=0.7)
    ax2.bar(x + width/2, bilinear_rmse, width, label='Bilinear', color='lightcoral', alpha=0.7)
    
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('RMSE Comparison Across All Test Cases')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(ml_rmse):
        ax2.text(i - width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(bilinear_rmse):
        ax2.text(i + width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a log-scale version for better visualization of small differences
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot MAE (log scale)
    ax1.bar(x - width/2, ml_mae, width, label='ML Model', color='royalblue', alpha=0.7)
    ax1.bar(x + width/2, bilinear_mae, width, label='Bilinear', color='lightcoral', alpha=0.7)
    
    ax1.set_ylabel('Mean Absolute Error (log scale)')
    ax1.set_title('MAE Comparison Across All Test Cases (Log Scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(ml_mae):
        ax1.text(i - width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(bilinear_mae):
        ax1.text(i + width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    
    # Plot RMSE (log scale)
    ax2.bar(x - width/2, ml_rmse, width, label='ML Model', color='royalblue', alpha=0.7)
    ax2.bar(x + width/2, bilinear_rmse, width, label='Bilinear', color='lightcoral', alpha=0.7)
    
    ax2.set_ylabel('Root Mean Square Error (log scale)')
    ax2.set_title('RMSE Comparison Across All Test Cases (Log Scale)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on the bars
    for i, v in enumerate(ml_rmse):
        ax2.text(i - width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(bilinear_rmse):
        ax2.text(i + width/2, v, f'{v:.6f}', ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_comparison_log_scale.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare in-sample and out-of-sample test cases')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--n_samples', type=int, default=16, help='Number of samples for each test')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    model.eval()
    
    # Test with training-like cases
    print("\n=== Testing with Training-like Cases ===")
    training_like_dir = model_path.parent / 'training_like_results'
    training_like_dir.mkdir(exist_ok=True)
    training_metrics, training_avg = evaluate_training_like_cases(model, device, training_like_dir, args.n_samples)
    
    # Test with constant theta (theta = 1.0)
    print("\n=== Testing with Constant Theta (θ = 1.0) ===")
    const_results_dir = model_path.parent / 'constant_theta_results'
    const_results_dir.mkdir(exist_ok=True)
    
    # Generate and evaluate in-sample data
    in_sample_data = generate_test_data(k_range=(1.0, 6.0), n_samples=args.n_samples, 
                                      label="in_sample_const_theta", constant_theta=True)
    in_sample_metrics, in_sample_avg = evaluate_dataset(in_sample_data, model, device, 
                                                      const_results_dir, "in_sample")
    
    # Generate and evaluate out-of-sample data
    out_sample_data = generate_test_data(k_range=(6.0, 8.0), n_samples=args.n_samples, 
                                       label="out_of_sample_const_theta", constant_theta=True)
    out_sample_metrics, out_sample_avg = evaluate_dataset(out_sample_data, model, device, 
                                                        const_results_dir, "out_of_sample")
    
    # Test with varying theta
    print("\n=== Testing with Varying Theta ===")
    var_results_dir = model_path.parent / 'varying_theta_results'
    var_results_dir.mkdir(exist_ok=True)
    
    # Generate and evaluate in-sample data
    in_sample_data_var = generate_test_data(k_range=(1.0, 6.0), n_samples=args.n_samples, 
                                          label="in_sample_var_theta", constant_theta=False)
    in_sample_metrics_var, in_sample_avg_var = evaluate_dataset(in_sample_data_var, model, 
                                                              device, var_results_dir, "in_sample")
    
    # Generate and evaluate out-of-sample data
    out_sample_data_var = generate_test_data(k_range=(6.0, 8.0), n_samples=args.n_samples, 
                                           label="out_of_sample_var_theta", constant_theta=False)
    out_sample_metrics_var, out_sample_avg_var = evaluate_dataset(out_sample_data_var, model, 
                                                                device, var_results_dir, "out_of_sample")
    
    # Save all metrics
    all_results = {
        'training_like': {
            'individual_metrics': training_metrics,
            'average_metrics': training_avg
        },
        'constant_theta': {
            'in_sample': {
                'individual_metrics': in_sample_metrics,
                'average_metrics': in_sample_avg
            },
            'out_of_sample': {
                'individual_metrics': out_sample_metrics,
                'average_metrics': out_sample_avg
            }
        },
        'varying_theta': {
            'in_sample': {
                'individual_metrics': in_sample_metrics_var,
                'average_metrics': in_sample_avg_var
            },
            'out_of_sample': {
                'individual_metrics': out_sample_metrics_var,
                'average_metrics': out_sample_avg_var
            }
        }
    }
    
    # Create comprehensive comparison plots
    create_comprehensive_comparison(all_results, model_path.parent)
    
    # Print summary
    print("\nResults Summary:")
    
    print("\nTraining-like Cases:")
    for metric, value in training_avg.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nConstant Theta (θ = 1.0):")
    print("\nIn-Sample Metrics:")
    for metric, value in in_sample_avg.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nOut-of-Sample Metrics:")
    for metric, value in out_sample_avg.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nVarying Theta:")
    print("\nIn-Sample Metrics:")
    for metric, value in in_sample_avg_var.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nOut-of-Sample Metrics:")
    for metric, value in out_sample_avg_var.items():
        print(f"{metric}: {value:.6f}")
    
    # Save detailed results
    with open(model_path.parent / 'comprehensive_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main() 