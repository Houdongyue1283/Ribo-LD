import numpy as np
import pandas as pd
import torch
import traceback
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.stats
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams['font.family'] = 'Arial'
import os
import random
from sklearn.manifold import TSNE
from datetime import datetime
from model import get_model
from pathlib import Path
from multiprocessing import cpu_count
from torch.optim import Adam
from tqdm.auto import tqdm

# Diffusion model imports (may need adjustment depending on the diffusion library)
try:
    from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
    from ema_pytorch import EMA
    DIFFUSION_AVAILABLE = True
except ImportError:
    print("Warning: Diffusion library not available. Diffusion training mode will be disabled.")
    DIFFUSION_AVAILABLE = False
# Data preprocessing
def one_hot_encode(seq, target_length=200):
    """
    One-hot encode DNA sequence and pad/truncate to target length
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3, 'P':4}  # Add encoding for P
    
    # Pad or truncate sequence to target length
    if len(seq) > target_length:
        seq = seq[:target_length]  # Truncate if too long
    elif len(seq) < target_length:
        seq = seq + 'P' * (target_length - len(seq))  # Pad with 'P' if too short
    
    arr = np.zeros((target_length, 5), dtype=np.float32)  # 5D now, including P
    for i, c in enumerate(seq):
        if c in mapping:
            arr[i, mapping[c]] = 1.0
        # For unknown characters, leave as all zeros
    return arr.flatten()

class actDataset(Dataset):
    def __init__(self, txt_file, target_length=200):
        df = pd.read_csv(txt_file, sep='\t', header=None)
        self.seqs = df[0].values
        # self.y = -np.log10(df[1].astype(float).values)
        self.y = df[1].astype(float).values
        self.target_length = target_length
        self.X = np.array([one_hot_encode(seq, target_length) for seq in self.seqs], dtype=np.float32)
        self.y = self.y.astype(np.float32).reshape(-1, 1)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cycle(dl):
    """Infinitely cycle through a dataloader."""
    while True:
        for data in dl:
            yield data

class ActivityEvaluationTrainer(object):
    """
    Custom diffusion trainer with activity evaluation and optional dataset optimization.

    This follows the overall design of Trainer1D in denoising_diffusion_pytorch_1d.
    """
    def __init__(
        self,
        diffusion_model,  # GaussianDiffusion1D
        ae_decoder,  # AE decoder used to decode latent vectors
        dataset: Dataset,
        val_dataset: Dataset = None,
        true_activities: list = None,  # Ground-truth activity values
        data_path: str = None,  # Original dataset path (for dataset optimization)
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 10000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 500,
        activity_eval_every = 100,  # Activity evaluation frequency
        num_samples = 5000,  # Increase samples to select the best 500
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        vocab = None,
        early_stopping_patience = 10,
        early_stopping_min_delta = 1e-4,
        early_stopping_eval_every = 500,
        enable_dataset_optimization = True,  # Enable dataset optimization
        replacement_count = 500,  # Number of samples to replace each round
        lower_is_better = True,  # Activity interpretation: True=lower is better, False=higher is better
        act_ts = None,
        seq_len=200
    ):
        super().__init__()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = diffusion_model.to(self.device)
        self.ae_decoder = ae_decoder.to(self.device)
        self.ae_decoder.eval()  # Set decoder to eval mode
        
        # Training parameters
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps
        self.vocab = vocab or {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'P': 4}
        
        # Activity evaluation
        self.true_activities = true_activities or []
        self.activity_eval_every = activity_eval_every
        self.num_samples = num_samples
        
        # Dataset optimization
        self.enable_dataset_optimization = enable_dataset_optimization
        self.replacement_count = replacement_count
        self.data_path = data_path
        self.lower_is_better = lower_is_better  # Activity interpretation
        self.original_dataset = None
        if data_path and enable_dataset_optimization:
            self.original_dataset = dataset
            # self.original_dataset = actDataset(data_path, target_length=100)
            print(f"Loaded original dataset for optimization: {len(self.original_dataset)} samples")
            print(f"Activity interpretation: {'Lower values = better activity' if lower_is_better else 'Higher values = better activity'}")
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_eval_every = early_stopping_eval_every
        
        # Activity tracking - used for plotting mean activity
        self.activity_tracking = {
            'steps': [],
            'mean_activities': [],
            'best_activities': [],  # Best activity in each evaluation
            'worst_activities': [],  # Worst activity in each evaluation
            'std_activities': [],   # Standard deviation of activity values
            'sample_counts': []     # Number of samples evaluated each time
        }
        self.act_ts=act_ts
        self.seq_len=seq_len
        # Data loader
        # dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, 
        #                pin_memory=True, num_workers=min(4, cpu_count()))
        dl = DataLoader(self.original_dataset, batch_size=train_batch_size, shuffle=True, 
                       pin_memory=True, num_workers=min(4, cpu_count()))
        self.dl = cycle(dl)
        
        if val_dataset is not None:
            self.val_dl = DataLoader(val_dataset, batch_size=train_batch_size, 
                                   shuffle=False, pin_memory=True, num_workers=min(4, cpu_count()))
        else:
            self.val_dl = None
        
        # Optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        
        # EMA (simplified)
        try:
            if DIFFUSION_AVAILABLE:
                self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
                self.ema.to(self.device)
            else:
                self.ema = None
        except:
            self.ema = None
        
        # Result saving
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.save_and_sample_every = save_and_sample_every
        
        # Step counter
        self.step = 0
    
    def save(self, milestone):
        """Save model checkpoint."""
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'version': '1.0'
        }
        
        if self.ema is not None:
            data['ema'] = self.ema.state_dict()
            
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    def load(self, milestone):
        """Load model checkpoint."""
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)
        
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        if self.ema is not None and 'ema' in data:
            self.ema.load_state_dict(data['ema'])
    
    def validate(self):
        """Validation function."""
        if self.val_dl is None:
            return float('inf')
            
        self.model.eval()
        total_val_loss = 0.
        num_batches = 0
        
        with torch.no_grad():
            for val_data in self.val_dl:
                val_data = val_data.to(self.device)
                loss = self.model(val_data)
                total_val_loss += loss.item()
                num_batches += 1
                
        avg_val_loss = total_val_loss / max(num_batches, 1)
        return avg_val_loss
    
    def evaluate_activity(self, num_eval_samples=None):
        """Traditional activity evaluation (no dataset optimization)."""
        if num_eval_samples is None:
            num_eval_samples = min(self.num_samples, 100)  # Use fewer samples for the traditional method
            
        if num_eval_samples == 0:
            print("No samples to evaluate activity for")
            return None
        
        self.model.eval()
        
        with torch.no_grad():
            try:
                # Generate latent vectors
                if self.ema is not None:
                    generated_latents = self.ema.ema_model.sample(batch_size=num_eval_samples)
                else:
                    generated_latents = self.model.sample(batch_size=num_eval_samples)
                
                # Handle generated latent shape
                if len(generated_latents.shape) == 3:
                    generated_latents = generated_latents.squeeze(1)  # Remove channel dimension
                
                # Use decoder and activity_predictor
                x_rec, dec_intermediate = self.ae_decoder.decoder(generated_latents)
                activity_pred = self.ae_decoder.activity_predictor(dec_intermediate)
                predicted_activities = activity_pred.detach().cpu().numpy().flatten()
                
                if len(predicted_activities) > 0:
                    print(f"Generated {len(predicted_activities)} samples with predicted activities:")
                    print(f"  Mean activity: {np.mean(predicted_activities):.4f}")
                    print(f"  Std activity: {np.std(predicted_activities):.4f}")
                    print(f"  Min activity: {np.min(predicted_activities):.4f}")
                    print(f"  Max activity: {np.max(predicted_activities):.4f}")
                    print(f"  Sample activities: {predicted_activities[:5]}")  # Show first 5 samples
                
                return predicted_activities
                
            except Exception as e:
                print(f"Warning: Activity prediction failed: {e}")
                return None

    def evaluate_activity_and_optimize_dataset(self, num_eval_samples=None):
        """
        Predict activities for generated latent vectors and optimize the dataset.

        Implements a bottom-elimination strategy: replace the worst samples in the
        original dataset with the best generated samples.
        """
        if num_eval_samples is None:
            num_eval_samples = self.num_samples
            
        if num_eval_samples == 0:
            print("No samples to evaluate activity for")
            return None
        
        self.model.eval()
        
        with torch.no_grad():
            try:
                # Generate latent vectors
                if self.ema is not None:
                    generated_latents = self.ema.ema_model.sample(batch_size=num_eval_samples)
                else:
                    generated_latents = self.model.sample(batch_size=num_eval_samples)
                
                # Handle generated latent shape
                if len(generated_latents.shape) == 3:
                    generated_latents = generated_latents.squeeze(1)  # Remove channel dimension
                
                # Reconstruct sequences and predict activity
                # x_rec, dec_intermediate = self.ae_decoder.decoder(generated_latents)
                # activity_pred = self.ae_decoder.activity_predictor(dec_intermediate)
                # print(generated_latents.shape)
                x_rec = self.ae_decoder.decoder(generated_latents)
                activity_pred = self.ae_decoder.activity_predictor(generated_latents)
                ori_pred = self.ae_decoder.activity_predictor(self.original_dataset.tensor.to(device=self.device))

                predicted_activities = activity_pred.cpu().numpy().flatten()
                ori_activities = ori_pred.cpu().numpy().flatten()

                # Reconstructed sequences (x_rec) have shape (batch_size, seq_len*5)
                x_rec_numpy = x_rec.cpu().numpy()
                
                if len(predicted_activities) > 0:
                    # Compute activity statistics
                    mean_activity = np.mean(predicted_activities)
                    std_activity = np.std(predicted_activities)
                    min_activity = np.min(predicted_activities)
                    max_activity = np.max(predicted_activities)
                    
                    print(f"Generated {len(predicted_activities)} samples with predicted activities:")
                    print(f"  Mean activity: {mean_activity:.4f}")
                    print(f"  Std activity: {std_activity:.4f}")
                    print(f"  Min activity: {min_activity:.4f}")
                    print(f"  Max activity: {max_activity:.4f}")
                    
                    # Update activity tracking
                    # Determine best and worst activity based on interpretation
                    if self.lower_is_better:
                        best_activity = min_activity  # Lower value = better activity
                        worst_activity = max_activity
                    else:
                        best_activity = max_activity  # Higher value = better activity
                        worst_activity = min_activity
                    
                    self.activity_tracking['steps'].append(self.step)
                    self.activity_tracking['mean_activities'].append(mean_activity)
                    self.activity_tracking['best_activities'].append(best_activity)
                    self.activity_tracking['worst_activities'].append(worst_activity)
                    self.activity_tracking['std_activities'].append(std_activity)
                    self.activity_tracking['sample_counts'].append(len(predicted_activities))
                    
                    # Save activity evaluation results to file
                    with open(self.results_folder / 'activity_evaluation_log.txt', 'a') as f:
                        f.write(f"Step {self.step}: Samples={len(predicted_activities)}, "
                               f"Mean={mean_activity:.4f}, Best={best_activity:.4f}, "
                               f"Worst={worst_activity:.4f}, Std={std_activity:.4f} "
                               f"({'Lower is better' if self.lower_is_better else 'Higher is better'})\n")
                    
                    # Plot activity tracking
                    self.plot_activity_tracking()
                    
                    # Dataset optimization: bottom-elimination
                    if self.enable_dataset_optimization and self.original_dataset is not None:
                        self.optimize_dataset_with_best_samples(x_rec_numpy, predicted_activities, ori_activities, generated_latents,self.act_ts)
                
                return predicted_activities
                
            except Exception as e:
                print(f"Warning: Activity prediction failed: {e}")
                return None

    def optimize_dataset_with_best_samples(self, generated_sequences, predicted_activities, ori_activities, generated_latents, threshold=2.9):
        """
        Replace low-activity samples in the original dataset with high-activity generated samples.

        The meaning of "high activity" is controlled by lower_is_better.
        """
        try:
            # 1) Select high-activity generated samples based on the interpretation
            if self.lower_is_better:
                # If lower value means higher activity, select samples below the threshold
                high_activity_indices = np.where(predicted_activities < threshold)[0]
                comparison_op = "<"
            else:
                # If higher value means higher activity, select samples above the threshold
                high_activity_indices = np.where(predicted_activities > threshold)[0]
                comparison_op = ">"
            
            N = len(high_activity_indices)
            if N == 0:
                print(f"No generated samples with activity {comparison_op} {threshold}")
                return
                
            best_latent = generated_latents[high_activity_indices]
            best_activities = predicted_activities[high_activity_indices]
            print(f"Selected {N} samples with activity {comparison_op} {threshold} for dataset optimization.")
            print(f"  Activity range: {best_activities.min():.4f} - {best_activities.max():.4f}")
            print(f"  Mean of selected activities: {np.mean(best_activities):.4f}")

            # 2) Find the N worst samples in the original dataset
            original_activities = ori_activities
            if self.lower_is_better:
                # If lower value means higher activity, worst samples have the largest values
                worst_indices = np.argsort(original_activities)[-N:]  # Take largest N
                worst_type = "highest"
            else:
                # If higher value means higher activity, worst samples have the smallest values
                worst_indices = np.argsort(original_activities)[:N]   # Take smallest N
                worst_type = "lowest"
                
            print(f"Replacing {worst_type} {N} samples from original dataset.")
            print(f"  Worst activity range: {original_activities[worst_indices].min():.4f} - {original_activities[worst_indices].max():.4f}")
            print(f"  Mean of worst activities: {np.mean(original_activities[worst_indices]):.4f}")

            # 3) (Optional) Reconstruct sequences (convert one-hot to strings)


            # 4) Replace the worst samples in the original dataset
            for i, worst_idx in enumerate(worst_indices):
                if i < len(best_latent):
                    # self.original_dataset.seqs[worst_idx] = best_latent[i]
                    self.original_dataset[worst_idx] = best_latent[i]
            # 6) Record optimization statistics
            # Compute improvement based on the interpretation
            if self.lower_is_better:
                # If lower value means higher activity: improvement = old - new (positive = better)
                improvement = np.mean(original_activities[worst_indices]) - np.mean(best_activities)
                best_sample_activity = best_activities.min()  # Best sample has the smallest value
                replaced_worst_activity = original_activities[worst_indices].max()  # Replaced worst sample has the largest value
            else:
                # If higher value means higher activity: improvement = new - old (positive = better)
                improvement = np.mean(best_activities) - np.mean(original_activities[worst_indices])
                best_sample_activity = best_activities.max()  # Best sample has the largest value
                replaced_worst_activity = original_activities[worst_indices].min()  # Replaced worst sample has the smallest value
                
            print(f"Dataset optimization completed:")
            print(f"  Average improvement in activity: {improvement:.4f}")
            print(f"  Best sample activity: {best_sample_activity:.4f}")
            print(f"  Replaced worst activity: {replaced_worst_activity:.4f}")
            print(f"  Activity interpretation: {'Lower is better' if self.lower_is_better else 'Higher is better'}")
            
            with open(self.results_folder / 'dataset_optimization_log.txt', 'a') as f:
                f.write(f"Step {self.step}: Replaced {N} samples (Activity mode: {'Lower is better' if self.lower_is_better else 'Higher is better'})\n")
                f.write(f"  Selected activity range: {best_activities.min():.4f} - {best_activities.max():.4f}\n")
                f.write(f"  Replaced activity range: {original_activities[worst_indices].min():.4f} - {original_activities[worst_indices].max():.4f}\n")
                f.write(f"  Average improvement: {improvement:.4f}\n")
                f.write(f"  Best sample activity: {best_sample_activity:.4f}\n")
                f.write(f"  Replaced worst activity: {replaced_worst_activity:.4f}\n")
                # f.write(f"  Optimized dataset saved to: {optimized_data_path}\n\n")
        except Exception as e:
            print(f"Warning: Dataset optimization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_activity_tracking(self):
        """Plot mean activity vs training steps (only keep the latest version)."""
        if len(self.activity_tracking['steps']) < 2:
            return  # Need at least two points to plot

        try:
            import matplotlib.pyplot as plt
            
            steps = self.activity_tracking['steps']
            mean_activities = self.activity_tracking['mean_activities']
            std_activities = self.activity_tracking['std_activities']

            # Create figure
            plt.figure(figsize=(8,6))
            
            # Main curve + uncertainty band
            plt.plot(steps, mean_activities, 'b-', linewidth=2, marker='o', markersize=4, label='Mean Activity')
            plt.fill_between(
                steps,
                [m - s for m, s in zip(mean_activities, std_activities)],
                [m + s for m, s in zip(mean_activities, std_activities)],
                alpha=0.3, color='blue', label='±1 Std'
            )
            
            plt.xlabel('Training Steps',fontsize=22)
            plt.ylabel('Predicted Activities',fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            # plt.title('Generated Samples: Mean Activity Evolution')
            plt.legend(fontsize=18)
            # plt.grid(True, alpha=0.3)
            
                # Optional: add stats textbox
            # stats_text = (
            #     f'Latest Stats (Step {steps[-1]}):\n'
            #     f'Mean: {mean_activities[-1]:.4f}\n'
            #     f'Std: {std_activities[-1]:.4f}'
            # )
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    # fontsize=9, verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save figure
            latest_plot_path = self.results_folder / "activity_tracking_latest.png"
            plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Activity tracking plot updated: {latest_plot_path}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to plot activity tracking: {e}")

    def plot_losses_and_activities(self, training_losses, validation_losses):
        """Plot loss curves together with activity metrics."""
        try:
            import matplotlib.pyplot as plt
            
            # Decide subplot layout
            if len(self.activity_tracking['steps']) > 1:
                # If there is activity data, use a 2x3 layout
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                ax1, ax2, ax3 = axes[0]
                ax4, ax5, ax6 = axes[1]
            else:
                # If there is no activity data, fall back to a 1x2 layout
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax3 = ax4 = ax5 = ax6 = None
            
            # Training and validation losses
            if training_losses:
                steps_train, train_loss_vals = zip(*training_losses)
                ax1.plot(steps_train, train_loss_vals, label="Training Loss", color='blue')
                
            if validation_losses:
                steps_val, val_loss_vals = zip(*validation_losses)
                ax1.plot(steps_val, val_loss_vals, label="Validation Loss", color='red')
                
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.set_title("Training and Validation Loss")
            ax1.grid(True)
            
            # Second subplot: loss on log scale
            if training_losses:
                ax2.semilogy(steps_train, train_loss_vals, label="Training Loss", color='blue')
            if validation_losses:
                ax2.semilogy(steps_val, val_loss_vals, label="Validation Loss", color='red')
                
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss (log scale)")
            ax2.legend()
            ax2.set_title("Training and Validation Loss (Log Scale)")
            ax2.grid(True)
            
            # If there is activity tracking data, add activity-related subplots
            if len(self.activity_tracking['steps']) > 1:
                steps = self.activity_tracking['steps']
                mean_activities = self.activity_tracking['mean_activities']
                best_activities = self.activity_tracking['best_activities']
                worst_activities = self.activity_tracking['worst_activities']
                std_activities = self.activity_tracking['std_activities']
                
                # Third subplot: mean activity
                ax3.plot(steps, mean_activities, 'b-', linewidth=2, marker='o', markersize=3, label='Mean Activity')
                ax3.fill_between(steps, 
                               [m - s for m, s in zip(mean_activities, std_activities)],
                               [m + s for m, s in zip(mean_activities, std_activities)],
                               alpha=0.3, color='blue', label='±1 Std')
                ax3.set_xlabel("Step")
                ax3.set_ylabel("Mean Activity")
                ax3.set_title("Generated Samples: Mean Activity")
                ax3.legend()
                ax3.grid(True)
                
                # Fourth subplot: best vs worst activity
                ax4.plot(steps, best_activities, 'g-', linewidth=2, marker='o', markersize=3, label='Best')
                ax4.plot(steps, worst_activities, 'r-', linewidth=2, marker='s', markersize=3, label='Worst')
                ax4.set_xlabel("Step")
                ax4.set_ylabel("Activity Value")
                ax4.set_title("Best vs Worst Activity")
                ax4.legend()
                ax4.grid(True)
                
                # Fifth subplot: activity standard deviation
                ax5.plot(steps, std_activities, 'purple', linewidth=2, marker='d', markersize=3)
                ax5.set_xlabel("Step")
                ax5.set_ylabel("Std Deviation")
                ax5.set_title("Activity Diversity")
                ax5.grid(True)
                
                # Sixth subplot: activity improvement trend (if multiple data points)
                if len(mean_activities) > 1:
                    # Compute improvement relative to the initial value
                    initial_mean = mean_activities[0]
                    if self.lower_is_better:
                        # Decrease in value is improvement
                        improvement = [(initial_mean - m) for m in mean_activities]
                        improvement_label = "Activity Improvement (Lower is Better)"
                    else:
                        # Increase in value is improvement
                        improvement = [(m - initial_mean) for m in mean_activities]
                        improvement_label = "Activity Improvement (Higher is Better)"
                        
                    ax6.plot(steps, improvement, 'orange', linewidth=2, marker='^', markersize=3)
                    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax6.set_xlabel("Step")
                    ax6.set_ylabel("Activity Improvement")
                    ax6.set_title(improvement_label)
                    ax6.grid(True)
                    
                    # Add improvement direction indicator
                    if improvement[-1] > 0:
                        ax6.text(0.7, 0.9, 'Improving ↑', transform=ax6.transAxes, 
                                color='green', fontweight='bold')
                    elif improvement[-1] < 0:
                        ax6.text(0.7, 0.9, 'Worsening ↓', transform=ax6.transAxes, 
                                color='red', fontweight='bold')
                    else:
                        ax6.text(0.7, 0.9, 'Stable →', transform=ax6.transAxes, 
                                color='blue', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.results_folder / "comprehensive_training_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to plot comprehensive metrics: {e}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting diffusion training with activity prediction...")
        print(f"Device: {self.device}")
        print(f"Results folder: {self.results_folder}")
        
        # Training logs
        training_losses = []
        validation_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Progress bar
        pbar = tqdm(initial=self.step, total=self.train_num_steps, desc="Training")
        
        try:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.
                
                # Training step
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(self.device)
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    loss.backward()
                
                # Update parameters
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                # Record training loss
                training_losses.append((self.step, total_loss))
                pbar.set_description(f'loss: {total_loss:.4f}')
                
                # Write training loss
                with open(self.results_folder / 'loss_train.txt', 'a') as f:
                    f.write(f"{self.step}\t{total_loss}\n")
                
                self.step += 1
                
                # Validation and early stopping
                if self.step % self.early_stopping_eval_every == 0:
                    val_loss = self.validate()
                    validation_losses.append((self.step, val_loss))
                    
                    with open(self.results_folder / 'loss_val.txt', 'a') as f:
                        f.write(f"{self.step}\t{val_loss}\n")
                    
                    print(f'Step {self.step}: Val Loss = {val_loss:.4f}')
                    
                    # Check if improved
                    if val_loss < best_val_loss - self.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save('best')
                    else:
                        patience_counter += 1
                    
                    # Early stopping check
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at step {self.step}")
                        break
                
                # Activity prediction and dataset optimization
                if self.step % self.activity_eval_every == 0:
                    print(f"\n--- Step {self.step}: Activity Prediction {'and Dataset Optimization' if self.enable_dataset_optimization else ''} ---")
                    
                    if self.enable_dataset_optimization:
                        predicted_activities = self.evaluate_activity_and_optimize_dataset()
                    else:
                        predicted_activities = self.evaluate_activity()
                    
                    if predicted_activities is not None:
                        # Save predicted activity values
                        with open(self.results_folder / 'predicted_activities.txt', 'a') as f:
                            f.write(f"Step {self.step}: {predicted_activities.tolist()}\n")
                        print(f"Activity prediction {'and dataset optimization' if self.enable_dataset_optimization else ''} completed.")
                    else:
                        print(f"Activity prediction failed at step {self.step}")
                    print(f"--- End Activity Prediction {'and Dataset Optimization' if self.enable_dataset_optimization else ''} ---\n")
                
                # Periodic checkpointing and sampling
                if self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)
                    
                    # Plot training curves
                    self.plot_losses_and_activities(training_losses, validation_losses)
                # Use decoder and activity_predictor
                if self.step % self.activity_eval_every == 0:
                    if self.ema is not None:
                        generated_latents = self.ema.ema_model.sample(batch_size=3200)
                    else:
                        generated_latents = self.model.sample(batch_size=3200)
                    
                    # Handle generated latent shape
                    if len(generated_latents.shape) == 3:
                        generated_latents = generated_latents.squeeze(1)  # Remove channel dimension
                    
                    # Use decoder and activity_predictor
                    x_rec = self.ae_decoder.decoder(generated_latents)
                    activity_pred = self.ae_decoder.activity_predictor(generated_latents)
                    predicted_activities = activity_pred.detach().cpu().numpy().flatten()
                    x_rec_numpy = x_rec.detach().cpu().numpy()
                    allseqs = []
                    for seq, activity in zip(x_rec_numpy, predicted_activities):
                        seq_reshaped = seq.reshape(self.seq_len, 5)
                        indices = np.argmax(seq_reshaped, axis=1)
                        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'P'}
                        seq_str = ''.join([mapping[idx] for idx in indices])
                        allseqs.append((seq_str, activity))
                    sample_path = self.results_folder / f'sampled_sequences_step_{self.step}.txt'
                    with open(sample_path, 'w') as f:
                        for seq, activity in allseqs:
                            seq = seq.split('P', 1)[0]
                            f.write(f"{seq}\t{activity:.6f}\n")
                
                pbar.update(1)



        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            pbar.close()
            
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Final save
        self.save('final')
        self.plot_losses_and_activities(training_losses, validation_losses)
        
        # Save activity tracking data
        self.save_activity_tracking_data()
        
        # Generate final activity report
        self.generate_final_activity_report()
        
        return {
            'final_step': self.step,
            'best_val_loss': best_val_loss,
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'activity_tracking': self.activity_tracking
        }
    
    def save_activity_tracking_data(self):
        """Save activity tracking data to files."""
        try:
            import json
            import numpy as np
            
            # Save as JSON (for downstream analysis)
            activity_data = {
                'steps': self.activity_tracking['steps'],
                'mean_activities': self.activity_tracking['mean_activities'],
                'best_activities': self.activity_tracking['best_activities'],
                'worst_activities': self.activity_tracking['worst_activities'],
                'std_activities': self.activity_tracking['std_activities'],
                'sample_counts': self.activity_tracking['sample_counts']
            }
            
            json_path = self.results_folder / 'activity_tracking_data.json'
            with open(json_path, 'w') as f:
                json.dump(activity_data, f, indent=2)
            
            # Save as NumPy format (convenient for downstream processing)
            np_path = self.results_folder / 'activity_tracking_data.npz'
            np.savez(np_path, **activity_data)
            
            print(f"Activity tracking data saved to: {json_path} and {np_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save activity tracking data: {e}")
    
    def generate_final_activity_report(self):
        """Generate a final activity tracking report."""
        try:
            if len(self.activity_tracking['steps']) == 0:
                print("No activity tracking data available for report generation")
                return
            
            steps = self.activity_tracking['steps']
            mean_activities = self.activity_tracking['mean_activities']
            best_activities = self.activity_tracking['best_activities']
            worst_activities = self.activity_tracking['worst_activities']
            std_activities = self.activity_tracking['std_activities']
            sample_counts = self.activity_tracking['sample_counts']
            
            report_path = self.results_folder / 'final_activity_report.txt'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DIFFUSION MODEL ACTIVITY TRACKING FINAL REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Training Period: Step {steps[0]} to Step {steps[-1]}\n")
                f.write(f"Total Evaluations: {len(steps)}\n")
                f.write(f"Evaluation Frequency: Every {self.activity_eval_every} steps\n")
                f.write(f"Total Samples Generated: {sum(sample_counts)}\n\n")
                
                f.write("ACTIVITY EVOLUTION SUMMARY:\n")
                f.write("-" * 50 + "\n")
                
                # Initial vs final comparison
                initial_mean = mean_activities[0]
                final_mean = mean_activities[-1]
                
                # Compute improvement based on the interpretation
                if self.lower_is_better:
                    improvement = initial_mean - final_mean  # Decrease in value is improvement
                    improvement_direction = "decreasing"
                else:
                    improvement = final_mean - initial_mean  # Increase in value is improvement
                    improvement_direction = "increasing"
                    
                improvement_pct = (improvement / abs(initial_mean)) * 100 if initial_mean != 0 else 0
                
                f.write(f"Activity Interpretation Mode: {'Lower values = better activity' if self.lower_is_better else 'Higher values = better activity'}\n")
                f.write(f"Initial Mean Activity: {initial_mean:.4f}\n")
                f.write(f"Final Mean Activity: {final_mean:.4f}\n")
                f.write(f"Overall Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)\n\n")
                
                # Best activity tracking
                initial_best = best_activities[0]
                final_best = best_activities[-1]
                
                if self.lower_is_better:
                    best_improvement = initial_best - final_best  # Decrease in value is improvement
                    best_overall = min(best_activities)
                else:
                    best_improvement = final_best - initial_best  # Increase in value is improvement
                    best_overall = max(best_activities)
                    
                best_step = steps[best_activities.index(best_overall)]
                
                f.write(f"Best Activity Values:\n")
                f.write(f"  Initial Best: {initial_best:.4f}\n")
                f.write(f"  Final Best: {final_best:.4f}\n")
                f.write(f"  Overall Best: {best_overall:.4f} (achieved at step {best_step})\n")
                f.write(f"  Best Value Improvement: {best_improvement:.4f}\n\n")
                
                # Activity diversity analysis
                initial_std = std_activities[0]
                final_std = std_activities[-1]
                avg_std = np.mean(std_activities)
                
                f.write(f"Activity Diversity (Standard Deviation):\n")
                f.write(f"  Initial Std: {initial_std:.4f}\n")
                f.write(f"  Final Std: {final_std:.4f}\n")
                f.write(f"  Average Std: {avg_std:.4f}\n\n")
                
                # Activity range analysis
                # Range is always positive: worst - best
                initial_range = abs(worst_activities[0] - best_activities[0])
                final_range = abs(worst_activities[-1] - best_activities[-1])
                avg_range = np.mean([abs(w - b) for w, b in zip(worst_activities, best_activities)])
                
                f.write(f"Activity Range (Worst - Best):\n")
                f.write(f"  Initial Range: {initial_range:.4f}\n")
                f.write(f"  Final Range: {final_range:.4f}\n")
                f.write(f"  Average Range: {avg_range:.4f}\n\n")
                
                # Trend analysis
                f.write("TREND ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                # Compute trend slope (simple linear fit)
                if len(steps) > 1:
                    x = np.array(steps)
                    y = np.array(mean_activities)
                    slope = np.polyfit(x, y, 1)[0]
                    
                    f.write(f"Mean Activity Trend: {slope:.2e} per step\n")
                    if self.lower_is_better:
                        if slope < -1e-6:
                            f.write("  → Improving (decreasing activity values)\n")
                        elif slope > 1e-6:
                            f.write("  → Worsening (increasing activity values)\n")
                        else:
                            f.write("  → Stable (minimal change)\n")
                    else:
                        if slope > 1e-6:
                            f.write("  → Improving (increasing activity values)\n")
                        elif slope < -1e-6:
                            f.write("  → Worsening (decreasing activity values)\n")
                        else:
                            f.write("  → Stable (minimal change)\n")
                
                # Recent trend (last 20% of the data)
                recent_portion = max(1, len(mean_activities) // 5)
                recent_mean = np.mean(mean_activities[-recent_portion:])
                early_mean = np.mean(mean_activities[:recent_portion]) if len(mean_activities) > recent_portion else mean_activities[0]
                
                # Compute recent improvement based on interpretation
                if self.lower_is_better:
                    recent_improvement = early_mean - recent_mean  # Decrease in value is improvement
                else:
                    recent_improvement = recent_mean - early_mean  # Increase in value is improvement
                
                f.write(f"\nRecent Performance (last {recent_portion} evaluations):\n")
                f.write(f"  Recent Mean Activity: {recent_mean:.4f}\n")
                f.write(f"  Recent Improvement: {recent_improvement:.4f}\n")
                
                # Performance milestones
                f.write(f"\nPERFORMANCE MILESTONES:\n")
                f.write("-" * 35 + "\n")
                
                # Find breakthrough points (significant improvements)
                significant_improvements = []
                for i in range(1, len(mean_activities)):
                    if self.lower_is_better:
                        improvement_at_step = mean_activities[i-1] - mean_activities[i]  # Decrease in value is improvement
                        improvement_threshold = 0.1
                        improvement_description = "activity reduction"
                    else:
                        improvement_at_step = mean_activities[i] - mean_activities[i-1]  # Increase in value is improvement
                        improvement_threshold = 0.1
                        improvement_description = "activity increase"
                        
                    if improvement_at_step > improvement_threshold:  # Significant improvement threshold
                        significant_improvements.append((steps[i], improvement_at_step, mean_activities[i]))
                
                if significant_improvements:
                    f.write(f"Significant Improvements (>{improvement_threshold} {improvement_description}):\n")
                    for step, improvement, activity in significant_improvements:
                        sign = "-" if self.lower_is_better else "+"
                        f.write(f"  Step {step}: {sign}{improvement:.4f} → {activity:.4f}\n")
                else:
                    f.write("No significant breakthrough improvements detected.\n")
                
                # Data quality metrics
                f.write(f"\nDATA QUALITY METRICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Evaluation Consistency: {len(set(sample_counts))} different sample sizes used\n")
                f.write(f"Most Common Sample Size: {max(set(sample_counts), key=sample_counts.count)}\n")
                f.write(f"Activity Value Stability: CV = {(np.std(mean_activities)/np.mean(mean_activities)*100):.2f}%\n")
                
                # Conclusions and recommendations
                f.write(f"\nCONCLUSIONS AND RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                
                if improvement > 0.5:
                    f.write("✓ EXCELLENT: Significant activity improvement achieved!\n")
                elif improvement > 0.1:
                    f.write("✓ GOOD: Moderate activity improvement observed.\n")
                elif improvement > -0.1:
                    f.write("~ STABLE: Activity levels maintained with minimal change.\n")
                else:
                    f.write("✗ CONCERNING: Activity levels have worsened.\n")
                
                if final_std < initial_std:
                    f.write("✓ Activity diversity has improved (lower std deviation).\n")
                elif final_std > initial_std * 1.5:
                    f.write("⚠ Activity diversity has increased significantly.\n")
                
                f.write(f"\nBest Generated Activity: {best_overall:.4f}\n")
                f.write(f"Recommended Next Steps:\n")
                if improvement > 0:
                    f.write("- Continue training to further improve activity\n")
                    f.write("- Consider increasing generation frequency\n")
                else:
                    f.write("- Review model architecture or training parameters\n")
                    f.write("- Consider adjusting learning rate or evaluation frequency\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("Report generated on training completion.\n")
                f.write("=" * 80 + "\n")
            
            print(f"Final activity report saved to: {report_path}")
            
            # Print key statistics to console
            print(f"\n" + "="*50)
            print(f"ACTIVITY TRACKING SUMMARY")
            print(f"="*50)
            print(f"Mean Activity: {initial_mean:.4f} → {final_mean:.4f} (Δ{improvement:+.4f})")
            print(f"Best Activity: {best_overall:.4f} (step {best_step})")
            print(f"Total Evaluations: {len(steps)}")
            print(f"Total Samples: {sum(sample_counts)}")
            print(f"="*50)
            
        except Exception as e:
            print(f"Warning: Failed to generate final activity report: {e}")

def count_parameters(model):
    """Count total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn):
    """
    Compute reconstruction loss with a unified output format across architectures.
    """
    # Unified handling: all architectures now output the same format
    # X has shape (batch_size, seq_len*5); reshape to (batch_size, seq_len, 5)
    X_reshaped = X.view(X.size(0), -1, 5)  # (B, seq_len, 5)
    X_targets = torch.argmax(X_reshaped, dim=2)  # (B, seq_len) - class indices
    
    # x_rec needs shape (batch_size, seq_len, 5) for CE loss
    x_rec_reshaped = x_rec.view(x_rec.size(0), -1, 5)  # (B, seq_len, 5)
    # CE loss expects (N, C, *), so permute
    x_rec_for_loss = x_rec_reshaped.permute(0, 2, 1)  # (B, 5, seq_len)
    
    loss_rec = recon_loss_fn(x_rec_for_loss, X_targets)
    
    return loss_rec

def train_and_evaluate(
    data_path='dataset/glmS.tsv',
    batch_size=64,
    latent_dim=128,
    learning_rate=1e-3,
    patience=10,
    max_epochs=100,
    arch='ribo-ld',
    seq_len=200):
    # Create checkpoint directory
    rbz_type = data_path.split('/')[-1].split('.')[0]
    checkpoint_dir = f"vae_checkpoint/{arch}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(checkpoint_dir, f"training_log_{rbz_type}.txt")
    print(seq_len)
    dataset = actDataset(data_path, target_length=seq_len)  # Use a unified sequence length
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = get_model(arch, seq_len, latent_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Count and print model parameter counts
    total_params, trainable_params = count_parameters(model)
    param_info = f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}"
    print(f"=== {arch.upper()} Architecture ===")
    print(param_info)
    print("=" * len(param_info))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    recon_loss_fn = nn.CrossEntropyLoss()  # Multi-class reconstruction loss
    reg_loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_rec_loss': [], 'train_reg_loss': [], 'train_pearson': [],
        'val_loss': [], 'val_rec_loss': [], 'val_reg_loss': [], 'val_pearson': [],
        'train_latent_loss': [], 'val_latent_loss': []
    }
    
    # Write training configuration
    with open(log_file, 'w') as f:
        f.write(f"Training Configuration for {arch.upper()} Architecture\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {rbz_type}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Latent dim: {latent_dim}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Max epochs: {max_epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}\n")
        f.write(f"Training set size: {len(train_set)}\n")
        f.write(f"Validation set size: {len(val_set)}\n")
        f.write(f"Test set size: {len(test_set)}\n\n")
        f.write("Training Progress:\n")
        f.write("-" * 100 + "\n")

    for epoch in range(max_epochs):
        # ----------- Train -----------
        model.train()
        total_loss = 0
        total_rec_loss = 0
        total_reg_loss = 0
        all_train_y = []
        all_train_pred = []
        total_latent_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            x_rec, y_pred = model(X)


            
            # Use the shared loss computation helper
            loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)
            loss_reg = reg_loss_fn(y_pred, y)

            z = model.encoder(X)
            latent_loss = torch.mean(z ** 2)

            # accumulate latent loss for epoch-level reporting
            total_latent_loss += latent_loss.item() * X.size(0)

            loss = loss_rec + loss_reg + latent_loss*1e-4

            # loss = loss_rec + loss_reg 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_rec_loss += loss_rec.item() * X.size(0)
            total_reg_loss += loss_reg.item() * X.size(0)
            all_train_y.append(y.detach().cpu().numpy())
            all_train_pred.append(y_pred.detach().cpu().numpy())
        train_loss = total_loss / len(train_set)
        train_rec_loss = total_rec_loss / len(train_set)
        train_reg_loss = total_reg_loss / len(train_set)
        train_latent_loss = total_latent_loss / len(train_set)
        train_y_true = np.concatenate(all_train_y).flatten()
        train_y_pred = np.concatenate(all_train_pred).flatten()
        train_pearson, _ = scipy.stats.pearsonr(train_y_true, train_y_pred)

        # ----------- Validation -----------
        model.eval()
        val_loss = 0
        val_rec_loss = 0
        val_reg_loss = 0
        total_val_latent_loss = 0.0
        test_rec_loss = 0
        test_latent_loss = 0.0  # Accumulate test-set latent loss
        all_val_y = []
        all_val_pred = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                x_rec, y_pred = model(X)
                
                # Use the shared loss computation helper
                loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)
                loss_reg = reg_loss_fn(y_pred, y)


                # loss = loss_rec + loss_reg

                z = model.encoder(X)
                latent_loss = torch.mean(z ** 2)

                # accumulate validation latent loss
                total_val_latent_loss += latent_loss.item() * X.size(0)

                loss = loss_rec + loss_reg + latent_loss*0.01


                val_loss += loss.item() * X.size(0)
                val_rec_loss += loss_rec.item() * X.size(0)
                val_reg_loss += loss_reg.item() * X.size(0)
                all_val_y.append(y.detach().cpu().numpy())
                all_val_pred.append(y_pred.detach().cpu().numpy())
        val_loss /= len(val_set)
        val_rec_loss /= len(val_set)
        val_reg_loss /= len(val_set)
        val_latent_loss = total_val_latent_loss / len(val_set)
        val_y_true = np.concatenate(all_val_y).flatten()
        val_y_pred = np.concatenate(all_val_pred).flatten()
        val_pearson, _ = scipy.stats.pearsonr(val_y_true, val_y_pred)

        # Record training history
        history['train_loss'].append(train_loss)
        history['train_rec_loss'].append(train_rec_loss)
        history['train_reg_loss'].append(train_reg_loss)
        history['train_pearson'].append(train_pearson)
        history['val_loss'].append(val_loss)
        history['val_rec_loss'].append(val_rec_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_pearson'].append(val_pearson)
        history['train_latent_loss'].append(train_latent_loss)
        history['val_latent_loss'].append(val_latent_loss)

        epoch_log = (f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} (Recon: {train_rec_loss:.4f}, MSE: {train_reg_loss:.4f}, Pearson: {train_pearson:.4f}, Latent: {train_latent_loss:.6f}) | "
                f"Val Loss: {val_loss:.4f} (Recon: {val_rec_loss:.4f}, MSE: {val_reg_loss:.4f}, Pearson: {val_pearson:.4f}, Latent: {val_latent_loss:.6f})")
        
        print(epoch_log)
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(epoch_log + "\n")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model to the corresponding architecture folder
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{arch}_{rbz_type}.pt")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history, 'train_latent_loss': train_latent_loss, 'val_latent_loss': val_latent_loss
                    }, best_model_path)
                print(f"Best model saved to {best_model_path}")
            except Exception as e:
                print(f"Warning: Failed to save model checkpoint: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop_log = f"Early stopping at epoch {epoch+1}"
                print(early_stop_log)
                with open(log_file, 'a') as f:
                    f.write(early_stop_log + "\n")
                break
    
    # Plot training curves
    def plot_training_curves():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        color1="#424585" 
        color2="#F14918"
        # 总损失
        ax1.plot(history['train_loss'], label='Train', color=color1, linewidth=2)
        ax1.plot(history['val_loss'], label='Val', color=color2, linewidth=2)
        ax1.set_title('Total Loss',fontsize=16)
        ax1.set_xlabel('Epoch',fontsize=16)
        ax1.set_ylabel('Loss',fontsize=16)
        ax1.legend(fontsize=16)
        ax1.grid(False)
        
        # 重构损失
        ax2.plot(history['train_rec_loss'], label='Train', color=color1, linewidth=2)
        ax2.plot(history['val_rec_loss'], label='Val', color=color2, linewidth=2)
        ax2.set_title('Reconstruction Loss',fontsize=16)
        ax2.set_xlabel('Epoch',fontsize=16)
        ax2.set_ylabel('Recon Loss',fontsize=16)
        ax2.legend(fontsize=16)
        ax2.grid(False)
        
        # 回归损失
        ax3.plot(history['train_reg_loss'], label='Train', color=color1, linewidth=2)
        ax3.plot(history['val_reg_loss'], label='Val', color=color2, linewidth=2)
        ax3.set_title('Regression Loss',fontsize=16)
        ax3.set_xlabel('Epoch',fontsize=16)
        ax3.set_ylabel('Reg Loss',fontsize=16)
        ax3.legend(fontsize=16)
        ax3.grid(False)
        
        # Pearson相关系数
        ax4.plot(history['train_pearson'], label='Train', color=color1, linewidth=2)
        ax4.plot(history['val_pearson'], label='Val', color=color2, linewidth=2)
        ax4.set_title('Pearson Correlation',fontsize=16)
        ax4.set_xlabel('Epoch',fontsize=16)
        ax4.set_ylabel('Pearson r',fontsize=16)
        ax4.legend(fontsize=16)
        ax4.grid(False)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(checkpoint_dir, f"training_curves_{arch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {plot_path}")
    
    # 调用绘图函数
    try:
        plot_training_curves()
    except Exception as e:
        print(f"Warning: Failed to plot training curves: {e}")
    # Test set evaluation
    try:
        # Try to load the best model
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"best_model_{arch}") and f.endswith('.pt')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {checkpoint_path}")
        else:
            print("No checkpoint found, using current model state for evaluation")
    except Exception as e:
        print(f"Warning: Failed to load best model checkpoint: {e}")
        print("Using current model state for evaluation")
        
        
    model.eval()
    test_rec_loss = 0
    test_latent_loss = 0.0
    all_y = []
    all_pred = []
    all_x_rec = []
    all_x_true = []
    all_latent_vecs = []  # 用于保存所有latent vector（全部数据集）
    
    # 测试集评估（用于计算指标）
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            # 获取模型输出
            x_rec, y_pred = model(X)
            
            # 使用通用损失计算函数
            loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)
            test_rec_loss += loss_rec.item() * X.size(0)

            # compute latent loss on test batch if possible
            if hasattr(model, 'encoder'):
                try:
                    z = model.encoder(X)
                    if isinstance(z, (tuple, list)):
                        z = z[0]
                    latent_loss = torch.mean(z ** 2)
                    test_latent_loss += latent_loss.item() * X.size(0)
                except Exception:
                    pass


            all_y.append(y.detach().cpu().numpy())
            all_pred.append(y_pred.detach().cpu().numpy())
            all_x_rec.append(x_rec.detach().cpu().numpy())
            all_x_true.append(X.detach().cpu().numpy())    
    # Extract latent vectors from the entire dataset
    print("Extracting latent vectors from entire dataset...")
    all_latent_vecs=[]
    full_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X, y in full_data_loader:
            X = X.to(device)
            
            # Directly call the encoder to get latent vectors
            if hasattr(model, 'encoder'):
                latent = model.encoder(X)
                if isinstance(latent, tuple) or isinstance(latent, list):
                    latent = latent[0]  # If multiple values are returned, take the first
                
                # Handle latent vectors with different shapes
                latent_np = latent.detach().cpu().numpy()
                if len(latent_np.shape) > 2:
                    # If 3D or higher, flatten all dims except batch
                    latent_np = latent_np.reshape(latent_np.shape[0], -1)
                elif len(latent_np.shape) == 1:
                    # If 1D, add batch dimension
                    latent_np = latent_np.reshape(1, -1)
                
                all_latent_vecs.append(latent_np)
    test_rec_loss /= len(test_set)
    # finalize test latent loss if we computed any
    try:
        test_latent_loss /= len(test_set)
    except Exception:
        test_latent_loss = float('nan')
    y_true = np.concatenate(all_y).flatten()
    y_pred = np.concatenate(all_pred).flatten()
    pearson_r, p_value = scipy.stats.pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Scatter plot (A4 three-column layout)
    plt.figure(figsize=(2.5, 2.5))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#00AA4A', s=8)
    plt.text(0.05, 0.98, f'Pearson r = {pearson_r:.4f}\nR² = {r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predicted Values', fontsize=10)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()

    # Save PNG and PDF versions
    scatter_plot_path_png = os.path.join(checkpoint_dir, f"scatter_plot_{arch}_{rbz_type}.png")
    scatter_plot_path_pdf = os.path.join(checkpoint_dir, f"scatter_plot_{arch}_{rbz_type}.pdf")
    plt.savefig(scatter_plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(scatter_plot_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved to {scatter_plot_path_png} and {scatter_plot_path_pdf}")
    
    # Compute reconstruction accuracy
    x_rec_all = np.concatenate(all_x_rec, axis=0)  # (N, seq_len*5)
    x_true_all = np.concatenate(all_x_true, axis=0)  # (N, seq_len*5)
    
    # Reshape to sequence format
    x_rec_seq = x_rec_all.reshape(-1, seq_len, 5)  # (N, seq_len, 5)
    x_true_seq = x_true_all.reshape(-1, seq_len, 5)  # (N, seq_len, 5)
    
    # Compute reconstruction accuracy - per-position prediction accuracy
    x_rec_pred = np.argmax(x_rec_seq, axis=2)  # (N, seq_len) - predicted classes
    x_true_labels = np.argmax(x_true_seq, axis=2)  # (N, seq_len) - ground-truth classes
    
    reconstruction_accuracy = np.mean(x_rec_pred == x_true_labels)  # overall per-position accuracy
    # Additionally save latent vectors with activity > 0.5
    high_activity_indices = np.where(y_true > 0.5)[0]
    if all_latent_vecs and len(high_activity_indices) > 0:
        latent_vectors = np.concatenate(all_latent_vecs, axis=0)
        high_activity_latents = latent_vectors[high_activity_indices]
        print(len(high_activity_latents))
        high_activity_save_path = os.path.join(checkpoint_dir, f"latent_vectors_high_activity_{arch}_{rbz_type}.npy")
        np.save(high_activity_save_path, high_activity_latents)
        print(f"High-activity latent vectors (activity > 0.5) saved to {high_activity_save_path}")
        print(f"Total high-activity latent vectors saved: {high_activity_latents.shape[0]}")
    # Save latent vectors to file
    if all_latent_vecs:
        latent_vectors = np.concatenate(all_latent_vecs, axis=0)
        latent_save_path = os.path.join(checkpoint_dir, f"latent_vectors_{arch}_{rbz_type}.npy")
        np.save(latent_save_path, latent_vectors)
        print(f"Latent vectors saved to {latent_save_path}")
        print(f"Total latent vectors extracted: {latent_vectors.shape[0]} (from entire dataset of {len(dataset)} samples)")
        print(f"Latent vector dimension: {latent_vectors.shape[1]}")
        
        # t-SNE visualization of latent vectors
        print("Performing t-SNE visualization of latent vectors...")
        try:
            # Get activity values for all samples for coloring
            all_activities = dataset.y.flatten()
            
            # Ensure latent_vectors is 2D
            print(f"Original latent vectors shape: {latent_vectors.shape}")
            if len(latent_vectors.shape) > 2:
                print(f"Reshaping latent vectors from {latent_vectors.shape} to 2D")
                latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
            elif len(latent_vectors.shape) == 1:
                print(f"Reshaping latent vectors from {latent_vectors.shape} to 2D")
                latent_vectors = latent_vectors.reshape(1, -1)
            
            print(f"Final latent vectors shape for t-SNE: {latent_vectors.shape}")
            
            # If there are too many samples, randomly subsample for visualization
            if len(latent_vectors) > 50000:
                print(f"Dataset too large ({len(latent_vectors)} samples), sampling 50000 for t-SNE visualization")
                sample_indices = np.random.choice(len(latent_vectors), 50000, replace=False)
                latent_for_tsne = latent_vectors[sample_indices]
                activities_for_tsne = all_activities[sample_indices]
            else:
                latent_for_tsne = latent_vectors
                activities_for_tsne = all_activities
            
            # Double-check dimensionality
            if len(latent_for_tsne.shape) != 2:
                raise ValueError(f"t-SNE expects 2D array, got {latent_for_tsne.shape}")
            
            # Run t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_for_tsne)-1))
            latent_2d = tsne.fit_transform(latent_for_tsne)
            
            # Plot t-SNE (A4 three-column layout)
            plt.figure(figsize=(2.76, 2.18))
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=activities_for_tsne, cmap='viridis', 
                                s=2, alpha=0.5)
            cbar = plt.colorbar(scatter, shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Activity Value', fontsize=10)
            plt.xlabel('t-SNE Component 1', fontsize=10)
            plt.ylabel('t-SNE Component 2', fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            
            plt.tight_layout()
            
            # Save PNG and PDF versions
            tsne_plot_path_png = os.path.join(checkpoint_dir, f"tsne_latent_{arch}_{rbz_type}.png")
            tsne_plot_path_pdf = os.path.join(checkpoint_dir, f"tsne_latent_{arch}_{rbz_type}.pdf")
            plt.savefig(tsne_plot_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(tsne_plot_path_pdf, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"t-SNE plots saved to {tsne_plot_path_png} and {tsne_plot_path_pdf}")
            
        except Exception as e:
            print(f"Warning: t-SNE visualization failed: {e}")
    else:
        print("Warning: No latent vectors were extracted or model does not support latent extraction.")
    
    # Save test results to file
    test_results_file = os.path.join(checkpoint_dir, f"test_results_{arch}_{rbz_type}.txt")
    with open(test_results_file, 'w') as f:
        f.write(f"Test Results for {arch.upper()} Architecture\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {rbz_type}\n")
        f.write(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}\n")
        f.write(f"Test Reconstruction Loss: {test_rec_loss:.6f}\n")
        f.write(f"Test Latent Loss: {test_latent_loss:.6f}\n")
        f.write(f"Test Reconstruction Accuracy: {reconstruction_accuracy:.6f}\n")
        f.write(f"Test Pearson Correlation: {pearson_r:.6f}\n")
        f.write(f"Test Pearson p-value: {p_value:.6e}\n")
        f.write(f"Test Set Size: {len(test_set)}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Total Training Epochs: {len(history['train_loss'])}\n")
        f.write(f"Full Dataset Size: {len(dataset)}\n")
        if all_latent_vecs:
            latent_vectors = np.concatenate(all_latent_vecs, axis=0)
            f.write(f"Latent Vectors Extracted: {latent_vectors.shape[0]} (from entire dataset)\n")
            f.write(f"Latent Vector Dimension: {latent_vectors.shape[1]}\n")
        f.write(f"\n")
        
        # Add some summary statistics
        f.write("Prediction Statistics (Test Set Only):\n")
        f.write(f"Mean True Value: {np.mean(y_true):.6f}\n")
        f.write(f"Std True Value: {np.std(y_true):.6f}\n")
        f.write(f"Mean Predicted Value: {np.mean(y_pred):.6f}\n")
        f.write(f"Std Predicted Value: {np.std(y_pred):.6f}\n")
        f.write(f"Mean Absolute Error: {np.mean(np.abs(y_true - y_pred)):.6f}\n")
        f.write(f"Root Mean Square Error: {np.sqrt(np.mean((y_true - y_pred)**2)):.6f}\n")
    
    print(f"Test results saved to {test_results_file}")
    print(f"Test Reconstruction Loss: {test_rec_loss:.4f}, Test Latent Loss: {test_latent_loss:.6f}, Test Reconstruction Accuracy: {reconstruction_accuracy:.4f}, Test Pearson r: {pearson_r:.4f}")
    
    return test_rec_loss, pearson_r

def train_diffusion_model(
    latent_vectors_path,
    arch='ribo-ld',
    latent_dim=128,
    batch_size=32,
    learning_rate=8e-5,
    train_steps=5000,
    timesteps=500,
    save_every=500,
    results_folder=None
):
    """
    Train a diffusion model over latent vectors.
    """
    if not DIFFUSION_AVAILABLE:
        raise ImportError("Diffusion library not available. Please install denoising_diffusion_pytorch")
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    if results_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"diffusion_results/{arch}_{rbz_type}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load latent vectors
    print(f"Loading latent vectors from {latent_vectors_path}")
    latent_vectors = np.load(latent_vectors_path)
    
    # Data preprocessing
    latent_vectors = torch.tensor(latent_vectors, dtype=torch.float32)
    print(f"Loaded latent vectors shape: {latent_vectors.shape}")
    
    # Ensure latent_vectors has the expected shape (N, latent_dim)
    if len(latent_vectors.shape) == 1:
        latent_vectors = latent_vectors.unsqueeze(0)
    if latent_vectors.shape[1] != latent_dim:
        print(f"Warning: Expected latent_dim {latent_dim}, but got {latent_vectors.shape[1]}")
        latent_dim = latent_vectors.shape[1]
    
    # Add channel dimension for diffusion
    latent_vectors = latent_vectors.unsqueeze(1)  # (N, 1, latent_dim)
    
    # Split train/validation sets
    train_latents, val_latents = train_test_split(latent_vectors, test_size=0.15, random_state=42)
    
    # Create Dataset1D instances
    train_dataset = Dataset1D(train_latents)
    val_dataset = Dataset1D(val_latents)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Define the Unet1D model
    model = Unet1D(
        dim=32,                    # model dimension
        dim_mults=(1, 2, 4, 8),   # layer multipliers
        channels=1                 # input channels, matching latent space
    )
    
    # Define the diffusion process
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=latent_dim,     # sequence length matches latent vector dimension
        timesteps=timesteps,       # number of diffusion steps
        objective='pred_v',         # objective
        sampling_timesteps=100
    )
    
    # Count model parameters
    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Diffusion Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Set up trainer
    trainer = Trainer1D(
        diffusion,
        dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=batch_size,
        train_lr=learning_rate,
        train_num_steps=train_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=results_folder,
        save_and_sample_every=save_every,
        num_samples=16,  # number of samples
    )
    
    # Save training configuration
    config_file = os.path.join(results_folder, "training_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Diffusion Model Training Configuration\n")
        f.write(f"{'='*50}\n")
        f.write(f"Architecture: {arch}\n")
        f.write(f"Latent vectors path: {latent_vectors_path}\n")
        f.write(f"Latent dimension: {latent_dim}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Training steps: {train_steps}\n")
        f.write(f"Timesteps: {timesteps}\n")
        f.write(f"Training set size: {len(train_dataset)}\n")
        f.write(f"Validation set size: {len(val_dataset)}\n")
        f.write(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}\n")
    
    print(f"Starting diffusion model training...")
    print(f"Results will be saved to: {results_folder}")
    
    # Start training
    trainer.train()
    
    print(f"Diffusion training completed!")
    print(f"Results saved to: {results_folder}")
    
    return results_folder

def train_diffusion_with_activity_evaluation(
    latent_vectors_path,
    ae_model_path,  # AE model path used to load the decoder
    arch='ribo-ld',
    latent_dim=128,
    batch_size=32,
    learning_rate=1e-6,
    train_steps=5000,
    timesteps=500,
    save_every=500, 
    activity_eval_every=100,
    results_folder=None,
    data_path='dataset/glms_log.tsv',  # used to load ground-truth activity values
    enable_dataset_optimization=False,  # new: whether to enable dataset optimization
    replacement_count=500,  # new: number of samples to replace
    generation_count=5000,  # new: number of samples to generate
    lower_is_better=True,
    act_ts=None,
    seq_len=200
):
    """
    Train a diffusion model with activity evaluation.
    """
    # Set random seed
    set_seed(42)
    rbz_type = data_path.split('/')[-1].split('.')[0]
    # Create output directory
    if results_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"diffusion_results_with_activity/{arch}_{rbz_type}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Load latent vectors
    print(f"Loading latent vectors from {latent_vectors_path}")
    latent_vectors = np.load(latent_vectors_path)
    
    # Data preprocessing
    latent_vectors = torch.tensor(latent_vectors, dtype=torch.float32)
    print(f"Loaded latent vectors shape: {latent_vectors.shape}")
    
    # Ensure latent_vectors has the expected shape (N, latent_dim)
    if len(latent_vectors.shape) == 1:
        latent_vectors = latent_vectors.unsqueeze(0)
    if latent_vectors.shape[1] != latent_dim:
        print(f"Warning: Expected latent_dim {latent_dim}, but got {latent_vectors.shape[1]}")
        latent_dim = latent_vectors.shape[1]
    
    # Add channel dimension for diffusion
    latent_vectors = latent_vectors.unsqueeze(1)  # (N, 1, latent_dim)
    
    # Split train/validation sets
    train_latents, val_latents = train_test_split(latent_vectors, test_size=0.15, random_state=42)
    
    # Create Dataset1D instances
    if DIFFUSION_AVAILABLE:
        train_dataset = Dataset1D(train_latents)
        val_dataset = Dataset1D(val_latents)
    else:
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_latents)
        val_dataset = TensorDataset(val_latents)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Load AE model to obtain decoder
    print(f"Loading AE model from {ae_model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_model = get_model(arch, seq_len, latent_dim)
    ae_checkpoint = torch.load(ae_model_path, map_location=device)
    ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
    ae_model.to(device)
    ae_model.eval()
    
    # Get decoder
    ae_decoder = ae_model  # use the full model since we also need to predict activity
    
    # Load ground-truth activity values (from the original dataset)
    print(f"Loading true activities from {data_path}")
    try:
        dataset_for_activities = actDataset(data_path, target_length=seq_len)
        # Get activity values for the test split (same split as AE training)
        idx = np.arange(len(dataset_for_activities))
        train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
        true_activities = [dataset_for_activities.y[i].item() for i in test_idx[:len(train_latents)]]
        print(f"Loaded {len(true_activities)} true activity values")
    except Exception as e:
        print(f"Warning: Could not load true activities: {e}")
        true_activities = []
    
    # Define the Unet1D model
    if DIFFUSION_AVAILABLE:
        model = Unet1D(
            dim=64,                   
            dim_mults=(1, 2, 4, 8),   # layer multipliers
            channels=1                 # input channels, matching latent space
        )
        
        # Define the diffusion process
        diffusion = GaussianDiffusion1D(
            model,
            seq_length=latent_dim,     # sequence length matches latent vector dimension
            timesteps=timesteps,       # number of diffusion steps
            objective='pred_v',         # objective
            sampling_timesteps=100      # number of sampling steps
        )
    else:
        raise ImportError("Diffusion library not available. Please install denoising_diffusion_pytorch")
    
    # Count model parameters
    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Diffusion Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Use custom ActivityEvaluationTrainer
    trainer = ActivityEvaluationTrainer(
        diffusion_model=diffusion,
        ae_decoder=ae_decoder,
        dataset=train_dataset,
        val_dataset=val_dataset,
        true_activities=true_activities,
        data_path=data_path if enable_dataset_optimization else None,  # only pass when optimization is enabled
        train_batch_size=batch_size,
        train_lr=learning_rate,
        train_num_steps=train_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        save_and_sample_every=save_every,
        activity_eval_every=activity_eval_every,
        num_samples=generation_count if enable_dataset_optimization else 100,  # adjust based on optimization flag
        results_folder=results_folder,
        vocab={'A': 0, 'C': 1, 'G': 2, 'T': 3, 'P': 4},
        early_stopping_patience=20,
        early_stopping_min_delta=1e-4,
        early_stopping_eval_every=500,
        enable_dataset_optimization=enable_dataset_optimization,  # controlled by argument
        replacement_count=replacement_count,  # controlled by argument
        lower_is_better=lower_is_better,
        act_ts=act_ts,
        seq_len=seq_len
    )
    
    # Save training configuration
    config_file = os.path.join(results_folder, "training_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Diffusion Model Training with Activity Evaluation and Dataset Optimization\n")
        f.write(f"{'='*80}\n")
        f.write(f"Architecture: {arch}\n")
        f.write(f"Latent vectors path: {latent_vectors_path}\n")
        f.write(f"AE model path: {ae_model_path}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Latent dimension: {latent_dim}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Training steps: {train_steps}\n")
        f.write(f"Timesteps: {timesteps}\n")
        f.write(f"Activity evaluation every: {activity_eval_every} steps\n")
        f.write(f"Training set size: {len(train_dataset)}\n")
        f.write(f"Validation set size: {len(val_dataset)}\n")
        f.write(f"True activities count: {len(true_activities)}\n")
        f.write(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}\n")
        f.write(f"\nActivity Tracking Features:\n")
        f.write(f"  Real-time activity monitoring: Enabled\n")
        f.write(f"  Activity plots generation: Enabled\n")
        f.write(f"  Comprehensive activity reports: Enabled\n")
        f.write(f"  Activity data logging: Every evaluation step\n")
        f.write(f"  Expected total evaluations: ~{train_steps // activity_eval_every}\n")
        f.write(f"\nDataset Optimization Settings:\n")
        f.write(f"  Enable dataset optimization: {enable_dataset_optimization}\n")
        if enable_dataset_optimization:
            f.write(f"  Samples generated per evaluation: {generation_count}\n")
            f.write(f"  Best samples selected for replacement: {replacement_count}\n")
            f.write(f"  Strategy: Replace worst {replacement_count} samples with best {replacement_count} generated samples\n")
            f.write(f"  Optimization frequency: Every {activity_eval_every} training steps\n")
        else:
            f.write(f"  Basic activity prediction only (no dataset optimization)\n")
    
    print(f"Starting diffusion model training with activity evaluation...")
    print(f"Results will be saved to: {results_folder}")
    
    # Start training
    training_results = trainer.train()
    
    print(f"Diffusion training with activity prediction completed!")
    print(f"Results saved to: {results_folder}")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    
    return results_folder, training_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ae', choices=['ae', 'diffusion', 'diffusion_activity'], 
                       help='Training mode: ae for autoencoder, diffusion for basic diffusion, diffusion_activity for diffusion with activity evaluation')
    parser.add_argument('--data_path', type=str, default='dataset/glms_log.tsv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--arch', type=str, default='ribo-ld')
    parser.add_argument('--select_all', action='store_true', help='Run all architectures and summarize')
    
    # Diffusion-specific parameters
    parser.add_argument('--latent_vectors_path', type=str, default=None,
                       help='Path to saved latent vectors .npy file for diffusion training')
    parser.add_argument('--ae_model_path', type=str, default=None,
                       help='Path to trained AE model for activity prediction (diffusion_activity mode)')
    parser.add_argument('--diffusion_lr', type=float, default=8e-5,
                       help='Learning rate for diffusion model training')
    parser.add_argument('--train_steps', type=int, default=5000,
                       help='Number of training steps for diffusion model')
    parser.add_argument('--timesteps', type=int, default=500,
                       help='Number of timesteps for diffusion process')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save model every N steps')
    parser.add_argument('--activity_eval_every', type=int, default=100,
                       help='Predict activity every N steps (diffusion_activity mode)')
    parser.add_argument('--enable_dataset_optimization', action='store_true',
                       help='Enable dataset optimization with best generated samples (bottom-elimination)')
    parser.add_argument('--replacement_count', type=int, default=500,
                       help='Number of samples to replace in dataset optimization')
    parser.add_argument('--generation_count', type=int, default=5000,
                       help='Number of samples to generate for selection in dataset optimization')
    parser.add_argument('--lower_is_better', action='store_true',
                        help='For glms dataset, lower activity is better. For cpeb3 and line1, higher is better')
    parser.add_argument('--act_ts',type=float,default=None,
                        help='Activity threshold')
    parser.add_argument('--seq_len',type=int,default=None,
                        help='Sequence length for input data')

    args = parser.parse_args()

    if args.mode == 'diffusion_activity':
        # Diffusion mode with activity evaluation
        if not DIFFUSION_AVAILABLE:
            print("Error: Diffusion library not available. Please install denoising_diffusion_pytorch")
            return
            
        if args.latent_vectors_path is None:
            print("Error: --latent_vectors_path is required for diffusion_activity mode")
            return
            
        if args.ae_model_path is None:
            print("Error: --ae_model_path is required for diffusion_activity mode")
            return
        
        if not os.path.exists(args.latent_vectors_path):
            print(f"Error: Latent vectors file not found: {args.latent_vectors_path}")
            return
            
        if not os.path.exists(args.ae_model_path):
            print(f"Error: AE model file not found: {args.ae_model_path}")
            return
        
        print(f"Starting diffusion model training with activity prediction for {args.arch} architecture")
        
        results_folder, training_results = train_diffusion_with_activity_evaluation(
            latent_vectors_path=args.latent_vectors_path,
            ae_model_path=args.ae_model_path,
            arch=args.arch,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            learning_rate=args.diffusion_lr,
            train_steps=args.train_steps,
            timesteps=args.timesteps,
            save_every=args.save_every,
            activity_eval_every=args.activity_eval_every,
            data_path=args.data_path,
            enable_dataset_optimization=args.enable_dataset_optimization,
            replacement_count=args.replacement_count,
            generation_count=args.generation_count,
            lower_is_better=args.lower_is_better,
            act_ts=args.act_ts,
            seq_len=args.seq_len
        )
        
        print(f"Diffusion training with activity prediction completed.")
        print(f"Results saved to: {results_folder}")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        
    elif args.mode == 'diffusion':
        # Basic diffusion mode
        if not DIFFUSION_AVAILABLE:
            print("Error: Diffusion library not available. Please install denoising_diffusion_pytorch")
            return
            
        if args.latent_vectors_path is None:
            print("Error: --latent_vectors_path is required for diffusion mode")
            return
        
        if not os.path.exists(args.latent_vectors_path):
            print(f"Error: Latent vectors file not found: {args.latent_vectors_path}")
            return
        
        print(f"Starting diffusion model training for {args.arch} architecture")
        
        results_folder = train_diffusion_model(
            latent_vectors_path=args.latent_vectors_path,
            arch=args.arch,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            learning_rate=args.diffusion_lr,
            train_steps=args.train_steps,
            timesteps=args.timesteps,
            save_every=args.save_every
        )
        
        print(f"Diffusion training completed. Results saved to: {results_folder}")

    elif args.mode == 'ae':
        # Original AE training mode


        rec_loss, pearson = train_and_evaluate(
            data_path=args.data_path,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            patience=args.patience,
            max_epochs=args.max_epochs,
            arch=args.arch,
            seq_len=args.seq_len
        )
        print(f"\nTest Recon Loss: {rec_loss:.4f}, Test Pearson r: {pearson:.4f}")

if __name__ == '__main__':
    main()