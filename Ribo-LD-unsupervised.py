import numpy as np
import pandas as pd
import torch
import traceback
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scipy.stats
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from datetime import datetime
from model import get_model
from pathlib import Path
from multiprocessing import cpu_count
from torch.optim import Adam
from tqdm.auto import tqdm

# Diffusion model imports (adjust according to the diffusion library you use)
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
    mapping = {'A':0, 'C':1, 'G':2, 'T':3, 'P':4}  # Add encoding for 'P'
    
    # Pad or truncate sequence to target length
    if len(seq) > target_length:
        seq = seq[:target_length]  # Truncate if too long
    elif len(seq) < target_length:
        seq = seq + 'P' * (target_length - len(seq))  # Pad with 'P' if too short
    
    arr = np.zeros((target_length, 5), dtype=np.float32)  # Now 5D, including 'P'
    for i, c in enumerate(seq):
        if c in mapping:
            arr[i, mapping[c]] = 1.0
        # For unknown characters, leave as all zeros
    return arr.flatten()
class rfamDataset(Dataset):
    def __init__(self, txt_file, target_length=200):
        df = pd.read_csv(txt_file, sep='\t', header=None)
        self.seqs = df[0].values
        self.target_length = target_length
        self.X = np.array([one_hot_encode(seq, target_length) for seq in self.seqs], dtype=np.float32)
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return (self.X[idx])
def set_seed(seed=42):
    """Set random seeds to ensure reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cycle(dl):
    """Iterate over a dataloader indefinitely."""
    while True:
        for data in dl:
            yield data

def count_parameters(model):
    """Count the total number of parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn):
    """
    Compute reconstruction loss while handling output formats across architectures.
    """
    # Unified handling: all architectures are treated as outputting the same format
    # X has shape (batch_size, seq_len*5) and needs reshaping to (batch_size, seq_len, 5)
    X_reshaped = X.view(X.size(0), -1, 5)  # (B, seq_len, 5)
    X_targets = torch.argmax(X_reshaped, dim=2)  # (B, seq_len) - class indices
    
    # x_rec needs shape (batch_size, seq_len, 5) for CE loss
    x_rec_reshaped = x_rec.view(x_rec.size(0), -1, 5)  # (B, seq_len, 5)
    # CE loss expects (N, C, *) so we permute
    x_rec_for_loss = x_rec_reshaped.permute(0, 2, 1)  # (B, 5, seq_len)
    
    loss_rec = recon_loss_fn(x_rec_for_loss, X_targets)
    
    return loss_rec

def train_and_evaluate(
    data_path='dataset/rfam.txt',
    train_path='dataset/rfam_training_set.txt',
    val_path='dataset/rfam_validation_set.txt',
    test_path='dataset/rfam_test_set.txt',
    batch_size=64,
    latent_dim=128,
    learning_rate=1e-3,
    patience=10,
    max_epochs=100,
    arch='ribo-ld-unsup'):
    # Create checkpoint directory
    checkpoint_dir = f"vae_checkpoint/{arch}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(checkpoint_dir, f"training_log_{timestamp}.txt")
    
    dataset = rfamDataset(data_path, target_length=200)  # Use a fixed sequence length
    # idx = np.arange(len(dataset))
    # train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    # train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
    # train_set = torch.utils.data.Subset(dataset, train_idx)
    # val_set = torch.utils.data.Subset(dataset, val_idx)
    # test_set = torch.utils.data.Subset(dataset, test_idx)
    train_set = rfamDataset(train_path, target_length=200)
    val_set = rfamDataset(val_path, target_length=200)
    test_set = rfamDataset(test_path, target_length=200)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    # test_label_loader = DataLoader(test_label, batch_size=test_set.__len__())

    seq_len = 200  # Fixed sequence length
    model = get_model(arch, seq_len, latent_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Count and print model parameters
    total_params, trainable_params = count_parameters(model)
    param_info = f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}"
    print(f"=== {arch.upper()} Architecture ===")
    print(param_info)
    print("=" * len(param_info))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    recon_loss_fn = nn.CrossEntropyLoss()  # Multi-class loss
    reg_loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [], 'train_rec_loss': [], 'train_reg_loss': [], 'train_pearson': [],
        'val_loss': [], 'val_rec_loss': [], 'val_reg_loss': [], 'val_pearson': [],
        'train_latent_loss': [], 'val_latent_loss': []  # Track latent loss
    }
    
    # Write training configuration
    with open(log_file, 'w') as f:
        f.write(f"Training Configuration for {arch.upper()} Architecture\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Latent dim: {latent_dim}\n")

        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Patience: {patience}\n")
        f.write(f"Max epochs: {max_epochs}\n")
        f.write(f"Device: {device}\n")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_latent_loss = 0.0

        for X in train_loader:
            X = X.to(device)
            x_rec = model(X)

            loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)

            z = model.encoder(X)
            if isinstance(z, (tuple, list)):
                z = z[0]
            latent_loss = torch.mean(z ** 2)
            total_latent_loss += latent_loss.item() * X.size(0)

            loss = loss_rec + latent_loss * 1e-4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_rec_loss += loss_rec.item() * X.size(0)

        train_loss = total_loss / len(train_set)
        train_rec_loss = total_rec_loss / len(train_set)
        train_latent_loss = total_latent_loss / len(train_set)  # Mean latent loss

        # ----------- Validation -----------
        model.eval()
        val_loss = 0
        val_rec_loss = 0
        # val_reg_loss = 0
        # all_val_y = []
        # all_val_pred = []
        total_val_latent_loss = 0.0  # Accumulate validation latent loss
        with torch.no_grad():
            for X in val_loader:
                X = X.to(device)
                x_rec = model(X)

                # Use the shared loss computation helper
                loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)
                
                # Compute validation latent loss
                z = model.encoder(X)
                if isinstance(z, (tuple, list)):
                    z = z[0]
                latent_loss = torch.mean(z ** 2)
                
                # Accumulate validation latent loss
                total_val_latent_loss += latent_loss.item() * X.size(0)
                
                loss = loss_rec + latent_loss * 1e-4
                
                val_loss += loss.item() * X.size(0)
                val_rec_loss += loss_rec.item() * X.size(0)
                # val_reg_loss += loss_reg.item() * X.size(0)
                # all_val_y.append(y.detach().cpu().numpy())
                # all_val_pred.append(y_pred.detach().cpu().numpy())
        val_loss /= len(val_set)
        val_rec_loss /= len(val_set)
        val_latent_loss = total_val_latent_loss / len(val_set)  # Mean validation latent loss
        # val_reg_loss /= len(val_set)
        # val_y_true = np.concatenate(all_val_y).flatten()
        # val_y_pred = np.concatenate(all_val_pred).flatten()
        # val_pearson, _ = scipy.stats.pearsonr(val_y_true, val_y_pred)

        # Record training history
        history['train_loss'].append(train_loss)
        history['train_rec_loss'].append(train_rec_loss)
        # history['train_reg_loss'].append(train_reg_loss)
        # history['train_pearson'].append(train_pearson)
        history['val_loss'].append(val_loss)
        history['val_rec_loss'].append(val_rec_loss)
        # history['val_reg_loss'].append(val_reg_loss)
        # history['val_pearson'].append(val_pearson)
        history['train_latent_loss'].append(train_latent_loss)  # Training latent loss
        history['val_latent_loss'].append(val_latent_loss)      # Validation latent loss

        epoch_log = (f"Epoch {epoch+1} | "
                    f"Train Loss: {train_loss:.4f} (Recon: {train_rec_loss:.4f}, Latent: {train_latent_loss:.6f}) | "
                    f"Val Loss: {val_loss:.4f} (Recon: {val_rec_loss:.4f}, Latent: {val_latent_loss:.6f})")
        print(epoch_log)
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(epoch_log + "\n")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoint under the architecture folder
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{arch}.pt")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history,
                    'train_latent_loss': train_latent_loss,  # Save latent-loss info
                    'val_latent_loss': val_latent_loss
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
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        # Total loss
        ax1.plot(history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(history['val_loss'], label='Val Loss', color='red')
        ax1.set_title(f'{arch.upper()} - Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        # ax1.grid(True)
        
        # Reconstruction loss
        ax2.plot(history['train_rec_loss'], label='Train Recon Loss', color='blue')
        ax2.plot(history['val_rec_loss'], label='Val Recon Loss', color='red')
        ax2.set_title('Reconstruction Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Recon Loss')
        ax2.legend()
        # ax2.grid(True)
        

        
        plt.tight_layout()
        
        # Save figures
        plot_path = os.path.join(checkpoint_dir, f"training_curves_{arch}.png")
        plot_path_pdf = os.path.join(checkpoint_dir, f"training_curves_{arch}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {plot_path} and {plot_path_pdf}")
    
    # Call plotting helper
    try:
        plot_training_curves()
    except Exception as e:
        print(f"Warning: Failed to plot training curves: {e}")

    # Test set evaluation
    try:
        # Try to load best checkpoint
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
    test_latent_loss = 0.0  # Accumulate test-set latent loss
    # all_y = []
    all_pred = []
    all_x_rec = []
    all_x_true = []
    all_latent_vecs = []  # Store latent vectors (entire dataset)

    # Test set evaluation (for metrics)
    with torch.no_grad():
        for X in test_loader: 
            # Forward pass
            X = X.to(device)
            x_rec = model(X)
            
            # Use the shared loss computation helper
            loss_rec = calculate_reconstruction_loss(x_rec, X, arch, recon_loss_fn)
            test_rec_loss += loss_rec.item() * X.size(0)
            
            # Compute test-set latent loss
            if hasattr(model, 'encoder'):
                try:
                    z = model.encoder(X)
                    if isinstance(z, (tuple, list)):
                        z = z[0]
                    latent_loss = torch.mean(z ** 2)
                    test_latent_loss += latent_loss.item() * X.size(0)
                except Exception:
                    pass
            
            all_x_rec.append(x_rec.detach().cpu().numpy())
            all_x_true.append(X.detach().cpu().numpy())

    
    # Extract latent vectors for the full dataset
    print("Extracting latent vectors from entire dataset...")
    full_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X in full_data_loader:
            X = X.to(device)
            if hasattr(model, 'encoder'):
                latent = model.encoder(X)
                if isinstance(latent, tuple) or isinstance(latent, list):
                    latent = latent[0]  # 如果返回多个值，取第一个
                latent_np = latent.detach().cpu().numpy()
                if len(latent_np.shape) > 2:
                    latent_np = latent_np.reshape(latent_np.shape[0], -1)
                elif len(latent_np.shape) == 1:
                    latent_np = latent_np.reshape(1, -1)
                all_latent_vecs.append(latent_np)

                
    test_rec_loss /= len(test_set)
    # Compute average test-set latent loss
    try:
        test_latent_loss /= len(test_set)
    except Exception:
        test_latent_loss = float('nan')
    # y_true = np.concatenate(all_y).flatten()
    # y_pred = np.concatenate(all_pred).flatten()
    # pearson_r, p_value = scipy.stats.pearsonr(y_true, y_pred)
    
    # Compute reconstruction accuracy
    x_rec_all = np.concatenate(all_x_rec, axis=0)  # (N, seq_len*5)
    x_true_all = np.concatenate(all_x_true, axis=0)  # (N, seq_len*5)

    
    # Reshape to sequence format
    x_rec_seq = x_rec_all.reshape(-1, 200, 5)  # (N, seq_len, 5)
    x_true_seq = x_true_all.reshape(-1, 200, 5)  # (N, seq_len, 5)
    print(x_true_seq.shape[0])
    
    # Reconstruction accuracy - per-position prediction accuracy
    x_rec_pred = np.argmax(x_rec_seq, axis=2)  # (N, seq_len) - predicted classes
    x_true_labels = np.argmax(x_true_seq, axis=2)  # (N, seq_len) - true classes
    
    # Per-sample mean positional accuracy
    sample_accuracies = np.mean(x_rec_pred == x_true_labels, axis=1)  # (N,) - accuracy per sample
    
    # Overall reconstruction accuracy (mean over samples)
    reconstruction_accuracy = np.mean(sample_accuracies)  # scalar
    
    # Print summary statistics
    print(f"Per-sample accuracy statistics:")
    print(f"  Mean accuracy: {reconstruction_accuracy:.4f}")
    print(f"  Std accuracy: {np.std(sample_accuracies):.4f}")
    print(f"  Min accuracy: {np.min(sample_accuracies):.4f}")
    print(f"  Max accuracy: {np.max(sample_accuracies):.4f}")
    print(f"  Median accuracy: {np.median(sample_accuracies):.4f}")

    if all_latent_vecs:
        latent_vectors = np.concatenate(all_latent_vecs, axis=0)
        latent_save_path = os.path.join(checkpoint_dir, f"latent_vectors_{arch}.npy")
        np.save(latent_save_path, latent_vectors)
        print(f"Latent vectors saved to {latent_save_path}")
        print(f"Total latent vectors extracted: {latent_vectors.shape[0]} (from entire dataset of {len(dataset)} samples)")
        print(f"Latent vector dimension: {latent_vectors.shape[1]}")
    # Save test results to file
    test_results_file = os.path.join(checkpoint_dir, f"test_results_{arch}.txt")
    with open(test_results_file, 'w') as f:
        f.write(f"Test Results for {arch.upper()} Architecture\n")
        f.write(f"{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}\n")
        f.write(f"Test Reconstruction Loss: {test_rec_loss:.6f}\n")
        f.write(f"Test Latent Loss: {test_latent_loss:.6f}\n")  # Include latent loss
        f.write(f"Test Reconstruction Accuracy: {reconstruction_accuracy:.6f}\n")
        f.write(f"Test Set Size: {len(test_set)}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Total Training Epochs: {len(history['train_loss'])}\n")
        f.write(f"Full Dataset Size: {len(dataset)}\n")
        if all_latent_vecs:
            latent_vectors = np.concatenate(all_latent_vecs, axis=0)
            f.write(f"Latent Vectors Extracted: {latent_vectors.shape[0]} (from entire dataset)\n")
            f.write(f"Latent Vector Dimension: {latent_vectors.shape[1]}\n")
        f.write(f"\n")
    print(f"Test results saved to {test_results_file}")
    print(f"Test Reconstruction Loss: {test_rec_loss:.4f}, Test Latent Loss: {test_latent_loss:.6f}, Test Reconstruction Accuracy: {reconstruction_accuracy:.4f}")
    
    return test_rec_loss

def train_diffusion_model(
    latent_vectors_path,
    arch='ribo-ld-unsup',
    latent_dim=128,
    batch_size=32,
    learning_rate=8e-5,
    train_steps=5000,
    timesteps=500,
    save_every=500,
    results_folder=None,
    ae_model_path=None
):
    """
    Train a diffusion model on latent vectors.
    """
    if not DIFFUSION_AVAILABLE:
        raise ImportError("Diffusion library not available. Please install denoising_diffusion_pytorch")
    
    # Set random seed
    set_seed(42)
    
    # Create output directory
    if results_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"diffusion_results/{arch}"
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
    
    # Split train/validation
    train_latents, val_latents = train_test_split(latent_vectors, test_size=0.15, random_state=42)
    
    # Create Dataset1D instances
    train_dataset = Dataset1D(train_latents)
    val_dataset = Dataset1D(val_latents)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Define Unet1D model
    model = Unet1D(
        dim=64,                    # Model dimension
        dim_mults=(1, 2, 4, 8),   # Layer multipliers
        channels=1                 # Input channels (matches latent space)
    )
    
    # Define diffusion process
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=latent_dim,     # Sequence length matches latent dimension
        timesteps=timesteps,       # Number of diffusion timesteps
        objective='pred_v',         # Objective
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
        num_samples=16,  # Number of samples
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
    
    # Train
    trainer.train()
    
    print(f"Diffusion training completed!")
    print(f"Results saved to: {results_folder}")


    if trainer.ema is not None:
        generated_latents = trainer.ema.ema_model.sample(batch_size=8630)
    else:
        generated_latents = trainer.model.sample(batch_size=8630)
    
    # Handle generated latent tensor shape
    if len(generated_latents.shape) == 3:
        generated_latents = generated_latents.squeeze(1)  # Remove channel dimension
    seq_len=200
    ae_model = get_model(arch, seq_len, latent_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_checkpoint = torch.load(ae_model_path, map_location=device)
    ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
    ae_model.to(device)
    ae_model.eval()
    
    # Get decoder
    ae_decoder = ae_model  # Use the full model since we may also need activity prediction
    # Decode sequences (and optionally predict activity)
    x_rec = ae_decoder.decoder(generated_latents)
    # activity_pred = ae_decoder.activity_predictor(generated_latents)
    # predicted_activities = activity_pred.detach().cpu().numpy().flatten()
    x_rec_numpy = x_rec.detach().cpu().numpy()
    allseqs = []
    for seq in x_rec_numpy:
        seq_reshaped = seq.reshape(200, 5)
        indices = np.argmax(seq_reshaped, axis=1)
        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'P'}
        seq_str = ''.join([mapping[idx] for idx in indices])
        allseqs.append(seq_str)
    sample_path = results_folder + f'/sampled_sequences_step_{trainer.step}.txt'
    with open(sample_path, 'w') as f:
        for seq in allseqs:
            # truncate at first 'P' (padding), do not write 'P' itself
            seq_trunc = seq.split('P', 1)[0]
            if seq_trunc:  # optional: skip empty sequences
                f.write(f"{seq_trunc}\n")
    


    return results_folder
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ae', choices=['ae', 'diffusion', 'diffusion_activity'], 
                       help='Training mode: ae for autoencoder, diffusion for basic diffusion, diffusion_activity for diffusion with activity evaluation')
    parser.add_argument('--data_path', type=str, default='dataset/rfam.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--arch', type=str, default='ribo-ld-unsup')
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
    
    args = parser.parse_args()

    if args.mode == 'diffusion':
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
            save_every=args.save_every,
            ae_model_path=args.ae_model_path  # Optional: used for later sequence generation
        )
        
        print(f"Diffusion training completed. Results saved to: {results_folder}")

    elif args.mode == 'ae':
        rec_loss = train_and_evaluate(
            data_path=args.data_path,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            patience=args.patience,
            max_epochs=args.max_epochs,
            arch=args.arch
        )
        print(f"\nTest Recon Loss: {rec_loss:.4f}")

if __name__ == '__main__':
    main()