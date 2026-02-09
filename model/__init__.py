"""
Model factory for importing and creating different autoencoder architectures
"""

from .ribo_ld import Ribo_LD_AEWithRegressor
from .ribo_ld_unsup import Ribo_LD_AE

def get_model(arch, seq_len, latent_dim):
    """
    Args:
        arch (str): Architecture type ('ribo-ld', 'ribo-ld-unsup')
        seq_len (int): Sequence length
        latent_dim (int): Latent dimension size
    
    Returns:
        nn.Module: Model instance
    """
    seq_len=int(seq_len)
    if arch == 'ribo-ld':
        return Ribo_LD_AEWithRegressor(seq_len, latent_dim)
    elif arch == 'ribo-ld-unsup':
        return Ribo_LD_AE(seq_len, latent_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: 'ribo-ld-unsup', 'ribo-ld'.")


# Export individual model classes for direct import
__all__ = [
    'get_model',
    'Ribo_LD_AEWithRegressor', 
    'Ribo_LD_AE'
]
