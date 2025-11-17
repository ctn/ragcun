"""
LeJEPA SIGReg Loss Implementation

Implements the LeJEPA loss using:
- Epps-Pulley test for univariate normality
- SlicingUnivariateTest for multivariate isotropy
- JEPA-style predictive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional

# Add external LeJEPA to path
external_path = Path(__file__).parent.parent / "external" / "lejepa"
sys.path.insert(0, str(external_path))

try:
    from lejepa.univariate.epps_pulley import EppsPulley
    from lejepa.multivariate.slicing import SlicingUnivariateTest
except ImportError as e:
    print(f"Warning: Could not import LeJEPA modules: {e}")
    print("Falling back to simplified implementation")
    EppsPulley = None
    SlicingUnivariateTest = None


class LeJEPALoss(nn.Module):
    """
    LeJEPA Loss combining:
    1. Predictive loss (JEPA-style)
    2. SIGReg loss (LeJEPA-style isotropy via Epps-Pulley + slicing)
    
    Args:
        lambda_predictive: Weight for predictive loss (default: 1.0)
        lambda_sigreg: Weight for SIGReg isotropy loss (default: 1.0)
        num_slices: Number of random slices for multivariate test (default: 1000)
        t_max: Maximum integration point for Epps-Pulley (default: 5.0)
        n_points: Number of integration points (default: 21)
        use_stopgrad: Use stop-gradient on target (default: True)
    """
    
    def __init__(
        self,
        lambda_predictive: float = 1.0,
        lambda_sigreg: float = 1.0,
        num_slices: int = 1000,
        t_max: float = 5.0,
        n_points: int = 21,
        use_stopgrad: bool = True
    ):
        super().__init__()
        
        self.lambda_predictive = lambda_predictive
        self.lambda_sigreg = lambda_sigreg
        self.use_stopgrad = use_stopgrad
        
        # Initialize LeJEPA SIGReg components
        if EppsPulley is not None and SlicingUnivariateTest is not None:
            try:
                # Create Epps-Pulley univariate test
                # Try the first implementation signature (t_max, n_points, integration)
                try:
                    epps_pulley = EppsPulley(t_max=t_max, n_points=n_points, integration="trapezoid")
                except TypeError:
                    # Fallback to second implementation signature (t_range, n_points, weight_type)
                    epps_pulley = EppsPulley(t_range=(-t_max, t_max), n_points=n_points, weight_type="gaussian")
                
                # Wrap with slicing for multivariate testing
                self.sigreg = SlicingUnivariateTest(
                    univariate_test=epps_pulley,
                    num_slices=num_slices,
                    reduction="mean",
                    sampler="gaussian",
                    clip_value=0.01
                )
                self.use_lejepa = True
                print(f"✅ Using LeJEPA SIGReg (Epps-Pulley + {num_slices} slices)")
            except Exception as e:
                print(f"⚠️  Failed to initialize LeJEPA SIGReg: {e}")
                print("   Falling back to simplified covariance-based isotropy")
                self.sigreg = None
                self.use_lejepa = False
        else:
            # Fallback to simplified covariance-based isotropy
            self.sigreg = None
            self.use_lejepa = False
            print("⚠️  Using simplified covariance-based isotropy (LeJEPA not available)")
    
    def compute_sigreg_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss using LeJEPA approach.
        
        Args:
            embeddings: [batch_size, proj_dim] - Embeddings to test for isotropy
        
        Returns:
            sigreg_loss: Scalar tensor
        """
        if self.use_lejepa and self.sigreg is not None:
            # LeJEPA approach: Epps-Pulley + slicing
            # Input shape: (*, N, D) where N is batch size, D is dimension
            # We need to reshape to (1, batch_size, proj_dim) for slicing
            embeddings_reshaped = embeddings.unsqueeze(0)  # (1, batch_size, proj_dim)
            
            # Compute SIGReg statistic (lower is better, so we use it directly as loss)
            sigreg_stat = self.sigreg(embeddings_reshaped)
            
            return sigreg_stat
        else:
            # Fallback: Simplified covariance-based isotropy
            # Center embeddings
            embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
            
            # Compute covariance
            cov = torch.mm(embeddings_centered.t(), embeddings_centered) / (embeddings.size(0) - 1)
            
            # Target: identity * variance
            variance = embeddings.var(dim=0).mean()
            target_cov = torch.eye(cov.shape[0], device=cov.device) * variance
            
            # Frobenius norm of difference
            isotropy_loss = torch.norm(cov - target_cov, p='fro') / cov.shape[0]
            
            return isotropy_loss
    
    def forward(
        self,
        p_online: torch.Tensor,
        z_target: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute LeJEPA loss.
        
        Args:
            p_online: [batch_size, proj_dim] - Predictor output from online branch
            z_target: [batch_size, proj_dim] - Target embedding (will be detached if use_stopgrad)
            embeddings: [batch_size, proj_dim] - All embeddings for SIGReg (if None, uses p_online)
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with loss components
        """
        # 1. Predictive Loss (JEPA core)
        if self.use_stopgrad:
            z_target_detached = z_target.detach()
        else:
            z_target_detached = z_target
        
        predictive_loss = F.mse_loss(p_online, z_target_detached)
        
        # 2. SIGReg Loss (LeJEPA isotropy)
        if embeddings is None:
            # Use predictor output for SIGReg
            sigreg_embeddings = p_online
        else:
            sigreg_embeddings = embeddings
        
        sigreg_loss = self.compute_sigreg_loss(sigreg_embeddings)
        
        # Total loss
        total_loss = (
            self.lambda_predictive * predictive_loss +
            self.lambda_sigreg * sigreg_loss
        )
        
        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'predictive': predictive_loss.item(),
            'sigreg': sigreg_loss.item() if isinstance(sigreg_loss, torch.Tensor) else sigreg_loss,
        }
        
        return total_loss, loss_dict

