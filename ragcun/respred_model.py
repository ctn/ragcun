"""
ResPred: Residual Predictor for Dense Retrieval

Key innovation: Learns residual corrections (delta) instead of full predictions.
- Predictor learns: delta = doc_emb - query_emb
- Forward pass: predicted_doc = query_emb + alpha * delta
- Alpha is learnable, starts at 0.1
- Residual is bounded with Tanh()
"""

import torch
import torch.nn as nn
from ragcun.model import IsotropicGaussianEncoder


class ResidualGaussianEncoder(IsotropicGaussianEncoder):
    """
    Encoder with bounded residual predictor for stable query→doc prediction.

    Architecture:
        Inherits: IsotropicGaussianEncoder
            ↓
        Input Text → Base → Projection → [output_dim]
            ↓
        ResidualPredictor (replaces default predictor):
            ├─ Linear(dim → dim * 2)
            ├─ GELU()
            ├─ Dropout(0.1)
            ├─ Linear(dim * 2 → dim)
            ├─ Tanh()  ← Bounds to [-1, 1]
            ↓ delta (bounded residual)
            ├─ alpha (learnable scale, init=0.1)
            ↓
        Output: query_emb + alpha * delta

    Key innovations:
    1. Bounded residuals: Tanh ensures δ ∈ [-1, 1]
    2. Learnable scale: α parameter prevents large deviations
    3. Residual connection: Small corrections from query embedding
    4. Stability: Prevents predictor collapse during training

    Training strategy:
    - Frozen base encoder (freeze_base=True by default)
    - Train projection + residual predictor only
    - Loss includes residual regularization (L2 on δ)

    Args:
        base_model: Pre-trained model (default: all-mpnet-base-v2)
        output_dim: Embedding dimension (default: 768)
        freeze_base: Freeze base encoder (default: True)
        freeze_early_layers: Freeze early layers if base trainable (default: False)
        normalize_embeddings: L2 normalize base embeddings (default: False)
        use_predictor: Enable residual predictor (default: True)
        residual_scale_init: Initial value of α scale factor (default: 0.1)
    """
    
    def __init__(
        self,
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=768,
        freeze_base=True,
        freeze_early_layers=False,
        normalize_embeddings=False,
        use_predictor=True,
        residual_scale_init=0.1
    ):
        # Initialize parent (but we'll replace predictor)
        super().__init__(
            base_model=base_model,
            output_dim=output_dim,
            freeze_base=freeze_base,
            freeze_early_layers=freeze_early_layers,
            normalize_embeddings=normalize_embeddings,
            use_predictor=False  # Don't create default predictor
        )
        
        # Create ResPred predictor with residual connection
        if use_predictor:
            self.predictor = ResidualPredictor(
                dim=output_dim,
                scale_init=residual_scale_init
            )
            print(f"✅ Added ResPred predictor (residual scale α={residual_scale_init})")
        else:
            self.predictor = None
    
    def predict_with_residual(self, query_emb):
        """
        Predict document embedding from query using residual connection.
        
        Args:
            query_emb: Query embeddings [batch, dim]
            
        Returns:
            predicted_doc: query_emb + alpha * delta
            delta: The residual correction (for loss computation)
        """
        if self.predictor is None:
            return query_emb, torch.zeros_like(query_emb)
        
        return self.predictor(query_emb)


class ResidualPredictor(nn.Module):
    """
    Predictor that learns residual corrections with bounded output.
    
    Architecture:
        Input (query_emb) [dim]
            ↓
        Linear(dim → dim*2)
            ↓
        GELU()
            ↓
        Dropout(0.1)
            ↓
        Linear(dim*2 → dim)
            ↓
        Tanh()  ← Bound to [-1, 1]
            ↓
        delta [dim]
            ↓
        output = input + alpha * delta  (learnable alpha)
    """
    
    def __init__(self, dim=768, scale_init=0.1):
        super().__init__()
        
        # Predictor network
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Tanh()  # Bound residual to [-1, 1]
        )
        
        # Learnable scale factor (start small)
        self.alpha = nn.Parameter(torch.tensor(scale_init))
        
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input embeddings [batch, dim]
            
        Returns:
            output: x + alpha * delta
            delta: The raw residual (for regularization loss)
        """
        delta = self.network(x)  # Bounded to [-1, 1]
        output = x + self.alpha * delta  # Scaled residual
        
        return output, delta
    
    def get_alpha(self):
        """Get current alpha value."""
        return self.alpha.item()

