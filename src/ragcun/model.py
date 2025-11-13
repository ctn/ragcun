"""
GaussianEmbeddingGemma - Isotropic Gaussian embeddings via LeJEPA.

This model wraps EmbeddingGemma-300M and projects to unnormalized
isotropic Gaussian space using LeJEPA's SIGReg loss.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class GaussianEmbeddingGemma(nn.Module):
    """
    EmbeddingGemma with LeJEPA-trained projection to isotropic Gaussian space.

    Key features:
    - Starts with EmbeddingGemma-300M (state-of-the-art 300M param model)
    - Projects to unnormalized Gaussian space (NO L2 normalization)
    - Trained with LeJEPA SIGReg loss for isotropy
    - Uses Euclidean distance for retrieval (not cosine similarity)

    Args:
        output_dim: Dimension of output embeddings (default: 512)
        freeze_early_layers: Whether to freeze first 4 transformer layers

    Example:
        >>> model = GaussianEmbeddingGemma(output_dim=512)
        >>> embeddings = model.encode(["Hello world", "Machine learning"])
        >>> print(embeddings.shape)  # (2, 512)
    """

    def __init__(self, output_dim=512, freeze_early_layers=True):
        super().__init__()

        print("Loading EmbeddingGemma-300M...")
        self.base = SentenceTransformer(
            'google/embeddinggemma-300m',
            trust_remote_code=True
        )

        # Projection: 768 (normalized) → output_dim (Gaussian)
        # CRITICAL: NO normalization layers!
        self.projection = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768 * 2, output_dim)
        )

        # Make base trainable
        for param in self.base.parameters():
            param.requires_grad = True

        # Optionally freeze early layers to preserve general knowledge
        if freeze_early_layers:
            frozen = 0
            for name, param in self.base.named_parameters():
                if any(f'encoder.layer.{i}.' in name for i in range(4)):
                    param.requires_grad = False
                    frozen += 1
            if frozen > 0:
                print(f"Froze {frozen} parameters in early layers")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

        self.output_dim = output_dim

    def encode(self, texts, batch_size=32, show_progress=False, convert_to_numpy=False):
        """
        Encode texts to isotropic Gaussian embeddings.

        Args:
            texts: String or list of strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            convert_to_numpy: Return numpy array instead of tensor

        Returns:
            Tensor or numpy array of shape (num_texts, output_dim)
            IMPORTANT: NOT normalized! Use Euclidean distance for similarity.
        """
        # Ensure list
        if isinstance(texts, str):
            texts = [texts]

        # Get base embeddings (L2 normalized by EmbeddingGemma)
        base_emb = self.base.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )

        # Project to Gaussian space (undoes normalization)
        with torch.set_grad_enabled(self.training):
            gaussian_emb = self.projection(base_emb)

        if convert_to_numpy:
            return gaussian_emb.detach().cpu().numpy()
        return gaussian_emb

    def forward(self, texts):
        """Forward pass for training."""
        return self.encode(texts, show_progress=False)

    @classmethod
    def from_pretrained(cls, path, output_dim=512):
        """
        Load model from saved weights.

        Args:
            path: Path to saved state dict (.pt file)
            output_dim: Output dimension (must match saved model)

        Returns:
            Loaded model
        """
        model = cls(output_dim=output_dim, freeze_early_layers=False)
        state_dict = torch.load(path, map_location='cpu')

        # Handle both full checkpoint and state_dict only
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Loaded model from {path}")
        return model
