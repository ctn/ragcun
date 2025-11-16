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
    - Starts with pre-trained model (EmbeddingGemma or Sentence-BERT)
    - Projects to unnormalized Gaussian space (NO L2 normalization)
    - Trained with LeJEPA SIGReg loss for isotropy
    - Uses Euclidean distance for retrieval (not cosine similarity)

    Args:
        output_dim: Dimension of output embeddings (default: 512)
        base_model: Pre-trained model to use (default: google/embeddinggemma-300m)
        freeze_base: Freeze entire base encoder (only train projection)
        freeze_early_layers: Freeze first 4 transformer layers (if not freeze_base)

    Example:
        >>> # Full training
        >>> model = GaussianEmbeddingGemma(output_dim=512)
        
        >>> # Smart hybrid (train projection only)
        >>> model = GaussianEmbeddingGemma(
        ...     output_dim=512,
        ...     base_model='sentence-transformers/all-mpnet-base-v2',
        ...     freeze_base=True
        ... )
        >>> print(embeddings.shape)  # (2, 512)
    """

    def __init__(
        self,
        output_dim=512,
        base_model=None,
        freeze_base=False,
        freeze_early_layers=True,
        normalize_embeddings=True,
        use_predictor=False
    ):
        super().__init__()

        # Default to EmbeddingGemma if not specified
        if base_model is None:
            base_model = 'google/embeddinggemma-300m'

        self.base_model_name = base_model
        self.freeze_base = freeze_base
        self.normalize_embeddings = normalize_embeddings

        print(f"Loading base model: {base_model}")
        print(f"Base embedding normalization: {normalize_embeddings}")
        self.base = SentenceTransformer(
            base_model,
            trust_remote_code=True
        )

        # If not normalizing, remove the Normalize layer from the model pipeline
        # Many sentence-transformers models have Normalize() as the last module
        if not normalize_embeddings:
            if '2' in self.base._modules and type(self.base._modules['2']).__name__ == 'Normalize':
                print("⚠️  Removing built-in Normalize layer from base model")
                del self.base._modules['2']
            elif hasattr(self.base, '_last_module') and type(self.base._last_module()).__name__ == 'Normalize':
                print("⚠️  Removing built-in Normalize layer from base model")
                # Remove last module
                keys = list(self.base._modules.keys())
                if keys:
                    del self.base._modules[keys[-1]]

        # Get embedding dimension from base model
        base_dim = self.base.get_sentence_embedding_dimension()
        print(f"Base embedding dimension: {base_dim}")

        # Projection: base_dim (normalized) → output_dim (Gaussian)
        # CRITICAL: NO normalization layers!
        self.projection = nn.Sequential(
            nn.Linear(base_dim, base_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim * 2, output_dim)
        )

        # Handle freezing
        if freeze_base:
            # Freeze entire base encoder (smart hybrid training)
            for param in self.base.parameters():
                param.requires_grad = False
            print(f"✅ Froze entire base encoder ({base_model})")
        else:
            # Make base trainable
            for param in self.base.parameters():
                param.requires_grad = True
            
            # Optionally freeze early layers to preserve general knowledge
            if freeze_early_layers:
                frozen = 0
                for name, param in self.base.named_parameters():
                    # Try to freeze early transformer layers
                    if any(f'encoder.layer.{i}.' in name for i in range(4)):
                        param.requires_grad = False
                        frozen += 1
                    # Also handle different architecture names
                    elif any(f'layers.{i}.' in name for i in range(4)):
                        param.requires_grad = False
                        frozen += 1
                
                if frozen > 0:
                    print(f"Froze {frozen} parameters in early layers")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

        self.output_dim = output_dim
        
        # Predictor network (JEPA-style): predicts document embedding from query embedding
        if use_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
            print(f"✅ Added predictor network (query → document embedding)")
        else:
            self.predictor = None

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

        # Get base embeddings
        if self.training:
            # During training, use forward pass to maintain gradients
            tokenized = self.base.tokenize(texts)
            # Move to same device as model
            tokenized = {k: v.to(next(self.parameters()).device) for k, v in tokenized.items()}
            base_emb = self.base(tokenized)['sentence_embedding']
            # Note: If normalize_embeddings=False, we've already removed the Normalize layer
            # so base_emb is already unnormalized. No need for manual normalization.
        else:
            # During inference, use encode method (more efficient)
            # Note: normalize_embeddings flag is ignored here because we've already
            # removed the Normalize layer from the model if needed
            base_emb = self.base.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=show_progress
            )

        # Project to Gaussian space (undoes normalization)
        gaussian_emb = self.projection(base_emb)

        if convert_to_numpy:
            return gaussian_emb.detach().cpu().numpy()
        return gaussian_emb

    def forward(self, texts):
        """Forward pass for training."""
        return self.encode(texts, show_progress=False)

    def get_trainable_parameters(self):
        """
        Get list of trainable parameters for optimizer groups.
        
        Returns:
            Dictionary with 'base', 'projection', and 'predictor' parameter lists
        """
        base_params = []
        projection_params = []
        predictor_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'projection' in name:
                    projection_params.append(param)
                elif 'predictor' in name:
                    predictor_params.append(param)
                else:
                    base_params.append(param)
        
        return {
            'base': base_params,
            'projection': projection_params,
            'predictor': predictor_params
        }
    
    @classmethod
    def from_pretrained(
        cls,
        path,
        output_dim=512,
        base_model=None,
        freeze_base=False,
        normalize_embeddings=True,
        use_predictor=None
    ):
        """
        Load model from saved weights.

        Args:
            path: Path to saved state dict (.pt file)
            output_dim: Output dimension (must match saved model)
            base_model: Base model to use (default: same as saved)
            freeze_base: Whether to freeze base encoder
            normalize_embeddings: Whether to normalize base model embeddings
            use_predictor: Whether to load predictor (None = auto-detect from checkpoint)

        Returns:
            Loaded model
        """
        state_dict = torch.load(path, map_location='cpu', weights_only=False)

        # Handle both full checkpoint and state_dict only
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Auto-detect if predictor exists in checkpoint
        if use_predictor is None:
            use_predictor = any('predictor' in key for key in state_dict.keys())

        model = cls(
            output_dim=output_dim,
            base_model=base_model,
            freeze_base=freeze_base,
            freeze_early_layers=False,
            normalize_embeddings=normalize_embeddings,
            use_predictor=use_predictor
        )
        
        # Load state dict (may have predictor or not)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Loaded model from {path}")
        return model
