"""
Asymmetric Projection Model WITH Predictor for Dense Retrieval

Key innovation: Combines explicit query/doc projections with predictive learning.
- Query projection: Learns "how to seek/ask"
- Document projection: Learns "how to provide/answer"
- Predictor: Learns query→doc transformation (JEPA-style)

Architecture: (1, 0, 1)
- Shared encoder (frozen)
- Separate projections (trainable)
- Predictor (trainable)
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List


class AsymmetricWithPredictor(nn.Module):
    """
    Asymmetric dual projection encoder with predictor for JEPA-style learning.
    
    Architecture:
        Input (Query or Document)
            ↓
        Frozen SentenceTransformer (shared)
            ↓ [base_dim]
            ├─────────────┬─────────────┤
            ↓             ↓             ↓
        Query Path     Doc Path
            ↓             ↓
        Query Projection  Doc Projection
        (2-layer MLP)    (2-layer MLP)
            ↓             ↓
        Linear(base → base*2)
        GELU()           GELU()
        Dropout(0.1)     Dropout(0.1)
        Linear(base*2 → out)
            ↓             ↓
        q_emb [out_dim]  d_emb [out_dim]
            ↓
        Predictor (query→doc)
        (2-layer MLP)
            ↓
        predicted_doc [out_dim]
    
    Key features:
    - Shared frozen encoder: Single base model for efficiency
    - Asymmetric projections: Separate query/doc semantic spaces
    - Predictor: JEPA-style query→doc transformation
    - Stop-gradient: On doc embeddings in predictive loss
    
    Training strategy:
    - Frozen base encoder (always)
    - Train both projections + predictor simultaneously
    - Triple loss: Contrastive + Isotropy + Predictive
    
    Args:
        base_model: Pre-trained model (default: all-mpnet-base-v2)
        output_dim: Projection dimension (default: 768)
        freeze_base: Freeze base encoder (default: True, always recommended)
        normalize_embeddings: Apply L2 norm to base embeddings (default: False)
    """
    
    def __init__(
        self,
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=768,
        freeze_base=True,
        normalize_embeddings=False
    ):
        super().__init__()
        
        self.base_model_name = base_model
        self.freeze_base = freeze_base
        self.normalize_embeddings = normalize_embeddings
        self.output_dim = output_dim
        
        print(f"Loading base model: {base_model}")
        self.base = SentenceTransformer(
            base_model,
            trust_remote_code=True
        )
        
        # Remove normalization if needed
        if not normalize_embeddings:
            if '2' in self.base._modules and type(self.base._modules['2']).__name__ == 'Normalize':
                print("⚠️  Removing built-in Normalize layer from base model")
                del self.base._modules['2']
        
        # Get embedding dimension
        base_dim = self.base.get_sentence_embedding_dimension()
        print(f"Base embedding dimension: {base_dim}")
        
        # Freeze base encoder
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
            print(f"✅ Froze entire base encoder ({base_model})")
        
        # Asymmetric projections
        self.query_projection = self._build_projection(base_dim, output_dim)
        self.doc_projection = self._build_projection(base_dim, output_dim)
        
        print(f"✅ Created asymmetric projections ({base_dim} → {output_dim})")
        print(f"   Query projection: {sum(p.numel() for p in self.query_projection.parameters()):,} params")
        print(f"   Doc projection: {sum(p.numel() for p in self.doc_projection.parameters()):,} params")
        
        # Predictor: query→doc transformation
        self.predictor = self._build_projection(output_dim, output_dim)
        print(f"✅ Created predictor: query_emb → predicted_doc_emb")
        print(f"   Predictor: {sum(p.numel() for p in self.predictor.parameters()):,} params")
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"Architecture: (1, 0, 1) - Shared encoder + Separate projections + Predictor")
    
    def _build_projection(self, input_dim, output_dim):
        """Build a 2-layer MLP projection head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )
    
    def encode(self, texts, batch_size=32, show_progress=False, convert_to_numpy=False, is_query=True):
        """
        Encode texts using appropriate projection.
        
        Args:
            texts: String or list of strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            convert_to_numpy: Return numpy array
            is_query: If True, use query_projection; else use doc_projection
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get base embeddings
        if self.training:
            tokenized = self.base.tokenize(texts)
            tokenized = {k: v.to(next(self.parameters()).device) for k, v in tokenized.items()}
            base_emb = self.base(tokenized)['sentence_embedding']
        else:
            base_emb = self.base.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=show_progress
            )
            # Ensure base embeddings are on same device as projection layers
            device = next(self.parameters()).device
            base_emb = base_emb.to(device)
        
        # Project with appropriate head
        if is_query:
            embeddings = self.query_projection(base_emb)
        else:
            embeddings = self.doc_projection(base_emb)
        
        if convert_to_numpy:
            return embeddings.detach().cpu().numpy()
        return embeddings
    
    def encode_queries(self, texts, **kwargs):
        """Encode queries."""
        return self.encode(texts, is_query=True, **kwargs)
    
    def encode_docs(self, texts, **kwargs):
        """Encode documents."""
        return self.encode(texts, is_query=False, **kwargs)
    
    def predict_doc_from_query(self, query_emb):
        """
        Predict document embedding from query embedding.
        
        Args:
            query_emb: Query embeddings [batch, dim]
            
        Returns:
            predicted_doc: Predicted document embeddings [batch, dim]
        """
        return self.predictor(query_emb)
    
    def forward(self, texts, is_query=True):
        """Forward pass for training."""
        return self.encode(texts, show_progress=False, is_query=is_query)
    
    def get_trainable_parameters(self):
        """Get trainable parameters grouped by component."""
        query_params = list(self.query_projection.parameters())
        doc_params = list(self.doc_projection.parameters())
        predictor_params = list(self.predictor.parameters())
        
        return {
            'query_projection': query_params,
            'doc_projection': doc_params,
            'predictor': predictor_params,
            'base': []  # Empty since base is frozen
        }
    
    @classmethod
    def from_pretrained(
        cls,
        path,
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=768,
        freeze_base=True,
        normalize_embeddings=False
    ):
        """
        Load model from saved weights.
        
        Args:
            path: Path to saved state dict (.pt file)
            base_model: Base model to use
            output_dim: Output dimension
            freeze_base: Whether to freeze base encoder
            normalize_embeddings: Whether to normalize base embeddings
            
        Returns:
            Loaded model
        """
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle both full checkpoint and state_dict only
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model = cls(
            base_model=base_model,
            output_dim=output_dim,
            freeze_base=freeze_base,
            normalize_embeddings=normalize_embeddings
        )
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"✅ Loaded model from {path}")
        return model


