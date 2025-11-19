"""
Asymmetric Projection Model for Dense Retrieval

Key innovation: Different projection heads for queries vs documents.
- Query projection: Learns "how to seek/ask"
- Document projection: Learns "how to provide/answer"
- No predictor needed - just contrastive learning

This avoids the identity mapping problem of ResPred.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List


class AsymmetricProjectionModel(nn.Module):
    """
    Model with asymmetric projections for queries and documents.
    
    Architecture:
        Frozen Encoder (shared)
            ↓
        Query Projection ← Different
            ↓           ↘
        z_q              z_d ← Different
                        ↗
        Doc Projection ← Different
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
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    
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
    
    def forward(self, texts, is_query=True):
        """Forward pass for training."""
        return self.encode(texts, show_progress=False, is_query=is_query)
    
    def get_trainable_parameters(self):
        """Get trainable parameters grouped by component."""
        query_params = list(self.query_projection.parameters())
        doc_params = list(self.doc_projection.parameters())
        
        return {
            'query_projection': query_params,
            'doc_projection': doc_params,
            'base': []  # Empty since base is frozen
        }

