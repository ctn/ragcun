"""
MPNet-JEPA Architecture

Implements a JEPA-style architecture with:
- Online and target encoders (MPNet)
- Mean pooling layer
- Online and target projection heads
- Predictor head (online only)
- EMA updates for target network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, List


class MeanPooling(nn.Module):
    """
    Mean pooling layer for token embeddings.
    
    Computes mean over sequence length, weighted by attention mask.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            pooled_embedding: [batch_size, hidden_size]
        """
        # Expand attention mask to match embeddings shape
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Masked embeddings
        masked_embeddings = token_embeddings * mask_expanded
        
        # Sum over sequence length
        summed = torch.sum(masked_embeddings, dim=1)
        
        # Count (sum of attention mask)
        count = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        
        # Mean pooling
        pooled = summed / count
        
        return pooled


class ProjectionHead(nn.Module):
    """
    Projection MLP: maps pooled embeddings to lower-dimensional latent space.
    
    Architecture:
        Linear(hidden_size → hidden_size)
        GELU
        Linear(hidden_size → proj_dim)
    """
    
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_size]
        
        Returns:
            z: [batch_size, proj_dim]
        """
        return self.projection(x)


class PredictorHead(nn.Module):
    """
    Predictor MLP: operates only on online branch's projected embeddings.
    
    Architecture:
        Linear(proj_dim → proj_dim)
        GELU
        Linear(proj_dim → proj_dim)
    """
    
    def __init__(self, proj_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, proj_dim]
        
        Returns:
            p: [batch_size, proj_dim]
        """
        return self.predictor(z)


class MPNetLeJEPA(nn.Module):
    """
    MPNet-LeJEPA Architecture
    
    Implements LeJEPA-style architecture with:
    - Online encoder (trainable or frozen)
    - Target encoder (EMA, no gradients)
    - Mean pooling
    - Online and target projection heads
    - Predictor (online only)
    - LeJEPA SIGReg loss for isotropy
    
    Args:
        base_model: Base model name (default: 'sentence-transformers/all-mpnet-base-v2')
        proj_dim: Projection dimension (default: 256)
        ema_decay: EMA decay rate for target network (default: 0.999)
        freeze_base: Freeze base encoder (default: True)
    """
    
    def __init__(
        self,
        base_model: str = 'sentence-transformers/all-mpnet-base-v2',
        proj_dim: int = 256,
        ema_decay: float = 0.999,
        freeze_base: bool = True
    ):
        super().__init__()
        
        self.base_model_name = base_model
        self.proj_dim = proj_dim
        self.ema_decay = ema_decay
        self.freeze_base = freeze_base
        
        print(f"Initializing MPNet-LeJEPA with base model: {base_model}")
        print(f"Projection dimension: {proj_dim}")
        print(f"EMA decay: {ema_decay}")
        print(f"Freeze base: {freeze_base}")
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        config = AutoConfig.from_pretrained(base_model)
        hidden_size = config.hidden_size
        
        print(f"Hidden size: {hidden_size}")
        
        # 1. Online Encoder
        print("Loading online encoder...")
        self.encoder_online = AutoModel.from_pretrained(base_model)
        if freeze_base:
            # Freeze base encoder (only train projection + predictor)
            for param in self.encoder_online.parameters():
                param.requires_grad = False
            print("✅ Frozen online encoder (base model)")
        else:
            # Trainable encoder
            for param in self.encoder_online.parameters():
                param.requires_grad = True
            print("✅ Online encoder is trainable")
        
        # 2. Target Encoder (EMA, no gradients)
        print("Loading target encoder (EMA)...")
        self.encoder_target = AutoModel.from_pretrained(base_model)
        for param in self.encoder_target.parameters():
            param.requires_grad = False  # No gradients for target (always frozen)
        
        # 3. Mean Pooling (shared)
        self.mean_pooling = MeanPooling()
        
        # 4. Projection Heads
        print("Initializing projection heads...")
        self.proj_online = ProjectionHead(hidden_size, proj_dim)
        self.proj_target = ProjectionHead(hidden_size, proj_dim)
        
        # Initialize target projection with same weights as online
        self.proj_target.load_state_dict(self.proj_online.state_dict())
        for param in self.proj_target.parameters():
            param.requires_grad = False  # No gradients for target
        
        # 5. Predictor Head (online only)
        print("Initializing predictor head...")
        self.predictor = PredictorHead(proj_dim)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        if freeze_base:
            print("📌 Training strategy: Frozen base + trainable projection + predictor")
    
    def update_target_network(self):
        """
        Update target network weights via EMA.
        
        Should be called after each optimizer step during training.
        Only updates projection head (encoder is frozen).
        """
        with torch.no_grad():
            # Update encoder (only if not frozen, but typically both are frozen)
            if not self.freeze_base:
                for online_param, target_param in zip(
                    self.encoder_online.parameters(),
                    self.encoder_target.parameters()
                ):
                    target_param.data.mul_(self.ema_decay).add_(
                        online_param.data, alpha=1 - self.ema_decay
                    )
            
            # Update projection head (always updated via EMA)
            for online_param, target_param in zip(
                self.proj_online.parameters(),
                self.proj_target.parameters()
            ):
                target_param.data.mul_(self.ema_decay).add_(
                    online_param.data, alpha=1 - self.ema_decay
                )
    
    def forward_online(
        self,
        texts: List[str],
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through online branch.
        
        Args:
            texts: List of input texts
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            p_online: [batch_size, proj_dim] - Predictor output
            intermediates: Optional dict with intermediate representations
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.encoder_online.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Encode
        with torch.set_grad_enabled(self.training):
            outputs = self.encoder_online(**encoded)
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Mean pool
        h_online = self.mean_pooling(token_embeddings, encoded['attention_mask'])
        
        # Project
        z_online = self.proj_online(h_online)
        
        # Predict
        p_online = self.predictor(z_online)
        
        if return_intermediates:
            intermediates = {
                'token_embeddings': token_embeddings,
                'h_online': h_online,
                'z_online': z_online,
                'p_online': p_online
            }
            return p_online, intermediates
        
        return p_online, None
    
    def forward_target(
        self,
        texts: List[str],
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through target branch.
        
        Args:
            texts: List of input texts
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            z_target: [batch_size, proj_dim] - Projected target embedding
            intermediates: Optional dict with intermediate representations
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        device = next(self.encoder_target.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Encode (no gradients)
        with torch.no_grad():
            outputs = self.encoder_target(**encoded)
            token_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Mean pool
        h_target = self.mean_pooling(token_embeddings, encoded['attention_mask'])
        
        # Project
        z_target = self.proj_target(h_target)
        
        if return_intermediates:
            intermediates = {
                'token_embeddings': token_embeddings,
                'h_target': h_target,
                'z_target': z_target
            }
            return z_target, intermediates
        
        return z_target, None
    
    def forward(
        self,
        texts_x: List[str],
        texts_y: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA training.
        
        Args:
            texts_x: Input texts for online branch
            texts_y: Input texts for target branch (if None, uses texts_x)
        
        Returns:
            Dictionary with:
                - 'p_online': Predictor output from online branch
                - 'z_target': Projected embedding from target branch
        """
        if texts_y is None:
            texts_y = texts_x
        
        # Online branch
        p_online, _ = self.forward_online(texts_x)
        
        # Target branch
        z_target, _ = self.forward_target(texts_y)
        
        return {
            'p_online': p_online,
            'z_target': z_target
        }
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_numpy: bool = False
    ) -> torch.Tensor:
        """
        Encode texts to final embeddings (for retrieval/downstream use).
        
        Uses: normalize(proj_online(mean_pool(encoder_online(x))))
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            convert_to_numpy: Whether to return numpy array
        
        Returns:
            embeddings: [num_texts, proj_dim]
        """
        self.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Forward through online branch (without predictor)
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                device = next(self.encoder_online.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Encode
                outputs = self.encoder_online(**encoded)
                token_embeddings = outputs.last_hidden_state
                
                # Mean pool
                h_online = self.mean_pooling(token_embeddings, encoded['attention_mask'])
                
                # Project (this is the final embedding)
                z_online = self.proj_online(h_online)
                
                # Normalize if requested
                if normalize:
                    z_online = F.normalize(z_online, p=2, dim=1)
                
                all_embeddings.append(z_online)
        
        # Concatenate
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return embeddings.cpu().numpy()
        
        return embeddings
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        base_model: Optional[str] = None,
        proj_dim: Optional[int] = None,
        ema_decay: float = 0.999,
        freeze_base: bool = True
    ):
        """
        Load model from saved checkpoint.
        
        Args:
            path: Path to checkpoint file
            base_model: Base model name (if None, uses saved value)
            proj_dim: Projection dimension (if None, uses saved value)
            ema_decay: EMA decay rate
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle both full checkpoint and state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Try to get config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                base_model = config.get('base_model', base_model)
                proj_dim = config.get('proj_dim', proj_dim)
        else:
            state_dict = checkpoint
        
        if base_model is None:
            base_model = 'sentence-transformers/all-mpnet-base-v2'
        if proj_dim is None:
            # Try to infer from state_dict
            for key in state_dict.keys():
                if 'proj_online.projection.2.weight' in key:
                    proj_dim = state_dict[key].shape[0]
                    break
            if proj_dim is None:
                proj_dim = 256  # Default
        
        model = cls(
            base_model=base_model,
            proj_dim=proj_dim,
            ema_decay=ema_decay,
            freeze_base=freeze_base
        )
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"✅ Loaded MPNet-LeJEPA from {path}")
        return model

