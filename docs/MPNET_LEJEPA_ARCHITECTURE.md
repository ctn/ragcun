# MPNet-LeJEPA Architecture

This document describes the MPNet-LeJEPA architecture implementation, which follows the LeJEPA (Joint Embedding Predictive Architecture with LeJEPA SIGReg) design pattern with online and target networks.

## Architecture Overview

The MPNet-LeJEPA architecture consists of:

1. **Two Encoders** (Online & Target, base frozen by default)
2. **Mean Pooling Layer** (shared)
3. **Two Projection Heads** (Online & Target)
4. **Predictor Head** (Online only)
5. **EMA Updates** for target network
6. **LeJEPA SIGReg Loss** for isotropy regularization

## Components

### 1. Encoders

#### Online Encoder (`encoder_online`)
- **Base Model**: `sentence-transformers/all-mpnet-base-v2`
- **Status**: Frozen by default (`freeze_base=True`), can be trainable
- **Initialization**: From pre-trained MPNet weights

#### Target Encoder (`encoder_target`)
- **Base Model**: Same as online encoder
- **Status**: No gradients (`requires_grad=False`)
- **Initialization**: Same weights as online encoder
- **Updates**: Via EMA (Exponential Moving Average)

### 2. Mean Pooling Layer

Computes mean over sequence length, weighted by attention mask:

```python
masked = token_embeddings * attention_mask[..., None]
summed = sum(masked over seq_len)
count = sum(attention_mask over seq_len)
pooled = summed / max(count, 1)
```

**Input**: `[batch_size, seq_len, hidden_size]` + `[batch_size, seq_len]`  
**Output**: `[batch_size, hidden_size]`

### 3. Projection Heads

#### Online Projection (`proj_online`)
- **Architecture**: 
  ```
  Linear(768 → 768)
  GELU
  Linear(768 → proj_dim)
  ```
- **Status**: Trainable
- **Output**: `[batch_size, proj_dim]`

#### Target Projection (`proj_target`)
- **Architecture**: Same as online
- **Status**: No gradients, updated via EMA
- **Initialization**: Same weights as online
- **Output**: `[batch_size, proj_dim]`

### 4. Predictor Head (Online Only)

- **Architecture**:
  ```
  Linear(proj_dim → proj_dim)
  GELU
  Linear(proj_dim → proj_dim)
  ```
- **Input**: `z_online = proj_online(pooled_online)`
- **Output**: `p_online = predictor(z_online)`
- **Status**: Trainable

## Forward Pass Structure

### Online Branch

For input text `x`:

```
tokens_x → encoder_online → token_embeddings_x
token_embeddings_x, attention_mask_x → mean_pool → h_x_online
h_x_online → proj_online → z_x_online
z_x_online → predictor → p_x_online
```

### Target Branch

For input text `y`:

```
tokens_y → encoder_target → token_embeddings_y
token_embeddings_y, attention_mask_y → mean_pool → h_y_target
h_y_target → proj_target → z_y_target
```

### LeJEPA Alignment

The predictor output `p_x_online` is aligned with the target embedding `z_y_target`:

- **Training**: Minimize distance between `p_x_online` and `z_y_target` (with stop-gradient on target)
- **Loss**: Combines predictive loss (MSE) + SIGReg loss (LeJEPA isotropy via Epps-Pulley + slicing)
- **Inference**: Use `z_online` (projected online embedding) as final representation

## Final Embedding Definition

For retrieval and downstream tasks, the model exposes:

```python
embedding(x) = normalize(proj_online(mean_pool(encoder_online(x))))
```

This is the **projected online embedding** (without the predictor), normalized for cosine similarity.

## EMA Updates

The target network is updated via Exponential Moving Average after each optimizer step:

```python
target_param = ema_decay * target_param + (1 - ema_decay) * online_param
```

**Default EMA decay**: `0.999`

## Usage

### Basic Usage

```python
from ragcun.mpnet_lejepa import MPNetLeJEPA
from ragcun.lejepa_loss import LeJEPALoss

# Create model
model = MPNetLeJEPA(
    base_model='sentence-transformers/all-mpnet-base-v2',
    proj_dim=256,
    ema_decay=0.999,
    freeze_base=True  # Default: frozen base
)

# Create loss function
criterion = LeJEPALoss(
    lambda_predictive=1.0,
    lambda_sigreg=1.0,
    num_slices=1000
)

# Forward pass (training)
texts_x = ['What is machine learning?']
texts_y = ['Machine learning is a subset of AI.']
output = model(texts_x, texts_y)
p_online = output['p_online']  # Predictor output
z_target = output['z_target']   # Target embedding

# Encode for retrieval
embeddings = model.encode(
    ['Query text', 'Document text'],
    normalize=True
)
```

### Training Loop

```python
from ragcun.lejepa_loss import LeJEPALoss

criterion = LeJEPALoss(
    lambda_predictive=1.0,
    lambda_sigreg=1.0,
    num_slices=1000
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Forward pass
    output = model(batch['queries'], batch['documents'])
    p_online = output['p_online']
    z_target = output['z_target']
    
    # Compute LeJEPA loss (predictive + SIGReg)
    loss, loss_dict = criterion(
        p_online=p_online,
        z_target=z_target,
        embeddings=p_online
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update target network via EMA
    model.update_target_network()
```

### Loading from Checkpoint

```python
model = MPNetLeJEPA.from_pretrained(
    path='checkpoints/mpnet_lejepa/best_model.pt',
    base_model='sentence-transformers/all-mpnet-base-v2',
    proj_dim=256,
    freeze_base=True
)
```

## Key Design Decisions

1. **Stop-Gradient on Target**: The target branch uses `.detach()` to prevent gradients from flowing back, preventing collapse.

2. **EMA Updates**: Target network is updated smoothly via EMA, providing stable targets for the predictor.

3. **Predictor Only on Online**: The predictor operates only on the online branch, forcing it to learn meaningful representations.

4. **Final Embedding**: Uses projected online embedding (without predictor) for inference, as the predictor is training-only.

## Parameters

- **Total Parameters**: ~220M (with MPNet-base)
- **Trainable Parameters** (with frozen base): ~330K (0.4%)
  - Online encoder: Frozen (0)
  - Online projection: ~200K
  - Predictor: ~130K
- **Non-trainable**: Base encoder (frozen), target encoder and projection (updated via EMA)

## LeJEPA SIGReg Loss

The architecture uses LeJEPA's SIGReg (Sketched Isotropic Gaussian Regularization) loss:

- **Epps-Pulley Test**: Univariate normality test using empirical characteristic function
- **SlicingUnivariateTest**: Extends Epps-Pulley to multivariate data via random 1D projections
- **Isotropy**: Encourages embeddings to follow isotropic Gaussian distribution
- **Integration**: Combined with predictive loss for joint training

## Differences from Previous Architecture

The previous `GaussianEmbeddingGemma` model:
- Used a single encoder
- Had an optional predictor
- Used simplified covariance-based isotropy loss

The new `MPNetLeJEPA` model:
- Uses dual encoders (online + target, base frozen by default)
- Always has a predictor (online only)
- Uses EMA for target network updates
- Uses LeJEPA SIGReg loss (Epps-Pulley + slicing) for isotropy
- Follows LeJEPA design pattern more closely

