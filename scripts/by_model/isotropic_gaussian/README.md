# IsotropicGaussianEncoder Scripts 🥇

**Best Performing Architecture** - NDCG@10: 0.4779

## Architecture

- **Pattern:** (1, 1, 1) or (1, 1, 0)
- **Encoder:** Shared, frozen MPNet base
- **Projection:** Shared for queries and documents
- **Predictor:** Optional (use `--use_predictor` flag)

## Model Class

`ragcun.model.IsotropicGaussianEncoder`

## Scripts

### 1. train_contrastive.py (Main Training)

Query-document pairs with contrastive + isotropy + predictive losses.

```bash
python scripts/by_model/isotropic_gaussian/train_contrastive.py \
  --train_data data/processed/msmarco_smoke/train.json \
  --use_predictor \
  --freeze_base \
  --output_dim 768 \
  --epochs 3 \
  --batch_size 32 \
  --lambda_contrastive 1.0 \
  --lambda_isotropy 1.5 \
  --lambda_predictive 1.2
```

### 2. train_xy_masked.py (Self-Supervised)

Original text (X) vs masked text (Y) - pure predictive learning.

```bash
python scripts/by_model/isotropic_gaussian/train_xy_masked.py \
  --input_xy_pairs data/processed/xy_masked_documents.json \
  --use_predictor \
  --epochs 3
```

### 3. train_self_supervised.py (Document Splitting)

Split documents into two parts - learn to predict one from the other.

```bash
python scripts/by_model/isotropic_gaussian/train_self_supervised.py \
  --documents data/processed/documents.json \
  --use_predictor \
  --split_method half
```

### 4. eval_beir.py (Evaluation)

Evaluate on BEIR benchmark datasets.

```bash
python scripts/by_model/isotropic_gaussian/eval_beir.py \
  --model_path checkpoints/jepa_10k/best_model.pt \
  --use_predictor \
  --datasets scifact nfcorpus arguana
```

## Best Models

| Model | NDCG@10 | Config |
|-------|---------|--------|
| jepa_10k | 0.4779 | predictor=True, iso=1.5, pred=1.2 |
| 512dim_10k | 0.4726 | predictor=True, dim=512 |
| jepa_iso15_pred12 | 0.4610 | predictor=True |
| pure_isotropy | 0.4562 | predictor=False |

## Key Parameters

- `--use_predictor`: Enable predictor head (recommended!)
- `--freeze_base`: Freeze base encoder (recommended!)
- `--lambda_isotropy`: Isotropy loss weight (1.5 works well)
- `--lambda_predictive`: Predictive loss weight (1.2 works well)
- `--output_dim`: Embedding dimension (768 or 512)
