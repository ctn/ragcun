# AsymmetricWithPredictor Scripts 🥈

**Explicit Query/Doc Separation + Predictor** - NDCG@10: ~0.48

## Architecture

- **Pattern:** (1, 0, 1)
- **Encoder:** Shared, frozen MPNet base
- **Projection:** Separate for queries and documents
- **Predictor:** Yes - learns query→document transformation

## Model Class

`ragcun.asymmetric_predictor_model.AsymmetricWithPredictor`

## Scripts

### train.py

Asymmetric projections with predictor for query→doc learning.

```bash
python scripts/by_model/asymmetric_predictor/train.py \
  --train_data data/processed/msmarco_smoke/train.json \
  --output_dim 768 \
  --epochs 3 \
  --batch_size 32 \
  --lambda_contrastive 1.0 \
  --lambda_isotropy_query 1.5 \
  --lambda_isotropy_doc 1.5 \
  --lambda_predictive 1.2
```

### eval_quick.py

Quick BEIR evaluation on selected datasets.

```bash
python scripts/by_model/asymmetric_predictor/eval_quick.py \
  --checkpoint checkpoints/asymmetric_pred_smoke/best_model.pt \
  --datasets scifact nfcorpus
```

## Best Models

| Model | NDCG@10 | Notes |
|-------|---------|-------|
| asymmetric_pred_smoke | ~0.48 | Good balance of performance |

## When to Use

- Want explicit query/document separation
- Need interpretable separate spaces
- Predictor helps close the gap to (1,1,1)

## Key Parameters

- `--lambda_isotropy_query`: Query space isotropy (1.5)
- `--lambda_isotropy_doc`: Doc space isotropy (1.5)
- `--lambda_predictive`: Predictive loss weight (1.2)
