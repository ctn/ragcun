# Archive Directory

This directory contains archived training runs, evaluations, and scripts that are no longer actively used but kept for reference.

## Structure

- `checkpoints/` - Old model checkpoints
- `results/` - Old evaluation results  
- `logs/` - Old training/evaluation logs
- `scripts/` - Old/unused scripts

## Archived Items

### Supervised Fine-Tuning (Without Predictor) - Nov 17, 2025
- **Issue**: Model collapsed during fine-tuning (embeddings became too similar)
- **Root Cause**: Missing predictor loss (lambda_predictive=0.0) led to embedding collapse
- **Results**: Very poor performance (NDCG@10 ~0.02-0.04 vs expected ~0.45-0.66)
- **Files**: 
  - `checkpoints/jepa_supervised_finetuned/` (old checkpoints without predictor)
  - `results/beir_standard/jepa_supervised_finetuned.json` (terrible results)
  - `logs/supervised_finetuning.log` (training log)
  - `logs/jepa_supervised_finetuned_eval.log` (evaluation log)

### Current Active Training
- **Location**: `checkpoints/jepa_supervised_finetuned/` (with predictor enabled)
- **Configuration**: `use_predictor=True`, `lambda_predictive=1.0`
- **Status**: Training in progress (Epoch 2/3)
