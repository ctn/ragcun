# Quick Reference Guide

## What's Active (Current Work)

### Training
- **Location**: `checkpoints/jepa_supervised_finetuned/`
- **Files**: 
  - `best_model.pt` (epoch 1, with predictor)
  - `checkpoint_epoch_1.pt` (with predictor)
- **Log**: `logs/supervised_finetuning.log`
- **Status**: Training in progress (Epoch 2/3)
- **Config**: `use_predictor=True`, `lambda_predictive=1.0`

### Evaluation
- **Results**: `results/beir_standard/jepa_supervised_finetuned_with_predictor.json` (in progress)
- **Log**: `logs/jepa_supervised_finetuned_with_predictor_eval.log`

### Reference Model
- **Self-Supervised**: `checkpoints/jepa_xy_masked/best_model.pt`
- **Results**: `results/beir_standard/jepa_xy_masked.json` (if exists)

## What's Archived

### Failed Supervised Fine-Tuning (No Predictor)
- **Issue**: Embedding collapse (cosine sim = 0.996)
- **Performance**: NDCG@10 ~0.02-0.04 (terrible)
- **Location**: `archive/checkpoints/jepa_supervised_finetuned_no_predictor/`
- **Logs**: `archive/logs/supervised_finetuning_no_predictor.log`

### Old Checkpoints
- **OLD_with_predictor**: `archive/checkpoints/jepa_supervised_finetuned_OLD_with_predictor/`
- **No_predictor**: `archive/checkpoints/jepa_supervised_finetuned_no_predictor/`

### Old Scripts
- **wait_and_restart_from_checkpoint.sh**: `archive/scripts/`

### Old Evaluation Logs
- Various evaluation logs from self-supervised model testing
- See `archive/INDEX.txt` for full list

## Key Lesson Learned

**Predictor is essential for supervised fine-tuning!**
- Without predictor: Model collapses (embeddings become identical)
- With predictor: Model maintains diversity (prevents collapse)
- The predictor loss with stop-gradient prevents embedding collapse during training
