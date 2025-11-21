# Script Index by Model Class

## 🏆 IsotropicGaussianEncoder (Winner!)

**Best Performing Architecture**

### Training Scripts:
1. **train_contrastive.py** - Main training with query-doc pairs
   - Command: `python by_model/isotropic_gaussian/train_contrastive.py --train_data data/processed/msmarco_smoke/train.json --use_predictor --freeze_base`
   - Best for: Production models
   - Result: jepa_10k (NDCG@10: 0.4779) 🥇

2. **train_xy_masked.py** - Self-supervised with masked text
   - Command: `python by_model/isotropic_gaussian/train_xy_masked.py --input_xy_pairs data/processed/xy_masked.json`
   - Best for: Unsupervised learning

3. **train_self_supervised.py** - Self-supervised with document splitting
   - Command: `python by_model/isotropic_gaussian/train_self_supervised.py --documents data/processed/documents.json`
   - Best for: When only documents available

### Evaluation:
- **eval_beir.py** - Evaluate on BEIR benchmark
  - Command: `python by_model/isotropic_gaussian/eval_beir.py --model_path checkpoints/jepa_10k/best_model.pt --use_predictor`

---

## 🥈 AsymmetricWithPredictor (Second Best)

**Explicit Query/Doc Separation + Predictor**

### Training:
- **train.py** - Asymmetric projections with predictor
  - Command: `python by_model/asymmetric_predictor/train.py --train_data data/processed/msmarco_smoke/train.json`
  - Result: ~0.48 NDCG@10 🥈

### Evaluation:
- **eval_quick.py** - Quick BEIR evaluation
  - Command: `python by_model/asymmetric_predictor/eval_quick.py --checkpoint checkpoints/asymmetric_pred_smoke/best_model.pt`

---

## 🥉 AsymmetricDualEncoder (Third)

**Separate Projections, No Predictor**

### Training:
- **train.py** - Asymmetric projections only
  - Command: `python by_model/asymmetric_dual/train.py --train_data data/processed/msmarco_smoke/train.json`
  - Result: ~0.47 NDCG@10 🥉

### Evaluation:
- **eval_quick.py** - Quick BEIR evaluation
  - Command: `python by_model/asymmetric_dual/eval_quick.py --checkpoint checkpoints/asymmetric_smoke/best_model.pt`

---

## ❌ ResidualGaussianEncoder (Failed)

**Residual Predictor with Bounded Updates**

### Training:
- **train.py** - ResPred with residual connections
  - Command: `python by_model/residual_gaussian/train.py --train_data data/processed/msmarco_smoke/train.json`
  - Result: 0.4416 NDCG@10 ❌ (identity trap)

### Evaluation:
- **eval_quick.py** - Quick BEIR evaluation
  - Command: `python by_model/residual_gaussian/eval_quick.py --checkpoint checkpoints/respred_smoke/best_model.pt`

---

## ❓ MPNetLeJEPA (Untested)

**Full BYOL/JEPA with Dual Encoders**

### Training:
- **train.py** - Full JEPA training with EMA
  - Command: `python by_model/mpnet_lejepa/train.py --train_data data/processed/msmarco_smoke/train.json`
  - Result: Untested

### Evaluation:
- Use: `by_model/isotropic_gaussian/eval_beir.py` (needs adaptation)

---

## 🎯 Quick Start

### For best results:
```bash
# Train
python scripts/by_model/isotropic_gaussian/train_contrastive.py \
  --train_data data/processed/msmarco_smoke/train.json \
  --use_predictor \
  --freeze_base \
  --output_dim 768 \
  --epochs 3

# Evaluate
python scripts/by_model/isotropic_gaussian/eval_beir.py \
  --model_path checkpoints/YOUR_MODEL/best_model.pt \
  --use_predictor \
  --datasets scifact nfcorpus
```

## 📊 Performance Summary

| Model Class | Pattern | NDCG@10 | Rank |
|------------|---------|---------|------|
| IsotropicGaussianEncoder | (1,1,1) | 0.4779 | 🥇 |
| AsymmetricWithPredictor | (1,0,1) | ~0.48 | 🥈 |
| AsymmetricDualEncoder | (1,0,0) | ~0.47 | 🥉 |
| ResidualGaussianEncoder | (1,1,1)* | 0.4416 | ❌ |
| MPNetLeJEPA | (0,0,1) | ❓ | ? |

*Special residual predictor
