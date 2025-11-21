# Migration Guide: Finding Your Scripts

If you're looking for a script that was previously in `scripts/train/` or `scripts/eval/`, here's where to find it now:

## 🔄 Training Scripts

### Old → New Locations

| Old Location | New Location | Model Class |
|-------------|--------------|-------------|
| `train/isotropic.py` | `by_model/isotropic_gaussian/train_contrastive.py` | IsotropicGaussianEncoder |
| `train/xy_masked.py` | `by_model/isotropic_gaussian/train_xy_masked.py` | IsotropicGaussianEncoder |
| `train/self_supervised.py` | `by_model/isotropic_gaussian/train_self_supervised.py` | IsotropicGaussianEncoder |
| `train/residual_gaussian.py` | `by_model/residual_gaussian/train.py` | ResidualGaussianEncoder |
| `train/asymmetric_dual.py` | `by_model/asymmetric_dual/train.py` | AsymmetricDualEncoder |
| `train/asymmetric_predictor.py` | `by_model/asymmetric_predictor/train.py` | AsymmetricWithPredictor |
| `train/mpnet_lejepa.py` | `by_model/mpnet_lejepa/train.py` | MPNetLeJEPA |

## 🔄 Evaluation Scripts

| Old Location | New Location | Model Class |
|-------------|--------------|-------------|
| `eval/beir.py` | `by_model/isotropic_gaussian/eval_beir.py` | IsotropicGaussianEncoder |
| `eval/residual_gaussian_quick.py` | `by_model/residual_gaussian/eval_quick.py` | ResidualGaussianEncoder |
| `eval/asymmetric_dual_quick.py` | `by_model/asymmetric_dual/eval_quick.py` | AsymmetricDualEncoder |
| `eval/asymmetric_predictor_quick.py` | `by_model/asymmetric_predictor/eval_quick.py` | AsymmetricWithPredictor |

## 📝 Note

The original scripts in `train/` and `eval/` are still present and functional. The `by_model/` directory provides an alternative, more organized view of the same scripts grouped by their model class.

## 🎯 Quick Reference

### "I want to train the best model"
→ Use `by_model/isotropic_gaussian/train_contrastive.py`

### "I want to evaluate on BEIR"
→ Use `by_model/isotropic_gaussian/eval_beir.py`

### "I want to try separate query/doc projections"
→ Use `by_model/asymmetric_predictor/train.py`

### "I want self-supervised learning"
→ Use `by_model/isotropic_gaussian/train_xy_masked.py` or `train_self_supervised.py`

## 🗂️ Directory Structure

```
scripts/
├── by_model/                    ← NEW ORGANIZED VIEW
│   ├── README.md               ← Start here!
│   ├── INDEX.md                ← Command examples
│   ├── MIGRATION_GUIDE.md      ← This file
│   │
│   ├── isotropic_gaussian/     ← 🥇 Best performer
│   │   ├── README.md
│   │   ├── train_contrastive.py
│   │   ├── train_xy_masked.py
│   │   ├── train_self_supervised.py
│   │   └── eval_beir.py
│   │
│   ├── asymmetric_predictor/   ← 🥈 Second best
│   │   ├── README.md
│   │   ├── train.py
│   │   └── eval_quick.py
│   │
│   ├── asymmetric_dual/        ← 🥉 Third place
│   │   ├── train.py
│   │   └── eval_quick.py
│   │
│   ├── residual_gaussian/      ← ❌ Failed experiment
│   │   ├── train.py
│   │   └── eval_quick.py
│   │
│   └── mpnet_lejepa/           ← ❓ Untested
│       └── train.py
│
├── train/                      ← OLD LOCATION (still works)
│   ├── isotropic.py
│   ├── xy_masked.py
│   ├── ...
│   └── (7 training scripts)
│
└── eval/                       ← OLD LOCATION (still works)
    ├── beir.py
    ├── asymmetric_dual_quick.py
    └── (more evaluation scripts)
```

## 💡 Benefits of New Organization

1. **Clarity**: Immediately see which scripts use which model class
2. **Documentation**: Each model class has its own README with examples
3. **Performance**: Ranked by actual NDCG@10 scores
4. **Completeness**: Training + evaluation scripts together
5. **Discovery**: INDEX.md provides copy-paste commands

## 🔗 Related Documentation

- Model implementations: `/home/ubuntu/ragcun/ragcun/`
- Results: `/home/ubuntu/ragcun/results/beir_standard/`
- Top-level docs: `/home/ubuntu/ragcun/SCRIPT_ORGANIZATION.md`
