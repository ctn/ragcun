# Scripts by Model Class

**Organized by base model architecture for easy discovery.**

## 📁 Directory Structure

```
by_model/
├── isotropic_gaussian/      🥇 Best performer (0.4779 NDCG@10)
├── asymmetric_predictor/    🥈 Second best (0.48 NDCG@10)
├── asymmetric_dual/         🥉 Third place (0.47 NDCG@10)
├── residual_gaussian/       ❌ Failed experiment (0.4416 NDCG@10)
└── mpnet_lejepa/            ❓ Untested (full BYOL/JEPA)
```

## 🎯 Quick Navigation

### Want the best results?
→ [`isotropic_gaussian/`](isotropic_gaussian/) - IsotropicGaussianEncoder

### Want explicit query/doc separation?
→ [`asymmetric_predictor/`](asymmetric_predictor/) - AsymmetricWithPredictor

### Want to try full JEPA/BYOL?
→ [`mpnet_lejepa/`](mpnet_lejepa/) - MPNetLeJEPA

## 📊 Performance Comparison

| Model Class | Pattern* | NDCG@10 | Description |
|------------|---------|---------|-------------|
| IsotropicGaussianEncoder | (1,1,1) | **0.4779** | Shared encoder + projection + predictor |
| AsymmetricWithPredictor | (1,0,1) | ~0.48 | Shared encoder + separate projections + predictor |
| AsymmetricDualEncoder | (1,0,0) | ~0.47 | Shared encoder + separate projections |
| ResidualGaussianEncoder | (1,1,1)* | 0.4416 | Failed - identity trap |

*Pattern: (Encoder Sharing, Projection Sharing, Has Predictor) where 1=shared/yes, 0=separate/no

## 📖 Documentation

- **INDEX.md** - Command examples for each model
- **MIGRATION_GUIDE.md** - Finding relocated scripts
- **[model]/README.md** - Detailed docs for each model class

## 🔗 Related

- **Models**: `/home/ubuntu/ragcun/ragcun/`
- **Workflows**: `/home/ubuntu/ragcun/scripts/workflows/`
- **Results**: `/home/ubuntu/ragcun/results/beir_standard/`

---

**Start here**: Read [`INDEX.md`](INDEX.md) for copy-paste commands!
