# Results Organization

**BEIR evaluation results reorganized by model class for easy discovery.**

## ✅ What Was Done

Results from `results/beir_standard/` have been reorganized into:
- `results/by_model/` - Organized by model class
- `results/experiments/` - Smoke tests, diagnostics, deprecated
- `results/analysis/` - Performance analysis documents

## 📊 New Structure

```
results/
│
├── by_model/                    ← MAIN RESULTS BY MODEL CLASS
│   ├── isotropic_gaussian/      🥇 15 result files (BEST: 0.4779)
│   │   ├── jepa_10k.json                  ⭐ WINNER
│   │   ├── jepa_10k_epoch2.json
│   │   ├── jepa_10k_epoch3.json
│   │   ├── 512dim_10k.json                0.4726 NDCG@10
│   │   ├── jepa_iso15_pred12.json         0.4610 NDCG@10
│   │   ├── iso15_epoch*.json (1-5)
│   │   ├── pure_isotropy_only.json        0.4562 NDCG@10
│   │   ├── pure_sigreg_margin01.json
│   │   └── *_frozen.json (baselines)
│   │
│   ├── asymmetric_dual/         🥉 4 result files (~0.47)
│   │   ├── asymmetric_epoch3_quick.json
│   │   └── asymmetric_smoke_*.json
│   │
│   ├── asymmetric_predictor/    🥈 0 files yet (new model)
│   │   └── (empty - model just created)
│   │
│   ├── residual_gaussian/       ❌ 2 result files (failed)
│   │   ├── respred_epoch1_quick.json
│   │   └── respred_epoch3_quick.json
│   │
│   └── mpnet_lejepa/            ❓ 0 files (untested)
│       └── (empty - not tested yet)
│
├── experiments/                 ← EXPERIMENTAL RESULTS
│   ├── smoke_tests/
│   │   ├── smoke_test/
│   │   ├── smoke_frozen/
│   │   ├── smoke_frozen_fixed/
│   │   └── smoke_multi/
│   ├── diagnostic/
│   ├── deprecated/
│   │   └── deprecated_fullft/
│   ├── frozen_msmarco_full_eval/
│   └── msmarco_full_eval/
│
└── analysis/                    ← ANALYSIS DOCUMENTS
    ├── ARCHITECTURE_SUMMARY.md
    ├── asymmetric_failure_analysis.md
    └── respred_failure_analysis.md
```

## 📈 Performance by Model Class

| Model Class | Best NDCG@10 | Files | Status |
|------------|--------------|-------|--------|
| IsotropicGaussianEncoder | **0.4779** | 15 | 🥇 BEST |
| AsymmetricWithPredictor | ~0.48 | 0 | 🥈 New |
| AsymmetricDualEncoder | ~0.47 | 4 | 🥉 Good |
| ResidualGaussianEncoder | 0.4416 | 2 | ❌ Failed |
| MPNetLeJEPA | ? | 0 | ❓ Untested |

## 🎯 Quick Access

### View best results:
```bash
cat results/by_model/isotropic_gaussian/jepa_10k.json
```

### Compare all IsotropicGaussian models:
```bash
cd results/by_model/isotropic_gaussian
ls -lh *.json
```

### Check smoke test history:
```bash
ls -lt results/experiments/smoke_tests/*/
```

## 📖 Documentation

- **results/README.md** - Main results overview
- **results/by_model/README.md** - Detailed model results guide
- **results/analysis/** - Performance analysis documents

## 🔗 Alignment with Scripts

Results organization now matches scripts organization:

| Location | Scripts | Results |
|----------|---------|---------|
| IsotropicGaussianEncoder | `scripts/by_model/isotropic_gaussian/` | `results/by_model/isotropic_gaussian/` |
| AsymmetricDualEncoder | `scripts/by_model/asymmetric_dual/` | `results/by_model/asymmetric_dual/` |
| AsymmetricWithPredictor | `scripts/by_model/asymmetric_predictor/` | `results/by_model/asymmetric_predictor/` |
| ResidualGaussianEncoder | `scripts/by_model/residual_gaussian/` | `results/by_model/residual_gaussian/` |
| MPNetLeJEPA | `scripts/by_model/mpnet_lejepa/` | `results/by_model/mpnet_lejepa/` |

## 🔄 What Changed

**Moved:**
- `beir_standard/jepa_*.json` → `by_model/isotropic_gaussian/`
- `beir_standard/asymmetric_*.json` → `by_model/asymmetric_dual/`
- `beir_standard/respred_*.json` → `by_model/residual_gaussian/`
- `smoke_*/` → `experiments/smoke_tests/`
- Old baselines → `experiments/deprecated/`

**Removed:**
- `beir_standard/` (now empty, removed)

**Created:**
- `by_model/` structure matching `scripts/by_model/`
- Documentation files (README.md in key locations)

## ✨ Benefits

1. **Parallel Structure**: Results match scripts organization
2. **Easy Discovery**: Find results by model class, not random names
3. **Clear Performance**: See best performers at a glance
4. **Clean Separation**: Main results vs experiments vs analysis

---

**🎉 Results are now organized by model class!**
