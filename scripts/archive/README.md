# Archive - Historical Experiment Scripts

This directory contains **38 historical experiment scripts** from previous training and evaluation runs. These are preserved for reference but are not part of the active codebase.

## What's in Archive

### Historical BEIR Evaluation Scripts (20 files)
Scripts for evaluating specific experiment checkpoints:

**Frozen base experiments:**
- `eval_beir_frozen_48k.sh`, `eval_beir_frozen_48k_epoch2.sh`
- `eval_beir_768dim.sh`

**JEPA experiments:**
- `eval_beir_jepa_10k.sh` (+ epoch2, epoch3)
- `eval_beir_jepa_pure.sh` (+ epoch1, epoch2, epoch3)
- `eval_beir_jepa_contrastive_05.sh` (+ epoch1, epoch2)
- `eval_beir_jepa_iso15_pred12.sh`

**Baseline experiments:**
- `eval_beir_pure_isotropy.sh`
- `eval_beir_standard.sh`
- `eval_vanilla_baseline.py`
- `eval_frozen_msmarco_full.py`, `eval_frozen_simple.py`
- `eval_msmarco_full.py`
- `evaluate.py`, `evaluate_all_beir.sh`

### Historical Training Scripts (17 files)
Scripts for specific hyperparameter experiments:

**JEPA training:**
- `train_jepa_10k.sh`, `train_jepa_10k_continue.sh`
- `train_jepa_pure.sh`
- `train_jepa_contrastive_05.sh`
- `train_jepa_iso15_pred12.sh`

**Frozen/hybrid training:**
- `train_frozen_48k.sh`
- `train_smart_hybrid.sh`

**Isotropy experiments:**
- `train_pure_isotropy.sh`
- `train_isotropy_contrastive_01.sh`

**Other:**
- `train_quick_768dim.sh`
- `train_pilot.sh`
- `train_parallel_p4d.sh`
- `train_publication_recommended.sh`
- `train.py`, `train_updates.py` (old versions)

### Other (1 file)
- `setup_p4d_instance.sh` - AWS p4d instance setup (AWS-specific)

---

## Active Scripts

Active scripts have been **reorganized by function** into:

```
scripts/
├── train/         - Training scripts by model type
├── eval/          - Evaluation scripts
├── data/          - Data preparation
├── workflows/     - End-to-end pipelines
├── monitoring/    - Status checks and scheduled jobs
├── analysis/      - Model comparison and diagnostics
└── smoke_tests/   - Quick validation tests
```

See `../` (parent directory) for current scripts.

---

## Why Keep Archive?

These scripts document:
- Hyperparameter search history
- Experiment configurations that were tried
- Evolution of training approaches
- Checkpoints from specific runs

They're kept for:
- Historical reference
- Reproducing past experiments
- Understanding what didn't work

---

## If You Need to Use an Archived Script

1. **Check if there's a newer version** in the organized directories first
2. **Update paths** - archived scripts may reference old file locations
3. **Check dependencies** - some may use deprecated APIs
4. **Consider updating** - the active scripts have better practices

---

For current workflows, see: `../README.md` (coming soon)
