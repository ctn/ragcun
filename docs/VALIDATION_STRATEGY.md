# Validation Strategy: 3-Tier Approach

Before committing to expensive full training, validate your approach in stages.

## 🎯 Quick Overview

| Stage | Time | Cost | What You Learn |
|-------|------|------|----------------|
| **0. Diagnostic** | 5 min | Free | Implementation works correctly |
| **1. Smoke Test** | 1-2 hours | Free | Isotropy helps retrieval |
| **2. Pilot Run** | 1-2 days | Free | Ready for publication |
| **3. Full Training** | 15 days or $220 | Variable | Publication-quality results |

---

## Stage 0: Quick Diagnostic (5 minutes) ⚡

**Goal:** Verify implementation is correct

```bash
python scripts/diagnostic_quick.py
```

**What it checks:**
- ✅ Model loads and runs
- ✅ Loss functions compute correctly
- ✅ Lambda affects training
- ✅ Isotropy improves with regularization (10 gradient steps)

**Output:**
```
✅ ALL CHECKS PASSED!

Your implementation is correct:
  ✓ Model works
  ✓ Losses compute correctly  
  ✓ Lambda affects training
  ✓ Isotropy improves (+0.0342)

→ Ready to train!
```

**Decision:**
- ✅ All pass → Proceed to Stage 1
- ❌ Any fail → Debug before training

---

## Stage 1: Smoke Test (1-2 hours) 🔬

**Goal:** Prove isotropy helps retrieval with meaningful data

```bash
./scripts/train_smoke_test.sh
```

**What it does:**
- Trains on 10K MS MARCO examples (1 epoch)
- Compares baseline vs. isotropy
- Measures isotropy scores directly
- Tests retrieval accuracy

**Output:**
```
ISOTROPY SCORES (higher = better)
  Baseline:   0.7234
  Isotropy:   0.8891
  Improvement: +0.1657 ✅

RETRIEVAL ACCURACY (500 queries)
  Baseline Acc@1: 32.1%
  Isotropy Acc@1: 35.8%
  Δ: +3.7% ✅

✅ SUCCESS: Isotropy helps!
```

**Decision:**
- ✅ Isotropy & retrieval improve → Proceed to Stage 2
- ⚠️  Mixed results → Tune hyperparameters, re-run
- ❌ No improvement → Debug before Stage 2

---

## Stage 2: Pilot Run (1-2 days) 🧪

**Goal:** Validate approach with enough data to predict full results

```bash
./scripts/train_pilot.sh
```

**What it does:**
- Trains on 50K MS MARCO examples (2 epochs)
- Evaluates on 3 representative BEIR datasets
- Predicts full training results

**Output:**
```
Quick Results (on 3 BEIR datasets):
  Baseline: 46.2% NDCG@10
  Isotropy: 47.9% NDCG@10
  Improvement: +1.7% ✅

✅ Isotropy is helping! Ready for full training.
```

**Decision:**
- ✅ >0.5% improvement → Proceed to full training
- ⚠️  <0.5% improvement → Check logs, consider tuning
- ❌ No improvement → Review approach

---

## Stage 3: Full Training 🚀

**Goal:** Publication-quality results

### Option A: Local (Free, Slow)
```bash
./scripts/train_publication_recommended.sh
./scripts/evaluate_all_beir.sh
```
- **Time:** 15 days
- **Cost:** Free
- **Result:** All 3 models + full BEIR evaluation

### Option B: AWS p4d (Fast, $$)
```bash
# On p4d.24xlarge
./scripts/train_parallel_p4d.sh
```
- **Time:** ~21 hours
- **Cost:** ~$220
- **Result:** All 3 models + full BEIR evaluation

---

## 📊 What Each Stage Proves

| Stage | Proves | Key Metrics |
|-------|--------|-------------|
| 0. Diagnostic | Implementation correct | Isotropy Δ > 0.01 |
| 1. Smoke Test | Isotropy helps | Retrieval Δ > 2% |
| 2. Pilot Run | Predicts full results | BEIR Δ > 0.5% |
| 3. Full Training | Publication quality | BEIR > 49%, 15 datasets |

---

## 🎓 Recommended Path

### For Research/Publication:
1. **Diagnostic** (5 min) - Verify implementation
2. **Smoke Test** (2 hours) - Quick proof of concept
3. **Pilot Run** (2 days) - Validate on meaningful data
4. **AWS p4d** (1 day) - Fast publication results

**Total time:** ~3 days from start to publication-ready results

### For Experimentation:
1. **Diagnostic** (5 min) - Verify implementation
2. **Smoke Test** (2 hours) - Test ideas quickly
3. Iterate on hyperparameters
4. **Pilot Run** when confident

### For Patience/Free Compute:
1. **Diagnostic** (5 min)
2. **Local Full Training** (15 days) - Set and forget

---

## 💡 Pro Tips

**Save time:**
- Always run Diagnostic first (catches bugs early)
- Smoke Test is fast enough to iterate on hyperparameters
- Pilot Run predicts full results well

**Save money:**
- Validate locally before AWS
- AWS spot instances can interrupt - use snapshots!
- Smoke Test + Pilot = high confidence for AWS

**For publication:**
- All 3 stages provide complementary evidence:
  - Diagnostic: Implementation is sound
  - Smoke: Core mechanism works
  - Pilot: Scales to real data
  - Full: Publication-quality evaluation

---

## 🚦 Current Status

After pre-flight tests passed, you're at:
```
[ ] Stage 0: Diagnostic
[ ] Stage 1: Smoke Test  
[ ] Stage 2: Pilot Run
[ ] Stage 3: Full Training
```

**Next command:**
```bash
# Start with diagnostic (5 min)
python scripts/diagnostic_quick.py
```

If it passes, you'll know your implementation is correct and can confidently proceed to training stages!

