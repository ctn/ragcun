# Asymmetric Projections Training - Status

**Started:** 2025-11-18 11:19 UTC  
**Expected completion:** ~11:35 UTC (~15 minutes)  
**Status:** ✅ Running in background

**Evaluation scheduled:** 11:52 UTC (30 min from start)  
**Evaluation completion:** ~12:00 UTC  
**Status:** ✅ Automated - will run automatically

---

## 🎯 What's Running

**Model:** Asymmetric Projection Model
- Different projection heads for queries vs documents
- No predictor, no residual connections
- Clean contrastive + isotropy learning

**Architecture:**
```
Frozen MPNet Encoder (shared, 110M params)
         ↓
    ┌────┴────┐
    ↓         ↓
Query Proj  Doc Proj  (2.4M params each, trainable)
  768→1536   768→1536
    →768      →768
    ↓         ↓
   z_q       z_d
```

**Total trainable params:** 4.7M (4.1% of total)

---

## 📊 Initial Training Metrics (First Batches)

| Batch | Loss | Contrastive | Isotropy | Accuracy |
|-------|------|-------------|----------|----------|
| 1 | 4.2020 | 4.2019 | 0.0000 | 1.6% |
| 2 | 3.4631 | 3.4630 | 0.0001 | 29.7% |
| 3 | 2.8512 | 2.8511 | 0.0001 | 59.4% |
| 4 | 2.2231 | 2.2230 | 0.0001 | 71.9% |

**Good signs:**
- ✅ Loss decreasing rapidly (4.2 → 2.2)
- ✅ Accuracy improving (1.6% → 71.9%)
- ✅ Isotropy constraint satisfied (near 0)
- ✅ Model is learning!

---

## 🎯 Expected Results

**Targets:**
- **Baseline (frozen MPNet):** 0.4628
- **ISO15_PRED12:** 0.4759 (+2.83%)
- **ResPred (failed):** 0.4416 (-4.59%)

**Goal for Asymmetric:**
- Conservative: > 0.48 (+3.5%)
- Optimistic: > 0.49 (+6%)

**Why it should work:**
- Acknowledges query/doc asymmetry
- Avoids identity mapping trap
- Simpler architecture (no predictor complexity)
- Clean learning objective

---

## 📂 Files and Logs

**Model checkpoint:** `checkpoints/asymmetric_smoke_20251118_111938/`

**Training log:** `logs/asymmetric_dual/asymmetric_smoke_20251118_111938.log`

**Monitor commands:**
```bash
# Watch live progress
tail -f logs/asymmetric_dual/asymmetric_smoke_20251118_111938.log

# Check process status
ps aux | grep "[t]rain_asymmetric"

# Quick status
tail -30 logs/asymmetric_dual/asymmetric_smoke_20251118_111938.log | grep -E "Epoch|Loss|Accuracy"
```

---

## 🔍 What to Check When Complete

### 1. Training Summary
```bash
grep -E "Epoch [0-9]/|Training Loss|Validation Loss|Accuracy" \
  logs/asymmetric_dual/asymmetric_smoke_20251118_111938.log
```

### 2. Final Metrics
- Training accuracy (should be > 90%)
- Validation accuracy (should be > 85%)
- Isotropy loss (should stay near 0)
- Query/Doc std (should both be ~1.0)

### 3. Run Evaluation
```bash
# Quick eval on 2 datasets (~5 min)
python scripts/eval/residual_gaussian_quick.py \
  --checkpoint checkpoints/asymmetric_smoke_20251118_111938/best_model.pt \
  --datasets scifact nfcorpus \
  --output_file results/beir_standard/asymmetric_epoch3_quick.json \
  --base_model sentence-transformers/all-mpnet-base-v2 \
  --output_dim 768
```

**Note:** Will need to adapt evaluation script to handle asymmetric model (query vs doc encoding)

---

## 🚀 Next Steps Based on Results

### If it WORKS (> 0.48):
1. ✅ Train on full dataset (88K examples, ~2 hours)
2. ✅ Evaluate on all 5 BEIR datasets
3. ✅ Write up findings
4. ✅ Consider: add back weak contrastive negatives

### If it's MARGINAL (0.465-0.48):
1. 🔧 Tune hyperparameters (temperature, isotropy weight)
2. 🔧 Try different projection architecture (3-layer, wider)
3. 🔧 Combine with partial encoder unfreezing

### If it FAILS (< 0.465):
1. ❌ Investigate: Are projections learning meaningful transformations?
2. ❌ Check: Query vs doc embedding statistics
3. ❌ Consider: Unfreeze encoder layers (Plan B)
4. ❌ Or: Accept that frozen encoder limits gains

---

## 📝 Key Differences from ResPred

| Feature | ResPred | Asymmetric |
|---------|---------|------------|
| **Projections** | 1 (shared) | 2 (separate) |
| **Predictor** | Yes (768→1536→768) | No |
| **Residual** | Yes (with regularization) | No |
| **Alpha** | Learnable (collapsed to 0.077) | N/A |
| **Query/Doc** | Symmetric | Asymmetric ✅ |
| **Trainable params** | 2.4M | 4.7M |
| **Loss components** | 4 (contr, iso, pred, res) | 2 (contr, iso) |
| **Identity trap** | Yes ❌ | No ✅ |

---

## ⏰ Timeline

- **11:19 UTC:** Training started
- **11:23 UTC:** Epoch 1 in progress (~3% complete)
- **11:25 UTC:** (estimated) Epoch 1 complete
- **11:29 UTC:** (estimated) Epoch 2 complete
- **11:33 UTC:** (estimated) Epoch 3 complete
- **11:35 UTC:** (estimated) Training finished

**Total time:** ~15 minutes

---

## 🎓 What We're Testing

**Hypothesis:** Query and document representations should be learned differently because they serve different roles in retrieval.

**Evidence for:**
- Queries are short, question-like, seeking information
- Documents are long, declarative, providing information
- Asymmetry is natural in IR

**Evidence against:**
- Base model already understands this
- Sentence-transformers work well with symmetry
- Production systems often use same encoder

**This experiment will show:** Whether explicit architectural asymmetry helps when encoder is frozen.

---

**Check this file when you return for quick status update!**

**Full analysis (if training completes):** Will be in training log and checkpoint directory.

---

## ⏰ SCHEDULED EVALUATION (AUTOMATED)

**Process:** Sleeping for 30 minutes, then will auto-evaluate

**Timeline:**
- 11:22 UTC: Scheduled evaluation process started
- 11:22-11:52 UTC: Waiting (sleep 1800 seconds)
- 11:52 UTC: Evaluation begins automatically
- ~12:00 UTC: Results ready!

**What it will do:**
1. Check if training completed (use best_model.pt)
2. If training incomplete, use latest checkpoint
3. Evaluate on scifact + nfcorpus (2 datasets)
4. Save results to: `results/beir_standard/asymmetric_epoch3_quick.json`
5. Generate comparison table automatically

**Output:**
- Results: `results/beir_standard/asymmetric_epoch3_quick.json`
- Log: `logs/asymmetric_dual/eval_asymmetric_dual_scheduled.log`

**Monitor:**
```bash
# Check if evaluation started (after 11:52)
tail -f logs/asymmetric_dual/eval_asymmetric_dual_scheduled.log

# Check when done (after 12:00)
cat results/beir_standard/asymmetric_epoch3_quick.json
```

**Expected comparison:**
```
Model              | Avg NDCG@10 | vs Baseline
-------------------+-------------+------------
Baseline (MPNet)   | 0.4628      | -
ISO15_PRED12       | 0.4759      | +2.83%
Asymmetric (NEW!)  | ???         | ???
```

---

## 🎉 COME BACK IN ~40 MINUTES

**At 12:00 UTC (or later), you'll have:**
- ✅ Fully trained model (3 epochs)
- ✅ Evaluation results on 2 BEIR datasets
- ✅ Automatic comparison with baseline & ISO15
- ✅ Ready for analysis and next steps

**Everything is automated - just come back and check the results!** 🚀

