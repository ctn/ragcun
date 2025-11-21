
## 512d vs 768d Comparison (2025-11-16)

**Config:** Both models trained identically with predictor loss:
- Dataset: msmarco_smoke (10k samples)
- Epochs: 1
- lambda_predictive: 1.0
- lambda_contrastive: 0.1
- lambda_isotropy: 1.0
- use_predictor: True

**NDCG@10 Results:**

| Dataset   | 512d      | 768d      | Baseline  | 512d vs Base | 768d vs Base |
|-----------|-----------|-----------|-----------|--------------|--------------|
| SciFact   | 0.6210    | 0.6382    | 0.6241    | -0.50%       | **+2.26%**   |
| NFCorpus  | 0.3242    | 0.3169    | 0.3016    | **+7.49%**   | +5.07%       |

**Conclusion:** 
- 768d wins on SciFact (+2.26% vs baseline, +1.72pts over 512d)
- 512d wins on NFCorpus (+7.49% vs baseline, +0.73pts over 768d)
- Overall: 768d preserves more information by avoiding dimension reduction
- Recommendation: Use 768d (native dimension) for best SciFact performance
