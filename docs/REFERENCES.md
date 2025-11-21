# References

## LeJEPA (Joint-Embedding Predictive Architecture)

**Paper:** Lean Joint-Embedding Predictive Architecture (LeJEPA): Provable and Scalable Self-Supervised Learning Without the Heuristics

**Links:**
- GitHub: https://github.com/rbalestr-lab/lejepa
- arXiv: https://arxiv.org/abs/2511.08544

**Key Contributions:**
- SIGReg loss: Isotropy regularization via Epps-Pulley divergence
- Sliced Wasserstein distance for efficient isotropy enforcement
- Provable convergence properties for self-supervised learning

**Our Implementation:**
We adapted LeJEPA's SIGReg isotropy loss for text retrieval:
- Applied to unnormalized Gaussian embeddings
- Combined with contrastive learning for supervised retrieval
- Extended to various architectures (see `ragcun/` models)

**Note:** The original LeJEPA implementation (computer vision focus) was previously
included as a git submodule at `external/lejepa/` but has been removed as it's not
actively used in our codebase. See the GitHub repo above for the reference implementation.

---

## Other References

### Sentence Transformers
- Reimers & Gurevych (2019): Sentence-BERT
- Used as base encoder in all our models
- https://www.sbert.net/

### MS MARCO
- Nguyen et al. (2016): MS MARCO dataset
- Primary training data for passage retrieval
- https://microsoft.github.io/msmarco/

### BEIR Benchmark
- Thakur et al. (2021): BEIR benchmark
- Zero-shot evaluation across 18 retrieval tasks
- https://github.com/beir-cellar/beir
