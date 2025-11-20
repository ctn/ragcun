#!/usr/bin/env python3
"""
Evaluate frozen base models using the SAME protocol as the multi-model smoke test:
- 1 positive + 9 random negatives per query
- 100 queries from MS MARCO dev set
"""

import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from ragcun.model import IsotropicGaussianEncoder

def test_retrieval(model, queries, passages):
    """Same evaluation as multi-model smoke test"""
    correct_at_1 = 0
    correct_at_5 = 0
    
    with torch.no_grad():
        for i, (query, pos_passage) in enumerate(zip(queries[:100], passages[:100])):
            # 1 positive + 9 random negatives
            negatives = [passages[j] for j in range(len(passages)) if j != i][:9]
            all_passages = [pos_passage] + negatives
            
            q_emb = model.encode(query, convert_to_numpy=False)
            p_embs = model.encode(all_passages, convert_to_numpy=False)
            
            q_emb = q_emb.unsqueeze(0) if q_emb.dim() == 1 else q_emb
            distances = torch.cdist(q_emb, p_embs).squeeze(0)
            ranks = distances.argsort()
            
            pos_rank = (ranks == 0).nonzero(as_tuple=True)[0].item()
            if pos_rank == 0:
                correct_at_1 += 1
                correct_at_5 += 1
            elif pos_rank < 5:
                correct_at_5 += 1
    
    return correct_at_1 / 100, correct_at_5 / 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
with open('data/processed/msmarco_smoke/dev.json') as f:
    data = json.load(f)[:500]

queries = [d['query'] for d in data]
passages = [d['positive'] for d in data]

# Models to evaluate
models_config = [
    ('mpnet', 'sentence-transformers/all-mpnet-base-v2'),
    ('minilm-l6', 'sentence-transformers/all-MiniLM-L6-v2'),
    ('minilm-l12', 'sentence-transformers/all-MiniLM-L12-v2'),
    ('distilroberta', 'sentence-transformers/all-distilroberta-v1'),
    ('paraphrase-minilm', 'sentence-transformers/paraphrase-MiniLM-L6-v2'),
]

print("Evaluating frozen base models (100 queries, 1+9 protocol)")
print("="*80)
print()

results = []
for model_key, base_model in models_config:
    print(f"{model_key}:")
    
    # Load baseline (from multi-model smoke test)
    try:
        baseline = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_multi/{model_key}_baseline/best_model.pt',
            base_model=base_model,
            output_dim=512
        ).to(device).eval()
        
        base_acc1, base_acc5 = test_retrieval(baseline, queries, passages)
        print(f"  Baseline:  Acc@1={base_acc1:.2%}, Acc@5={base_acc5:.2%}")
        
        del baseline
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Baseline: ERROR - {e}")
        base_acc1, base_acc5 = None, None
    
    # Load full FT (from multi-model smoke test)
    try:
        full_ft = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_multi/{model_key}_isotropy/best_model.pt',
            base_model=base_model,
            output_dim=512
        ).to(device).eval()
        
        full_acc1, full_acc5 = test_retrieval(full_ft, queries, passages)
        print(f"  Full FT:   Acc@1={full_acc1:.2%}, Acc@5={full_acc5:.2%}")
        
        del full_ft
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Full FT: ERROR - {e}")
        full_acc1, full_acc5 = None, None
    
    # Load frozen base
    try:
        frozen = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_frozen/{model_key}_frozen_isotropy/best_model.pt',
            base_model=base_model,
            output_dim=512,
            freeze_base=True
        ).to(device).eval()
        
        frozen_acc1, frozen_acc5 = test_retrieval(frozen, queries, passages)
        print(f"  Frozen:    Acc@1={frozen_acc1:.2%}, Acc@5={frozen_acc5:.2%}")
        
        del frozen
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Frozen: ERROR - {e}")
        frozen_acc1, frozen_acc5 = None, None
    
    # Compute deltas
    if base_acc5 and full_acc5 and frozen_acc5:
        full_delta = (full_acc5 - base_acc5) * 100
        frozen_delta = (frozen_acc5 - base_acc5) * 100
        print(f"  Δ vs Base: Full FT={full_delta:+.1f}%, Frozen={frozen_delta:+.1f}%")
        
        if frozen_acc5 > full_acc5:
            winner = "Frozen"
        elif full_acc5 > frozen_acc5:
            winner = "Full FT"
        else:
            winner = "Tie"
        print(f"  Winner: {winner}")
        
        results.append({
            'model': model_key,
            'baseline_acc5': base_acc5,
            'full_ft_acc5': full_acc5,
            'frozen_acc5': frozen_acc5,
            'full_ft_delta': full_delta,
            'frozen_delta': frozen_delta,
            'winner': winner
        })
    
    print()

# Summary
if results:
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_full_delta = sum(r['full_ft_delta'] for r in results) / len(results)
    avg_frozen_delta = sum(r['frozen_delta'] for r in results) / len(results)
    
    print(f"Average Acc@5 improvement:")
    print(f"  Full fine-tune: {avg_full_delta:+.1f}%")
    print(f"  Frozen base:    {avg_frozen_delta:+.1f}%")
    print()
    
    if avg_frozen_delta > avg_full_delta + 1.0:
        print("✅ FROZEN BASE WINS! Better performance with 3x faster training")
    elif avg_full_delta > avg_frozen_delta + 1.0:
        print("✅ FULL FINE-TUNE WINS! Worth the extra training time")
    else:
        print("✅ TIE! Both approaches perform similarly")

