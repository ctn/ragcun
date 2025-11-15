#!/usr/bin/env python3
"""
Quick Diagnostic: Verify isotropy regularization in ~5 minutes

This script doesn't train - it just checks if your implementation is correct:
  1. Does lambda_isotropy affect the loss?
  2. Does it improve isotropy in a few gradient steps?
  3. Are the loss components computed correctly?

Run this first before any training!
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.models import GaussianEmbeddingModel
from ragcun.losses import InfoNCELoss, SIGRegLoss

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def measure_isotropy(embeddings):
    """Compute isotropy score from embeddings"""
    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) > 0:
        return eigenvalues.min() / eigenvalues.max()
    return 0.0

def main():
    print("🔬 Quick Diagnostic: Isotropy Regularization")
    print("Expected time: ~5 minutes")
    print("")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test 1: Model initialization
    print_section("Test 1: Model Initialization")
    
    try:
        model = GaussianEmbeddingModel(
            base_model='sentence-transformers/all-mpnet-base-v2',
            output_dim=512
        )
        model = model.to(device)
        model.train()
        print("✅ Model loads successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Model failed to load: {e}")
        return
    
    # Test 2: Loss functions
    print_section("Test 2: Loss Functions")
    
    try:
        infonce_loss = InfoNCELoss(temperature=0.05)
        sigreg_loss = SIGRegLoss()
        print("✅ Loss functions initialized")
    except Exception as e:
        print(f"❌ Loss functions failed: {e}")
        return
    
    # Test 3: Forward pass
    print_section("Test 3: Forward Pass")
    
    # Create dummy batch
    queries = [
        "What is machine learning?",
        "How does gradient descent work?",
        "What is a neural network?"
    ]
    positives = [
        "Machine learning is a subset of AI that learns from data.",
        "Gradient descent optimizes by following the negative gradient.",
        "Neural networks are composed of layers of interconnected nodes."
    ]
    
    try:
        # Encode queries
        q_inputs = model.tokenizer(queries, return_tensors='pt', 
                                   padding=True, truncation=True, max_length=512)
        q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
        q_out = model(**q_inputs)
        
        # Encode positives
        p_inputs = model.tokenizer(positives, return_tensors='pt',
                                   padding=True, truncation=True, max_length=512)
        p_inputs = {k: v.to(device) for k, v in p_inputs.items()}
        p_out = model(**p_inputs)
        
        print("✅ Forward pass works")
        print(f"   Query shape: {q_out['mean'].shape}")
        print(f"   Positive shape: {p_out['mean'].shape}")
        
        # Initial isotropy
        q_embeddings = q_out['mean'].detach().cpu().numpy()
        initial_isotropy = measure_isotropy(q_embeddings)
        print(f"   Initial isotropy: {initial_isotropy:.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Test 4: Loss computation
    print_section("Test 4: Loss Computation")
    
    try:
        # InfoNCE loss (supervised)
        loss_infonce = infonce_loss(q_out, p_out)
        print(f"✅ InfoNCE loss: {loss_infonce.item():.4f}")
        
        # SIGReg loss (isotropy regularization)
        loss_sigreg = sigreg_loss(q_out)
        print(f"✅ SIGReg loss: {loss_sigreg.item():.4f}")
        
        # Combined loss
        loss_combined = loss_infonce + 1.0 * loss_sigreg
        print(f"✅ Combined loss: {loss_combined.item():.4f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return
    
    # Test 5: Effect of lambda_isotropy
    print_section("Test 5: Lambda Isotropy Effect")
    
    print("Testing different lambda values...")
    lambda_values = [0.0, 0.5, 1.0, 2.0]
    losses = []
    
    for lam in lambda_values:
        loss = loss_infonce + lam * loss_sigreg
        losses.append(loss.item())
        print(f"   λ={lam:.1f}: loss={loss.item():.4f}")
    
    if losses[1] > losses[0] and losses[2] > losses[1]:
        print("✅ Lambda increases loss as expected")
    else:
        print("⚠️  Unexpected lambda behavior")
    
    # Test 6: Gradient updates improve isotropy
    print_section("Test 6: Gradient Updates on Isotropy")
    
    print("Running 10 gradient steps with isotropy regularization...")
    
    # Fresh model for fair test
    model_test = GaussianEmbeddingModel(
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=512
    )
    model_test = model_test.to(device)
    model_test.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model_test.parameters(), lr=1e-3)
    
    # Sample more text for better isotropy measurement
    texts = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks",
        "What is gradient descent?",
        "How do transformers work?",
        "What is attention mechanism?",
        "Explain backpropagation",
        "What is overfitting?",
        "How to prevent overfitting?",
        "What is regularization?"
    ]
    
    isotropy_scores = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Encode
        inputs = model_test.tokenizer(texts, return_tensors='pt',
                                     padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model_test(**inputs)
        
        # Only isotropy loss
        loss = sigreg_loss(outputs)
        loss.backward()
        optimizer.step()
        
        # Measure isotropy
        embeddings = outputs['mean'].detach().cpu().numpy()
        isotropy = measure_isotropy(embeddings)
        isotropy_scores.append(isotropy)
        
        if step % 3 == 0:
            print(f"   Step {step}: isotropy={isotropy:.4f}, loss={loss.item():.4f}")
    
    improvement = isotropy_scores[-1] - isotropy_scores[0]
    print(f"\nIsotropy change: {isotropy_scores[0]:.4f} → {isotropy_scores[-1]:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    
    if improvement > 0.01:
        print("✅ Isotropy IMPROVES with regularization!")
    elif improvement > 0:
        print("⚠️  Small improvement, might need more steps")
    else:
        print("❌ No improvement - check implementation")
    
    # Final verdict
    print_section("VERDICT")
    
    results = {
        'model_loads': True,
        'forward_pass_works': True,
        'loss_computed': True,
        'lambda_affects_loss': losses[-1] > losses[0],
        'isotropy_improves': improvement > 0.01,
        'isotropy_improvement': float(improvement),
        'initial_isotropy': float(isotropy_scores[0]),
        'final_isotropy': float(isotropy_scores[-1])
    }
    
    # Save results
    Path('results/diagnostic').mkdir(parents=True, exist_ok=True)
    with open('results/diagnostic/quick_check.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    all_pass = all([
        results['model_loads'],
        results['forward_pass_works'],
        results['loss_computed'],
        results['lambda_affects_loss'],
        results['isotropy_improves']
    ])
    
    if all_pass:
        print("✅ ALL CHECKS PASSED!")
        print("")
        print("Your implementation is correct:")
        print("  ✓ Model works")
        print("  ✓ Losses compute correctly")
        print("  ✓ Lambda affects training")
        print(f"  ✓ Isotropy improves ({improvement:+.4f})")
        print("")
        print("→ Ready to train!")
        print("")
        print("Next steps:")
        print("  1. Smoke test (2 hours):  ./scripts/train_smoke_test.sh")
        print("  2. Pilot run (1-2 days):  ./scripts/train_pilot.sh")
        print("  3. Full training (15d):   ./scripts/train_publication_recommended.sh")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("")
        print("Issues found:")
        if not results['lambda_affects_loss']:
            print("  ❌ Lambda doesn't affect loss")
        if not results['isotropy_improves']:
            print(f"  ❌ Isotropy doesn't improve (Δ={improvement:.4f})")
        print("")
        print("Debug before training!")
    
    print("")
    print(f"Results saved: results/diagnostic/quick_check.json")

if __name__ == '__main__':
    main()

