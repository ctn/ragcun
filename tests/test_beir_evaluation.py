"""
Unit tests for BEIR evaluation script.

Tests metric computation logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from evaluate_beir import compute_metrics


class TestMetricComputation:
    """Test retrieval metric computation."""
    
    def test_perfect_ranking(self):
        """Test metrics with perfect ranking."""
        # Query with 2 relevant docs ranked at top
        results = {
            'q1': {
                'doc1': 1.0,  # Relevant
                'doc2': 0.9,  # Relevant
                'doc3': 0.5,  # Not relevant
                'doc4': 0.3   # Not relevant
            }
        }
        
        qrels = {
            'q1': {
                'doc1': 1,
                'doc2': 1
            }
        }
        
        metrics = compute_metrics(results, qrels, k_values=[2, 5, 10])
        
        # Perfect ranking should have:
        assert metrics['MRR'] == 1.0  # First doc is relevant
        assert metrics['Recall@2'] == 1.0  # All relevant docs in top 2
        assert metrics['Recall@5'] == 1.0  # All relevant docs in top 5
        assert metrics['P@2'] == 1.0  # Both top 2 are relevant
    
    def test_partial_ranking(self):
        """Test metrics with partial ranking."""
        results = {
            'q1': {
                'doc1': 1.0,  # Not relevant
                'doc2': 0.9,  # Relevant
                'doc3': 0.8,  # Relevant
                'doc4': 0.7   # Not relevant
            }
        }
        
        qrels = {
            'q1': {
                'doc2': 1,
                'doc3': 1
            }
        }
        
        metrics = compute_metrics(results, qrels, k_values=[2, 4])
        
        # Second doc is first relevant
        assert metrics['MRR'] == 0.5  # 1/2
        
        # Only 1 out of 2 relevant in top 2
        assert metrics['Recall@2'] == 0.5
        
        # Both relevant docs in top 4
        assert metrics['Recall@4'] == 1.0
        
        # 1 out of 2 in top 2 is relevant
        assert metrics['P@2'] == 0.5
    
    def test_no_relevant_found(self):
        """Test metrics when no relevant docs are found."""
        results = {
            'q1': {
                'doc1': 1.0,
                'doc2': 0.9,
                'doc3': 0.8
            }
        }
        
        qrels = {
            'q1': {
                'doc4': 1,  # Relevant doc not in results
                'doc5': 1
            }
        }
        
        metrics = compute_metrics(results, qrels, k_values=[5, 10])
        
        assert metrics['MRR'] == 0.0
        assert metrics['Recall@5'] == 0.0
        assert metrics['Recall@10'] == 0.0
        assert metrics['NDCG@10'] == 0.0
    
    def test_multiple_queries(self):
        """Test averaging across multiple queries."""
        results = {
            'q1': {
                'doc1': 1.0,  # Relevant
                'doc2': 0.5   # Not relevant
            },
            'q2': {
                'doc3': 1.0,  # Not relevant
                'doc4': 0.9   # Relevant
            }
        }
        
        qrels = {
            'q1': {'doc1': 1},
            'q2': {'doc4': 1}
        }
        
        metrics = compute_metrics(results, qrels, k_values=[1, 5])
        
        # MRR: q1=1.0 (rank 1), q2=0.5 (rank 2), avg=0.75
        assert metrics['MRR'] == pytest.approx(0.75, abs=1e-6)
        
        # Recall@5: both queries find their relevant doc
        assert metrics['Recall@5'] == 1.0
    
    def test_recall_computation(self):
        """Test recall computation specifically."""
        results = {
            'q1': {
                'doc1': 1.0,  # Relevant
                'doc2': 0.9,  # Relevant
                'doc3': 0.8,  # Not relevant
                'doc4': 0.7,  # Relevant
                'doc5': 0.6   # Not relevant
            }
        }
        
        qrels = {
            'q1': {
                'doc1': 1,
                'doc2': 1,
                'doc4': 1,
                'doc6': 1  # Not in results
            }
        }
        
        metrics = compute_metrics(results, qrels, k_values=[1, 2, 3, 5])
        
        # 4 total relevant docs
        # Recall@1: 1/4 = 0.25
        assert metrics['Recall@1'] == 0.25
        
        # Recall@2: 2/4 = 0.5
        assert metrics['Recall@2'] == 0.5
        
        # Recall@3: 2/4 = 0.5 (doc3 not relevant)
        assert metrics['Recall@3'] == 0.5
        
        # Recall@5: 3/4 = 0.75 (doc6 not found)
        assert metrics['Recall@5'] == 0.75
    
    def test_ndcg_computation(self):
        """Test NDCG computation."""
        results = {
            'q1': {
                'doc1': 1.0,  # Relevance = 2
                'doc2': 0.9,  # Relevance = 1
                'doc3': 0.8   # Relevance = 0
            }
        }
        
        qrels = {
            'q1': {
                'doc1': 2,  # Graded relevance
                'doc2': 1
            }
        }
        
        metrics = compute_metrics(results, qrels, k_values=[3])
        
        # NDCG should be 1.0 (perfect ranking)
        assert metrics['NDCG@3'] == pytest.approx(1.0, abs=1e-6)
    
    def test_empty_results(self):
        """Test handling of empty results."""
        results = {}
        qrels = {'q1': {'doc1': 1}}
        
        metrics = compute_metrics(results, qrels, k_values=[10])
        
        # Should return 0 for all metrics
        assert metrics['MRR'] == 0.0
        assert metrics['Recall@10'] == 0.0
    
    def test_query_without_qrels(self):
        """Test query that has results but no qrels."""
        results = {
            'q1': {'doc1': 1.0, 'doc2': 0.9}
        }
        
        qrels = {}
        
        metrics = compute_metrics(results, qrels, k_values=[10])
        
        # Should handle gracefully
        assert metrics['MRR'] == 0.0
    
    def test_different_k_values(self):
        """Test with various k values."""
        results = {
            'q1': {
                f'doc{i}': 1.0 - i*0.1 for i in range(100)
            }
        }
        
        qrels = {
            'q1': {'doc5': 1, 'doc10': 1, 'doc49': 1}  # Changed doc50 to doc49 so all within top 50
        }
        
        k_values = [1, 5, 10, 20, 50, 100]
        metrics = compute_metrics(results, qrels, k_values=k_values)
        
        # Recall should increase with k
        recalls = [metrics[f'Recall@{k}'] for k in k_values]
        
        # Check monotonicity
        for i in range(len(recalls) - 1):
            assert recalls[i] <= recalls[i+1], "Recall should be monotonic in k"
        
        # Recall@50 should find all 3 docs (doc5, doc10, doc49)
        assert metrics['Recall@50'] == 1.0


class TestEdgeCases:
    """Test edge cases in evaluation."""
    
    def test_single_relevant_doc(self):
        """Test with single relevant document."""
        results = {
            'q1': {'doc1': 1.0, 'doc2': 0.9}
        }
        
        qrels = {
            'q1': {'doc2': 1}
        }
        
        metrics = compute_metrics(results, qrels, k_values=[1, 2])
        
        assert metrics['MRR'] == 0.5  # Second position
        assert metrics['Recall@1'] == 0.0
        assert metrics['Recall@2'] == 1.0
    
    def test_many_relevant_docs(self):
        """Test with many relevant documents."""
        results = {
            'q1': {f'doc{i}': 1.0 - i*0.01 for i in range(20)}
        }
        
        # Half are relevant
        qrels = {
            'q1': {f'doc{i}': 1 for i in range(10)}
        }
        
        metrics = compute_metrics(results, qrels, k_values=[10, 20])
        
        # All relevant in top 10
        assert metrics['Recall@10'] == 1.0
        
        # Precision@10 = 10/10 = 1.0
        assert metrics['P@10'] == 1.0
    
    def test_tied_scores(self):
        """Test handling of tied scores."""
        results = {
            'q1': {
                'doc1': 1.0,
                'doc2': 1.0,  # Tied
                'doc3': 0.5
            }
        }
        
        qrels = {
            'q1': {'doc2': 1}
        }
        
        # Should handle gracefully (order depends on dict ordering)
        metrics = compute_metrics(results, qrels, k_values=[2])
        
        # Relevant doc is in top 2
        assert metrics['Recall@2'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

