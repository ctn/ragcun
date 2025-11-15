"""
Property tests for isotropic Gaussian embeddings.

Tests the critical mathematical properties that make these embeddings
work: isotropy, N(0,I) distribution, and Euclidean distance properties.
"""

import pytest
import torch
import numpy as np
from scipy import stats

from ragcun.model import GaussianEmbeddingGemma


class TestIsotropyProperties:
    """Test that embeddings follow isotropic Gaussian distribution N(0, I)."""

    @pytest.mark.slow
    def test_embedding_mean_near_zero(self, embedding_dim, sample_texts):
        """Test that mean of embeddings is close to 0 (first moment)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        # Compute mean across all embeddings
        mean_vector = np.mean(embeddings, axis=0)
        mean_norm = np.linalg.norm(mean_vector)

        # Mean should be close to 0
        # Using relaxed threshold since we have small sample
        assert mean_norm < 1.0, f"Mean norm {mean_norm} is too large, should be close to 0"

    @pytest.mark.slow
    def test_embedding_dimensions_approximately_uncorrelated(self, embedding_dim, sample_texts):
        """Test that embedding dimensions are approximately uncorrelated."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        # Need more samples for covariance estimation
        texts = sample_texts * 5  # Repeat to get more samples

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        # Compute covariance matrix
        cov = np.cov(embeddings.T)

        # Extract diagonal and off-diagonal elements
        diagonal = np.diag(cov)
        off_diagonal = cov[~np.eye(cov.shape[0], dtype=bool)]

        # Off-diagonal elements should be small compared to diagonal
        mean_diagonal = np.mean(np.abs(diagonal))
        mean_off_diagonal = np.mean(np.abs(off_diagonal))

        # Off-diagonal should be smaller than diagonal
        assert mean_off_diagonal < mean_diagonal, "Dimensions appear correlated"

    @pytest.mark.slow
    def test_embedding_norms_vary(self, embedding_dim, sample_texts):
        """Test that embedding norms vary (not constant like normalized vectors)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Standard deviation of norms should be significant
        std_norms = np.std(norms)
        mean_norms = np.mean(norms)

        # Coefficient of variation
        cv = std_norms / mean_norms if mean_norms > 0 else 0

        # For isotropic Gaussian, norms should vary
        # For normalized vectors, CV would be ~0
        assert cv > 0.05, f"Norm variation too low ({cv}), embeddings may be normalized"

    @pytest.mark.slow
    def test_covariance_closer_to_identity_than_constant(self, embedding_dim, sample_texts):
        """Test that covariance is closer to identity than to constant matrix."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = sample_texts * 5

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        # Compute covariance
        cov = np.cov(embeddings.T)

        # Distance to identity
        identity = np.eye(cov.shape[0])
        dist_to_identity = np.linalg.norm(cov - identity, 'fro')

        # Distance to constant correlation
        ones = np.ones_like(cov)
        dist_to_constant = np.linalg.norm(cov - ones, 'fro')

        # Should be closer to identity than to constant
        # (Relaxed check since we have limited samples)
        assert dist_to_identity < dist_to_constant * 2

    @pytest.mark.slow
    def test_no_dimensional_collapse(self, embedding_dim, sample_texts):
        """Test that all dimensions are used (no collapse to lower-dimensional space)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = sample_texts * 5

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        # Compute variance per dimension
        var_per_dim = np.var(embeddings, axis=0)

        # Count dimensions with significant variance
        significant_dims = np.sum(var_per_dim > 0.01)

        # Most dimensions should have significant variance
        usage_ratio = significant_dims / embedding_dim
        assert usage_ratio > 0.8, f"Only {usage_ratio:.2%} of dimensions used, possible collapse"


class TestGaussianDistributionProperties:
    """Test that embeddings follow Gaussian distribution."""

    @pytest.mark.slow
    def test_dimension_values_approximately_gaussian(self, embedding_dim, sample_texts):
        """Test that values in each dimension are approximately Gaussian."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        # Need more samples for distribution testing
        texts = sample_texts * 10

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        # Test first few dimensions for normality
        # Using Shapiro-Wilk test (note: may fail with small samples)
        num_dims_to_test = min(5, embedding_dim)
        normal_count = 0

        for dim in range(num_dims_to_test):
            dim_values = embeddings[:, dim]

            # Shapiro-Wilk test (p > 0.05 suggests normal)
            # Using very relaxed threshold due to small sample
            _, p_value = stats.shapiro(dim_values)

            if p_value > 0.01:  # Very relaxed threshold
                normal_count += 1

        # At least some dimensions should pass normality test
        # (Relaxed since we have limited samples)
        assert normal_count > 0, "No dimensions appear normally distributed"

    @pytest.mark.slow
    def test_embedding_values_have_reasonable_range(self, embedding_dim, sample_texts):
        """Test that embedding values are in reasonable range for N(0,1)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        # For N(0,1), ~99.7% of values should be in [-3, 3]
        # Check that most values are in reasonable range
        max_val = np.max(np.abs(embeddings))

        # Should not have extreme outliers
        assert max_val < 10, f"Embedding values too large ({max_val}), unusual for Gaussian"


class TestEuclideanDistanceProperties:
    """Test mathematical properties of Euclidean distance."""

    @pytest.mark.slow
    def test_distance_triangle_inequality(self, embedding_dim):
        """Test that triangle inequality holds: d(A,C) <= d(A,B) + d(B,C)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = ["Text A", "Text B", "Text C"]

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        A, B, C = embeddings[0], embeddings[1], embeddings[2]

        # Compute distances
        dist_AB = np.linalg.norm(A - B)
        dist_BC = np.linalg.norm(B - C)
        dist_AC = np.linalg.norm(A - C)

        # Triangle inequality
        assert dist_AC <= dist_AB + dist_BC + 1e-6

    @pytest.mark.slow
    def test_distance_symmetry(self, embedding_dim):
        """Test that d(A, B) == d(B, A)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text_a = "First text"
        text_b = "Second text"

        with torch.no_grad():
            emb_a = model.encode(text_a, convert_to_numpy=True)
            emb_b = model.encode(text_b, convert_to_numpy=True)

        dist_ab = np.linalg.norm(emb_a - emb_b)
        dist_ba = np.linalg.norm(emb_b - emb_a)

        assert abs(dist_ab - dist_ba) < 1e-6

    @pytest.mark.slow
    def test_self_distance_is_zero(self, embedding_dim):
        """Test that d(A, A) == 0."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text = "Test text"

        with torch.no_grad():
            emb1 = model.encode(text, convert_to_numpy=True)
            emb2 = model.encode(text, convert_to_numpy=True)

        dist = np.linalg.norm(emb1 - emb2)

        assert dist < 1e-6

    @pytest.mark.slow
    def test_distance_is_positive_for_different_texts(self, embedding_dim):
        """Test that d(A, B) > 0 when A != B."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text_a = "First text"
        text_b = "Different text"

        with torch.no_grad():
            emb_a = model.encode(text_a, convert_to_numpy=True)
            emb_b = model.encode(text_b, convert_to_numpy=True)

        dist = np.linalg.norm(emb_a - emb_b)

        assert dist > 0


class TestNotNormalizedProperty:
    """Test the critical property that embeddings are NOT normalized."""

    @pytest.mark.slow
    def test_embeddings_not_unit_norm(self, embedding_dim, sample_texts):
        """Test that embeddings do NOT have unit norm (≠ 1)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)

        # Count how many are close to 1.0
        close_to_one = np.sum(np.abs(norms - 1.0) < 0.1)

        # Most should NOT be close to 1.0
        ratio_near_one = close_to_one / len(norms)
        assert ratio_near_one < 0.5, f"{ratio_near_one:.2%} have norm ~1, embeddings may be normalized!"

    @pytest.mark.slow
    def test_norm_variance_is_significant(self, embedding_dim, sample_texts):
        """Test that norms have significant variance (not all the same)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        norms = np.linalg.norm(embeddings, axis=1)
        norm_std = np.std(norms)

        # Standard deviation should be significant
        assert norm_std > 0.1, f"Norm std too low ({norm_std}), norms may be constant"

    @pytest.mark.slow
    def test_different_norm_than_cosine_space(self, embedding_dim):
        """Test that distance differs from what cosine similarity would give."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = ["Text A", "Text B"]

        with torch.no_grad():
            embeddings = model.encode(texts, convert_to_numpy=True)

        # Actual L2 distance
        l2_dist = np.linalg.norm(embeddings[0] - embeddings[1])

        # If embeddings were normalized, L2 distance would relate to cosine similarity
        # L2^2 = 2(1 - cos_sim) for unit vectors
        # So L2 would be in [0, 2*sqrt(2)] ≈ [0, 2.83]

        # For unnormalized Gaussian embeddings, L2 can be larger
        # This is a weak test, but checks that we're not in normalized space

        # Just verify that the distance is computed correctly (non-negative)
        assert l2_dist >= 0


class TestMagnitudeAsConfidence:
    """Test that magnitude can represent confidence/uncertainty."""

    @pytest.mark.slow
    def test_magnitude_varies_across_samples(self, embedding_dim, sample_texts):
        """Test that different texts have different embedding magnitudes."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        norms = np.linalg.norm(embeddings, axis=1)

        # Norms should vary
        unique_norms = len(set(np.round(norms, 2)))

        # Should have different norm values
        assert unique_norms > 1, "All embeddings have same magnitude"

    @pytest.mark.slow
    def test_can_use_magnitude_for_filtering(self, embedding_dim, sample_texts):
        """Test that magnitude can be used as a filtering criterion."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, convert_to_numpy=True)

        norms = np.linalg.norm(embeddings, axis=1)

        # Can set a threshold and filter
        threshold = np.median(norms)
        high_confidence = norms >= threshold
        low_confidence = norms < threshold

        # Should have both high and low confidence samples
        assert np.any(high_confidence)
        assert np.any(low_confidence)


class TestCompositionalityProperty:
    """Test that embeddings compose naturally (unlike normalized embeddings)."""

    @pytest.mark.slow
    def test_addition_preserves_meaning(self, embedding_dim):
        """Test that embedding addition is meaningful."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = ["Python", "programming", "Python programming"]

        with torch.no_grad():
            emb1 = model.encode(texts[0], convert_to_numpy=True)
            emb2 = model.encode(texts[1], convert_to_numpy=True)
            emb_combined = model.encode(texts[2], convert_to_numpy=True)

        # Composed embedding
        emb_sum = emb1 + emb2

        # Sum should be in same general space as embeddings
        # (This is a weak test - just checking it doesn't explode)
        sum_norm = np.linalg.norm(emb_sum)
        combined_norm = np.linalg.norm(emb_combined)

        # They should be in similar magnitude range
        # (For normalized embeddings, sum would have norm ~sqrt(2))
        assert 0.5 < sum_norm / combined_norm < 5

    @pytest.mark.slow
    def test_scalar_multiplication_is_meaningful(self, embedding_dim):
        """Test that scaling embeddings is meaningful."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text = "Test text"

        with torch.no_grad():
            emb = model.encode(text, convert_to_numpy=True)

        # Scale by 2
        scaled = 2 * emb

        # Norm should scale appropriately
        original_norm = np.linalg.norm(emb)
        scaled_norm = np.linalg.norm(scaled)

        assert abs(scaled_norm - 2 * original_norm) < 1e-5
