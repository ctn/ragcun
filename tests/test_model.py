"""
Unit tests for IsotropicGaussianEncoder model.

Tests the core model functionality including initialization, encoding,
and the critical property that embeddings are NOT normalized.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from ragcun.model import IsotropicGaussianEncoder


class TestIsotropicGaussianEncoderInitialization:
    """Test model initialization."""

    @pytest.mark.slow
    def test_model_loads_successfully(self, embedding_dim):
        """Test that model initializes without errors."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        assert model is not None
        assert isinstance(model, IsotropicGaussianEncoder)

    @pytest.mark.slow
    def test_output_dimension_is_correct(self, embedding_dim):
        """Test that output dimension matches parameter."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        assert model.output_dim == embedding_dim

    @pytest.mark.slow
    def test_projection_layer_exists(self, embedding_dim):
        """Test that projection layer is created."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        assert hasattr(model, 'projection')
        assert isinstance(model.projection, torch.nn.Sequential)

    @pytest.mark.slow
    def test_freeze_early_layers_parameter(self):
        """Test freeze_early_layers parameter works."""
        # Note: This creates the full model, so it's slow
        model_frozen = IsotropicGaussianEncoder(output_dim=128, freeze_early_layers=True)
        model_unfrozen = IsotropicGaussianEncoder(output_dim=128, freeze_early_layers=False)

        # Count trainable params
        frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        unfrozen_trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)

        # Frozen model should have fewer trainable params
        assert frozen_trainable < unfrozen_trainable

    @pytest.mark.slow
    def test_model_has_base_and_projection(self, embedding_dim):
        """Test model has both base encoder and projection."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        assert hasattr(model, 'base')
        assert hasattr(model, 'projection')


class TestIsotropicGaussianEncoderEncoding:
    """Test encoding functionality."""

    @pytest.mark.slow
    def test_encode_single_string(self, embedding_dim):
        """Test encoding a single string."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text = "Hello world"
        with torch.no_grad():
            embedding = model.encode(text)

        assert embedding is not None
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, embedding_dim)

    @pytest.mark.slow
    def test_encode_batch_of_strings(self, embedding_dim, sample_texts):
        """Test encoding multiple strings."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts)

        assert embeddings.shape == (len(sample_texts), embedding_dim)

    @pytest.mark.slow
    def test_output_shape_is_correct(self, embedding_dim):
        """Test that output has correct shape."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        texts = ["Text 1", "Text 2", "Text 3"]
        with torch.no_grad():
            embeddings = model.encode(texts)

        assert embeddings.shape == (3, embedding_dim)

    @pytest.mark.slow
    def test_embeddings_are_not_normalized(self, embedding_dim, sample_texts):
        """CRITICAL: Test that embeddings are NOT L2 normalized."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts)

        # Calculate L2 norms
        norms = torch.norm(embeddings, dim=1)

        # For L2 normalized vectors, all norms would be 1.0
        # For Gaussian embeddings, norms should vary
        norm_std = norms.std().item()

        # Check that norms have significant variation (not all ~1.0)
        assert norm_std > 0.1, f"Norms have too little variation ({norm_std}), embeddings may be normalized!"

        # Check that not all norms are close to 1.0
        close_to_one = (torch.abs(norms - 1.0) < 0.01).sum().item()
        assert close_to_one < len(sample_texts), "All norms are ~1.0, embeddings appear normalized!"

    @pytest.mark.slow
    def test_convert_to_numpy_flag(self, embedding_dim):
        """Test convert_to_numpy parameter."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text = "Test text"

        with torch.no_grad():
            # As tensor
            tensor_emb = model.encode(text, convert_to_numpy=False)
            assert isinstance(tensor_emb, torch.Tensor)

            # As numpy
            numpy_emb = model.encode(text, convert_to_numpy=True)
            assert isinstance(numpy_emb, np.ndarray)

    @pytest.mark.slow
    def test_batch_size_parameter(self, embedding_dim, sample_texts):
        """Test that batch_size parameter is accepted."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts, batch_size=2)

        assert embeddings.shape == (len(sample_texts), embedding_dim)

    @pytest.mark.slow
    def test_empty_input_handling(self, embedding_dim):
        """Test handling of empty input."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode([])

        assert embeddings.shape == (0, embedding_dim)

    @pytest.mark.slow
    def test_forward_method(self, embedding_dim):
        """Test forward() method for training."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.train()

        texts = ["Test 1", "Test 2"]
        embeddings = model.forward(texts)

        assert embeddings.shape == (2, embedding_dim)
        assert embeddings.requires_grad  # Should support gradients in training mode


class TestIsotropicGaussianEncoderLoading:
    """Test model loading from checkpoints."""

    @pytest.mark.slow
    def test_from_pretrained_with_checkpoint(self, mock_model_checkpoint, embedding_dim):
        """Test loading model from checkpoint file."""
        model = IsotropicGaussianEncoder.from_pretrained(
            str(mock_model_checkpoint),
            output_dim=embedding_dim
        )

        assert model is not None
        assert model.output_dim == embedding_dim
        assert not model.training  # Should be in eval mode

    @pytest.mark.slow
    def test_from_pretrained_with_state_dict_only(self, temp_dir, embedding_dim):
        """Test loading when checkpoint contains only state_dict."""
        from ragcun.model import IsotropicGaussianEncoder

        # Create model and save state_dict only
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        checkpoint_path = temp_dir / "state_dict_only.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Load it back
        loaded_model = IsotropicGaussianEncoder.from_pretrained(
            str(checkpoint_path),
            output_dim=embedding_dim
        )

        assert loaded_model is not None

    @pytest.mark.slow
    def test_from_pretrained_invalid_path(self, embedding_dim):
        """Test loading from non-existent path."""
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            IsotropicGaussianEncoder.from_pretrained(
                "/nonexistent/path/model.pt",
                output_dim=embedding_dim
            )

    @pytest.mark.slow
    def test_loaded_model_can_encode(self, mock_model_checkpoint, embedding_dim):
        """Test that loaded model can perform encoding."""
        model = IsotropicGaussianEncoder.from_pretrained(
            str(mock_model_checkpoint),
            output_dim=embedding_dim
        )

        with torch.no_grad():
            embedding = model.encode("Test text")

        assert embedding.shape == (1, embedding_dim)


class TestIsotropicGaussianEncoderProperties:
    """Test critical properties of Gaussian embeddings."""

    @pytest.mark.slow
    def test_embeddings_have_varying_norms(self, embedding_dim, sample_texts):
        """Test that embedding norms vary (not constant like normalized embeddings)."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(sample_texts)

        norms = torch.norm(embeddings, dim=1)

        # Calculate coefficient of variation
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        cv = std_norm / mean_norm if mean_norm > 0 else 0

        # Coefficient of variation should be significant for Gaussian embeddings
        # For normalized embeddings, CV would be ~0
        assert cv > 0.05, f"Coefficient of variation too low ({cv}), norms may be constant"

    @pytest.mark.slow
    def test_projection_has_no_normalization_layer(self, embedding_dim):
        """Test that projection layer doesn't contain normalization."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)

        # Check that projection doesn't have LayerNorm or BatchNorm
        for module in model.projection.modules():
            assert not isinstance(module, torch.nn.LayerNorm), "Projection should not have LayerNorm!"
            assert not isinstance(module, torch.nn.BatchNorm1d), "Projection should not have BatchNorm!"

    @pytest.mark.slow
    def test_different_texts_produce_different_embeddings(self, embedding_dim):
        """Test that different texts produce different embeddings."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text1 = "Python programming"
        text2 = "Machine learning"

        with torch.no_grad():
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)

        # Embeddings should be different
        distance = torch.norm(emb1 - emb2).item()
        assert distance > 0.1, "Different texts should produce different embeddings"

    @pytest.mark.slow
    def test_same_text_produces_same_embedding(self, embedding_dim):
        """Test that same text produces same embedding (deterministic)."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()

        text = "Test consistency"

        with torch.no_grad():
            emb1 = model.encode(text)
            emb2 = model.encode(text)

        # Should be identical
        assert torch.allclose(emb1, emb2, atol=1e-6)
