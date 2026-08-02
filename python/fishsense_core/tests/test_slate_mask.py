"""Tests for the slate board-mask checkpoint slot (ported from
tests/test_mask.py in UCSD-E4E/2026-07-31_slate_training).

The important property is that this module stays importable and the classical
path stays usable **without torch installed** — the data-worker must be able to
ship the activity before the checkpoint exists.
"""

import numpy as np
import pytest

from fishsense_core.slate.mask import MASK_MEAN, MASK_STD, BoardMasker, preprocess

HAS_TORCH = True
try:  # pragma: no cover - environment dependent
    import torch as _torch  # noqa: F401
except ImportError:  # pragma: no cover
    HAS_TORCH = False


class TestImportableWithoutTorch:
    def test_module_imports_without_touching_torch(self):
        # The import at the top of this file already proves it, but make the
        # intent explicit: nothing at module scope may pull torch in.
        import fishsense_core.slate.mask as m
        assert m.BoardMasker is not None


class TestPreprocess:
    def test_outputs_chw_float32_at_the_requested_size(self):
        out = preprocess(np.zeros((300, 400, 3), np.uint8), (512, 384))
        assert out.shape == (3, 384, 512)
        assert out.dtype == np.float32

    def test_applies_training_normalization(self):
        # A mid-grey frame maps to (0.5 - mean) / std.
        out = preprocess(np.full((10, 10, 3), 128, np.uint8), (32, 32))
        expected = (128 / 255.0 - MASK_MEAN) / MASK_STD
        assert out.mean() == pytest.approx(expected, abs=1e-3)

    def test_is_contiguous(self):
        # torch.from_numpy on a transposed view would otherwise copy or fail.
        assert preprocess(np.zeros((64, 64, 3), np.uint8), (32, 32)).flags["C_CONTIGUOUS"]


class TestFromCheckpointErrors:
    def test_missing_file_raises_filenotfound(self):
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            BoardMasker.from_checkpoint("/nonexistent/board_unet.pt")

    def test_wrong_payload_shape_raises_valueerror(self, tmp_path):
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        import torch

        bad = tmp_path / "bad.pt"
        torch.save({"weights": {}}, bad)
        with pytest.raises(ValueError, match="not a board-mask checkpoint"):
            BoardMasker.from_checkpoint(str(bad))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRoundTrip:
    def test_predict_returns_probabilities_at_model_resolution(self, tmp_path):
        import torch
        from fishsense_core.slate.mask import build_unet

        model = build_unet()
        path = tmp_path / "ck.pt"
        torch.save({"model": model.state_dict(), "size": (128, 96)}, path)

        masker = BoardMasker.from_checkpoint(str(path))
        assert masker.size == (128, 96)

        probs = masker.predict(np.random.randint(0, 255, (300, 400, 3), np.uint8))
        assert probs.shape == (96, 128)          # model resolution, not frame
        assert probs.dtype == np.float32
        assert 0.0 <= probs.min() and probs.max() <= 1.0

    def test_output_feeds_mask_candidates_directly(self, tmp_path):
        import torch
        from fishsense_core.slate.estimator import mask_candidates
        from fishsense_core.slate.mask import build_unet

        model = build_unet()
        path = tmp_path / "ck.pt"
        torch.save({"model": model.state_dict(), "size": (128, 96)}, path)
        probs = BoardMasker.from_checkpoint(str(path)).predict(
            np.zeros((300, 400, 3), np.uint8))
        # Must not raise on the float-probability path, whatever it contains.
        mask_candidates(probs, (300, 400))


class TestFromPretrained:
    """Mirrors LaserDetector.from_pretrained so both read alike at call sites."""

    def test_defaults_match_the_e4e_convention(self):
        from fishsense_core.slate.mask import DEFAULT_CHECKPOINT, DEFAULT_HF_REPO
        assert DEFAULT_HF_REPO.startswith("ucsde4e/")
        assert DEFAULT_CHECKPOINT.endswith(".pt")

    def test_missing_hub_raises_a_pointed_import_error(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def blocked(name, *a, **kw):
            if name == "huggingface_hub":
                raise ImportError("no hub")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match=r"fishsense_core\[slate\]"):
            BoardMasker.from_pretrained()
