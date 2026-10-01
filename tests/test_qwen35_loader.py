from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from nanovllm.utils.loader import load_model


class _PackedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 2))
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, parameter, loaded_weight, shard_id):
        parameter.data[shard_id * 2 : (shard_id + 1) * 2].copy_(loaded_weight)


class _LanguageBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.gate_up_proj = _PackedLinear()


class _FakeQwen35(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    checkpoint_prefix_mapping = (("model.language_model.", "model."),)
    explicitly_skipped_prefixes = ("model.visual.", "mtp.")

    def __init__(self):
        super().__init__()
        self.model = _LanguageBody()
        self.unloaded = nn.Parameter(torch.empty(1))


class _TiedCheckpointModel(nn.Module):
    checkpoint_prefix_mapping = (("model.language_model.", "model."),)

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(3, 2)
        self.lm_head = nn.Linear(2, 3, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def _write_checkpoint(path: Path, *, include_unexpected: bool = False):
    tensors = {
        "model.language_model.proj.weight": torch.arange(4).reshape(2, 2).float(),
        "model.language_model.gate_proj.weight": torch.ones(2, 2),
        "model.language_model.up_proj.weight": torch.full((2, 2), 2.0),
        "model.visual.patch.weight": torch.ones(1),
        "mtp.layer.weight": torch.ones(1),
    }
    if include_unexpected:
        tensors["model.language_model.typo.weight"] = torch.ones(1)
    save_file(tensors, path / "model.safetensors")


def test_loader_reports_prefix_mapping_packing_skips_and_missing(tmp_path):
    model = _FakeQwen35()
    _write_checkpoint(tmp_path)

    report = load_model(model, str(tmp_path), strict=False)
    assert report.loaded == ("model.language_model.proj.weight",)
    assert report.packed == (
        "model.language_model.gate_proj.weight",
        "model.language_model.up_proj.weight",
    )
    assert report.skipped == ("model.visual.patch.weight", "mtp.layer.weight")
    assert report.missing == ("unloaded",)
    assert report.unexpected == ()
    torch.testing.assert_close(
        model.model.proj.weight,
        torch.arange(4).reshape(2, 2).float(),
    )
    torch.testing.assert_close(
        model.model.gate_up_proj.weight,
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
    )


def test_strict_loader_rejects_missing_and_unexpected_parameters(tmp_path):
    model = _FakeQwen35()
    _write_checkpoint(tmp_path, include_unexpected=True)

    try:
        load_model(model, str(tmp_path), strict=True)
    except RuntimeError as error:
        message = str(error)
        assert "unloaded" in message
        assert "model.language_model.typo.weight" in message
        assert "model.visual.patch.weight" in message
    else:
        raise AssertionError("strict loader accepted an incomplete checkpoint")


def test_strict_loader_accepts_parameter_alias_from_tied_embedding(tmp_path):
    expected = torch.arange(6).reshape(3, 2).float()
    save_file(
        {"model.language_model.embed_tokens.weight": expected},
        tmp_path / "model.safetensors",
    )
    model = _TiedCheckpointModel()

    report = load_model(model, str(tmp_path), strict=True)

    assert report.missing == ()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(model.lm_head.weight, expected)
