"""Audit Qwen3.5 checkpoint names against a meta-device nano-vLLM model.

Usage:
    PYTHONSAFEPATH=1 PYTHONPATH=/path/to/nano-vllm-infra \
      python tests/audit_qwen35_official_index.py /path/to/qwen35-model-dir

The directory only needs config.json and model.safetensors.index.json. Model
weight shards are not loaded or allocated.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP


def audit(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous.name}",
        rank=0,
        world_size=1,
    )
    old_device = torch.get_default_device()
    old_dtype = torch.get_default_dtype()
    try:
        full_config = AutoConfig.from_pretrained(model_dir)
        text_config, capabilities = normalize_hf_config(full_config)
        torch.set_default_device("meta")
        torch.set_default_dtype(text_config.dtype)
        get_rope.cache_clear()
        model = create_model(text_config, capabilities)
        parameter_names = set()
        aliases_by_id = {}
        parameters_by_name = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            parameter_names.add(name)
            parameters_by_name[name] = parameter
            aliases_by_id.setdefault(id(parameter), set()).add(name)
        checkpoint_names = set(
            json.loads(index_path.read_text())["weight_map"]
        )

        loaded_targets = set()
        unexpected = []
        skipped = []
        for source_name in checkpoint_names:
            if source_name.startswith(model.explicitly_skipped_prefixes):
                skipped.append(source_name)
                continue
            mapped_name = source_name
            for source_prefix, target_prefix in model.checkpoint_prefix_mapping:
                if mapped_name.startswith(source_prefix):
                    mapped_name = target_prefix + mapped_name[len(source_prefix) :]
                    break
            components = mapped_name.split(".")
            for source_component, (target_component, _) in model.packed_modules_mapping.items():
                if source_component in components:
                    components[components.index(source_component)] = target_component
                    mapped_name = ".".join(components)
                    break
            if mapped_name not in parameter_names:
                unexpected.append((source_name, mapped_name))
            else:
                loaded_targets.update(
                    aliases_by_id[id(parameters_by_name[mapped_name])]
                )

        missing = sorted(parameter_names - loaded_targets)
        mtp = Qwen3_5MTP(text_config, model.model.embed_tokens)
        mtp_parameter_names = {
            name for name, _ in mtp.named_parameters(remove_duplicate=False)
        }
        mtp_loaded_targets = set()
        mtp_unexpected = []
        for source_name in checkpoint_names:
            if not source_name.startswith("mtp."):
                continue
            mapped_name = source_name[len("mtp.") :]
            components = mapped_name.split(".")
            for source_component, (target_component, _) in mtp.packed_modules_mapping.items():
                if source_component in components:
                    components[components.index(source_component)] = target_component
                    mapped_name = ".".join(components)
                    break
            if mapped_name not in mtp_parameter_names:
                mtp_unexpected.append((source_name, mapped_name))
            else:
                mtp_loaded_targets.add(mapped_name)
        mtp_missing = sorted(mtp_parameter_names - mtp_loaded_targets)
        return {
            "architecture": capabilities.architecture,
            "num_layers": len(capabilities.layer_types),
            "num_linear_layers": len(capabilities.linear_attention_layer_indices),
            "num_full_layers": len(capabilities.full_attention_layer_indices),
            "checkpoint_keys": len(checkpoint_names),
            "model_parameters": len(parameter_names),
            "skipped": len(skipped),
            "unexpected": unexpected,
            "missing": missing,
            "mtp_checkpoint_keys": sum(
                name.startswith("mtp.") for name in checkpoint_names
            ),
            "mtp_model_parameters": len(mtp_parameter_names),
            "mtp_unexpected": mtp_unexpected,
            "mtp_missing": mtp_missing,
        }
    finally:
        get_rope.cache_clear()
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    report = audit(args.model_dir)
    print(json.dumps(report, indent=2))
    if (
        report["unexpected"]
        or report["missing"]
        or report["mtp_unexpected"]
        or report["mtp_missing"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
