import os
from dataclasses import dataclass
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class LoadReport:
    loaded: tuple[str, ...]
    packed: tuple[str, ...]
    skipped: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _mapped_parameter_name(model: nn.Module, checkpoint_name: str) -> str:
    for source_prefix, target_prefix in getattr(
        model,
        "checkpoint_prefix_mapping",
        (),
    ):
        if checkpoint_name.startswith(source_prefix):
            return target_prefix + checkpoint_name[len(source_prefix) :]
    return checkpoint_name


def _replace_path_component(
    parameter_name: str,
    source_component: str,
    target_component: str,
) -> str | None:
    components = parameter_name.split(".")
    try:
        index = components.index(source_component)
    except ValueError:
        return None
    components[index] = target_component
    return ".".join(components)


def load_model(
    model: nn.Module,
    path: str,
    *,
    strict: bool = True,
) -> LoadReport:
    """Load safetensors with explicit packed/skip/missing reporting."""
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found under {path}")

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    skipped_prefixes = tuple(getattr(model, "explicitly_skipped_prefixes", ()))
    included_prefixes = tuple(getattr(model, "checkpoint_included_prefixes", ()))
    all_parameter_names = {
        name for name, _ in model.named_parameters(remove_duplicate=False)
    }
    aliases_by_parameter_id: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases_by_parameter_id.setdefault(id(parameter), set()).add(name)

    loaded_sources: list[str] = []
    packed_sources: list[str] = []
    skipped_sources: list[str] = []
    unexpected_sources: list[str] = []
    loaded_targets: set[str] = set()

    def mark_parameter_loaded(parameter: nn.Parameter) -> None:
        loaded_targets.update(aliases_by_parameter_id.get(id(parameter), ()))

    for file in files:
        with safe_open(file, "pt", "cpu") as tensors:
            for checkpoint_name in tensors.keys():
                if included_prefixes and not checkpoint_name.startswith(included_prefixes):
                    skipped_sources.append(checkpoint_name)
                    continue
                if checkpoint_name.endswith("_scale_inv"):
                    # Quantization scale loading is deferred to its backend.
                    skipped_sources.append(checkpoint_name)
                    continue
                if any(checkpoint_name.startswith(prefix) for prefix in skipped_prefixes):
                    skipped_sources.append(checkpoint_name)
                    continue

                mapped_name = _mapped_parameter_name(model, checkpoint_name)
                packed_match = None
                for source_fragment, (target_fragment, shard_id) in packed_modules_mapping.items():
                    packed_parameter_name = _replace_path_component(
                        mapped_name,
                        source_fragment,
                        target_fragment,
                    )
                    if packed_parameter_name is not None:
                        packed_match = (packed_parameter_name, shard_id)
                        break

                try:
                    if packed_match is not None:
                        parameter_name, shard_id = packed_match
                        parameter = model.get_parameter(parameter_name)
                        weight_loader = getattr(parameter, "weight_loader", None)
                        if weight_loader is None:
                            raise AttributeError(
                                f"packed parameter {parameter_name} has no weight_loader"
                            )
                        weight_loader(
                            parameter,
                            tensors.get_tensor(checkpoint_name),
                            shard_id,
                        )
                        packed_sources.append(checkpoint_name)
                    else:
                        parameter = model.get_parameter(mapped_name)
                        weight_loader = getattr(
                            parameter,
                            "weight_loader",
                            default_weight_loader,
                        )
                        weight_loader(parameter, tensors.get_tensor(checkpoint_name))
                        loaded_sources.append(checkpoint_name)
                except (AttributeError, KeyError):
                    unexpected_sources.append(checkpoint_name)
                    continue
                mark_parameter_loaded(parameter)

    report = LoadReport(
        loaded=tuple(sorted(loaded_sources)),
        packed=tuple(sorted(packed_sources)),
        skipped=tuple(sorted(skipped_sources)),
        missing=tuple(sorted(all_parameter_names - loaded_targets)),
        unexpected=tuple(sorted(unexpected_sources)),
    )
    if strict and not report.ok:
        raise RuntimeError(
            "strict checkpoint load failed: "
            f"missing={list(report.missing)}, "
            f"unexpected={list(report.unexpected)}, "
            f"explicitly_skipped={list(report.skipped)}"
        )
    return report
