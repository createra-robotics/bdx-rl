#!/usr/bin/env python3
"""Export a BDX RSL-RL actor checkpoint to ONNX.

The exporter intentionally avoids Isaac Sim imports so it can run in a normal
Python environment that has PyTorch and ONNX installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
SIM2SIM_DIR = SCRIPT_DIR.parent
DEFAULT_ONNX_DIR = SIM2SIM_DIR / "onnx"


class ActorMlp(nn.Module):
    """MLP actor matching the RSL-RL checkpoint keys used by this project."""

    def __init__(self, layer_sizes: list[tuple[int, int]], activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        for layer_index, (in_features, out_features) in enumerate(layer_sizes):
            layers.append(nn.Linear(in_features, out_features))
            if layer_index != len(layer_sizes) - 1:
                layers.append(_make_activation(activation))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}


def _extract_actor_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported actor state_dict type: {type(state_dict)!r}")

    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "actor.")
    state_dict = {key: value for key, value in state_dict.items() if key.startswith("mlp.")}
    if not state_dict:
        raise ValueError("Could not find actor MLP weights in checkpoint.")

    return state_dict


def _infer_layer_sizes(actor_state_dict: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    weight_items: list[tuple[int, torch.Tensor]] = []
    for key, value in actor_state_dict.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "mlp" and parts[2] == "weight":
            weight_items.append((int(parts[1]), value))

    if not weight_items:
        raise ValueError("Could not infer MLP layers from state_dict.")

    layer_sizes = []
    for _, weight in sorted(weight_items):
        if weight.ndim != 2:
            raise ValueError(f"Expected 2D linear weight, got shape {tuple(weight.shape)}")
        layer_sizes.append((int(weight.shape[1]), int(weight.shape[0])))
    return layer_sizes


def _default_output_path(checkpoint_path: Path) -> Path:
    return DEFAULT_ONNX_DIR / f"{checkpoint_path.stem}.onnx"


def export_checkpoint(checkpoint_path: Path, output_path: Path, activation: str, opset: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_state_dict = _extract_actor_state_dict(checkpoint)
    layer_sizes = _infer_layer_sizes(actor_state_dict)

    actor = ActorMlp(layer_sizes=layer_sizes, activation=activation)
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    obs_dim = layer_sizes[0][0]
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    except ImportError:
        print("[WARN] onnx package is not installed; skipped exported model validation.")

    action_dim = layer_sizes[-1][1]
    print(f"[INFO] Exported {checkpoint_path} -> {output_path}")
    print(f"[INFO] ONNX input obs_dim={obs_dim}, output action_dim={action_dim}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a BDX RSL-RL actor checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the RSL-RL .pt checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Defaults to sim2sim/onnx/<name>.onnx.",
    )
    parser.add_argument(
        "--activation",
        default="elu",
        choices=["elu", "relu", "tanh", "leaky_relu"],
        help="Actor MLP activation.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output is not None else _default_output_path(checkpoint_path)
    )
    export_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        activation=args.activation,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
