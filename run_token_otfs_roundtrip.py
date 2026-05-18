"""Run the token -> DD -> OTFS -> DD -> token closed loop.

This script stitches together the first two independent stages:

    visual tokens -> TokenDDMapper.encode -> X_DD
                  -> OTFSModem.modulate -> time waveform
                  -> OTFSModem.demodulate -> Y_DD
                  -> TokenDDMapper.decode -> recovered visual tokens

No channel, noise, training, bit conversion, or QAM is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from otfs_modem import OTFSModem, max_abs_error, normalized_mse
from token_dd_mapper import TokenDDMapper


def _torch_load(path: Path) -> Any:
    """Load a PyTorch object on CPU, compatible with old and new torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _shape_list(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def _load_token_payload(path: Path) -> tuple[torch.Tensor, int, tuple[int, int], int, int]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a dict.")

    required_keys = {
        "token_ids",
        "codebook_size",
        "grid_size",
        "image_size",
        "downsample_ratio",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}.")

    token_ids = payload["token_ids"]
    if not torch.is_tensor(token_ids):
        raise TypeError("payload['token_ids'] must be a torch.Tensor.")
    if token_ids.ndim != 3:
        raise ValueError(
            "payload['token_ids'] must have shape [B, Ht, Wt]; "
            f"got {_shape_list(token_ids)}."
        )
    if not token_ids.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"payload['token_ids'] must be integer; got {token_ids.dtype}.")
    if token_ids.numel() == 0:
        raise ValueError("payload['token_ids'] must not be empty.")

    codebook_size = int(payload["codebook_size"])
    grid_size = tuple(payload["grid_size"])
    if (
        len(grid_size) != 2
        or not all(isinstance(x, int) and x > 0 for x in grid_size)
    ):
        raise ValueError(f"payload['grid_size'] must be two positive ints; got {grid_size}.")
    if grid_size != tuple(token_ids.shape[1:]):
        raise ValueError(
            f"payload grid_size {grid_size} does not match token shape "
            f"{tuple(token_ids.shape[1:])}."
        )

    token_min = int(token_ids.min().item())
    token_max = int(token_ids.max().item())
    if token_min < 0 or token_max >= codebook_size:
        raise ValueError(
            "token_ids values must be in [0, codebook_size - 1]; "
            f"got min={token_min}, max={token_max}, codebook_size={codebook_size}."
        )

    return (
        token_ids,
        codebook_size,
        grid_size,
        int(payload["image_size"]),
        int(payload["downsample_ratio"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run token-DD-OTFS-DD-token roundtrip without channel or noise.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-dd-input", type=Path, required=True)
    parser.add_argument("--output-time", type=Path, required=True)
    parser.add_argument("--output-dd-output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=Path, required=True)
    parser.add_argument("--symbols-per-token", type=int, default=4)
    parser.add_argument("--dd-shape", type=int, nargs=2, metavar=("M", "N"), default=(32, 32))
    parser.add_argument("--cp-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dd_shape = tuple(args.dd_shape)
    (
        token_ids,
        codebook_size,
        grid_size,
        image_size,
        downsample_ratio,
    ) = _load_token_payload(args.tokens)

    mapper = TokenDDMapper(
        codebook_size=codebook_size,
        symbols_per_token=args.symbols_per_token,
        dd_shape=dd_shape,
        seed=args.seed,
        device=args.device,
    )
    modem = OTFSModem(
        dd_shape=dd_shape,
        cp_len=args.cp_len,
        device=args.device,
    )

    x_dd = mapper.encode(token_ids)
    time_signal = modem.modulate(x_dd)
    y_dd = modem.demodulate(time_signal)
    recovered = mapper.decode(y_dd, token_shape=grid_size)

    dd_nmse = normalized_mse(x_dd, y_dd)
    dd_max_err = max_abs_error(x_dd, y_dd)
    acc = TokenDDMapper.token_accuracy(token_ids, recovered)
    ter = TokenDDMapper.token_error_rate(token_ids, recovered)

    for path in (
        args.output_dd_input,
        args.output_time,
        args.output_dd_output,
        args.output_tokens,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dd_grid": x_dd.cpu(),
            "dd_shape": dd_shape,
            "symbols_per_token": args.symbols_per_token,
            "codebook_size": codebook_size,
            "token_shape": grid_size,
            "seed": args.seed,
        },
        args.output_dd_input,
    )
    torch.save(
        {
            "time_signal": time_signal.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "symbols_per_token": args.symbols_per_token,
            "codebook_size": codebook_size,
            "token_shape": grid_size,
            "seed": args.seed,
        },
        args.output_time,
    )
    torch.save(
        {
            "dd_grid": y_dd.cpu(),
            "source_dd_grid": x_dd.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "normalized_mse": dd_nmse,
            "max_abs_error": dd_max_err,
            "symbols_per_token": args.symbols_per_token,
            "codebook_size": codebook_size,
            "token_shape": grid_size,
            "seed": args.seed,
        },
        args.output_dd_output,
    )
    torch.save(
        {
            "token_ids": recovered.cpu(),
            "grid_size": grid_size,
            "codebook_size": codebook_size,
            "image_size": image_size,
            "downsample_ratio": downsample_ratio,
            "symbols_per_token": args.symbols_per_token,
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "seed": args.seed,
            "dd_normalized_mse": dd_nmse,
            "dd_max_abs_error": dd_max_err,
            "token_accuracy": acc,
            "token_error_rate": ter,
        },
        args.output_tokens,
    )

    print(f"original token shape: {_shape_list(token_ids)}")
    print(f"token min/max: {int(token_ids.min().item())}/{int(token_ids.max().item())}")
    print(f"codebook size: {codebook_size}")
    print(f"token grid size: {grid_size}")
    print(f"symbols per token: {args.symbols_per_token}")
    print(f"DD shape: {dd_shape}")
    print(f"cp_len: {args.cp_len}")
    print(f"DD grid before OTFS shape: {_shape_list(x_dd)}")
    print(f"time signal shape: {_shape_list(time_signal)}")
    print(f"DD grid after OTFS shape: {_shape_list(y_dd)}")
    print(f"DD normalized MSE: {dd_nmse}")
    print(f"DD max abs error: {dd_max_err}")
    print(f"recovered token shape: {_shape_list(recovered)}")
    print(f"token accuracy: {acc}")
    print(f"token error rate: {ter}")
    print(f"output dd input path: {args.output_dd_input}")
    print(f"output time path: {args.output_time}")
    print(f"output dd output path: {args.output_dd_output}")
    print(f"output tokens path: {args.output_tokens}")


if __name__ == "__main__":
    main()
