"""Run token-OTFS-token experiments with complex AWGN.

Pipeline:

    tokens -> DD grid -> OTFS modulation -> AWGN -> OTFS demodulation -> tokens

This script intentionally does not add a multipath/Doppler channel, neural
network training, bit conversion, or QAM modulation.
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


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty.")
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = tensor.mean().item()
    std = tensor.std(unbiased=False).item()
    return mean, std


def add_complex_awgn(
    x: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor.")
    if not torch.is_complex(x):
        raise TypeError(f"x must be a complex tensor; got {x.dtype}.")

    real_dtype = torch.float32 if x.dtype == torch.complex64 else torch.float64
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    if snr_linear <= 0:
        raise ValueError(f"snr_db must produce a positive linear SNR; got {snr_db}.")

    signal_power = x.abs().pow(2).mean()
    noise_std = torch.sqrt(signal_power / (2.0 * snr_linear))
    real_noise = torch.randn(
        x.shape,
        device=x.device,
        dtype=real_dtype,
        generator=generator,
    )
    imag_noise = torch.randn(
        x.shape,
        device=x.device,
        dtype=real_dtype,
        generator=generator,
    )
    noise = noise_std.to(dtype=real_dtype) * torch.complex(real_noise, imag_noise)
    noise = noise.to(dtype=x.dtype)
    return x + noise, noise


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
        description="Sweep SNR for token-OTFS-token with complex AWGN only.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("token_otfs_awgn_sweep.pt"))
    parser.add_argument(
        "--snr-db-list",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    )
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--symbols-per-token", type=int, default=4)
    parser.add_argument("--dd-shape", type=int, nargs=2, metavar=("M", "N"), default=(32, 32))
    parser.add_argument("--cp-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("num_trials must be a positive integer.")
    if not args.snr_db_list:
        raise ValueError("snr_db_list must contain at least one value.")

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

    generator_device = mapper.device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(args.seed)

    x_dd = mapper.encode(token_ids)
    x_time = modem.modulate(x_dd)

    print(f"token shape: {_shape_list(token_ids)}")
    print(f"codebook size: {codebook_size}")
    print(f"symbols per token: {args.symbols_per_token}")
    print(f"DD shape: {dd_shape}")
    print(f"cp_len: {args.cp_len}")
    print(f"time signal shape: {_shape_list(x_time)}")
    print(f"num trials: {args.num_trials}")
    print(f"output path: {args.output}")
    print()
    print("SNR(dB) | Acc mean | TER mean | DD NMSE mean | DD max err mean")
    print("---------------------------------------------------------------")

    results: list[dict[str, float]] = []
    last_details: dict[str, Any] | None = None

    for snr_db in args.snr_db_list:
        accuracies: list[float] = []
        error_rates: list[float] = []
        dd_nmses: list[float] = []
        dd_max_errors: list[float] = []

        for _trial in range(args.num_trials):
            y_time, noise = add_complex_awgn(x_time, float(snr_db), generator)
            y_dd = modem.demodulate(y_time)
            recovered = mapper.decode(y_dd, token_shape=grid_size)

            acc = TokenDDMapper.token_accuracy(token_ids, recovered)
            ter = TokenDDMapper.token_error_rate(token_ids, recovered)
            dd_nmse = normalized_mse(x_dd, y_dd)
            dd_max_err = max_abs_error(x_dd, y_dd)

            accuracies.append(acc)
            error_rates.append(ter)
            dd_nmses.append(dd_nmse)
            dd_max_errors.append(dd_max_err)

            last_details = {
                "x_dd": x_dd,
                "x_time": x_time,
                "y_time": y_time,
                "noise": noise,
                "y_dd": y_dd,
                "recovered_tokens": recovered,
                "snr_db": float(snr_db),
                "token_accuracy": acc,
                "token_error_rate": ter,
                "dd_nmse": dd_nmse,
                "dd_max_abs_error": dd_max_err,
            }

        acc_mean, acc_std = mean_std(accuracies)
        ter_mean, ter_std = mean_std(error_rates)
        dd_nmse_mean, dd_nmse_std = mean_std(dd_nmses)
        dd_max_err_mean, dd_max_err_std = mean_std(dd_max_errors)
        result = {
            "snr_db": float(snr_db),
            "token_accuracy_mean": acc_mean,
            "token_accuracy_std": acc_std,
            "token_error_rate_mean": ter_mean,
            "token_error_rate_std": ter_std,
            "dd_nmse_mean": dd_nmse_mean,
            "dd_nmse_std": dd_nmse_std,
            "dd_max_abs_error_mean": dd_max_err_mean,
            "dd_max_abs_error_std": dd_max_err_std,
        }
        results.append(result)
        print(
            f"{float(snr_db):7.1f} | "
            f"{acc_mean:8.6f} | "
            f"{ter_mean:8.6f} | "
            f"{dd_nmse_mean:12.6e} | "
            f"{dd_max_err_mean:15.6e}"
        )

    output_payload = {
        "results": results,
        "config": {
            "tokens": str(args.tokens),
            "symbols_per_token": args.symbols_per_token,
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "seed": args.seed,
            "num_trials": args.num_trials,
            "codebook_size": codebook_size,
            "token_shape": grid_size,
            "image_size": image_size,
            "downsample_ratio": downsample_ratio,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    if args.save_last:
        if last_details is None:
            raise RuntimeError("save-last requested but no trial was executed.")
        last_path = args.output.with_name("token_otfs_awgn_last.pt")
        torch.save(
            {
                "x_dd": last_details["x_dd"].cpu(),
                "x_time": last_details["x_time"].cpu(),
                "y_time": last_details["y_time"].cpu(),
                "noise": last_details["noise"].cpu(),
                "y_dd": last_details["y_dd"].cpu(),
                "recovered_tokens": last_details["recovered_tokens"].cpu(),
                "snr_db": last_details["snr_db"],
                "token_accuracy": last_details["token_accuracy"],
                "token_error_rate": last_details["token_error_rate"],
                "dd_nmse": last_details["dd_nmse"],
                "dd_max_abs_error": last_details["dd_max_abs_error"],
            },
            last_path,
        )
        print(f"last trial path: {last_path}")


if __name__ == "__main__":
    main()
