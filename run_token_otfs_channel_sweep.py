"""Run token-OTFS-token experiments through the existing time-varying channel.

Pipeline:

    tokens -> DD grid -> OTFS modulation -> TimeVaryingMultipathChannel
           -> OTFS demodulation -> DD grid -> recovered tokens

This is a baseline with no equalizer and no training. The channel itself is
implemented in channel_model.py and is only reused here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from channel_model import ChannelConfig, TimeVaryingMultipathChannel
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
    return tensor.mean().item(), tensor.std(unbiased=False).item()


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


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def _trial_seed(base_seed: int, snr_db: float, trial_index: int) -> int:
    return int(base_seed + trial_index + round(float(snr_db) * 1000.0))


def _uses_identity_single_path(args: argparse.Namespace) -> bool:
    return (
        args.num_paths == 1
        and float(args.max_delay_samples) == 0.0
        and float(args.max_doppler_hz) == 0.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SNR for token-OTFS-token through channel_model.py.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage5_channel/token_otfs_channel_sweep.pt"),
    )
    parser.add_argument(
        "--snr-db-list",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    )
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--symbols-per-token", type=int, default=4)
    parser.add_argument("--dd-shape", type=int, nargs=2, metavar=("M", "N"), default=(32, 32))
    parser.add_argument("--cp-len", type=int, default=4)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=15.36e6)
    parser.add_argument("--max-delay-samples", type=float, default=3.0)
    parser.add_argument("--max-doppler-hz", type=float, default=500.0)
    parser.add_argument("--fading", type=str, default="rayleigh", choices=["rayleigh", "rician", "fixed"])
    parser.add_argument("--rician-k-db", type=float, default=8.0)
    parser.add_argument("--doppler-distribution", type=str, default="jakes", choices=["jakes", "uniform"])
    parser.add_argument("--randomize-each-forward", action="store_true")
    parser.add_argument("--integer-delays", action="store_true")
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

    device = _resolve_device(args.device)
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
        device=str(device),
    )
    modem = OTFSModem(
        dd_shape=dd_shape,
        cp_len=args.cp_len,
        device=str(device),
    )

    x_dd = mapper.encode(token_ids)
    x_time = modem.modulate(x_dd)
    identity_single_path = _uses_identity_single_path(args)

    print(f"token shape: {_shape_list(token_ids)}")
    print(f"codebook size: {codebook_size}")
    print(f"symbols per token: {args.symbols_per_token}")
    print(f"DD shape: {dd_shape}")
    print(f"cp_len: {args.cp_len}")
    print(f"time signal shape: {_shape_list(x_time)}")
    print(f"num paths: {args.num_paths}")
    print(f"sample rate: {args.sample_rate}")
    print(f"max delay samples: {args.max_delay_samples}")
    print(f"max doppler Hz: {args.max_doppler_hz}")
    print(f"fading: {args.fading}")
    print(f"doppler distribution: {args.doppler_distribution}")
    print(f"randomize each forward: {args.randomize_each_forward}")
    print(f"delay mode: {'integer' if args.integer_delays else 'fractional'}")
    print(f"identity single-path sanity override: {identity_single_path}")
    print(f"num trials: {args.num_trials}")
    print(f"output path: {args.output}")
    print()
    print("Sanity notes:")
    print("  1. num_paths=1, max_delay=0, max_doppler=0, high SNR should be near 1.0 accuracy.")
    print("  2. The same single-path setting over 0/10/20/30 dB should resemble the AWGN sweep.")
    print("  3. Multipath Doppler without an equalizer can reduce token accuracy; that is expected.")
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

        for trial_index in range(args.num_trials):
            seed = _trial_seed(args.seed, float(snr_db), trial_index)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            config = ChannelConfig(
                num_paths=args.num_paths,
                sample_rate=args.sample_rate,
                snr_db=float(snr_db),
                max_delay_samples=args.max_delay_samples,
                max_doppler_hz=args.max_doppler_hz,
                fading=args.fading,
                rician_k_db=args.rician_k_db,
                doppler_distribution=args.doppler_distribution,
                randomize_each_forward=args.randomize_each_forward,
                fractional_delays=not args.integer_delays,
                complex_dtype="complex64",
                seed=seed,
            )
            channel = TimeVaryingMultipathChannel(config).to(device)

            # For the documented no-channel sanity check, force the existing
            # channel layer to use exactly one unit-gain, zero-delay, zero-Doppler
            # path. Otherwise a Rayleigh single path is an unknown complex scalar.
            if identity_single_path:
                path_gains = torch.ones(args.num_paths, device=device, dtype=torch.complex64)
                path_delays = torch.zeros(args.num_paths, device=device, dtype=torch.float32)
                path_dopplers_hz = torch.zeros(args.num_paths, device=device, dtype=torch.float32)
                out = channel(
                    x_time,
                    snr_db=float(snr_db),
                    path_gains=path_gains,
                    path_delays=path_delays,
                    path_dopplers_hz=path_dopplers_hz,
                    return_info=True,
                )
            else:
                out = channel(x_time, snr_db=float(snr_db), return_info=True)

            y_time = out.y
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
                "clean_time": out.clean,
                "noise": out.noise,
                "y_dd": y_dd,
                "recovered_tokens": recovered,
                "path_gains": out.path_gains,
                "delays": out.delays,
                "dopplers_hz": out.dopplers_hz,
                "conditioning": out.conditioning,
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
            "num_trials": args.num_trials,
            "num_paths": args.num_paths,
            "sample_rate": args.sample_rate,
            "max_delay_samples": args.max_delay_samples,
            "max_doppler_hz": args.max_doppler_hz,
            "fading": args.fading,
            "rician_k_db": args.rician_k_db,
            "doppler_distribution": args.doppler_distribution,
            "randomize_each_forward": args.randomize_each_forward,
            "fractional_delays": not args.integer_delays,
            "identity_single_path_sanity_override": identity_single_path,
            "seed": args.seed,
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
        last_path = args.output.parent / "token_otfs_channel_last.pt"
        torch.save(
            {
                "x_dd": last_details["x_dd"].cpu(),
                "x_time": last_details["x_time"].cpu(),
                "y_time": last_details["y_time"].cpu(),
                "clean_time": last_details["clean_time"].cpu(),
                "noise": last_details["noise"].cpu(),
                "y_dd": last_details["y_dd"].cpu(),
                "recovered_tokens": last_details["recovered_tokens"].cpu(),
                "path_gains": last_details["path_gains"].cpu(),
                "delays": last_details["delays"].cpu(),
                "dopplers_hz": last_details["dopplers_hz"].cpu(),
                "conditioning": last_details["conditioning"].cpu(),
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
