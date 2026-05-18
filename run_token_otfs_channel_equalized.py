"""Compare raw decoding and oracle scalar equalization for token-OTFS-channel.

Pipeline:

    tokens -> DD grid -> OTFS modulation -> TimeVaryingMultipathChannel
           -> OTFS demodulation -> raw token decode
           -> oracle scalar equalization -> equalized token decode

The scalar equalizer uses the transmitted X_DD and is therefore a diagnostic
oracle, not a deployable receiver. There is no training and no complex DD
equalizer here.
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


def scalar_equalize_oracle(
    reference_dd: torch.Tensor,
    received_dd: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(reference_dd) or not torch.is_tensor(received_dd):
        raise TypeError("reference_dd and received_dd must be torch.Tensor objects.")
    if reference_dd.shape != received_dd.shape:
        raise ValueError(
            "reference_dd and received_dd must have the same shape; "
            f"got {_shape_list(reference_dd)} and {_shape_list(received_dd)}."
        )
    if reference_dd.ndim != 3:
        raise ValueError(
            "reference_dd and received_dd must have shape [B, M, N]; "
            f"got {_shape_list(reference_dd)}."
        )
    if not torch.is_complex(reference_dd) or not torch.is_complex(received_dd):
        raise TypeError("reference_dd and received_dd must be complex tensors.")

    numerator = (received_dd * reference_dd.conj()).sum(dim=(1, 2))
    denominator = reference_dd.abs().pow(2).sum(dim=(1, 2)).clamp_min(eps)
    alpha = numerator / denominator
    safe_alpha = torch.where(
        alpha.abs() < eps,
        torch.full_like(alpha, complex(eps, 0.0)),
        alpha,
    )
    equalized_dd = received_dd / safe_alpha.reshape(-1, 1, 1)
    return equalized_dd, alpha


def distortion_metrics(
    reference_dd: torch.Tensor,
    received_dd: torch.Tensor,
    equalized_dd: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, float]:
    if reference_dd.shape != received_dd.shape or reference_dd.shape != equalized_dd.shape:
        raise ValueError("reference_dd, received_dd, and equalized_dd must share shape.")
    if alpha.ndim not in (1, 3):
        raise ValueError(f"alpha must have shape [B] or [B, 1, 1]; got {_shape_list(alpha)}.")

    return {
        "dd_nmse_raw": normalized_mse(reference_dd, received_dd),
        "dd_nmse_equalized": normalized_mse(reference_dd, equalized_dd),
        "dd_max_abs_error_raw": max_abs_error(reference_dd, received_dd),
        "dd_max_abs_error_equalized": max_abs_error(reference_dd, equalized_dd),
        "alpha_abs_mean": alpha.abs().mean().item(),
        "alpha_phase_mean": torch.angle(alpha).mean().item(),
        "residual_power_raw": (received_dd - reference_dd).abs().pow(2).mean().item(),
        "residual_power_equalized": (equalized_dd - reference_dd).abs().pow(2).mean().item(),
    }


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
        description="Compare raw and oracle scalar equalized token-OTFS-channel decoding.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage6_equalized/token_otfs_channel_equalized.pt"),
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
    print("Note: scalar equalization is oracle-aided because it uses transmitted X_DD. It is for diagnostics, not a deployable receiver.")
    print()
    print("SNR | Acc raw | Acc eq | TER raw | TER eq | NMSE raw | NMSE eq | |alpha|")
    print("------------------------------------------------------------------------")

    results: list[dict[str, float]] = []
    last_details: dict[str, Any] | None = None

    for snr_db in args.snr_db_list:
        acc_raw_values: list[float] = []
        ter_raw_values: list[float] = []
        acc_eq_values: list[float] = []
        ter_eq_values: list[float] = []
        dd_nmse_raw_values: list[float] = []
        dd_nmse_eq_values: list[float] = []
        dd_max_raw_values: list[float] = []
        dd_max_eq_values: list[float] = []
        alpha_abs_values: list[float] = []
        alpha_phase_values: list[float] = []
        residual_raw_values: list[float] = []
        residual_eq_values: list[float] = []

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

            recovered_raw = mapper.decode(y_dd, token_shape=grid_size)
            acc_raw = TokenDDMapper.token_accuracy(token_ids, recovered_raw)
            ter_raw = TokenDDMapper.token_error_rate(token_ids, recovered_raw)

            y_dd_eq, alpha = scalar_equalize_oracle(x_dd, y_dd)
            recovered_eq = mapper.decode(y_dd_eq, token_shape=grid_size)
            acc_eq = TokenDDMapper.token_accuracy(token_ids, recovered_eq)
            ter_eq = TokenDDMapper.token_error_rate(token_ids, recovered_eq)

            diagnostic_dict = distortion_metrics(x_dd, y_dd, y_dd_eq, alpha)

            acc_raw_values.append(acc_raw)
            ter_raw_values.append(ter_raw)
            acc_eq_values.append(acc_eq)
            ter_eq_values.append(ter_eq)
            dd_nmse_raw_values.append(diagnostic_dict["dd_nmse_raw"])
            dd_nmse_eq_values.append(diagnostic_dict["dd_nmse_equalized"])
            dd_max_raw_values.append(diagnostic_dict["dd_max_abs_error_raw"])
            dd_max_eq_values.append(diagnostic_dict["dd_max_abs_error_equalized"])
            alpha_abs_values.append(diagnostic_dict["alpha_abs_mean"])
            alpha_phase_values.append(diagnostic_dict["alpha_phase_mean"])
            residual_raw_values.append(diagnostic_dict["residual_power_raw"])
            residual_eq_values.append(diagnostic_dict["residual_power_equalized"])

            last_details = {
                "x_dd": x_dd,
                "x_time": x_time,
                "y_time": y_time,
                "clean_time": out.clean,
                "noise": out.noise,
                "y_dd_raw": y_dd,
                "y_dd_equalized": y_dd_eq,
                "alpha": alpha,
                "recovered_tokens_raw": recovered_raw,
                "recovered_tokens_equalized": recovered_eq,
                "path_gains": out.path_gains,
                "delays": out.delays,
                "dopplers_hz": out.dopplers_hz,
                "conditioning": out.conditioning,
                "snr_db": float(snr_db),
                "token_accuracy_raw": acc_raw,
                "token_error_rate_raw": ter_raw,
                "token_accuracy_equalized": acc_eq,
                "token_error_rate_equalized": ter_eq,
                "diagnostics": diagnostic_dict,
            }

        acc_raw_mean, acc_raw_std = mean_std(acc_raw_values)
        ter_raw_mean, ter_raw_std = mean_std(ter_raw_values)
        acc_eq_mean, acc_eq_std = mean_std(acc_eq_values)
        ter_eq_mean, ter_eq_std = mean_std(ter_eq_values)
        dd_nmse_raw_mean, dd_nmse_raw_std = mean_std(dd_nmse_raw_values)
        dd_nmse_eq_mean, dd_nmse_eq_std = mean_std(dd_nmse_eq_values)
        dd_max_raw_mean, dd_max_raw_std = mean_std(dd_max_raw_values)
        dd_max_eq_mean, dd_max_eq_std = mean_std(dd_max_eq_values)
        alpha_abs_mean, alpha_abs_std = mean_std(alpha_abs_values)
        alpha_phase_mean, alpha_phase_std = mean_std(alpha_phase_values)
        residual_raw_mean, residual_raw_std = mean_std(residual_raw_values)
        residual_eq_mean, residual_eq_std = mean_std(residual_eq_values)

        result = {
            "snr_db": float(snr_db),
            "token_accuracy_raw_mean": acc_raw_mean,
            "token_accuracy_raw_std": acc_raw_std,
            "token_error_rate_raw_mean": ter_raw_mean,
            "token_error_rate_raw_std": ter_raw_std,
            "token_accuracy_eq_mean": acc_eq_mean,
            "token_accuracy_eq_std": acc_eq_std,
            "token_error_rate_eq_mean": ter_eq_mean,
            "token_error_rate_eq_std": ter_eq_std,
            "dd_nmse_raw_mean": dd_nmse_raw_mean,
            "dd_nmse_raw_std": dd_nmse_raw_std,
            "dd_nmse_eq_mean": dd_nmse_eq_mean,
            "dd_nmse_eq_std": dd_nmse_eq_std,
            "dd_max_abs_error_raw_mean": dd_max_raw_mean,
            "dd_max_abs_error_raw_std": dd_max_raw_std,
            "dd_max_abs_error_eq_mean": dd_max_eq_mean,
            "dd_max_abs_error_eq_std": dd_max_eq_std,
            "alpha_abs_mean": alpha_abs_mean,
            "alpha_abs_std": alpha_abs_std,
            "alpha_phase_mean": alpha_phase_mean,
            "alpha_phase_std": alpha_phase_std,
            "residual_power_raw_mean": residual_raw_mean,
            "residual_power_raw_std": residual_raw_std,
            "residual_power_eq_mean": residual_eq_mean,
            "residual_power_eq_std": residual_eq_std,
        }
        results.append(result)
        print(
            f"{float(snr_db):3.0f} | "
            f"{acc_raw_mean:7.4f} | "
            f"{acc_eq_mean:6.4f} | "
            f"{ter_raw_mean:7.4f} | "
            f"{ter_eq_mean:6.4f} | "
            f"{dd_nmse_raw_mean:8.3e} | "
            f"{dd_nmse_eq_mean:7.3e} | "
            f"{alpha_abs_mean:7.4f}"
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
            "equalizer": "oracle_scalar",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    if args.save_last:
        if last_details is None:
            raise RuntimeError("save-last requested but no trial was executed.")
        last_path = args.output.parent / "token_otfs_channel_equalized_last.pt"
        torch.save(
            {
                "x_dd": last_details["x_dd"].cpu(),
                "x_time": last_details["x_time"].cpu(),
                "y_time": last_details["y_time"].cpu(),
                "clean_time": last_details["clean_time"].cpu(),
                "noise": last_details["noise"].cpu(),
                "y_dd_raw": last_details["y_dd_raw"].cpu(),
                "y_dd_equalized": last_details["y_dd_equalized"].cpu(),
                "alpha": last_details["alpha"].cpu(),
                "recovered_tokens_raw": last_details["recovered_tokens_raw"].cpu(),
                "recovered_tokens_equalized": last_details["recovered_tokens_equalized"].cpu(),
                "path_gains": last_details["path_gains"].cpu(),
                "delays": last_details["delays"].cpu(),
                "dopplers_hz": last_details["dopplers_hz"].cpu(),
                "conditioning": last_details["conditioning"].cpu(),
                "snr_db": last_details["snr_db"],
                "token_accuracy_raw": last_details["token_accuracy_raw"],
                "token_error_rate_raw": last_details["token_error_rate_raw"],
                "token_accuracy_equalized": last_details["token_accuracy_equalized"],
                "token_error_rate_equalized": last_details["token_error_rate_equalized"],
                "diagnostics": last_details["diagnostics"],
            },
            last_path,
        )
        print(f"last trial path: {last_path}")


if __name__ == "__main__":
    main()
