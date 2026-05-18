"""Re-evaluate a trained DD-token receiver checkpoint.

This script reloads a receiver checkpoint and runs the same synthetic token ->
DD -> OTFS -> channel -> receiver chain used during training, but with a larger
number of eval batches for a less noisy accuracy estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from dd_token_perceiver_receiver import DDTokenPerceiverReceiver, count_parameters
from otfs_modem import OTFSModem
from token_dd_mapper import TokenDDMapper
from train_dd_token_perceiver_receiver import evaluate, make_channel, validate_args


def _default_config() -> dict[str, Any]:
    return {
        "output_dir": "outputs/stage7_perceiver_receiver",
        "codebook_size": 1024,
        "token_shape": (16, 16),
        "symbols_per_token": 4,
        "dd_shape": (32, 32),
        "cp_len": 4,
        "batch_size": 8,
        "num_steps": 0,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "embed_dim": 256,
        "num_heads": 8,
        "self_attn_layers": 4,
        "dropout": 0.1,
        "use_packed_local": False,
        "use_channel_features": False,
        "max_channel_paths": 3,
        "snr_db_min": 15.0,
        "snr_db_max": 30.0,
        "channel_mode": "channel",
        "num_paths": 3,
        "sample_rate": 15.36e6,
        "max_delay_samples": 3.0,
        "max_doppler_hz": 500.0,
        "fading": "rayleigh",
        "rician_k_db": 8.0,
        "doppler_distribution": "jakes",
        "fixed_channel": False,
        "no_awgn": False,
        "integer_delays": False,
        "seed": 0,
        "device": None,
        "eval_every": 100,
        "eval_batches": 5,
        "save_every": 500,
        "resume_checkpoint": None,
        "dry_run": False,
    }


def _load_eval_args(cli_args: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    config = _default_config()
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict):
        config.update(checkpoint_config)

    config["output_dir"] = Path(config["output_dir"])
    config["token_shape"] = tuple(config["token_shape"])
    config["dd_shape"] = tuple(config["dd_shape"])
    config["resume_checkpoint"] = None
    config["num_steps"] = 0
    config["eval_batches"] = cli_args.eval_batches
    if cli_args.batch_size is not None:
        config["batch_size"] = cli_args.batch_size
    if cli_args.seed is not None:
        config["seed"] = cli_args.seed
    config["device"] = cli_args.device
    if cli_args.snr_db_min is not None:
        config["snr_db_min"] = cli_args.snr_db_min
    if cli_args.snr_db_max is not None:
        config["snr_db_max"] = cli_args.snr_db_max
    if cli_args.no_awgn is not None:
        config["no_awgn"] = cli_args.no_awgn
    if cli_args.max_doppler_hz is not None:
        config["max_doppler_hz"] = cli_args.max_doppler_hz
    if cli_args.max_delay_samples is not None:
        config["max_delay_samples"] = cli_args.max_delay_samples

    return argparse.Namespace(**config)


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a DD-token receiver checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--snr-db-min", type=float, default=None)
    parser.add_argument("--snr-db-max", type=float, default=None)
    parser.add_argument("--max-delay-samples", type=float, default=None)
    parser.add_argument("--max-doppler-hz", type=float, default=None)
    awgn_group = parser.add_mutually_exclusive_group()
    awgn_group.add_argument("--no-awgn", dest="no_awgn", action="store_true")
    awgn_group.add_argument("--with-awgn", dest="no_awgn", action="store_false")
    parser.set_defaults(no_awgn=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    if not cli_args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {cli_args.checkpoint}")
    if cli_args.eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")
    eval_args = _load_eval_args(cli_args, checkpoint)
    validate_args(eval_args)

    device = _resolve_device(eval_args.device)
    torch.manual_seed(eval_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(eval_args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(eval_args.seed)

    dd_shape = tuple(eval_args.dd_shape)
    token_shape = tuple(eval_args.token_shape)
    mapper = TokenDDMapper(
        codebook_size=eval_args.codebook_size,
        symbols_per_token=eval_args.symbols_per_token,
        dd_shape=dd_shape,
        seed=eval_args.seed,
        device=str(device),
    )
    modem = OTFSModem(dd_shape=dd_shape, cp_len=eval_args.cp_len, device=str(device))
    channel = None
    if eval_args.channel_mode == "channel":
        channel = make_channel(eval_args, snr_db=float(eval_args.snr_db_max)).to(device)
    receiver = DDTokenPerceiverReceiver(
        codebook_size=eval_args.codebook_size,
        dd_shape=dd_shape,
        token_shape=token_shape,
        embed_dim=eval_args.embed_dim,
        num_heads=eval_args.num_heads,
        self_attn_layers=eval_args.self_attn_layers,
        dropout=eval_args.dropout,
        symbols_per_token=eval_args.symbols_per_token,
        use_packed_local=eval_args.use_packed_local,
        use_channel_features=eval_args.use_channel_features,
        max_channel_paths=eval_args.max_channel_paths,
    ).to(device)
    receiver.load_state_dict(checkpoint["receiver_state_dict"])

    metrics = evaluate(eval_args, mapper, modem, channel, receiver, generator, device)
    result = {
        "checkpoint": str(cli_args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_step": int(checkpoint.get("best_step", -1)),
        "checkpoint_best_eval_token_accuracy": checkpoint.get("best_eval_token_accuracy"),
        "metrics": metrics,
        "config": vars(eval_args).copy(),
        "parameter_count": count_parameters(receiver),
    }
    result["config"]["output_dir"] = str(result["config"]["output_dir"])

    output_path = cli_args.output
    if output_path is None:
        output_path = cli_args.checkpoint.with_name(f"{cli_args.checkpoint.stem}_reeval.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    print(f"checkpoint: {cli_args.checkpoint}")
    print(f"checkpoint step: {result['checkpoint_step']}")
    print(f"eval batches: {eval_args.eval_batches}")
    print(f"batch size: {eval_args.batch_size}")
    print(f"device: {device}")
    print(f"eval_loss: {metrics['eval_loss']:.6f}")
    print(f"eval_token_accuracy: {metrics['eval_token_accuracy']:.6f}")
    print(f"eval_snr_db_mean: {metrics['eval_snr_db_mean']:.3f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
