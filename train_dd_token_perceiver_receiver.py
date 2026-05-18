"""Training skeleton for the DD-domain Perceiver token receiver.

The default workflow is a dry-run shape/loss check:

    random tokens -> TokenDDMapper -> OTFSModem -> channel -> OTFSModem
                  -> DDTokenPerceiverReceiver -> CE loss

This file provides a minimal training loop skeleton, but it does not start a
long training run by default.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from channel_model import ChannelConfig, TimeVaryingMultipathChannel
from dd_token_perceiver_receiver import DDTokenPerceiverReceiver, count_parameters
from otfs_modem import OTFSModem
from token_dd_mapper import TokenDDMapper


def _shape_list(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def _args_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    if config.get("resume_checkpoint") is not None:
        config["resume_checkpoint"] = str(config["resume_checkpoint"])
    config["token_shape"] = tuple(config["token_shape"])
    config["dd_shape"] = tuple(config["dd_shape"])
    return config


def token_accuracy(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    targets = token_ids.reshape(token_ids.shape[0], -1)
    predictions = logits.argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def grad_norm(parameters: Any) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().float().norm(2)
        squared_norm += float(parameter_norm.item() ** 2)
    return squared_norm**0.5


def trainable_parameter_snapshot(receiver: DDTokenPerceiverReceiver) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in receiver.parameters() if parameter.requires_grad]


def parameter_delta(before: list[torch.Tensor], receiver: DDTokenPerceiverReceiver) -> float:
    if not before:
        return 0.0
    max_delta = 0.0
    index = 0
    for parameter in receiver.parameters():
        if parameter.requires_grad:
            delta = float((parameter.detach() - before[index]).abs().max().item())
            max_delta = max(max_delta, delta)
            index += 1
    return max_delta


def validate_args(args: argparse.Namespace) -> None:
    token_shape = tuple(args.token_shape)
    dd_shape = tuple(args.dd_shape)
    if len(token_shape) != 2 or not all(x > 0 for x in token_shape):
        raise ValueError("token_shape must contain two positive integers.")
    if len(dd_shape) != 2 or not all(x > 0 for x in dd_shape):
        raise ValueError("dd_shape must contain two positive integers.")
    required_bins = token_shape[0] * token_shape[1] * args.symbols_per_token
    available_bins = dd_shape[0] * dd_shape[1]
    if required_bins > available_bins:
        raise ValueError(
            "token grid does not fit in DD grid: "
            f"{token_shape[0]}*{token_shape[1]}*{args.symbols_per_token}="
            f"{required_bins}, but dd_shape has {available_bins} bins."
        )
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if args.eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if args.eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative.")
    if args.snr_db_min > args.snr_db_max:
        raise ValueError("snr_db_min must be <= snr_db_max.")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume_checkpoint}")
    if args.channel_mode == "identity" and args.fixed_channel:
        raise ValueError("--fixed-channel is only meaningful when --channel-mode channel.")
    if args.channel_mode == "identity" and args.no_awgn:
        raise ValueError("--no-awgn is only meaningful when --channel-mode channel.")
    if args.max_channel_paths <= 0:
        raise ValueError("max_channel_paths must be positive.")
    if args.delay_window_radius < 0:
        raise ValueError("delay_window_radius must be non-negative.")


def make_channel(args: argparse.Namespace, snr_db: float) -> TimeVaryingMultipathChannel:
    config = ChannelConfig(
        num_paths=args.num_paths,
        sample_rate=args.sample_rate,
        snr_db=snr_db,
        max_delay_samples=args.max_delay_samples,
        max_doppler_hz=args.max_doppler_hz,
        fading=args.fading,
        rician_k_db=args.rician_k_db,
        doppler_distribution=args.doppler_distribution,
        add_awgn=not args.no_awgn,
        randomize_each_forward=not args.fixed_channel,
        fractional_delays=not args.integer_delays,
        complex_dtype="complex64",
        seed=args.seed,
    )
    return TimeVaryingMultipathChannel(config)


def sample_snr_db(args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> float:
    if args.snr_db_min == args.snr_db_max:
        return float(args.snr_db_min)
    value = torch.empty((), device=device).uniform_(
        float(args.snr_db_min),
        float(args.snr_db_max),
        generator=generator,
    )
    return float(value.item())


def sample_token_batch(
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=args.codebook_size,
        size=(args.batch_size, args.token_shape[0], args.token_shape[1]),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def make_channel_features(
    args: argparse.Namespace,
    path_delays: torch.Tensor | None,
    path_gains: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not args.use_channel_features:
        return None

    features = torch.zeros(
        batch_size,
        args.max_channel_paths,
        3,
        device=device,
        dtype=dtype,
    )
    if path_delays is None or path_gains is None:
        return features

    if path_delays.ndim != 2 or path_gains.ndim != 2:
        raise ValueError("path_delays and path_gains must have shape [B, paths].")
    if path_delays.shape[0] != batch_size or path_gains.shape[0] != batch_size:
        raise ValueError("channel batch size does not match token batch size.")
    num_paths = min(args.max_channel_paths, path_delays.shape[1], path_gains.shape[1])
    delay_scale = max(float(args.max_delay_samples), 1.0)
    features[:, :num_paths, 0] = path_delays[:, :num_paths].to(device=device, dtype=dtype) / delay_scale
    features[:, :num_paths, 1] = path_gains[:, :num_paths].real.to(device=device, dtype=dtype)
    features[:, :num_paths, 2] = path_gains[:, :num_paths].imag.to(device=device, dtype=dtype)
    return features


def forward_batch(
    args: argparse.Namespace,
    mapper: TokenDDMapper,
    modem: OTFSModem,
    channel: TimeVaryingMultipathChannel | None,
    receiver: DDTokenPerceiverReceiver,
    token_ids: torch.Tensor,
    snr_db: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    x_dd = mapper.encode(token_ids)
    x_time = modem.modulate(x_dd)
    path_delays = None
    path_gains = None
    if args.channel_mode == "identity":
        y_time = x_time
    else:
        if channel is None:
            raise RuntimeError("channel must be initialized when channel_mode='channel'.")
        out = channel(x_time, snr_db=snr_db, return_info=True)
        y_time = out.y
        path_delays = out.delays
        path_gains = out.path_gains
    y_dd = modem.demodulate(y_time)
    channel_features = make_channel_features(
        args,
        path_delays,
        path_gains,
        batch_size=token_ids.shape[0],
        device=token_ids.device,
        dtype=y_dd.real.dtype,
    )
    logits = receiver(y_dd, channel_features=channel_features)
    targets = token_ids.reshape(token_ids.shape[0], -1)
    loss = F.cross_entropy(
        logits.reshape(-1, args.codebook_size),
        targets.reshape(-1),
    )
    return x_dd, x_time, y_time, y_dd, channel_features, logits, loss


def make_checkpoint(
    args: argparse.Namespace,
    step: int,
    receiver: DDTokenPerceiverReceiver,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    log_rows: list[dict[str, float]],
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "step": step,
        "receiver_state_dict": receiver.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _args_config(args),
        "log": log_rows,
        "torch_rng_state": torch.get_rng_state(),
        "generator_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return checkpoint


def best_eval_from_log(log_rows: list[dict[str, float]]) -> tuple[float, int]:
    best_accuracy = float("-inf")
    best_step = 0
    for row in log_rows:
        if "eval_token_accuracy" not in row or "step" not in row:
            continue
        accuracy = float(row["eval_token_accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_step = int(row["step"])
    return best_accuracy, best_step


def restore_checkpoint(
    args: argparse.Namespace,
    receiver: DDTokenPerceiverReceiver,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[int, list[dict[str, float]]]:
    if args.resume_checkpoint is None:
        return 1, []

    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
    receiver.load_state_dict(checkpoint["receiver_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        parameter_device = next(receiver.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=parameter_device)

    log_rows = checkpoint.get("log", [])
    if not isinstance(log_rows, list):
        log_rows = []

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "generator_state" in checkpoint:
        try:
            generator.set_state(checkpoint["generator_state"])
        except RuntimeError as exc:
            print(f"warning: could not restore local generator state: {exc}")
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    resume_step = int(checkpoint.get("step", 0))
    start_step = resume_step + 1
    print(f"resumed checkpoint: {args.resume_checkpoint}")
    print(f"resume step: {resume_step}; next step: {start_step}")
    return start_step, log_rows


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    mapper: TokenDDMapper,
    modem: OTFSModem,
    channel: TimeVaryingMultipathChannel | None,
    receiver: DDTokenPerceiverReceiver,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, float]:
    receiver.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    snrs: list[float] = []
    num_batches = max(1, args.eval_batches)
    for _ in range(num_batches):
        token_ids = sample_token_batch(args, generator, device)
        snr_db = sample_snr_db(args, generator, device)
        _x_dd, _x_time, _y_time, _y_dd, _channel_features, logits, loss = forward_batch(
            args,
            mapper,
            modem,
            channel,
            receiver,
            token_ids,
            snr_db,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"eval loss is not finite: {loss.item()}.")
        losses.append(float(loss.item()))
        accuracies.append(token_accuracy(logits, token_ids))
        snrs.append(float(snr_db))

    return {
        "eval_loss": sum(losses) / len(losses),
        "eval_token_accuracy": sum(accuracies) / len(accuracies),
        "eval_snr_db_mean": sum(snrs) / len(snrs),
    }


def run_dry_run(args: argparse.Namespace) -> None:
    validate_args(args)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    dd_shape = tuple(args.dd_shape)
    token_shape = tuple(args.token_shape)
    mapper = TokenDDMapper(
        codebook_size=args.codebook_size,
        symbols_per_token=args.symbols_per_token,
        dd_shape=dd_shape,
        seed=args.seed,
        device=str(device),
    )
    modem = OTFSModem(dd_shape=dd_shape, cp_len=args.cp_len, device=str(device))
    channel = None
    if args.channel_mode == "channel":
        channel = make_channel(args, snr_db=float(args.snr_db_max)).to(device)
    receiver = DDTokenPerceiverReceiver(
        codebook_size=args.codebook_size,
        dd_shape=dd_shape,
        token_shape=token_shape,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        self_attn_layers=args.self_attn_layers,
        dropout=args.dropout,
        symbols_per_token=args.symbols_per_token,
        use_packed_local=args.use_packed_local,
        use_channel_features=args.use_channel_features,
        max_channel_paths=args.max_channel_paths,
        use_delay_window_local=args.use_delay_window_local,
        delay_window_radius=args.delay_window_radius,
    ).to(device)

    token_ids = sample_token_batch(args, generator, device)
    snr_db = sample_snr_db(args, generator, device)
    x_dd, x_time, y_time, y_dd, channel_features, logits, loss = forward_batch(
        args,
        mapper,
        modem,
        channel,
        receiver,
        token_ids,
        snr_db,
    )
    targets = token_ids.reshape(args.batch_size, -1)
    if not torch.isfinite(loss):
        raise RuntimeError(f"dry-run loss is not finite: {loss.item()}.")

    parameter_count = count_parameters(receiver)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "perceiver_dry_run.pt"
    torch.save(
        {
            "token_ids_shape": tuple(token_ids.shape),
            "x_dd_shape": tuple(x_dd.shape),
            "x_time_shape": tuple(x_time.shape),
            "y_time_shape": tuple(y_time.shape),
            "y_dd_shape": tuple(y_dd.shape),
            "channel_features_shape": None if channel_features is None else tuple(channel_features.shape),
            "logits_shape": tuple(logits.shape),
            "target_shape": tuple(targets.shape),
            "loss": float(loss.item()),
            "receiver_parameter_count": int(parameter_count),
            "config": _args_config(args),
        },
        output_path,
    )

    print(f"token_ids shape: {_shape_list(token_ids)}")
    print(f"x_dd shape: {_shape_list(x_dd)}")
    print(f"x_time shape: {_shape_list(x_time)}")
    print(f"y_time shape: {_shape_list(y_time)}")
    print(f"y_dd shape: {_shape_list(y_dd)}")
    if channel_features is not None:
        print(f"channel_features shape: {_shape_list(channel_features)}")
    print(f"logits shape: {_shape_list(logits)}")
    print(f"target shape: {_shape_list(targets)}")
    print(f"loss value: {float(loss.item())}")
    print(f"receiver parameter count: {parameter_count}")
    print(f"device: {device}")
    print(f"dtype: x_dd={x_dd.dtype}, y_dd={y_dd.dtype}, logits={logits.dtype}")
    print(f"dry-run output path: {output_path}")


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_steps == 0:
        print("num_steps is 0; no training will be run.")
        return

    dd_shape = tuple(args.dd_shape)
    token_shape = tuple(args.token_shape)
    mapper = TokenDDMapper(
        codebook_size=args.codebook_size,
        symbols_per_token=args.symbols_per_token,
        dd_shape=dd_shape,
        seed=args.seed,
        device=str(device),
    )
    modem = OTFSModem(dd_shape=dd_shape, cp_len=args.cp_len, device=str(device))
    channel = None
    if args.channel_mode == "channel":
        channel = make_channel(args, snr_db=float(args.snr_db_max)).to(device)
    receiver = DDTokenPerceiverReceiver(
        codebook_size=args.codebook_size,
        dd_shape=dd_shape,
        token_shape=token_shape,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        self_attn_layers=args.self_attn_layers,
        dropout=args.dropout,
        symbols_per_token=args.symbols_per_token,
        use_packed_local=args.use_packed_local,
        use_channel_features=args.use_channel_features,
        max_channel_paths=args.max_channel_paths,
        use_delay_window_local=args.use_delay_window_local,
        delay_window_radius=args.delay_window_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(
        receiver.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step, log_rows = restore_checkpoint(args, receiver, optimizer, generator)
    best_eval_accuracy, best_eval_step = best_eval_from_log(log_rows)
    if start_step > args.num_steps:
        print(
            f"resume checkpoint is already at step {start_step - 1}, "
            f"which is >= requested num_steps {args.num_steps}; no additional training will be run."
        )
    start_time = time.perf_counter()
    for step in range(start_step, args.num_steps + 1):
        receiver.train()
        token_ids = sample_token_batch(args, generator, device)
        snr_db = sample_snr_db(args, generator, device)
        _x_dd, _x_time, _y_time, _y_dd, _channel_features, logits, loss = forward_batch(
            args,
            mapper,
            modem,
            channel,
            receiver,
            token_ids,
            snr_db,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"train loss is not finite at step {step}: {loss.item()}.")
        train_accuracy = token_accuracy(logits, token_ids)
        params_before = trainable_parameter_snapshot(receiver)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        current_grad_norm = grad_norm(receiver.parameters())
        if not torch.isfinite(torch.tensor(current_grad_norm)):
            raise RuntimeError(f"grad_norm is not finite at step {step}: {current_grad_norm}.")
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(receiver.parameters(), args.grad_clip)
        optimizer.step()
        param_delta = parameter_delta(params_before, receiver)

        if step == 1 or step % args.eval_every == 0:
            eval_metrics = evaluate(
                args,
                mapper,
                modem,
                channel,
                receiver,
                generator,
                device,
            )
            elapsed_sec = time.perf_counter() - start_time
            completed_steps = step - start_step + 1
            log_row = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "train_token_accuracy": float(train_accuracy),
                "eval_loss": float(eval_metrics["eval_loss"]),
                "eval_token_accuracy": float(eval_metrics["eval_token_accuracy"]),
                "snr_db": float(snr_db),
                "eval_snr_db_mean": float(eval_metrics["eval_snr_db_mean"]),
                "grad_norm": float(current_grad_norm),
                "param_delta": float(param_delta),
                "elapsed_sec": float(elapsed_sec),
                "sec_per_step": float(elapsed_sec / max(1, completed_steps)),
            }
            print(
                f"step {step} | train_loss {log_row['train_loss']:.6f} "
                f"| train_acc {log_row['train_token_accuracy']:.4f} "
                f"| eval_loss {log_row['eval_loss']:.6f} "
                f"| eval_acc {log_row['eval_token_accuracy']:.4f} "
                f"| grad_norm {log_row['grad_norm']:.4f} "
                f"| param_delta {log_row['param_delta']:.3e} "
                f"| snr_db {snr_db:.2f}"
            )
            log_rows.append(log_row)
            if log_row["eval_token_accuracy"] > best_eval_accuracy:
                best_eval_accuracy = log_row["eval_token_accuracy"]
                best_eval_step = step
                best_checkpoint = make_checkpoint(args, step, receiver, optimizer, generator, log_rows)
                best_checkpoint["best_eval_token_accuracy"] = float(best_eval_accuracy)
                best_checkpoint["best_step"] = int(best_eval_step)
                torch.save(best_checkpoint, args.output_dir / "receiver_best.pt")
                print(
                    f"new best eval_acc {best_eval_accuracy:.4f} "
                    f"at step {best_eval_step}; saved receiver_best.pt"
                )
            torch.save(
                {
                    "log": log_rows,
                    "config": _args_config(args),
                    "best_eval_token_accuracy": float(best_eval_accuracy),
                    "best_step": int(best_eval_step),
                },
                args.output_dir / "training_log.pt",
            )

        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint_path = args.output_dir / f"checkpoint_step_{step}.pt"
            checkpoint = make_checkpoint(args, step, receiver, optimizer, generator, log_rows)
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, args.output_dir / "receiver_checkpoint.pt")

    final_path = args.output_dir / "receiver_final.pt"
    training_log_path = args.output_dir / "training_log.pt"
    torch.save(
        {
            "receiver_state_dict": receiver.state_dict(),
            "config": _args_config(args),
            "parameter_count": count_parameters(receiver),
            "log": log_rows,
            "best_eval_token_accuracy": None if best_eval_accuracy == float("-inf") else float(best_eval_accuracy),
            "best_step": int(best_eval_step),
        },
        final_path,
    )
    torch.save(
        {
            "log": log_rows,
            "config": _args_config(args),
            "best_eval_token_accuracy": None if best_eval_accuracy == float("-inf") else float(best_eval_accuracy),
            "best_step": int(best_eval_step),
        },
        training_log_path,
    )
    torch.save(
        make_checkpoint(args, args.num_steps, receiver, optimizer, generator, log_rows),
        args.output_dir / "receiver_checkpoint.pt",
    )
    print(f"final checkpoint path: {final_path}")
    print(f"training log path: {training_log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DD Token Perceiver receiver training skeleton.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage7_perceiver_receiver"))
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--token-shape", type=int, nargs=2, metavar=("H", "W"), default=(16, 16))
    parser.add_argument("--symbols-per-token", type=int, default=4)
    parser.add_argument("--dd-shape", type=int, nargs=2, metavar=("M", "N"), default=(32, 32))
    parser.add_argument("--cp-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--self-attn-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-packed-local", action="store_true")
    parser.add_argument("--use-channel-features", action="store_true")
    parser.add_argument("--max-channel-paths", type=int, default=3)
    parser.add_argument("--use-delay-window-local", action="store_true")
    parser.add_argument("--delay-window-radius", type=int, default=0)
    parser.add_argument("--snr-db-min", type=float, default=15.0)
    parser.add_argument("--snr-db-max", type=float, default=30.0)
    parser.add_argument("--channel-mode", type=str, default="channel", choices=["channel", "identity"])
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=15.36e6)
    parser.add_argument("--max-delay-samples", type=float, default=3.0)
    parser.add_argument("--max-doppler-hz", type=float, default=500.0)
    parser.add_argument("--fading", type=str, default="rayleigh", choices=["rayleigh", "rician", "fixed"])
    parser.add_argument("--rician-k-db", type=float, default=8.0)
    parser.add_argument("--doppler-distribution", type=str, default="jakes", choices=["jakes", "uniform"])
    parser.add_argument("--fixed-channel", action="store_true")
    parser.add_argument("--no-awgn", action="store_true")
    parser.add_argument("--integer-delays", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
