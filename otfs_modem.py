"""Minimal OTFS modulation/demodulation numerical loop.

This module implements only the deterministic transform chain:

    X_DD -> ISFFT -> X_TF -> OFDM modulation -> time waveform
         -> OFDM demodulation -> Y_TF -> SFFT -> Y_DD

There is no wireless channel, AWGN, token decoding, bit conversion, QAM, or
training. The transform definitions are chosen to be self-consistent inverses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


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


def normalized_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    if not torch.is_tensor(reference) or not torch.is_tensor(estimate):
        raise TypeError("reference and estimate must be torch.Tensor objects.")
    if reference.shape != estimate.shape:
        raise ValueError(
            "reference and estimate must have the same shape; "
            f"got {_shape_list(reference)} and {_shape_list(estimate)}."
        )
    denom = reference.abs().pow(2).sum()
    if denom.item() == 0:
        raise ValueError("normalized_mse is undefined for a zero-power reference.")
    error = (estimate - reference).abs().pow(2).sum()
    return (error / denom).item()


def max_abs_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    if not torch.is_tensor(reference) or not torch.is_tensor(estimate):
        raise TypeError("reference and estimate must be torch.Tensor objects.")
    if reference.shape != estimate.shape:
        raise ValueError(
            "reference and estimate must have the same shape; "
            f"got {_shape_list(reference)} and {_shape_list(estimate)}."
        )
    return (estimate - reference).abs().max().item()


class OTFSModem:
    def __init__(
        self,
        dd_shape: tuple[int, int] = (32, 32),
        cp_len: int = 0,
        device: str | None = None,
        dtype: torch.dtype = torch.complex64,
    ):
        if (
            not isinstance(dd_shape, tuple)
            or len(dd_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in dd_shape)
        ):
            raise ValueError("dd_shape must be a tuple of two positive integers.")
        if not isinstance(cp_len, int) or cp_len < 0:
            raise ValueError("cp_len must be a non-negative integer.")
        if cp_len > dd_shape[0]:
            raise ValueError(
                f"cp_len must be <= M ({dd_shape[0]}) so a full cyclic prefix exists; "
                f"got cp_len={cp_len}."
            )
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128.")

        self.dd_shape = dd_shape
        self.M = dd_shape[0]
        self.N = dd_shape[1]
        self.cp_len = cp_len
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

    def _validate_dd_tensor(self, x_dd: torch.Tensor, name: str = "x_dd") -> None:
        if not torch.is_tensor(x_dd):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if x_dd.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, M, N]; got shape {_shape_list(x_dd)}."
            )
        if tuple(x_dd.shape[1:]) != self.dd_shape:
            raise ValueError(
                f"{name} spatial shape must match dd_shape {self.dd_shape}; "
                f"got {tuple(x_dd.shape[1:])}."
            )
        if not torch.is_complex(x_dd):
            raise TypeError(f"{name} must be a complex tensor; got {x_dd.dtype}.")

    def isfft(self, x_dd: torch.Tensor) -> torch.Tensor:
        self._validate_dd_tensor(x_dd, "x_dd")
        x_dd = x_dd.to(device=self.device, dtype=self.dtype)
        # ISFFT: inverse FFT over Doppler slots, then FFT over delay/subcarriers.
        return torch.fft.fft(
            torch.fft.ifft(x_dd, dim=2, norm="ortho"),
            dim=1,
            norm="ortho",
        )

    def sfft(self, x_tf: torch.Tensor) -> torch.Tensor:
        self._validate_dd_tensor(x_tf, "x_tf")
        x_tf = x_tf.to(device=self.device, dtype=self.dtype)
        # SFFT is the inverse of the ISFFT definition above.
        return torch.fft.fft(
            torch.fft.ifft(x_tf, dim=1, norm="ortho"),
            dim=2,
            norm="ortho",
        )

    def modulate(self, x_dd: torch.Tensor) -> torch.Tensor:
        self._validate_dd_tensor(x_dd, "x_dd")
        x_tf = self.isfft(x_dd)

        # OFDM/Heisenberg step: IFFT over subcarriers for each time slot.
        x_symbols = torch.fft.ifft(x_tf, dim=1, norm="ortho")
        x_symbols = x_symbols.transpose(1, 2)  # [B, N, M]

        if self.cp_len > 0:
            cp = x_symbols[:, :, -self.cp_len :]
            x_symbols = torch.cat([cp, x_symbols], dim=-1)

        return x_symbols.reshape(x_symbols.shape[0], -1)

    def demodulate(self, time_signal: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(time_signal):
            raise TypeError("time_signal must be a torch.Tensor.")
        if time_signal.ndim == 1:
            time_signal = time_signal.unsqueeze(0)
        elif time_signal.ndim != 2:
            raise ValueError(
                "time_signal must have shape [B, time] or [time]; "
                f"got shape {_shape_list(time_signal)}."
            )
        if not torch.is_complex(time_signal):
            raise TypeError(f"time_signal must be a complex tensor; got {time_signal.dtype}.")

        expected_len = self.N * (self.M + self.cp_len)
        if time_signal.shape[1] != expected_len:
            raise ValueError(
                f"time_signal length must be N * (M + cp_len) = {expected_len}; "
                f"got {time_signal.shape[1]}."
            )

        time_signal = time_signal.to(device=self.device, dtype=self.dtype)
        y_symbols = time_signal.reshape(time_signal.shape[0], self.N, self.M + self.cp_len)
        if self.cp_len > 0:
            y_symbols = y_symbols[:, :, self.cp_len :]

        y_symbols = y_symbols.transpose(1, 2)  # [B, M, N]
        y_tf = torch.fft.fft(y_symbols, dim=1, norm="ortho")
        return self.sfft(y_tf)

    def roundtrip(self, x_dd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_signal = self.modulate(x_dd)
        recovered_dd = self.demodulate(time_signal)
        return time_signal, recovered_dd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal OTFS modulation/demodulation numerical loop.",
    )
    parser.add_argument("--input-dd", type=Path, required=True)
    parser.add_argument("--output-time", type=Path, required=True)
    parser.add_argument("--output-dd", type=Path, required=True)
    parser.add_argument("--cp-len", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _torch_load(args.input_dd)
    if not isinstance(payload, dict):
        raise ValueError(f"{args.input_dd} must contain a dict.")
    if "dd_grid" not in payload:
        raise KeyError(f"{args.input_dd} is missing required key: 'dd_grid'.")

    dd_grid = payload["dd_grid"]
    if not torch.is_tensor(dd_grid):
        raise TypeError("payload['dd_grid'] must be a torch.Tensor.")
    if dd_grid.ndim != 3:
        raise ValueError(
            "payload['dd_grid'] must have shape [B, M, N]; "
            f"got {_shape_list(dd_grid)}."
        )
    if not torch.is_complex(dd_grid):
        raise TypeError(f"payload['dd_grid'] must be complex; got {dd_grid.dtype}.")

    if "dd_shape" in payload:
        dd_shape = tuple(payload["dd_shape"])
    else:
        dd_shape = tuple(dd_grid.shape[-2:])

    modem = OTFSModem(
        dd_shape=dd_shape,
        cp_len=args.cp_len,
        device=args.device,
        dtype=dd_grid.dtype,
    )
    time_signal, recovered_dd = modem.roundtrip(dd_grid)
    nmse = normalized_mse(dd_grid.to(recovered_dd.device), recovered_dd)
    max_err = max_abs_error(dd_grid.to(recovered_dd.device), recovered_dd)

    input_power = dd_grid.abs().pow(2).mean().item()
    time_power = time_signal.abs().pow(2).mean().item()
    recovered_power = recovered_dd.abs().pow(2).mean().item()

    args.output_time.parent.mkdir(parents=True, exist_ok=True)
    args.output_dd.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time_signal": time_signal.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "input_dd_shape": tuple(dd_grid.shape),
        },
        args.output_time,
    )
    torch.save(
        {
            "dd_grid": recovered_dd.cpu(),
            "source_dd_grid": dd_grid.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "normalized_mse": nmse,
            "max_abs_error": max_err,
        },
        args.output_dd,
    )

    print(f"input DD shape: {_shape_list(dd_grid)}")
    print(f"DD dtype: {dd_grid.dtype}")
    print(f"DD shape M,N: {modem.M},{modem.N}")
    print(f"cp_len: {modem.cp_len}")
    print(f"time signal shape: {_shape_list(time_signal)}")
    print(f"recovered DD shape: {_shape_list(recovered_dd)}")
    print(f"input average power: {input_power}")
    print(f"time average power: {time_power}")
    print(f"recovered average power: {recovered_power}")
    print(f"normalized DD MSE: {nmse}")
    print(f"max abs error: {max_err}")
    print(f"output time path: {args.output_time}")
    print(f"output dd path: {args.output_dd}")


if __name__ == "__main__":
    main()
