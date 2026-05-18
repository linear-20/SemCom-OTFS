"""Direct visual-token to delay-Doppler grid mapper.

This module intentionally stops at the smallest closed loop:

    visual token ids -> complex DD grid -> recovered visual token ids

There is no OTFS modulation, channel model, bit conversion, QAM mapping, or
training. Each token id indexes a fixed random complex codeword, and decoding
uses nearest-neighbor matching against the same deterministic codebook.
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


class TokenDDMapper:
    def __init__(
        self,
        codebook_size: int = 16384,
        symbols_per_token: int = 4,
        dd_shape: tuple[int, int] = (32, 32),
        seed: int = 0,
        device: str | None = None,
        dtype: torch.dtype = torch.complex64,
    ):
        if not isinstance(codebook_size, int) or codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if not isinstance(symbols_per_token, int) or symbols_per_token <= 0:
            raise ValueError("symbols_per_token must be a positive integer.")
        if (
            not isinstance(dd_shape, tuple)
            or len(dd_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in dd_shape)
        ):
            raise ValueError("dd_shape must be a tuple of two positive integers.")
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128.")

        self.codebook_size = codebook_size
        self.symbols_per_token = symbols_per_token
        self.dd_shape = dd_shape
        self.num_dd_bins = dd_shape[0] * dd_shape[1]
        self.seed = seed
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.codebook = self._build_codebook()

    def _build_codebook(self) -> torch.Tensor:
        real_dtype = torch.float32 if self.dtype == torch.complex64 else torch.float64
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        real = torch.randn(
            self.codebook_size,
            self.symbols_per_token,
            generator=generator,
            dtype=real_dtype,
        )
        imag = torch.randn(
            self.codebook_size,
            self.symbols_per_token,
            generator=generator,
            dtype=real_dtype,
        )
        codebook = torch.complex(real, imag).to(dtype=self.dtype)

        # Normalize each token codeword so its K complex symbols have mean power 1.
        power = codebook.abs().pow(2).mean(dim=1, keepdim=True)
        eps = torch.finfo(real_dtype).eps
        codebook = codebook / torch.sqrt(power.clamp_min(eps))
        return codebook.to(self.device)

    def encode(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if not torch.is_tensor(token_ids):
            raise TypeError("token_ids must be a torch.Tensor.")
        if token_ids.ndim not in (2, 3):
            raise ValueError(
                "token_ids must have shape [B, N] or [B, Ht, Wt]; "
                f"got shape {_shape_list(token_ids)}."
            )
        if not token_ids.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(f"token_ids must be an integer tensor; got {token_ids.dtype}.")
        if token_ids.numel() == 0:
            raise ValueError("token_ids must not be empty.")

        token_min = int(token_ids.min().item())
        token_max = int(token_ids.max().item())
        if token_min < 0 or token_max >= self.codebook_size:
            raise ValueError(
                "token_ids values must be in [0, codebook_size - 1]; "
                f"got min={token_min}, max={token_max}, codebook_size={self.codebook_size}."
            )

        batch_size = token_ids.shape[0]
        flat_tokens = token_ids.reshape(batch_size, -1).to(
            device=self.device,
            dtype=torch.long,
        )
        num_tokens = flat_tokens.shape[1]
        num_symbols = num_tokens * self.symbols_per_token
        if num_symbols > self.num_dd_bins:
            raise ValueError(
                "token sequence does not fit in DD grid: "
                f"{num_tokens} tokens * {self.symbols_per_token} symbols/token "
                f"= {num_symbols} symbols, but dd_shape {self.dd_shape} has "
                f"{self.num_dd_bins} bins."
            )

        # Lookup maps token ids directly to complex DD-domain codewords.
        mapped = self.codebook[flat_tokens].reshape(batch_size, num_symbols)
        if num_symbols < self.num_dd_bins:
            padded = torch.zeros(
                batch_size,
                self.num_dd_bins,
                device=self.device,
                dtype=self.dtype,
            )
            padded[:, :num_symbols] = mapped
            mapped = padded

        return mapped.reshape(batch_size, self.dd_shape[0], self.dd_shape[1])

    def decode(
        self,
        dd_grid: torch.Tensor,
        token_shape: tuple[int, int] | None = None,
        num_tokens: int | None = None,
    ) -> torch.LongTensor:
        if not torch.is_tensor(dd_grid):
            raise TypeError("dd_grid must be a torch.Tensor.")
        if dd_grid.ndim != 3:
            raise ValueError(
                f"dd_grid must have shape [B, M, N]; got shape {_shape_list(dd_grid)}."
            )
        if not torch.is_complex(dd_grid):
            raise TypeError(f"dd_grid must be a complex tensor; got {dd_grid.dtype}.")
        if tuple(dd_grid.shape[1:]) != self.dd_shape:
            raise ValueError(
                f"dd_grid spatial shape must match dd_shape {self.dd_shape}; "
                f"got {tuple(dd_grid.shape[1:])}."
            )

        if token_shape is not None:
            if (
                not isinstance(token_shape, tuple)
                or len(token_shape) != 2
                or not all(isinstance(x, int) and x > 0 for x in token_shape)
            ):
                raise ValueError("token_shape must be a tuple of two positive integers.")
            inferred_num_tokens = token_shape[0] * token_shape[1]
            if num_tokens is not None and num_tokens != inferred_num_tokens:
                raise ValueError(
                    "num_tokens does not match token_shape: "
                    f"num_tokens={num_tokens}, token_shape={token_shape}."
                )
            num_tokens = inferred_num_tokens
        elif num_tokens is None:
            num_tokens = self.num_dd_bins // self.symbols_per_token

        if not isinstance(num_tokens, int) or num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer.")

        num_symbols = num_tokens * self.symbols_per_token
        if num_symbols > self.num_dd_bins:
            raise ValueError(
                "requested tokens do not fit in DD grid: "
                f"{num_tokens} tokens * {self.symbols_per_token} symbols/token "
                f"= {num_symbols} symbols, but dd_shape {self.dd_shape} has "
                f"{self.num_dd_bins} bins."
            )

        flat_grid = dd_grid.to(device=self.device, dtype=self.dtype).reshape(
            dd_grid.shape[0],
            self.num_dd_bins,
        )
        received = flat_grid[:, :num_symbols].reshape(
            dd_grid.shape[0],
            num_tokens,
            self.symbols_per_token,
        )

        # Nearest-neighbor recovery in complex space, chunked over codebook rows.
        recovered = self._nearest_codeword_indices(received)
        if token_shape is not None:
            return recovered.reshape(dd_grid.shape[0], token_shape[0], token_shape[1])
        return recovered

    def _nearest_codeword_indices(
        self,
        received: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.LongTensor:
        flat_received = received.reshape(-1, self.symbols_per_token)
        recv_norm = flat_received.abs().pow(2).sum(dim=1)
        best_dist = torch.full(
            (flat_received.shape[0],),
            float("inf"),
            device=self.device,
            dtype=torch.float32 if self.dtype == torch.complex64 else torch.float64,
        )
        best_idx = torch.zeros(
            flat_received.shape[0],
            device=self.device,
            dtype=torch.long,
        )

        for start in range(0, self.codebook_size, chunk_size):
            stop = min(start + chunk_size, self.codebook_size)
            codebook_chunk = self.codebook[start:stop]
            code_norm = codebook_chunk.abs().pow(2).sum(dim=1)
            inner = flat_received @ codebook_chunk.conj().transpose(0, 1)
            dist = recv_norm[:, None] + code_norm[None, :] - 2.0 * inner.real
            chunk_dist, chunk_idx = dist.min(dim=1)
            update = chunk_dist < best_dist
            best_dist[update] = chunk_dist[update]
            best_idx[update] = chunk_idx[update] + start

        return best_idx.reshape(received.shape[0], received.shape[1])

    def roundtrip(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        dd_grid = self.encode(token_ids)
        token_shape = tuple(token_ids.shape[1:]) if token_ids.ndim == 3 else None
        recovered = self.decode(
            dd_grid,
            token_shape=token_shape,
            num_tokens=token_ids.shape[1] if token_ids.ndim == 2 else None,
        )
        return dd_grid, recovered

    @staticmethod
    def token_accuracy(original: torch.Tensor, recovered: torch.Tensor) -> float:
        if not torch.is_tensor(original) or not torch.is_tensor(recovered):
            raise TypeError("original and recovered must be torch.Tensor objects.")
        if original.shape != recovered.shape:
            raise ValueError(
                "original and recovered must have the same shape; "
                f"got {_shape_list(original)} and {_shape_list(recovered)}."
            )
        if original.numel() == 0:
            raise ValueError("original and recovered must not be empty.")
        return (original.cpu().long() == recovered.cpu().long()).float().mean().item()

    @staticmethod
    def token_error_rate(original: torch.Tensor, recovered: torch.Tensor) -> float:
        return 1.0 - TokenDDMapper.token_accuracy(original, recovered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map visual token ids to a complex DD grid and recover them.",
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-dd", type=Path, required=True)
    parser.add_argument("--output-tokens", type=Path, required=True)
    parser.add_argument("--symbols-per-token", type=int, default=4)
    parser.add_argument("--dd-shape", type=int, nargs=2, metavar=("M", "N"), default=(32, 32))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _torch_load(args.tokens)
    if not isinstance(payload, dict):
        raise ValueError(f"{args.tokens} must contain a dict.")
    required_keys = {
        "token_ids",
        "codebook_size",
        "grid_size",
        "image_size",
        "downsample_ratio",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise KeyError(f"{args.tokens} is missing required keys: {missing}.")

    token_ids = payload["token_ids"]
    if not torch.is_tensor(token_ids):
        raise TypeError("payload['token_ids'] must be a torch.Tensor.")
    if token_ids.ndim != 3:
        raise ValueError(
            "CLI expects payload['token_ids'] shape [B, Ht, Wt]; "
            f"got {_shape_list(token_ids)}."
        )

    grid_size = tuple(payload["grid_size"])
    token_shape = tuple(token_ids.shape[1:])
    if grid_size != token_shape:
        raise ValueError(
            f"payload grid_size {grid_size} does not match token shape {token_shape}."
        )

    mapper = TokenDDMapper(
        codebook_size=int(payload["codebook_size"]),
        symbols_per_token=args.symbols_per_token,
        dd_shape=tuple(args.dd_shape),
        seed=args.seed,
        device=args.device,
    )
    dd_grid = mapper.encode(token_ids)
    recovered = mapper.decode(dd_grid, token_shape=token_shape)
    accuracy = TokenDDMapper.token_accuracy(token_ids, recovered)
    error_rate = TokenDDMapper.token_error_rate(token_ids, recovered)
    average_power = dd_grid.abs().pow(2).mean().item()

    args.output_dd.parent.mkdir(parents=True, exist_ok=True)
    args.output_tokens.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dd_grid": dd_grid.cpu(),
            "dd_shape": tuple(args.dd_shape),
            "symbols_per_token": args.symbols_per_token,
            "codebook_size": int(payload["codebook_size"]),
            "token_shape": token_shape,
            "seed": args.seed,
        },
        args.output_dd,
    )
    torch.save(
        {
            "token_ids": recovered.cpu(),
            "grid_size": grid_size,
            "codebook_size": int(payload["codebook_size"]),
            "image_size": int(payload["image_size"]),
            "downsample_ratio": int(payload["downsample_ratio"]),
        },
        args.output_tokens,
    )

    print(f"original token shape: {_shape_list(token_ids)}")
    print(f"token min/max: {int(token_ids.min().item())}/{int(token_ids.max().item())}")
    print(f"codebook size: {mapper.codebook_size}")
    print(f"symbols per token: {mapper.symbols_per_token}")
    print(f"dd_grid shape: {_shape_list(dd_grid)}")
    print(f"recovered token shape: {_shape_list(recovered)}")
    print(f"average DD power: {average_power}")
    print(f"token accuracy: {accuracy}")
    print(f"token error rate: {error_rate}")
    print(f"output dd path: {args.output_dd}")
    print(f"output tokens path: {args.output_tokens}")


if __name__ == "__main__":
    main()
