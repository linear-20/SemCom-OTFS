"""Minimal image <-> visual token wrapper for LlamaGen VQ tokenizers.

This module intentionally contains no OTFS or wireless-channel logic.  It is a
thin adapter around LlamaGen's pretrained VQ-VAE/VQGAN tokenizer interface:

    image -> token ids -> reconstructed image

Expected LlamaGen model API:
    from tokenizer.tokenizer_image.vq_model import VQ_models
    latent, _, [_, _, indices] = model.encode(x)      # x in [-1, 1]
    image = model.decode_code(indices, latent.shape)  # output in [-1, 1]
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


LLAMAGEN_IMPORT_CANDIDATES = (
    "tokenizer.tokenizer_image.vq_model",
    "tokenizer_image.vq_model",
    "vq_model",
)


def _torch_load(path: str | os.PathLike[str]) -> Any:
    """Load a checkpoint on CPU while staying compatible with older PyTorch."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as weights_only_error:  # noqa: BLE001 - report both attempts clearly.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as full_load_error:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load checkpoint with torch.load(). The weights-only "
                f"load error was: {weights_only_error}. The full load error was: "
                f"{full_load_error}."
            ) from full_load_error


def _select_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Select the model weights from common LlamaGen checkpoint layouts."""

    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not find model weights in checkpoint. Expected one of the keys "
        "'ema', 'model', or 'state_dict', or a raw PyTorch state_dict."
    )


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _infer_codebook_shape(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int | None, int | None]:
    for key, value in state_dict.items():
        if key.endswith("quantize.embedding.weight") and value.ndim == 2:
            return int(value.shape[0]), int(value.shape[1])
    return None, None


def _normalise_model_type(model_type: str, available_models: dict[str, Any]) -> str:
    """Map friendly names like 'vqvae' to LlamaGen's concrete VQ model name."""

    if model_type in available_models:
        return model_type

    lower = model_type.lower()
    case_map = {name.lower(): name for name in available_models}
    if lower in case_map:
        return case_map[lower]

    if lower in {"vqvae", "vq-vae", "vqgan", "vq-gan"}:
        if "VQ-16" in available_models:
            return "VQ-16"
        if available_models:
            return next(iter(available_models))

    raise ValueError(
        f"Unsupported model_type={model_type!r}. Available LlamaGen VQ models: "
        f"{sorted(available_models)}. Use model_type='vqvae' only when a default "
        "VQ model such as 'VQ-16' is available."
    )


def _import_llamagen_vq_models(
    llamagen_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Import LlamaGen VQ_models from the current project or an explicit root."""

    if llamagen_root is not None:
        root = str(Path(llamagen_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    errors: list[str] = []
    for module_name in LLAMAGEN_IMPORT_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            models = getattr(module, "VQ_models")
            if not isinstance(models, dict) or not models:
                raise AttributeError("VQ_models is missing or empty")
            return models
        except Exception as exc:  # noqa: BLE001 - preserve all import failures.
            errors.append(f"{module_name}: {exc}")

    joined_errors = "\n  - ".join(errors)
    raise ImportError(
        "Could not import LlamaGen VQ_models.\n"
        "Put the LlamaGen repository on PYTHONPATH, run from a checkout that "
        "contains tokenizer/tokenizer_image/vq_model.py, or pass "
        "--llamagen-root /path/to/LlamaGen.\n"
        f"Tried:\n  - {joined_errors}"
    )


def _center_crop_resize_pil(image: Image.Image, image_size: int) -> Image.Image:
    """Match LlamaGen's center-crop-to-square preprocessing intent."""

    image = image.convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return ImageOps.fit(
        image,
        (image_size, image_size),
        method=resample,
        centering=(0.5, 0.5),
    )


def _tensor_center_crop_resize(x: torch.Tensor, image_size: int) -> torch.Tensor:
    _, _, height, width = x.shape
    crop = min(height, width)
    top = (height - crop) // 2
    left = (width - crop) // 2
    x = x[:, :, top : top + crop, left : left + crop]
    if crop != image_size:
        x = F.interpolate(
            x,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )
    return x


class ImageTokenizer:
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vqvae",
        image_size: int = 256,
        device: str | None = None,
        *,
        codebook_size: int | None = None,
        codebook_embed_dim: int | None = None,
        llamagen_root: str | None = None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.model_type = model_type
        self.image_size = int(image_size)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint_file = Path(self.checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_file}\n"
                "Please provide a local LlamaGen VQ checkpoint, for example "
                "./checkpoints/vq_ds16_c2i.pt. This module does not download weights."
            )

        checkpoint = _torch_load(checkpoint_file)
        state_dict = _select_state_dict(checkpoint)
        state_dict = _strip_prefix_if_present(state_dict, "module.")
        inferred_size, inferred_embed_dim = _infer_codebook_shape(state_dict)

        self.codebook_size = int(codebook_size or inferred_size or 16384)
        self.codebook_embed_dim = int(codebook_embed_dim or inferred_embed_dim or 8)

        vq_models = _import_llamagen_vq_models(llamagen_root)
        self.vq_model_name = _normalise_model_type(model_type, vq_models)
        self.model = vq_models[self.vq_model_name](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim,
        )

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint into the LlamaGen VQ model. If this "
                "checkpoint uses non-default architecture settings, pass the "
                "matching --model-type, --codebook-size, and --codebook-embed-dim.\n"
                f"Model type: {self.vq_model_name}, codebook_size: "
                f"{self.codebook_size}, codebook_embed_dim: {self.codebook_embed_dim}\n"
                f"Original load error:\n{exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        self.downsample_ratio = self._infer_downsample_ratio()
        self.token_grid_size: tuple[int, int] | None = self._initial_grid_size()
        self.num_tokens: int | None = (
            self.token_grid_size[0] * self.token_grid_size[1]
            if self.token_grid_size is not None
            else None
        )

    def _infer_downsample_ratio(self) -> int | None:
        config = getattr(self.model, "config", None)
        ch_mult = getattr(config, "encoder_ch_mult", None)
        if ch_mult is not None:
            return 2 ** (len(ch_mult) - 1)

        name = self.vq_model_name.upper()
        if "16" in name:
            return 16
        if "8" in name:
            return 8
        return None

    def _initial_grid_size(self) -> tuple[int, int] | None:
        if self.downsample_ratio is None:
            return None
        if self.image_size % self.downsample_ratio != 0:
            raise ValueError(
                f"image_size={self.image_size} is not divisible by "
                f"downsample_ratio={self.downsample_ratio}."
            )
        token_side = self.image_size // self.downsample_ratio
        return token_side, token_side

    def _preprocess(self, image: str | os.PathLike[str] | Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, (str, os.PathLike)):
            pil_image = Image.open(image).convert("RGB")
            return self._preprocess_pil(pil_image)

        if isinstance(image, Image.Image):
            return self._preprocess_pil(image)

        if torch.is_tensor(image):
            return self._preprocess_tensor(image)

        raise TypeError(
            "Unsupported image input. Expected PIL.Image.Image, path-like str, "
            "or torch.Tensor with shape [C,H,W] or [B,C,H,W]."
        )

    def _preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        image = _center_crop_resize_pil(image, self.image_size)
        array = np.asarray(image).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(self.device)

    def _preprocess_tensor(self, image: torch.Tensor) -> torch.Tensor:
        tensor = image.detach()
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3):
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("3D tensor input must be [C,H,W] or [H,W,C].")
        elif tensor.ndim == 4:
            if tensor.shape[1] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
            if tensor.shape[1] not in (1, 3):
                raise ValueError("4D tensor input must be [B,C,H,W] or [B,H,W,C].")
        else:
            raise ValueError("Tensor input must have shape [C,H,W] or [B,C,H,W].")

        tensor = tensor.float()
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        tensor = _tensor_center_crop_resize(tensor, self.image_size)
        min_value = float(tensor.amin().cpu())
        max_value = float(tensor.amax().cpu())

        if min_value < -0.1:
            model_input = tensor
        else:
            if max_value > 1.5:
                tensor = tensor / 255.0
            model_input = tensor * 2.0 - 1.0

        return model_input.to(self.device)

    def _extract_indices(
        self,
        encoded: Any,
        batch_size: int,
    ) -> tuple[torch.LongTensor, tuple[int, int] | None]:
        latent_shape: torch.Size | tuple[int, ...] | None = None

        if isinstance(encoded, dict):
            for key in ("indices", "token_ids", "tokens", "codebook_indices"):
                if key in encoded:
                    indices = encoded[key]
                    break
            else:
                raise ValueError("Model encode() returned a dict without token indices.")
            latent = encoded.get("latent") if "latent" in encoded else encoded.get("quant")
            if torch.is_tensor(latent):
                latent_shape = latent.shape
        elif isinstance(encoded, (tuple, list)):
            if encoded and torch.is_tensor(encoded[0]):
                latent_shape = encoded[0].shape
            info = encoded[-1]
            if isinstance(info, (tuple, list)) and info:
                indices = info[-1]
            elif torch.is_tensor(info):
                indices = info
            else:
                raise ValueError("Could not locate codebook indices in encode() output.")
        elif torch.is_tensor(encoded):
            indices = encoded
        else:
            raise ValueError("Unsupported encode() return type from tokenizer model.")

        if not torch.is_tensor(indices):
            raise ValueError("Tokenizer model returned non-tensor token indices.")

        indices = indices.to(dtype=torch.long, device=self.device)

        grid_size = None
        if latent_shape is not None and len(latent_shape) == 4:
            grid_size = (int(latent_shape[2]), int(latent_shape[3]))

        if indices.ndim == 1 and grid_size is not None:
            expected = batch_size * grid_size[0] * grid_size[1]
            if indices.numel() != expected:
                raise ValueError(
                    f"Token count mismatch: got {indices.numel()}, expected {expected}."
                )
            indices = indices.reshape(batch_size, grid_size[0], grid_size[1])
        elif indices.ndim == 2 and grid_size is not None:
            expected_per_image = grid_size[0] * grid_size[1]
            if indices.shape[1] == expected_per_image:
                indices = indices.reshape(batch_size, grid_size[0], grid_size[1])

        return indices, grid_size

    @torch.no_grad()
    def encode(self, image: str | os.PathLike[str] | Image.Image | torch.Tensor, *, flatten: bool = False) -> torch.LongTensor:
        """Encode image into discrete visual token ids.

        Accepted input:
        - PIL.Image.Image
        - path-like str
        - torch.Tensor with shape [C, H, W] or [B, C, H, W]

        Returns:
        - [B, Ht, Wt] by default, preserving the 2D token grid for OTFS mapping.
        - [B, N] when flatten=True.
        """

        x = self._preprocess(image)
        encoded = self.model.encode(x)
        token_ids, grid_size = self._extract_indices(encoded, batch_size=x.shape[0])

        if grid_size is not None:
            self.token_grid_size = grid_size
            self.num_tokens = grid_size[0] * grid_size[1]
            self.downsample_ratio = self.image_size // grid_size[0]

        if flatten:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)

        return token_ids.long()

    @torch.no_grad()
    def decode(
        self,
        token_ids: torch.LongTensor,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode discrete visual token ids to reconstructed images in [0, 1]."""

        if not torch.is_tensor(token_ids):
            raise TypeError("token_ids must be a torch.Tensor.")

        tokens = token_ids.to(device=self.device, dtype=torch.long)
        if tokens.ndim == 3:
            batch_size, height_tokens, width_tokens = map(int, tokens.shape)
            flat_tokens = tokens.reshape(-1)
        elif tokens.ndim == 2:
            batch_size, num_tokens = int(tokens.shape[0]), int(tokens.shape[1])
            if grid_size is None:
                grid_size = self.token_grid_size
            if grid_size is None:
                side = int(math.isqrt(num_tokens))
                if side * side != num_tokens:
                    raise ValueError(
                        "grid_size is required for flattened non-square token ids."
                    )
                grid_size = (side, side)
            height_tokens, width_tokens = map(int, grid_size)
            if height_tokens * width_tokens != num_tokens:
                raise ValueError(
                    f"grid_size={grid_size} does not match flattened token count "
                    f"N={num_tokens}."
                )
            flat_tokens = tokens.reshape(-1)
        else:
            raise ValueError("token_ids must have shape [B,N] or [B,Ht,Wt].")

        latent_shape = (
            batch_size,
            self.codebook_embed_dim,
            height_tokens,
            width_tokens,
        )

        if hasattr(self.model, "decode_code"):
            output = self.model.decode_code(flat_tokens, latent_shape)
        elif hasattr(self.model, "quantize") and hasattr(self.model, "decode"):
            quant = self.model.quantize.get_codebook_entry(flat_tokens, latent_shape)
            output = self.model.decode(quant)
        else:
            raise AttributeError(
                "Tokenizer model must provide decode_code() or quantize.get_codebook_entry() "
                "plus decode()."
            )

        if output.shape[-2:] != (self.image_size, self.image_size):
            output = F.interpolate(
                output,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )

        self.token_grid_size = (height_tokens, width_tokens)
        self.num_tokens = height_tokens * width_tokens
        if height_tokens > 0:
            self.downsample_ratio = self.image_size // height_tokens

        return ((output + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def reconstruct(
        self,
        image: str | os.PathLike[str] | Image.Image | torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Convenience method: image -> token ids -> reconstructed image."""

        token_ids = self.encode(image)
        reconstructed = self.decode(token_ids)
        return token_ids, reconstructed


def save_image_tensor(image: torch.Tensor, path: str | os.PathLike[str]) -> None:
    """Save a [B,3,H,W] or [3,H,W] tensor in [0, 1] as an image."""

    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor with shape [3,H,W] or [B,3,H,W].")

    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode an image with a LlamaGen VQ tokenizer and decode it back."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="Local LlamaGen VQ checkpoint path.")
    parser.add_argument(
        "--model-type",
        default="vqvae",
        help="Friendly type ('vqvae'/'vqgan') or concrete LlamaGen VQ model ('VQ-16'/'VQ-8').",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", required=True, help="Output reconstruction image path.")
    parser.add_argument("--save-tokens", default=None, help="Optional .pt path for token metadata.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--codebook-embed-dim", type=int, default=None)
    parser.add_argument(
        "--llamagen-root",
        default=os.environ.get("LLAMAGEN_ROOT"),
        help="Optional path to a LlamaGen checkout. Defaults to LLAMAGEN_ROOT if set.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Save/print flattened [B,N] tokens instead of the default [B,Ht,Wt] grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = ImageTokenizer(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        image_size=args.image_size,
        device=args.device,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        llamagen_root=args.llamagen_root,
    )

    token_ids, reconstructed = tokenizer.reconstruct(args.input)
    if args.flatten:
        token_ids = token_ids.reshape(token_ids.shape[0], -1)

    save_image_tensor(reconstructed, args.output)

    print(f"token_ids shape: {tuple(token_ids.shape)}")
    print(f"token_ids min/max: {int(token_ids.min())}/{int(token_ids.max())}")
    print(f"codebook size: {tokenizer.codebook_size}")
    print(f"token grid size: {tokenizer.token_grid_size}")
    print(f"downsample ratio: {tokenizer.downsample_ratio}")
    print(f"reconstruction saved to: {args.output}")

    if args.save_tokens:
        token_path = Path(args.save_tokens)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "token_ids": token_ids.cpu(),
                "grid_size": tokenizer.token_grid_size,
                "codebook_size": tokenizer.codebook_size,
                "image_size": tokenizer.image_size,
                "downsample_ratio": tokenizer.downsample_ratio,
            },
            token_path,
        )
        print(f"tokens saved to: {token_path}")


if __name__ == "__main__":
    main()
