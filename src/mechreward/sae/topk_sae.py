"""Load a TopK SAE trained by ``scripts/train_sae_qwen35.py``.

These SAEs are stored as a single ``.pt`` file with ``W_enc``, ``W_dec``,
``b_enc``, ``b_dec`` tensors plus scalar metadata (d_model, d_sae, k).
They're simpler than sae_lens releases — no release/sae_id structure,
just a flat file that can be downloaded from HuggingFace Hub.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mechreward.sae.cache import ensure_cached
from mechreward.sae.loader import SAEHandle


class TopKSAEBackend(nn.Module):
    """Minimal TopK SAE implementing the SAEBackend protocol.

    This is the inference-time backend that ``load_topk_sae`` populates
    from a trained checkpoint. The weights are default-initialized to
    Kaiming so that shape-only unit tests (without a checkpoint) still
    produce non-degenerate activations.
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone().contiguous()
            norms = self.W_dec.data.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.W_dec.data /= norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre = x_centered @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, topk_vals)
        return F.relu(acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def device(self) -> torch.device:
        return self.W_enc.device


def load_topk_sae(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
    model_name: str = "Qwen/Qwen3.5-4B",
    layer: int | None = None,
    release: str = "mechreward/qwen3.5-4b-topk",
    sae_id: str = "layer_18",
) -> SAEHandle:
    """Load a TopK SAE checkpoint saved by ``train_sae_qwen35.py``.

    Args:
        checkpoint: Path to the ``.pt`` file, OR a HuggingFace repo id
            like ``"caiovicentino/Qwen3.5-4B-SAE-L18"``.
        device: CUDA device to place the SAE on.
        model_name: Base model this SAE was trained on.
        layer: Layer index if not already encoded in the checkpoint.
        release: Label for the SAEHandle.
        sae_id: Label for the SAEHandle.

    Returns:
        A populated ``SAEHandle``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = str(checkpoint)
    local_path: Path

    if Path(checkpoint).exists():
        local_path = Path(checkpoint)
    elif "/" in checkpoint and not checkpoint.endswith(".pt"):
        # Assume HF repo id: download the final checkpoint
        from huggingface_hub import hf_hub_download

        cache = ensure_cached(release, sae_id)
        filename = "sae_final.pt"
        try:
            local_path = Path(
                hf_hub_download(
                    repo_id=checkpoint,
                    filename=filename,
                    cache_dir=str(cache),
                )
            )
        except Exception:
            # Try "pytorch_model.bin" as fallback
            local_path = Path(
                hf_hub_download(
                    repo_id=checkpoint,
                    filename="pytorch_model.bin",
                    cache_dir=str(cache),
                )
            )
    else:
        raise FileNotFoundError(f"Cannot resolve SAE checkpoint: {checkpoint}")

    state = torch.load(local_path, map_location="cpu", weights_only=True)

    d_model = int(state["d_model"])
    d_sae = int(state["d_sae"])
    k = int(state["k"])

    backend = TopKSAEBackend(d_model=d_model, d_sae=d_sae, k=k)
    backend.W_enc.data = state["W_enc"].to(dtype=torch.float32)
    backend.W_dec.data = state["W_dec"].to(dtype=torch.float32)
    backend.b_enc.data = state["b_enc"].to(dtype=torch.float32)
    backend.b_dec.data = state["b_dec"].to(dtype=torch.float32)
    backend = backend.to(device)
    backend.eval()

    hook_name = f"blocks.{layer if layer is not None else -1}.hook_resid_post"

    return SAEHandle(
        backend=backend,
        release=release,
        sae_id=sae_id,
        hook_name=hook_name,
        layer=int(layer) if layer is not None else -1,
        d_model=d_model,
        d_sae=d_sae,
        model_name=model_name,
    )
