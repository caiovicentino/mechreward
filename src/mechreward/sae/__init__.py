"""SAE loading, caching, and batched encoding utilities."""

from mechreward.sae.batched_encode import batched_encode
from mechreward.sae.cache import cache_dir, ensure_cached
from mechreward.sae.loader import SAEHandle, load_sae
from mechreward.sae.topk_sae import TopKSAEBackend, load_topk_sae

__all__ = [
    "SAEHandle",
    "load_sae",
    "load_topk_sae",
    "TopKSAEBackend",
    "cache_dir",
    "ensure_cached",
    "batched_encode",
]
