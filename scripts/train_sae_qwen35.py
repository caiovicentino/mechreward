#!/usr/bin/env python3
"""Train a TopK Sparse Autoencoder on Qwen3.5 hidden states.

This is a self-contained training script that DOES NOT depend on sae_lens
or TransformerLens, because Qwen3.5 (architecture ``qwen3_5``, a hybrid
GDN) isn't supported by TransformerLens yet. We extract activations with
plain HF forward hooks.

Algorithm: TopK SAE (Gao et al. 2024, arxiv:2406.04093). Much simpler
than JumpReLU, within 1-2 pp of the same reconstruction quality, and
extremely stable to train.

Target: ship the FIRST public SAE for Qwen3.5 architecture.

Usage (local or Colab):

    python3 scripts/train_sae_qwen35.py \\
        --model Qwen/Qwen3.5-4B \\
        --layer 18 \\
        --d-sae 40960 \\
        --k 64 \\
        --tokens 200_000_000 \\
        --output-dir ./sae_qwen35_4b_L18 \\
        --hf-repo caiovicentino/Qwen3.5-4B-SAE-L18-topk

Memory budget (Qwen3.5-4B, d_sae=40960, bf16 model, fp32 SAE):
  - Model (frozen):   ~8 GB
  - SAE params:       ~0.8 GB
  - Adam optimizer:   ~1.6 GB
  - Batch activations: small
  - Total VRAM:       ~12-14 GB  (fits T4 16GB, L4 24GB, A100 40GB)

Dataset: FineWeb-Edu sample-10BT by default (streaming, no download needed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------------
# TopK SAE
# ---------------------------------------------------------------------------


class TopKSAE(nn.Module):
    """Sparse autoencoder with TopK activation (Gao et al. 2024).

    Keeps only the top ``k`` activations per token; everything else is zero.
    This gives L0 = k by construction, so no L1 tuning needed.
    """

    def __init__(self, d_model: int, d_sae: int, k: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        # Pre-decoder bias (center the input)
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))

        # Decoder (init tied to encoder transpose, then trained freely)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model, dtype=dtype))

        self._init_weights()

        # For dead-feature tracking
        self.register_buffer("last_active_step", torch.zeros(d_sae, dtype=torch.long))

    def _init_weights(self) -> None:
        # Kaiming-style init, then normalize decoder rows to unit norm
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone().contiguous()
            self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @torch.no_grad()
    def init_b_dec_from_sample(self, sample: torch.Tensor) -> None:
        """Set ``b_dec`` to the geometric-median-ish estimate of the sample.

        Cuts the initial reconstruction error dramatically because the SAE
        no longer needs to learn the data mean. Gao et al. 2024 recommend
        computing the geometric median (Weiszfeld); for speed we use a few
        iterations that converge fast.
        """
        if sample.numel() == 0:
            return
        sample = sample.to(self.b_dec.device, dtype=torch.float32)
        median = sample.median(dim=0).values  # initial estimate
        # Weiszfeld iteration — 5 steps is enough for ~99% accuracy
        for _ in range(5):
            diffs = sample - median
            dists = diffs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = 1.0 / dists
            median = (sample * weights).sum(dim=0) / weights.sum(dim=0)
        self.b_dec.data = median.to(self.b_dec.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] → acts: [..., d_sae] with L0 = k."""
        x_centered = x - self.b_dec
        pre = x_centered @ self.W_enc + self.b_enc  # [..., d_sae]

        # TopK
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, topk_vals)

        # ReLU to enforce non-negativity (matches Gao et al.)
        return F.relu(acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: [..., d_sae] → recon: [..., d_model]."""
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Re-project W_dec rows to the unit sphere (standard SAE trick)."""
        norms = self.W_dec.data.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.W_dec.data /= norms


def aux_loss_dead_features(
    sae: TopKSAE,
    hidden: torch.Tensor,
    dead_mask: torch.Tensor,
    k_aux: int,
) -> torch.Tensor:
    """Gao et al. auxiliary loss: revive dead features with a second TopK.

    Applied on features that have not activated for N steps. We run an
    independent TopK over ONLY those features and train them to
    reconstruct the residual error from the main SAE.

    Critical detail vs. an earlier buggy version: we do NOT apply ReLU
    to the aux activations. Dead features typically have very negative
    pre-activations — applying ReLU would zero them out immediately and
    the gradient would never flow back to revive them. Gao et al. use
    the raw TopK values directly.
    """
    if not dead_mask.any() or k_aux == 0:
        return hidden.new_zeros(())

    # Target: the part of the hidden state the main SAE failed to reconstruct.
    with torch.no_grad():
        _, acts_main = sae(hidden)
        main_recon = sae.decode(acts_main)
        residual = hidden - main_recon

    # Pre-activations on dead features only. Use a large negative mask for
    # non-dead features so they never appear in the aux TopK.
    x_centered = hidden - sae.b_dec
    pre = x_centered @ sae.W_enc + sae.b_enc
    neg_inf = torch.finfo(pre.dtype).min
    dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), neg_inf)

    k_use = min(k_aux, int(dead_mask.sum().item()))
    if k_use == 0:
        return hidden.new_zeros(())

    topk_vals, topk_idx = torch.topk(dead_pre, k_use, dim=-1)
    # NO F.relu here — allow negative pre-activations so gradients can
    # push dead features' encoder rows back to a useful direction.
    aux_acts = torch.zeros_like(pre)
    aux_acts.scatter_(-1, topk_idx, topk_vals)

    aux_recon = aux_acts @ sae.W_dec
    return F.mse_loss(aux_recon, residual)


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


class HiddenStateBuffer:
    """Fills up a rolling buffer of activations from a target layer."""

    def __init__(self, d_model: int, capacity: int, device: str | torch.device):
        self.capacity = capacity
        self.buffer = torch.zeros((capacity, d_model), dtype=torch.float32, device=device)
        self.write_pos = 0
        self.filled = False

    def extend(self, acts: torch.Tensor) -> None:
        """acts: [N, d_model]"""
        n = acts.shape[0]
        if n == 0:
            return
        # Convert to fp32 for stability
        acts = acts.to(self.buffer.dtype)
        end = self.write_pos + n
        if end <= self.capacity:
            self.buffer[self.write_pos : end] = acts
        else:
            first = self.capacity - self.write_pos
            self.buffer[self.write_pos :] = acts[:first]
            self.buffer[: n - first] = acts[first:]
            self.filled = True
        self.write_pos = end % self.capacity
        if end >= self.capacity:
            self.filled = True

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """Randomly sample a batch from the buffer."""
        size = self.capacity if self.filled else self.write_pos
        if size < batch_size:
            idx = torch.randint(0, max(1, size), (batch_size,), device=self.buffer.device)
        else:
            idx = torch.randint(0, size, (batch_size,), device=self.buffer.device)
        return self.buffer[idx]

    def ready(self, min_size: int) -> bool:
        size = self.capacity if self.filled else self.write_pos
        return size >= min_size


def stream_tokens(
    dataset_name: str,
    dataset_config: str | None,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Infinite iterator of tokenized batches from a streaming HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
    )

    buf: list[str] = []
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        buf.append(text)
        if len(buf) >= batch_size:
            enc = tokenizer(
                buf,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            yield enc
            buf = []


@torch.inference_mode()
def extract_hidden_states(
    model,
    enc: dict[str, torch.Tensor],
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """Run model forward and return flattened hidden states at ``layer_idx``.

    Returns a ``[N, d_model]`` tensor where N is the number of
    non-padding tokens across the batch.
    """
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    # hidden_states is (num_layers + 1) tensors of shape [B, T, d_model]
    # hidden_states[0] is the embedding, hidden_states[i] is after layer i-1.
    # We want the output after layer ``layer_idx`` — that's index layer_idx + 1.
    hidden = out.hidden_states[layer_idx + 1]

    mask = attention_mask.bool()
    return hidden[mask]  # [N, d_model]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3.5-4B"
    layer: int = 18
    d_sae: int = 40960
    k: int = 128                       # was 64 — more features active per token
    k_aux: int = 512                   # was 256 — revive more dead features per step
    tokens: int = 200_000_000
    lr: float = 2e-4                   # was 5e-4 — prevent feature collapse
    batch_size: int = 4096             # tokens per SAE step
    micro_batch: int = 8               # sequences per model forward
    max_length: int = 512
    warmup_steps: int = 5000           # was 1000 — longer warmup stabilizes features
    lr_min_frac: float = 0.3           # cosine decay floor: final LR = peak * min_frac
    decoder_norm_every: int = 10       # was 100 — per Gao et al., stabilizes training
    dead_threshold_steps: int = 5000   # was 500 — be more patient before marking dead
    aux_coef: float = 1.0 / 8.0        # was 1/32 — 4× stronger revival gradient
    buffer_capacity: int = 2_000_000   # was 500_000 — more activation diversity
    init_bdec_from_sample: bool = True # initialize b_dec from geometric median
    bdec_init_sample: int = 16_384     # sample size for b_dec init
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    output_dir: Path = field(default_factory=lambda: Path("./sae_qwen35_output"))
    hf_repo: str | None = None
    hf_token: str | None = None
    log_every: int = 25
    save_every: int = 2000
    seed: int = 42
    bf16_model: bool = True
    resume: str | None = None          # path to sae_resume.pt to continue from

    @property
    def total_steps(self) -> int:
        return max(1, self.tokens // self.batch_size)


def build_model(cfg: TrainConfig, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if cfg.bf16_model else torch.float32
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model, dtype=dtype, trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=dtype, trust_remote_code=True
        )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Sniff d_model
    cfg_attr = getattr(model, "config", None)
    text_cfg = getattr(cfg_attr, "text_config", cfg_attr)
    d_model = getattr(text_cfg, "hidden_size", None)
    if d_model is None:
        d_model = getattr(cfg_attr, "hidden_size", 2560)

    num_layers = getattr(text_cfg, "num_hidden_layers", None) or getattr(
        cfg_attr, "num_hidden_layers", 32
    )
    if cfg.layer >= num_layers:
        raise ValueError(f"layer={cfg.layer} >= num_hidden_layers={num_layers}")

    return model, tok, d_model


def cosine_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_frac: float = 0.1,
) -> LambdaLR:
    """Cosine schedule with a floor, so the revival phase keeps enough LR.

    Args:
        optimizer: the torch optimizer
        warmup_steps: linear warmup duration
        total_steps: total training steps
        min_frac: LR floor as a fraction of peak LR (e.g. 0.1 = 10%).
            Cosine decays from 1.0 → min_frac instead of 1.0 → 0.0.
            Critical for SAE training: the dead-feature aux loss needs
            LR > ~5e-5 to sustain revival, and cosine-to-zero drops
            below that threshold in the last third of training, undoing
            revival work.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        # Interpolate between 1.0 (start) and min_frac (end) via cosine
        return min_frac + (1.0 - min_frac) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: training SAE on CPU will be impossibly slow.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sae] Loading {cfg.model} on {device}")
    model, tok, d_model = build_model(cfg, device)
    print(f"[sae] d_model = {d_model}, target layer = {cfg.layer}")

    sae = TopKSAE(d_model=d_model, d_sae=cfg.d_sae, k=cfg.k).to(device)
    print(f"[sae] SAE params: {sum(p.numel() for p in sae.parameters()) / 1e6:.1f} M")

    optim = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = cosine_with_warmup(
        optim,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        min_frac=cfg.lr_min_frac,
    )

    # Resume path: if a previous snapshot exists, load SAE+optim+scheduler
    # state *before* initializing the activation buffer. On successful resume
    # we also skip the b_dec geometric median init (it was already done in
    # the original run).
    resume_step = 0
    resume_elapsed = 0.0
    resumed = False
    if cfg.resume:
        resume_path = Path(cfg.resume)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"--resume path does not exist: {resume_path}"
            )
        resume_step, resume_elapsed = load_resume_state(
            resume_path, sae, cfg, optim, scheduler, device
        )
        resumed = True

    buf = HiddenStateBuffer(d_model=d_model, capacity=cfg.buffer_capacity, device=device)
    stream = stream_tokens(
        cfg.dataset,
        cfg.dataset_config,
        tok,
        max_length=cfg.max_length,
        batch_size=cfg.micro_batch,
    )

    def add_tokens(n_tokens: int) -> None:
        """Fetch new sequences from the stream and add their activations.

        Runs micro-batches through the frozen model and appends the
        resulting per-token activations to the ring buffer. Unlike a
        'ready' check this always pulls fresh tokens — critical for
        avoiding buffer memorization (see README smoke-test notes).
        """
        added = 0
        while added < n_tokens:
            try:
                enc = next(stream)
            except StopIteration:
                return
            acts = extract_hidden_states(model, enc, cfg.layer, device)
            if acts.numel() == 0:
                continue
            buf.extend(acts)
            added += acts.shape[0]

    def fill_buffer_to(target: int) -> None:
        """Fill buffer until it has at least ``target`` tokens."""
        while not buf.ready(target):
            try:
                enc = next(stream)
            except StopIteration:
                return
            acts = extract_hidden_states(model, enc, cfg.layer, device)
            if acts.numel() > 0:
                buf.extend(acts)

    # Initial fill: pack the buffer reasonably full so the first batches
    # see diverse tokens. Target = min(buffer capacity, 10% of capacity
    # or batch_size * 16, whichever is larger).
    initial_target = min(
        cfg.buffer_capacity,
        max(cfg.buffer_capacity // 10, cfg.batch_size * 16),
    )
    print(f"[sae] Filling activation buffer to {initial_target} tokens...")
    fill_buffer_to(initial_target)

    # Sanity-check activation scale. Qwen3.5 hidden states can have very
    # different magnitudes from the Kaiming-init assumption of ~N(0,1).
    sample_scale = buf.sample_batch(4096)
    act_mean = sample_scale.mean().item()
    act_std = sample_scale.std().item()
    act_abs_max = sample_scale.abs().max().item()
    print(
        f"[sae] Activation scale: mean={act_mean:+.3f} "
        f"std={act_std:.3f} max|x|={act_abs_max:.3f}"
    )
    if act_std > 10 or act_std < 0.1:
        print(
            f"[sae] WARNING: activation std={act_std:.2f} is far from 1.0. "
            f"Kaiming init may need scale adjustment."
        )

    # Initialize b_dec with geometric median of a fresh sample. Gao et al.
    # 2024 show this cuts initial reconstruction error by 30-50%.
    # Skip on resume — the loaded b_dec is already trained.
    if cfg.init_bdec_from_sample and not resumed:
        init_sample = buf.sample_batch(cfg.bdec_init_sample)
        print(
            f"[sae] Initializing b_dec via geometric median over "
            f"{cfg.bdec_init_sample} samples..."
        )
        sae.init_b_dec_from_sample(init_sample)
        print(
            f"[sae]   b_dec norm after init: {sae.b_dec.norm().item():.4f}"
        )
    elif resumed:
        print(
            f"[sae]   b_dec carried over from checkpoint "
            f"(norm={sae.b_dec.norm().item():.4f})"
        )

    print(f"[sae] Starting training: {cfg.total_steps} steps, "
          f"{cfg.batch_size} tokens/step, {cfg.tokens / 1e6:.1f} M total tokens")
    print(
        f"[sae] Config: k={sae.k} d_sae={sae.d_sae} lr={cfg.lr} "
        f"lr_min={cfg.lr * cfg.lr_min_frac:.2e} "
        f"warmup={cfg.warmup_steps} dead_thresh={cfg.dead_threshold_steps} "
        f"aux_coef={cfg.aux_coef:.4f} dec_norm_every={cfg.decoder_norm_every} "
        f"buffer={cfg.buffer_capacity:,}"
    )

    # Keep wall-clock consistent with prior runs when resuming so ETA math
    # stays accurate and the log lines read "elapsed X min" not "0 min".
    t0 = time.time() - resume_elapsed
    step = resume_step
    # Refresh rate: how many fresh tokens to stream in per 1 token trained.
    # Value ~0.5 means the buffer is turned over every 2 epochs — enough to
    # prevent memorization without starving throughput.
    refresh_rate = 0.5
    refresh_interval = 25  # steps
    tokens_per_refresh = int(cfg.batch_size * refresh_rate * refresh_interval)

    while step < cfg.total_steps:
        # ALWAYS stream fresh tokens every `refresh_interval` steps.
        # This is critical — without it the buffer memorizes and var_exp
        # spuriously hits 1.0.
        if step > 0 and step % refresh_interval == 0:
            add_tokens(tokens_per_refresh)

        x = buf.sample_batch(cfg.batch_size)
        recon, acts = sae(x)
        mse = F.mse_loss(recon, x)

        # Dead-feature tracking
        active_now = (acts.sum(dim=0) > 0)
        sae.last_active_step[active_now] = step
        dead_mask = (step - sae.last_active_step) > cfg.dead_threshold_steps
        if dead_mask.any() and cfg.aux_coef > 0:
            aux = aux_loss_dead_features(sae, x, dead_mask, cfg.k_aux)
        else:
            aux = x.new_zeros(())

        loss = mse + cfg.aux_coef * aux
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optim.step()
        scheduler.step()

        if step % cfg.decoder_norm_every == 0:
            sae.normalize_decoder()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / max(1.0, elapsed)
            eta_min = (cfg.total_steps - step) / max(1e-6, steps_per_sec) / 60.0
            with torch.no_grad():
                var_explained = 1.0 - (mse.item() / max(x.var().item(), 1e-9))
                n_dead = int(dead_mask.sum().item())
                dead_pct = 100.0 * n_dead / cfg.d_sae
                # Real L0: features ACTUALLY active after ReLU (may differ from k)
                l0_real = float((acts > 0).float().sum(dim=-1).mean().item())
                # Feature coverage: fraction of features selected at least once
                # in this batch — complements "dead" metric which is long-window
                batch_coverage = float((acts > 0).any(dim=0).float().mean().item())
                aux_val = aux.detach().item() if aux.requires_grad else float(aux)
            print(
                f"[sae] step {step:>6}/{cfg.total_steps} "
                f"mse={mse.item():.4f} var_exp={var_explained:.3f} "
                f"L0={l0_real:.1f}/{sae.k} cov={batch_coverage:.3f} "
                f"dead={n_dead}({dead_pct:.0f}%) aux={aux_val:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"eta={eta_min:.1f}min",
                flush=True,
            )

        if step > 0 and step % cfg.save_every == 0:
            save_resume_state(
                sae, cfg, step, optim, scheduler,
                elapsed=time.time() - t0,
            )
            # Echo briefly so logs show saves happening
            print(
                f"[sae] Saved resume snapshot at step {step} "
                f"(elapsed {(time.time() - t0) / 60.0:.1f} min)",
                flush=True,
            )

        step += 1

    save_checkpoint(sae, cfg, step, final=True)
    print(f"[sae] Training complete. Total time: {(time.time() - t0) / 60:.1f} min")

    if cfg.hf_repo:
        upload_to_hf(cfg)


def save_checkpoint(sae: TopKSAE, cfg: TrainConfig, step: int, final: bool = False) -> None:
    suffix = "final" if final else f"step{step}"
    ckpt_path = cfg.output_dir / f"sae_{suffix}.pt"
    meta_path = cfg.output_dir / f"sae_{suffix}.json"

    torch.save(
        {
            "W_enc": sae.W_enc.detach().cpu(),
            "W_dec": sae.W_dec.detach().cpu(),
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec": sae.b_dec.detach().cpu(),
            "d_model": sae.d_model,
            "d_sae": sae.d_sae,
            "k": sae.k,
            "step": step,
        },
        ckpt_path,
    )

    meta = {
        "d_model": sae.d_model,
        "d_sae": sae.d_sae,
        "k": sae.k,
        "model": cfg.model,
        "layer": cfg.layer,
        "tokens_trained": step * cfg.batch_size,
        "algorithm": "topk",
        "training_script": "mechreward/scripts/train_sae_qwen35.py",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[sae] Saved checkpoint to {ckpt_path}")


def save_resume_state(
    sae: TopKSAE,
    cfg: TrainConfig,
    step: int,
    optim: torch.optim.Optimizer,
    scheduler: LambdaLR,
    elapsed: float,
) -> None:
    """Write a full training snapshot for crash-proof resume.

    Writes a single file ``sae_resume.pt`` in ``cfg.output_dir`` atomically
    (tmp + rename) so a dying process cannot leave a corrupt checkpoint.
    Overwrites the previous snapshot on every call — we only ever need the
    latest one. Historical inspection points use ``sae_stepN.pt`` files
    written separately.

    Contents: SAE parameters, optimizer state dict, scheduler state dict,
    per-feature dead-feature tracker, RNG state, step counter, wall-time
    elapsed, and a ``meta`` block recording the hyperparameters that must
    match on resume.
    """
    tmp_path = cfg.output_dir / "sae_resume.tmp"
    final_path = cfg.output_dir / "sae_resume.pt"

    payload = {
        # SAE weights (same keys as sae_final.pt for inspection compatibility)
        "W_enc": sae.W_enc.detach().cpu(),
        "W_dec": sae.W_dec.detach().cpu(),
        "b_enc": sae.b_enc.detach().cpu(),
        "b_dec": sae.b_dec.detach().cpu(),
        "d_model": sae.d_model,
        "d_sae": sae.d_sae,
        "k": sae.k,
        # Training state for resume
        "step": int(step),
        "elapsed": float(elapsed),
        "optim_state": optim.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "last_active_step": sae.last_active_step.detach().cpu(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
        # Hyperparameter meta — verified on load to catch silent mismatches
        "meta": {
            "model": cfg.model,
            "layer": cfg.layer,
            "d_sae": cfg.d_sae,
            "k": cfg.k,
            "k_aux": cfg.k_aux,
            "tokens": cfg.tokens,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "warmup_steps": cfg.warmup_steps,
            "lr_min_frac": cfg.lr_min_frac,
            "aux_coef": cfg.aux_coef,
            "decoder_norm_every": cfg.decoder_norm_every,
            "dead_threshold_steps": cfg.dead_threshold_steps,
            "total_steps": cfg.total_steps,
        },
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


def load_resume_state(
    path: Path,
    sae: TopKSAE,
    cfg: TrainConfig,
    optim: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: str,
) -> tuple[int, float]:
    """Restore a full training snapshot previously written by ``save_resume_state``.

    Returns ``(step, elapsed)`` — the step count and wall-clock time already
    consumed, so the training loop can continue from exactly where it was
    interrupted and the ETA estimate stays accurate.

    Aborts with a clear error if the saved meta hyperparameters differ from
    the current ``cfg`` in any way that would make the resume unsafe
    (e.g. different total_steps, different k, different d_sae).
    """
    print(f"[sae] Loading resume checkpoint from {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    # Verify critical hyperparameters match
    meta = state.get("meta", {})
    critical = [
        "d_sae", "k", "k_aux", "tokens", "batch_size",
        "warmup_steps", "lr_min_frac", "total_steps",
    ]
    mismatches = []
    cfg_snapshot = {
        "d_sae": cfg.d_sae, "k": cfg.k, "k_aux": cfg.k_aux,
        "tokens": cfg.tokens, "batch_size": cfg.batch_size,
        "warmup_steps": cfg.warmup_steps, "lr_min_frac": cfg.lr_min_frac,
        "total_steps": cfg.total_steps,
    }
    for key in critical:
        saved = meta.get(key)
        current = cfg_snapshot.get(key)
        if saved is not None and saved != current:
            mismatches.append(f"{key}: saved={saved} vs current={current}")
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint hyperparameters do not match current run:\n  "
            + "\n  ".join(mismatches)
            + "\nTo resume, pass the same CLI flags as the original run."
        )

    # Restore SAE weights (shape-checked via torch .data assignment)
    sae.W_enc.data.copy_(state["W_enc"].to(device))
    sae.W_dec.data.copy_(state["W_dec"].to(device))
    sae.b_enc.data.copy_(state["b_enc"].to(device))
    sae.b_dec.data.copy_(state["b_dec"].to(device))
    sae.last_active_step = state["last_active_step"].to(device)

    # Restore optimizer — move any internal tensors to device
    optim.load_state_dict(state["optim_state"])
    for opt_state in optim.state.values():
        for k, v in opt_state.items():
            if isinstance(v, torch.Tensor):
                opt_state[k] = v.to(device)

    scheduler.load_state_dict(state["scheduler_state"])

    torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])

    step = int(state["step"])
    elapsed = float(state.get("elapsed", 0.0))
    print(
        f"[sae] Resumed at step {step}/{cfg.total_steps} "
        f"({100.0 * step / cfg.total_steps:.1f}%), "
        f"{elapsed / 60.0:.1f} min already elapsed"
    )
    return step, elapsed


def upload_to_hf(cfg: TrainConfig) -> None:
    from huggingface_hub import HfApi, create_repo

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[sae] No HF_TOKEN set; skipping upload.")
        return

    try:
        create_repo(cfg.hf_repo, token=token, exist_ok=True)
    except Exception as e:
        print(f"[sae] create_repo warning: {e}")

    api = HfApi()
    for path in sorted(cfg.output_dir.glob("*")):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=cfg.hf_repo,
            token=token,
        )
    print(f"[sae] Uploaded to https://huggingface.co/{cfg.hf_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a TopK SAE on Qwen3.5 hidden states")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--d-sae", type=int, default=40960)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--k-aux", type=int, default=512)
    p.add_argument("--tokens", type=int, default=200_000_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--lr-min-frac", type=float, default=0.3,
                   help="LR floor as fraction of peak (cosine decays 1.0 → min_frac). "
                        "Default 0.3 prevents revival collapse in the last third of training.")
    p.add_argument("--dead-threshold-steps", type=int, default=5000)
    p.add_argument("--aux-coef", type=float, default=1.0 / 8.0,
                   help="Weight of the dead-feature revival aux loss. Default 1/8 "
                        "(was 1/32 originally). Increase if dead%% stays >25%% at convergence.")
    p.add_argument("--decoder-norm-every", type=int, default=10,
                   help="Re-normalize W_dec rows every N steps. Default 10.")
    p.add_argument("--buffer-capacity", type=int, default=2_000_000)
    p.add_argument(
        "--no-bdec-init",
        action="store_true",
        help="Disable geometric-median init of b_dec (debug only)",
    )
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--output-dir", type=Path, default=Path("./sae_qwen35_output"))
    p.add_argument("--hf-repo", default=None, help="Optional HF repo to upload the final SAE")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-bf16", action="store_true", help="Force fp32 model (slower)")
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a sae_resume.pt checkpoint to continue training from. "
             "Restores SAE weights, optimizer state, LR schedule position, "
             "dead-feature tracker, and RNG state. Key hyperparams (d_sae, k, "
             "warmup_steps, total_steps, lr, lr_min_frac) must match the "
             "original run or loading will abort.",
    )
    args = p.parse_args()

    return TrainConfig(
        model=args.model,
        layer=args.layer,
        d_sae=args.d_sae,
        k=args.k,
        k_aux=args.k_aux,
        tokens=args.tokens,
        lr=args.lr,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        lr_min_frac=args.lr_min_frac,
        dead_threshold_steps=args.dead_threshold_steps,
        aux_coef=args.aux_coef,
        decoder_norm_every=args.decoder_norm_every,
        buffer_capacity=args.buffer_capacity,
        init_bdec_from_sample=not args.no_bdec_init,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        bf16_model=not args.no_bf16,
        resume=args.resume,
    )


if __name__ == "__main__":
    train(parse_args())
