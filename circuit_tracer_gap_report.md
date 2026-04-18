# Support for single-layer residual-stream TopK SAEs on non-TL-supported models

**Tested against**: `circuit-tracer` main (as of 2026-04-17, commits up through PR #94).

**Context**: I have a TopK SAE trained on the residual stream post-layer-18 of Qwen3.5-4B (hybrid Gated DeltaNet architecture — not currently supported by TransformerLens). I attempted to load it into `circuit-tracer` to generate attribution graphs on GSM8K reasoning, treating it as a `SingleLayerTranscoder` with `feature_input_hook == feature_output_hook`. Below are the 4 concrete gaps I hit. Each is a narrow ask; together they describe what's missing for the single-layer residual-SAE + non-TL-model case.

## 1. `TranscoderSet` requires contiguous layers 0..max

**File**: `circuit_tracer/transcoder/single_layer_transcoder.py:254`

```python
assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
    f"Each layer should have a transcoder, but got transcoders for layers "
    f"{set(transcoders.keys())}"
)
```

**Reproducer**:
```python
from circuit_tracer.transcoder import TranscoderSet
ts = TranscoderSet(
    transcoders={18: my_single_layer_transcoder},
    feature_input_hook='blocks.18.hook_resid_post',
    feature_output_hook='blocks.18.hook_resid_post',
)
# → AssertionError: Each layer should have a transcoder, but got transcoders for layers {18}
```

**Why this matters**: many SAE researchers train a single SAE at one target layer (not a full per-layer stack like GemmaScope). This assertion rules them out without a clean opt-out.

**Proposed fix**: add `allow_partial: bool = False` kwarg to `TranscoderSet.__init__`. When True, skip the assertion and the `layers` property reports only the layers that have transcoders. Downstream iteration should already tolerate missing layers because the class is indexed by `layer_id`.

---

## 2. `ReplacementModel.from_pretrained` treats `transcoder_set` only as HF repo ID, not local path

**File**: `circuit_tracer/utils/hf_utils.py:95` (called from `replacement_model.py:55`).

**Reproducer**:
```python
ReplacementModel.from_pretrained(
    model_name='Qwen/Qwen3.5-4B',
    transcoder_set='/content/drive/MyDrive/.../transcoder_set_dir',  # local path
    backend='nnsight',
)
# → huggingface_hub.errors.HFValidationError:
#     Repo id must use alphanumeric chars, '-', '_' or '.'.
#     '/content' (the code splits the path and takes '/content' as repo_id)
```

**Workaround that works**: `ReplacementModel.from_pretrained_and_transcoders(model_name=..., transcoders=ts_instance, ...)` accepts an in-memory `TranscoderSet`. But the documentation isn't explicit that this is the path for locally-built transcoder sets; I had to find it by reading the class methods.

**Proposed fix**: either (a) detect that `transcoder_set` starts with `/` or `./` and treat as path; or (b) add a large doc note explaining `from_pretrained_and_transcoders` is the entry point for local/custom transcoder sets.

---

## 3. `huggingface_hub` version conflict between transformers (current main) and circuit-tracer

**Observed**: `circuit-tracer` imports `HF_HUB_ENABLE_HF_TRANSFER` from `huggingface_hub.constants` (available in hf_hub ≥ ~1.6). But `transformers` main (5.6.0.dev0) still imports `is_offline_mode` from `huggingface_hub` (available only in hf_hub ≤ 1.5.x where it's at top level; moved in newer versions).

**Reproducer** (in a fresh env):
```bash
pip install git+https://github.com/huggingface/transformers.git  # installs hf_hub 1.11+ as dep
pip install circuit-tracer                                         # requires old behavior
python -c "import transformers"
# → ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

If you then pin `hf_hub==1.5.0`:
```python
from circuit_tracer import ReplacementModel
# → ImportError: cannot import name 'HF_HUB_ENABLE_HF_TRANSFER' from 'huggingface_hub.constants'
```

**Workaround**: pin `hf_hub==1.5.0` and monkey-patch: `huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False`.

**Proposed fix**: guard the import:
```python
try:
    from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
except ImportError:
    HF_HUB_ENABLE_HF_TRANSFER = False
```

---

## 4. `convert_nnsight_config_to_transformerlens` fails on multimodal/hybrid configs — `config.architectures=None`

**File**: `circuit_tracer/utils/tl_nnsight_mapping.py:275`.

```python
config_dict["original_architecture"] = config.architectures[0]
# → TypeError: 'NoneType' object is not subscriptable
```

**Reproducer**:
```python
ReplacementModel.from_pretrained_and_transcoders(
    model_name='Qwen/Qwen3.5-4B',   # multimodal, Qwen3_5ForConditionalGeneration
    transcoders=partial_set_with_one_transcoder_at_L18,
    backend='nnsight',
)
# Model weights load successfully (9.32 GB downloaded, 426 weight shards loaded)
# Then fails during config conversion.
```

**Why**: `Qwen/Qwen3.5-4B`'s top-level config has `architectures=["Qwen3_5ForConditionalGeneration"]`, but its `text_config` sub-config (which is what represents the LM backbone) does not have `architectures` set. NNSight's config wrapper probably points at the text sub-config, so `config.architectures` is None.

More generally, `convert_nnsight_config_to_transformerlens` assumes the HF config has standard transformer fields. Qwen3.5 uses Gated DeltaNet (linear attention with `layer_types` list alternating GDN and full attention) — the TL `HookedTransformerConfig` doesn't have fields for this.

**Proposed fix (short)**: at the failing line, fall back gracefully:
```python
archs = getattr(config, 'architectures', None) or ['unknown']
config_dict["original_architecture"] = archs[0]
```

**Proposed fix (deeper)**: `convert_nnsight_config_to_transformerlens` should either (a) raise a clear "this architecture is not TL-compatible" error upfront, or (b) set a minimal config that tells downstream code to skip TL-specific operations when the model is hybrid/non-standard.

---

## What I'd like to contribute

Each gap above is narrow. I'm happy to file PRs for #1 (partial allowed), #2 (path-or-name detection) and #3 (hf_hub import guard) if maintainers are open to them — those three together unblock single-layer residual-stream SAEs on TL-supported models.

Gap #4 is architecturally deeper (supporting hybrid/multimodal base models in the nnsight backend). I'd prefer to discuss the desired scope before attempting a PR — possibly a separate `HybridNNSightBackend` that doesn't go through TL-compatibility mapping at all.

## Why this matters for the broader SAE interp community

Single-layer residual-stream SAEs are the most common form trained today (Anthropic/Gemma Scope/Llama Scope all publish them). The current circuit-tracer API is strongly coupled to the per-layer MLP-transcoder paradigm (e.g., GemmaScope-2). Extending to the common single-layer case would significantly widen the library's applicability.

My test SAE: [`caiovicentino1/Qwen3.5-4B-SAE-L18-topk`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk) — TopK residual-stream SAE on Qwen3.5-4B (hybrid Gated DeltaNet). The nearest prior public SAE on this family is [`kroonen-ai/sae-qwen3.5-9b`](https://huggingface.co/kroonen-ai/sae-qwen3.5-9b) (ReLU, Qwen3.5-9B, layer-16 MLP output) — different hook, different activation, different expansion factor. The trained policy that validates the feature pack as a dense RL reward signal is at [`caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400). Write-up: [LessWrong](https://www.lesswrong.com/posts/H7mnTT7aPPijpjLAS/per-token-sae-features-as-online-rl-reward-breaking-the-g2).
