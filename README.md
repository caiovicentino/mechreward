# mechreward

**Mechanistic interpretability as a reward signal for RL training of LLMs.**

Most RL-for-reasoning methods reward the *output*: "did the final answer match?" (outcome reward), "did each step look correct?" (PRM), "did a judge like it?" (LLM-as-judge).

`mechreward` rewards the **process inside the model**. Using sparse-autoencoder (SAE) features from interpretability research, we ask a fundamentally different question:

> *Is the model actually doing the cognitive work we want it to do, at the circuit level?*

A model trained against a feature-reward like `+1 × fact_retrieval_active - 0.5 × hedging` can't trivially game the reward without actually activating those circuits — which requires actually doing retrieval and not hedging. The gradient signal is grounded in the model's internal state, not just its text output.

## Status

**Alpha (v0.1.0).** API is subject to change. Integrations with `trl` (GRPO), `openrlhf`, and `verl` available.

**Thesis validated** on Qwen3.5-4B (2026-04-17): three-stage empirical pipeline completed.
- **Stage Gate 1** (correlation): SAE features predict GSM8K correctness at Spearman ρ=0.540 on 100 held-out questions.
- **Stage Gate 2** (tiny RL): R1 (outcome + SAE features) 76 % vs R0 (outcome only) 74 % vs R2 (outcome + raw L13 direction) 65 %. The 11 pp R1-vs-R2 gap is direct evidence that sparse SAE decomposition is necessary, not cosmetic.
- **Stage Gate 3 Phase A** (full RL): 64 % → **83 % on GSM8K** in 168 effective training steps at LR=3e-6 (LoRA r=32). Breaks the trajectory-level G2 R1 ceiling (76 %) by +7 pp with per-token mech-reward; MMLU non-regressed; adversarial-canary hack rate within 95 % CI of baseline.

**Write-up**: [LessWrong post](https://www.lesswrong.com/posts/H7mnTT7aPPijpjLAS/per-token-sae-features-as-online-rl-reward-breaking-the-g2). Trained adapter at [`caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400).

**In progress**: Stage 4 extension to [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (first SAE on triple-hybrid MoE + GDN + Gated Attention architecture, SuperGPQA benchmark).

**Closest prior work**: Goodfire's [RLFR](https://arxiv.org/abs/2602.10067) (Prasad et al., Feb 2026) established feature-as-reward for online RL using linear probes on Gemma-3-12B-IT for hallucination. `mechreward` extends that paradigm in three directions: sparse TopK SAE features instead of raw probes, per-token dense reward instead of span-level, and hybrid architectures (GDN/MoE). See [RESEARCH.md](RESEARCH.md) for the full prior-art audit.

## Install

```bash
# Core
pip install mechreward

# With SAE support (sae_lens integration)
pip install "mechreward[sae]"

# With TRL integration (GRPOTrainer hook)
pip install "mechreward[sae,trl]"

# Everything
pip install "mechreward[all]"
```

## Quickstart — feature reward in 10 lines

```python
import mechreward as mr
from trl import GRPOConfig, GRPOTrainer

# 1. Load the validated SAE (trained by us on Qwen3.5-4B hybrid GDN, 200M tokens)
sae = mr.load_sae(
    release="caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
    sae_id="layer_18",
)

# 2. Build a feature reward from the validated pack (ρ=0.540 on GSM8K)
reward = mr.FeatureReward.from_pack(
    "qwen3.5-4b/reasoning_pack",
    sae=sae,
    aggregation="per_token",  # per-token dense — see Stage Gate 3
)

# 3. Combine with an outcome reward (math verifier)
composite = mr.CompositeReward(
    rewards=[
        reward,
        mr.OutcomeReward(verifier=mr.verifiers.gsm8k_exact_match),
    ],
    weights=[0.1, 1.0],  # λ_mech=0.1 — validated ratio from Stage Gate 2
)

# 4. Plug into TRL GRPOTrainer (unchanged API)
trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-4B",
    args=GRPOConfig(output_dir="./out", num_generations=4, learning_rate=3e-6),
    train_dataset=gsm8k_train,
    reward_funcs=composite,
)
trainer.train()
```

That's it. This is the exact configuration that took Qwen3.5-4B from 64 % to 83 % on GSM8K. The feature-reward runs alongside outcome-reward during each GRPO step, with anti-hacking detection and KL regularization enabled by default.

> Note: Qwen3.5-4B is multimodal. Use `AutoModelForImageTextToText` and freeze the vision tower before LoRA — see `notebooks/stage3_qwen35_4b_rl_v2.ipynb` for the full working example.

## Why this could work

Every published post-training technique for reasoning today rewards either (a) the final answer or (b) a human-labeled intermediate step. Both have brittle failure modes:

- **Outcome reward** gives sparse signal, doesn't distinguish lucky guesses from real reasoning.
- **Process reward models** get hacked within a few thousand steps — DeepSeek-R1 explicitly abandoned them.
- **LLM-as-judge** is adversarially fragile (arxiv:2507.08794 — "One Token to Fool LLM-as-a-Judge").

Meanwhile, mechanistic interpretability research has shown that specific SAE features reliably light up during specific cognitive operations:
- **fact retrieval** — arxiv:2408.05147 (Gemma Scope)
- **confidence vs hedging** — arxiv:2411.11296 (Microsoft refusal steering)
- **chain-of-reasoning** — well-documented in Anthropic's Claude 3 Sonnet interpretability work

If we reward the *internal pattern* instead of the *output token*, we're reaching a different layer of the stack — one that's harder to game at the surface, and that lines up more directly with what we actually want the model to learn.

## What makes this different from RLFR / SARM / CRL

The closest prior work is Goodfire's RLFR (Prasad et al., Feb 2026), which established the feature-as-reward-for-online-RL paradigm two months before this work. `mechreward` is a methodologically distinct instance within that paradigm, with a direct empirical argument for the specific design choices.

| Method | What it does | What `mechreward` adds |
|---|---|---|
| [**RLFR — Goodfire (Prasad et al., Feb 2026)**](https://arxiv.org/abs/2602.10067) | **Linear probes on activations** as online RL reward; trajectory/span-level; dense Gemma-3-12B-IT; hallucination task (58 % ↓) | **Sparse TopK SAE decomposition** instead of raw probes (validated: raw-direction reward is **−9 pp** worse than outcome-only on GSM8K; SAE-sparse is **+2 pp**, an 11 pp gap). **Per-token dense** reward instead of span-level (+7 pp in G3). **Hybrid architectures** (GDN, MoE, triple-hybrid) where no public SAEs existed. |
| [SARM](https://arxiv.org/abs/2508.08746) (AAAI 26) | SAE features → linear head → reward model, used in offline RLHF | Online GRPO use; multi-objective; composability with outcome verifier |
| [SparseRM](https://arxiv.org/abs/2511.07896) | Preference modeling via frequency-diff features | Reward is per-token, not pairwise |
| [CRL](https://arxiv.org/abs/2602.10437) | Token-level feature amplification via RL | Reward is feature activation, not action selection |
| [YaPO](https://arxiv.org/abs/2601.08441) | SAE-sparse steering vectors | We don't modify inference-time activations |
| [Wilhelm et al.](https://arxiv.org/abs/2603.04069) | SAE features **detect** reward hacking | We use the same probes *during training* to prevent it |

The novel contribution is the **combination within the feature-as-reward paradigm**: sparse TopK decomposition + per-token dense GRPO + hybrid architecture generalization + anti-hacking dual verification, shipped as a drop-in library. See [RESEARCH.md](RESEARCH.md) for the full positioning and verified prior-art audit.

## Anti-Goodhart is built in

The central risk of any reward signal is [Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law): the model learns to maximize the measure without doing the underlying work. Feature reward is *especially* vulnerable because SAE features are effectively linear probes, and linear probes are trivially gameable in the limit.

`mechreward` addresses this with **dual verification**:

```python
from mechreward.hacking import DualVerifier, AdversarialSuite

# A second, independent signal (a linear probe trained on L18 residuals of
# correct vs wrong GSM8K responses) checks whether the feature activation is "honest".
dual = DualVerifier(
    feature_reward=reward,
    independent_probe=mr.load_probe("qwen3.5-4b/correctness_probe_l18"),
    disagreement_threshold=0.3,  # if they disagree >30%, downweight
)

# And an adversarial red-team suite flags suspicious rollouts during training.
# In G3 Phase A with 50 canaries × 5 hack patterns, trained policy hack rate was
# 8 % (vs 4 % baseline, within 95 % CI) — no Goodhart collapse observed.
detector = AdversarialSuite.from_preset("standard")
```

Each GRPO step runs the detector in parallel with the main reward computation. If it fires, the affected rollouts are downweighted or dropped. See `src/mechreward/hacking/` for the full framework.

## Supported models

| Model | SAE source | Status |
|---|---|---|
| **Qwen3.5-4B** (hybrid GDN) | [`caiovicentino1/Qwen3.5-4B-SAE-L18-topk`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk) — trained by us, 200 M tokens, var_exp 0.866 | ✅ **Primary validated target (G1+G2+G3 passed)** |
| **Gemma-4-E4B** (ensemble MoE) | [`caiovicentino1/Gemma-4-E4B-SAE-L21-topk`](https://huggingface.co/caiovicentino1/Gemma-4-E4B-SAE-L21-topk) — trained by us, 1 B tokens, var_exp 0.939 | ✅ SAE published; G1–G3 pending |
| **Qwen3.6-35B-A3B** (triple-hybrid MoE+GDN+GA) | Training in progress at L23 | 🚧 S4 in flight |
| Gemma-2-9B / 2B / 27B | Gemma Scope (DeepMind) | ✅ Library-supported (Gemma Scope), not our primary test |
| Llama-3.1-8B | Llama Scope / Goodfire SAE | ✅ Library-supported, not our primary test |
| Qwen3.5-9B | [`kroonen-ai/sae-qwen3.5-9b`](https://huggingface.co/kroonen-ai/sae-qwen3.5-9b) — third-party ReLU MLP SAE | ⚠️ Alternative SAE exists (ReLU, not TopK); untested in mechreward |
| Mistral, DeepSeek-V3 MoE | No public SAE | ❌ Needs custom training |

To add a new model, see `docs/training_new_sae.md` for the `sae_lens`-based training recipe, or `scripts/train_sae_qwen35.py` for our hybrid-architecture TopK recipe that bypasses TransformerLens.

## Repository layout

```
mechreward/
├── src/mechreward/
│   ├── sae/            # SAE loading, caching, batched encoding
│   ├── features/       # Feature catalogs, Neuronpedia client, auto-interp
│   ├── reward/         # FeatureReward core, aggregation, composition
│   ├── hacking/        # Dual verification, adversarial, regularization
│   ├── probes/         # Linear probe baseline + training utilities
│   ├── rollout/        # HF and vLLM integration with hidden-state capture
│   └── integrations/   # TRL, OpenRLHF, verl adapters
├── catalogs/           # Pre-validated feature packs (JSON)
├── experiments/        # The 7 reference experiments from the research plan
├── benchmarks/         # Evaluation harnesses
└── tests/              # Unit + integration tests
```

## The 7 reference experiments

`experiments/` contains a full research pipeline:

1. **01_baseline_outcome_only.py** — outcome-reward GRPO baseline on GSM8K+MATH
2. **02_mechreward_only.py** — the tenuous experiment: mechreward alone, no outcome
3. **03_hybrid_outcome_plus_mech.py** — the commercially relevant combination
4. **04_sarm_reproduction.py** — reproduces Liu et al. 2508.08746 for comparison
5. **05_crl_reproduction.py** — reproduces Cho/Wu/Koshiyama 2602.10437
6. **06_adversarial_hacking_suite.py** — red-team suite + detection
7. **07_capability_preservation.py** — MMLU/HellaSwag pre/post RL

Run any of them after install:

```bash
python experiments/03_hybrid_outcome_plus_mech.py --config configs/hybrid.yaml
```

## How it talks to TRL

The tricky part of integrating feature rewards with a GRPO trainer is that the standard `reward_funcs` API only gets strings and token IDs — not hidden states. `mechreward` solves this by providing a TRL-compatible wrapper that registers a forward hook on the policy and extracts the residual stream at the target layer during the reward computation:

```python
from mechreward.integrations.trl_grpo import MechRewardGRPOTrainer

trainer = MechRewardGRPOTrainer(
    model="Qwen/Qwen3.5-4B",  # multimodal; loader handles AutoModelForImageTextToText
    reward_funcs=[feature_reward, outcome_reward],
    ...,
)
```

`MechRewardGRPOTrainer` wraps `trl.GRPOTrainer` and adds:
- Forward-hook registration on the SAE layer
- Residual-stream capture during rollout
- SAE encoding of hidden states
- Feature-reward computation from activations
- Hacking detection on the side

The rest of GRPO (policy gradient, KL, advantage computation) is unchanged.

## Testing

```bash
pip install "mechreward[dev]"
pytest
```

Integration tests require small SAEs and use `Gemma-2-2B` by default to stay under laptop compute.

## Research context

This library exists because of a specific empirical observation: fine-tuning Qwen3.5-9B on ProcessFlow v1.7 (108k synthetic reasoning samples) gave a 93% loss reduction but **zero PFE-Eval improvement**. The model learned the format, not the skill. Neither Full FT nor LoRA moved the Judge delta by more than 0.005.

The hypothesis: we need reward signals that point at the *cognitive circuits* we want to strengthen, not at the *output distribution*. Mech interp gives us a handle on those circuits. This library is the infrastructure to test that hypothesis.

**As of 2026-04-17, the hypothesis is validated** on Qwen3.5-4B: mech-reward GRPO took GSM8K from 64 % → 83 % in 168 effective steps, with MMLU preserved and adversarial-canary hack rate within the 95 % CI of the baseline model. The Stage Gate 2 ablation also shows that the sparse SAE decomposition is causally necessary: the same contrastive signal used as a raw direction (R2) is −9 pp worse than outcome-only, while its sparse SAE projection (R1) is +2 pp better — an 11 pp gap that directly rebuts the "just use a linear probe" alternative.

Closest concurrent work: Goodfire's [RLFR](https://arxiv.org/abs/2602.10067) (Feb 2026) established the feature-as-reward paradigm on dense Gemma-3-12B-IT for hallucination reduction. `mechreward` extends it to sparse-SAE features, per-token granularity, and hybrid architectures — three distinct methodological axes.

Full write-up: [LessWrong post](https://www.lesswrong.com/posts/H7mnTT7aPPijpjLAS/per-token-sae-features-as-online-rl-reward-breaking-the-g2) · [RESEARCH.md](RESEARCH.md).

## Contributing

This is alpha software. Issues and PRs welcome, but expect rapid breakage. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use `mechreward` in research, please cite:

```bibtex
@software{mechreward2026,
  author = {Vicentino, Caio},
  title = {mechreward: Mechanistic interpretability as reward signal for RL},
  year = {2026},
  url = {https://github.com/caiovicentino/mechreward}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Related projects

- [SAE Lens](https://github.com/jbloomAus/SAELens) — SAE training and loading
- [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) — pre-trained SAEs for Gemma
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — interpretability primitives
- [nnsight](https://nnsight.net/) — model internals API
- [TRL](https://github.com/huggingface/trl) — HuggingFace RL library
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — scalable RLHF
- [verl](https://github.com/verl-project/verl) — ByteDance reasoning RL
- [Neuronpedia](https://neuronpedia.org) — interactive SAE feature explorer
- [Delphi](https://github.com/EleutherAI/delphi) — automated interpretability
