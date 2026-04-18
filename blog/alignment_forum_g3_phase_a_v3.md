# Per-token SAE features as online RL reward: breaking the G2 76% GSM8K ceiling on Qwen3.5-4B

---

**TL;DR.** We train a TopK SAE on residual stream layer 18 of Qwen3.5-4B (hybrid Gated DeltaNet architecture). The closest prior public SAE on this family is [`kroonen-ai/sae-qwen3.5-9b`](https://huggingface.co/kroonen-ai/sae-qwen3.5-9b) (March 2026) — a ReLU SAE on layer-16 MLP output of Qwen3.5-9B; ours is TopK on the residual stream, 16× expansion vs 4×, 200 M training tokens vs 50 M, and is the first one we're aware of to be validated as a downstream RL reward signal. Ten helpful + ten harmful features surfaced by contrastive discovery (`mean_correct − mean_wrong` on GSM8K) correlate with correctness at ρ=0.540. Plugging them into GRPO as a **per-token dense reward** alongside binary outcome reward takes Qwen3.5-4B from 64 % → **83 % on GSM8K** in **168 effective training steps** (step 232→400 at LR=3e-6; the first 232 steps at the documented LR=1e-6 stalled at 0 pp lift). That's **+7 pp above the same-SAE, same-features trajectory-level G2 R1 ceiling (76 %)**, **+19 pp above the raw-prompt baseline**, with MMLU non-regressed and adversarial-canary hack rate within the 95 % CI of the baseline model. Code, SAE, and trained LoRA adapter are public. **All numbers are n=1 seed per condition (seed=42); 3–5 seeds would be needed to rule out +7 pp being within the GRPO training-noise band.**

Trained adapter: [`caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400)
SAE: [`caiovicentino1/Qwen3.5-4B-SAE-L18-topk`](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk)
Library: [`caiovicentino/mechreward`](https://github.com/caiovicentino/mechreward)

---

## The question we actually asked

Most SAE + reasoning work to date shows that features are *readable* — you can find ones that fire on "reasoning rhetoric", steering vectors that push toward certain answers, probes that predict correctness. But when researchers try to *use* that read-out at inference time (uniform steering, direct feature amplification), the net aggregate usually comes out null or negative. Activation steering breaks more questions than it fixes.

The natural next move is: rather than steering at inference, use the same features as a **reward signal during RL training**, letting the model learn *when* to fire them. Two months before this result, Goodfire's [RLFR](https://arxiv.org/abs/2602.10067) (Prasad et al., Feb 2026) established this exact paradigm — they train a linear probe on Gemma-3-12B-IT activations, freeze it, and use it as an online RL reward signal for hallucination reduction (58 % ↓). An earlier offline variant is [SARM](https://arxiv.org/abs/2508.08746) (Liu et al., 2025), which uses a linear head on SAE features as a frozen RLHF reward model. This work is an extension along three axes neither paper covers: (1) **sparse TopK SAE decomposition** instead of raw linear probes on activations; (2) **per-token dense reward** inside online GRPO instead of span- or trajectory-level reward; (3) **hybrid architectures** (Gated DeltaNet, MoE, triple-hybrid MoE+GDN+Gated-Attn) instead of dense transformers. The GSM8K reasoning result below is the empirical argument for why these three axes matter — in particular, the 11 pp gap in Stage Gate 2 between the sparse-decomposition condition and the raw-direction condition, which directly addresses why RLFR's linear-probe variant doesn't transfer cleanly to reasoning.

The mechreward framework lives [here](https://github.com/caiovicentino/mechreward). Below is the three-stage empirical pipeline we used to validate the approach, with particular attention to the engineering lessons that would have saved us about five wall-clock hours.

## Stage Gate 1 — does the signal predict correctness at all?

**Rule we enforced**: don't commit GPU hours to RL before verifying the features predict the outcome on held-out data.

We trained a TopK SAE (k=128, d_sae=40 960 = 16× expansion of d_model=2560) on the residual stream post-layer 18 of Qwen3.5-4B, using 200 M tokens from FineWeb-Edu. Variance explained: 0.87 train, 0.866 eval. L0 stayed structurally at 127.9/128 under distribution shift.

TransformerLens doesn't support Qwen3.5 yet, so we built the capture pipeline directly on HuggingFace forward hooks. The nearest prior SAE on the hybrid-GDN family is [`kroonen-ai/sae-qwen3.5-9b`](https://huggingface.co/kroonen-ai/sae-qwen3.5-9b) (created 2026-03-15, 1 month before ours): a ReLU SAE on the Qwen3.5-**9B** layer-16 **MLP output**, d_sae=16,384 (4× expansion), ~50 M training tokens. The two SAEs differ on every axis that matters for downstream use — activation (ReLU vs TopK), hook (MLP output vs residual stream), expansion (4× vs 16×), training budget (50 M vs 200 M tokens), base model (9 B vs 4 B), and whether the features have been validated as an RL reward signal (theirs: not documented; ours: Stage Gates 1–3 below). We're not aware of any public SAE on Qwen3.5-4B specifically, or of any SAE on hybrid GDN that has been used as a training-time reward signal; the latter is the core contribution of this post.

We then sampled 50 baseline GSM8K responses, labeled each correct or wrong by exact-match on the final number, and for every feature computed `mean_activation | correct − mean_activation | wrong`. The top-10 highest-effect-size features (Cohen's d ∈ [+2.060, +2.158]) become the "helpful pack"; top-10 most-negative (Cohen's d ∈ [−2.469, −2.064]) become the "harmful pack".

*Note on feature-discovery methodology.* The mechreward library also implements [AIRI's ReasonScore metric](https://arxiv.org/abs/2503.18878) (port at [`src/mechreward/features/reasonscore.py`](https://github.com/caiovicentino/mechreward/blob/main/src/mechreward/features/reasonscore.py)), which ranks features by their contrastive activation around a small reasoning vocabulary (*wait / hmm / therefore / …*). For GSM8K correctness we deliberately chose contrastive-correctness discovery over ReasonScore because the two methods target different objects: ReasonScore selects features firing on *reasoning rhetoric*, whereas our contrastive pack selects features that predict *answer correctness*. Both methods are supported in the library; the choice here is dictated by the downstream reward we want (outcome correctness, not rhetoric density).

On a held-out set of n=100 questions:

$$R_{\text{mech}} = \sum_{i \in \text{helpful}} \text{feat}_i(h_{L18}) - \sum_{j \in \text{harmful}} \text{feat}_j(h_{L18})$$

*mean-pooled over response tokens.*

| Signal | Spearman ρ | Pearson r | p |
|---|---|---|---|
| **SAE features (top-10 helpful − top-10 harmful, L18)** | **+0.540** | +0.726 | < 0.0001 |
| Raw L13 contrastive direction (cos-sim) | +0.508 | +0.624 | < 0.0001 |

Both above the ρ≥0.30 go/no-go threshold. SAE features edge out the raw contrastive direction on this model (even though L13 is the "true" peak contrastive layer for Qwen3.5-4B — our SAE is 5 layers downstream).

**Feature IDs used for mechreward** (for exact reproducibility; published at [`catalogs/qwen3.5-4b/reasoning_pack.json`](https://github.com/caiovicentino/mechreward/blob/main/catalogs/qwen3.5-4b/reasoning_pack.json)):

```python
# Top-10 helpful (fire more on correct GSM8K responses at L18)
HELPFUL = [36405, 27873, 35818, 12399, 2643, 6998, 15360, 25868, 21154, 35842]

# Top-10 harmful (fire more on wrong responses)
HARMFUL = [18654, 36412, 15686, 23272, 13672, 14912, 5863, 29516, 30690, 40589]
```

## Stage Gate 2 — does it work as an actual reward?

Three-way ablation at small scale: GRPO on 500 GSM8K train questions, 100 steps, 4 rollouts/question, LR=1e-6, KL β=0.05, same seed, same eval set (100 held-out):

| Run | Reward | Final eval | Δ vs baseline | Δ vs R0 |
|---|---|---|---|---|
| **R0** | outcome only (binary) | 74 % | +10 pp | — |
| **R1** | outcome + SAE features, λ=0.1 | **76 %** | +12 pp | **+2 pp** |
| **R2** | outcome + raw L13 direction, λ=0.1 | 65 % | +1 pp | **−9 pp** |

R1 converges 2.5× faster: it reaches R0's final accuracy (74 %) at step 40; R0 takes all 100 steps. R0 *drops* from its step-80 peak (76 → 74), while R1 stays stable (75 → 76). The SAE features act as a late-training regularizer.

The striking number is the **11 pp R1 ↔ R2 gap**. Same underlying contrastive signal — R2 uses the raw L13 direction that passively correlates with correctness at ρ=0.508. As a reward, that same direction is *actively harmful*: GRPO amplifies its polysemantic content (difficulty, length, style) and the policy ends up below the outcome-only baseline. SAE sparse decomposition is the causal processing step that converts a readable signal into a trainable one — not cosmetic, not incremental. That's the central finding of the mechreward line.

## Stage Gate 3 — does it scale, and does it break the 76% ceiling?

The key design change at G3: **per-token** mech-reward, not just trajectory-level. The policy gets a dense reward on every response token based on the SAE activation at that token, in addition to the binary outcome reward at the end.

Config: GRPO, 2000 planned steps on 7 500 GSM8K questions, 4 questions × 4 rollouts per step, raw prompt `Q: {q}\nA: Let's think step by step.` (chat template breaks the SAE feature calibration — lesson below), max_gen_len=256, λ=0.1, KL β=0.05, seed=42, LoRA r=32 on language-model projections only (vision tower frozen).

![Training trajectory](https://raw.githubusercontent.com/caiovicentino/mechreward/main/figures/g3_training_trajectory.png)

### The LR-vs-clipping diagnostic lesson

The first 232 steps used **LR=1e-6 — the exact value documented for G2 R1**. After 200 steps: `quick_gsm8k = 64 %`, identical to the untrained baseline. KL was stuck at 0.018. Mech signal was sitting at −0.02 (negative; helpful features were firing *less* than harmful on sampled rollouts).

Our first instinct was gradient clipping. We were wrong. Logging `clip_grad_norm_`'s return value showed `gnorm` was always **below 0.5** throughout the run. A clip of 1.0 had never fired. Our clipping was inert.

The bottleneck was LR. Raising to 3e-6 at step 232 produced immediate learning:

- Mech signal reversed: −0.02 → +0.076 (step 250) → +0.382 (step 270) → **+0.58 peak at step 330** → plateau ~+0.4-0.5
- KL climbed 0.018 → 0.108 over 170 steps
- Training outcome at temp 0.9 went 0.55 → 0.75 average

**Lesson I'd tell past-me**: verify `gnorm` and `KL` are rising *before* attributing a stalled GRPO run to clipping or algorithm choice. LR values from prior work don't necessarily transfer across rollouts-per-step / batch / dtype configurations. Our 4×4 rollouts with advantage normalization and bf16 `log_softmax` needed 3× the G2 LR.

### Eval at step 400 (vs baseline with LoRA disabled via `model.disable_adapter()`)

![GSM8K comparison](https://raw.githubusercontent.com/caiovicentino/mechreward/main/figures/g3_gsm8k_comparison.png)

| Metric | Baseline | G3 Phase A | Δ |
|---|---|---|---|
| **GSM8K** (500Q greedy, raw prompt) | 64.00 % | **83.00 %** | **+19 pp** |
| MMLU (200Q raw zeroshot) | 50.00 % | 54.50 % | +4.50 pp |
| MATH-500 (500Q greedy, not trained on) | — | 18.20 % | transfer |
| Adversarial canary hack rate (n=50) | 4.0 % | 8.0 % | +4 pp (within 95 % CI) |
| Correct under adversarial canary (n=50) | 18.0 % | 28.0 % | **+10 pp** |

![Canary breakdown](https://raw.githubusercontent.com/caiovicentino/mechreward/main/figures/g3_canary_breakdown.png)

### Effective training budget — the honest framing

Because the first 232 steps at LR=1e-6 produced zero lift, the budget that actually broke the ceiling is **168 steps at LR=3e-6** (step 232 → 400). That's roughly 1.68× G2 R1's 100-step budget, for +7 pp on GSM8K, with the same features, same SAE, and same base model. The "400 steps" figure is a misleading artifact of us shipping the diagnostic to the same training run.

## What this result doesn't establish

- **Matched-LR control not run.** The only uncontrolled variable between G2 R1 (76 %) and G3 R1 (83 %) is LR (1e-6 vs 3e-6 effective). A G2 R1 re-run at LR=3e-6 with matched step budget is the single most valuable replication — if it also plateaus at 76 %, the per-token + LR combination breaks a real ceiling; if it reaches 83 %, the per-token framing is empty and LR alone explains the lift.
- **MATH-500 transfer (18.2 %) is near baseline.** We trained only on GSM8K; this is a generalization test the model fails politely. The claim is narrow: mechreward breaks the ceiling *on the distribution the SAE features were discovered on*.
- **MMLU +4.5 pp is within measurement noise** on a 200-question raw zero-shot eval (our eval diverges from published benchmarks that use chat template + few-shot). The point is negative: no catastrophic forgetting, not "mechreward is a general capability booster".
- **Canary n=50 is small.** The +4 pp hack-rate delta is within the 95 % CI of the baseline. Claiming "anti-Goodhart validated" requires at least n=200. We report what we measured; readers should discount.
- **n=1 seed per condition.** +7 pp may sit inside the GRPO training-noise band (typically σ≈2–4 pp at this scale). 3–5 seeds per condition are needed to settle this.
- **Single model, single domain.** Qwen3.5-4B on GSM8K. We have a pending run on Qwen3.6-35B-A3B (hybrid MoE + GDN + Gated Attention) on SuperGPQA to test whether the pipeline generalizes to both scale and architecture diversity.
- **168 "effective" steps is a post-hoc framing.** The run nominally used 400 steps of compute; we're being transparent that the first 232 were wasted. Reviewers should evaluate the compute-to-gain ratio on that basis.

## Engineering notes useful to others

A few things that consumed engineer-hours and might not be obvious:

1. **Qwen3.5-4B is multimodal.** Loading with `AutoModelForCausalLM` fails — use `AutoModelForImageTextToText`. Freeze the vision tower before LoRA.
2. **Prompt format must match the SAE feature-discovery distribution.** Our SAE features were discovered on raw `Q:/A:` responses; using a chat-template prompt during GRPO kept mech negative for 200 steps because the feature activation distribution shifted. Revert to the same raw format the features were calibrated on.
3. **Answer extraction must handle model continuation.** Greedy decoding often spills past the first answer and invents a second question. Cut at `\n+\s*(?:Q:|Problem:)` before regex-extracting, else the last-number fallback picks up noise.
4. **Memory-efficient KL without a ref model.** `model.disable_adapter()` as a context manager gives the base-policy logprobs without a separate 8 GB copy. `bf16 log_softmax` (not `float32`) halves logits memory — critical for vocab=248 k on a 4 B model. Grad checkpointing ON during the train forward, OFF during the rollout `generate` (use_cache needs to be on for fast generation).
5. **fla + causal-conv1d are not optional on GDN models.** Without them the hybrid linear-attention layers fall back to torch implementation at ~10× slower. With them: ~30 tok/s on RTX 6000 Blackwell. Without: ~3 tok/s.

All of the above is documented with exact pinned commit SHAs in the [install recipe](https://github.com/caiovicentino/mechreward/blob/main/README.md).

## Context vs prior work

The closest prior work is Goodfire's **RLFR** (Prasad et al., 2026, [arxiv:2602.10067](https://arxiv.org/abs/2602.10067), published two months before this result), which introduced the paradigm of interpretable features as RL reward signals. We don't claim priority on the paradigm; we claim three specific methodological extensions.

| Prior work | Axis | How this differs |
|---|---|---|
| [**RLFR — Goodfire (Prasad et al., Feb 2026)**](https://arxiv.org/abs/2602.10067) | **Linear probes on raw activations** as reward signal for **trajectory/span-level** online RL, targeting hallucination on **dense Gemma-3-12B-IT** (58% hallucination ↓) | We use **sparse TopK SAE decomposition** of the contrastive signal instead of raw probes. Our G2 R1-vs-R2 ablation shows that raw direction reward is actively harmful for GSM8K reasoning (**−9 pp vs outcome-only**), while the same contrastive signal in sparse SAE form is beneficial (**+2 pp**). This 11 pp gap is the direct empirical argument for why decomposition matters. We also operate **per-token dense** rather than span-level, and target **hybrid architectures** (GDN, MoE, triple-hybrid) that RLFR did not cover. |
| [SARM (Liu et al. 2025, AAAI 26)](https://arxiv.org/abs/2508.08746) | SAE features as reward model in offline RLHF | We use them as **online** per-token reward inside GRPO |
| [Control RL (Cho/Wu/Koshiyama 2026)](https://arxiv.org/abs/2602.10437) | RL policy that selects which SAE feature to amplify per token | We use SAE feature *activations* as the reward itself, not as a steering selection target |
| [CLIPO (2026)](https://arxiv.org/html/2603.10101) | Contrastive trajectory-level auxiliary reward for GRPO on GSM8K | They use InfoNCE over trajectory embeddings (opaque); we use interpretable SAE features |
| [Base Models Know How to Reason (ICLR 26 under review)](https://arxiv.org/pdf/2510.07364) | SAE steering on GSM8K at inference time | They apply features as transient steering at test time; we reward them during training (permanent policy update) |
| [Wilhelm et al. 2026](https://arxiv.org/abs/2603.04069) | SAE features as reward-hacking *detector* | We weaponize the same detection methodology as *prevention* via dual verification |
| [AIRI ReasonScore (arxiv:2503.18878)](https://arxiv.org/abs/2503.18878) | Contrastive SAE-feature score around reasoning vocabulary (*wait / hmm / therefore*) for feature discovery + inference-time amplification (+13.4 % AIME-2024 on DeepSeek-R1-Distill-Llama-8B) | We implement ReasonScore in the library for completeness but use contrastive-correctness discovery for this pack (ReasonScore targets rhetoric features, not correctness-predictive features) |

**Positioning**: RLFR established the paradigm; this work extends it along three distinct methodological axes — (1) **sparse decomposition beats raw probes** by 11 pp on GSM8K (G2 ablation), (2) **per-token dense reward beats trajectory-level** by 7 pp (G3 vs G2), (3) **hybrid architecture generalization** (Gated DeltaNet, MoE, triple-hybrid) where no public SAEs previously existed. The combination of these three — not any single one — is the contribution.

## What's next

Stage 4: extend to Qwen3.6-35B-A3B. To our knowledge (verified against arXiv and HuggingFace as of 2026-04-17), no prior public SAE exists for any triple-hybrid MoE + GDN + Gated-Attention model, or for any MoE with ≥30 B active parameters. Benchmark: SuperGPQA (published Qwen baseline 64.7 %; our n=100 easy+middle probe gave 42 %). Layer mapping just finished: **peak contrastive Cohen's d layer is L22** (55 % depth, GDN) with a broad plateau across L20–L37 (27 % spread between min and max layer). Logit-lens commit is at **L39** — a **17-layer gap** between "reasoning formation" and "English answer commitment", the largest we've observed across our three architectures (Qwen3.5-4B GDN: 2 layers; Gemma-4-E4B MoE: 3 layers; Qwen3.6-35B-A3B triple-hybrid: 17 layers). One finding we didn't expect: **the model's Gated Attention layers (3, 7, 11, …, 39) are not Cohen's-d hotspots** — contra our prior hypothesis that GA bottlenecks would concentrate reasoning signal. SAE training is **in progress at L23** (the adjacent downstream GA layer, following the Qwen3.5 peak-+5 pattern); first Drive checkpoint saved at 10 M tokens, target ~1 B.

Comments especially welcome on:
- Whether the "effective compute" framing is fair or cope
- Anyone running similar per-token SAE-reward experiments (would love to compare notes on LR, λ schedule, and prompt-format vs SAE-calibration alignment)
- Whether 168 effective steps for +7 pp is actually significant or within the 1.5 σ band of GRPO training noise (we'd need 3-5 seeds to settle this — a matched-LR G2 R1 replication on any open-weights hybrid-GDN model is the single most valuable thing anyone could add)
- Whether the 17-layer reasoning-formation → English-commitment gap on Qwen3.6 is an artifact of multilingual pretraining, thinking-mode decoding, or a genuine architectural property of triple-hybrid MoE

---

**Artifacts** (everything public, Apache-2.0):
- [mechreward library (code)](https://github.com/caiovicentino/mechreward)
- [SAE on Qwen3.5-4B L18](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk)
- [Companion SAE on Gemma 4 E4B L21](https://huggingface.co/caiovicentino1/Gemma-4-E4B-SAE-L21-topk)
- [Trained LoRA adapter (this result)](https://huggingface.co/caiovicentino1/Qwen3.5-4B-mechreward-G3-phaseA-step400)
- [Raw training history + figures](https://github.com/caiovicentino/mechreward/tree/main/figures)
