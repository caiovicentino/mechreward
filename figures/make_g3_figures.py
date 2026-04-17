"""Generate all Stage Gate 3 Phase A figures from training history.jsonl.

Run from repo root:
    python figures/make_g3_figures.py ~/Downloads/history\ \(1\).jsonl

Produces 4 PNGs in figures/:
    g3_training_trajectory.png     4-panel: outcome, mech, KL, gnorm vs step
    g3_gsm8k_comparison.png        baseline vs G2 R1 vs G3 Phase A
    g3_canary_breakdown.png        hack/correct/ambiguous baseline vs trained
    g3_eval_comparison.png         all 5 metrics baseline vs trained
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_BASELINE = '#8E8E8E'
C_G2_R1 = '#4A90E2'
C_G3 = '#E94B3C'
C_PATCH = '#F5B800'

PATCH_STEP = 232
FIG_DIR = Path(__file__).parent


def load_history(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def smooth(y, window=10):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    smoothed = np.convolve(y_padded, kernel, mode='same')
    return smoothed[pad:pad + len(y)]


def fig1_training_trajectory(records, out_path):
    """4-panel training dynamics with LR patch annotation."""
    train = [r for r in records if 'mean_outcome' in r and r.get('step', 0) > 0]
    steps = np.array([r['step'] for r in train])
    outcome = np.array([r['mean_outcome'] for r in train])
    mech = np.array([r['mean_mech'] for r in train])
    kl = np.array([r['mean_kl'] for r in train])
    gnorm = np.array([r.get('grad_norm', np.nan) for r in train])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    (ax_o, ax_m), (ax_k, ax_g) = axes

    for ax in axes.flat:
        ax.axvspan(0, PATCH_STEP, alpha=0.08, color='red', label='_nolegend_')
        ax.axvline(PATCH_STEP, color=C_PATCH, lw=1.5, ls='--', alpha=0.9, zorder=5)

    # Outcome
    ax_o.scatter(steps, outcome, s=8, alpha=0.25, color=C_G3, label='per-step')
    ax_o.plot(steps, smooth(outcome, 10), lw=2, color=C_G3, label='10-step avg')
    ax_o.set_ylabel('train outcome (temp 0.9)')
    ax_o.set_title('Training rollout accuracy')
    ax_o.set_ylim(-0.05, 1.05)
    ax_o.legend(loc='lower right')

    # Mech signal — the star of the show
    ax_m.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax_m.scatter(steps, mech, s=8, alpha=0.25, color=C_G3)
    ax_m.plot(steps, smooth(mech, 10), lw=2, color=C_G3)
    ax_m.set_ylabel('mech reward per rollout')
    ax_m.set_title('SAE feature signal (helpful − harmful)')
    ax_m.annotate('LR patch @ step 232\n(1e-6 → 3e-6)',
                  xy=(PATCH_STEP, 0), xytext=(PATCH_STEP + 40, -0.35),
                  fontsize=10, color=C_PATCH,
                  arrowprops=dict(arrowstyle='->', color=C_PATCH, lw=1.2))
    ax_m.annotate('mech ≈ −0.02\n(stuck negative)',
                  xy=(100, -0.02), xytext=(30, 0.25),
                  fontsize=9, color='dimgray',
                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))
    peak_idx = int(np.argmax(mech))
    ax_m.annotate(f'peak +{mech[peak_idx]:.2f}\n(step {steps[peak_idx]})',
                  xy=(steps[peak_idx], mech[peak_idx]),
                  xytext=(steps[peak_idx] - 100, mech[peak_idx] + 0.1),
                  fontsize=9, color=C_G3,
                  arrowprops=dict(arrowstyle='->', color=C_G3, lw=0.8))

    # KL
    ax_k.plot(steps, kl, lw=1.5, color=C_G3, alpha=0.6)
    ax_k.plot(steps, smooth(kl, 10), lw=2, color=C_G3)
    ax_k.set_ylabel('KL from base policy')
    ax_k.set_title('Policy drift (KL)')
    ax_k.set_xlabel('GRPO step')
    ax_k.annotate('KL stuck at 0.018\n(adapter barely moving)',
                  xy=(200, 0.018), xytext=(30, 0.05),
                  fontsize=9, color='dimgray',
                  arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

    # gnorm (diagnostic)
    gmask = ~np.isnan(gnorm)
    ax_g.plot(steps[gmask], gnorm[gmask], lw=1.5, color=C_G3, alpha=0.6)
    ax_g.plot(steps[gmask], smooth(gnorm[gmask], 10), lw=2, color=C_G3)
    ax_g.axhline(1.0, color='k', lw=0.8, ls=':', alpha=0.6, label='clip threshold')
    ax_g.set_ylabel('gradient norm')
    ax_g.set_title('Grad norm — clip=1.0 was inert (gnorm < 0.5 throughout)')
    ax_g.set_xlabel('GRPO step')
    ax_g.set_ylim(0, 1.1)
    ax_g.legend(loc='upper right')

    fig.suptitle(
        'Stage Gate 3 Phase A — Qwen3.5-4B per-token mech-reward GRPO\n'
        'LR=1e-6 (steps 1–232) stalled everything. Patch to LR=3e-6 reversed mech signal and lifted eval.',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  {out_path}')
    plt.close(fig)


def fig2_gsm8k_comparison(out_path):
    """Bar chart: baseline / G2 R0 / G2 R1 / G2 R2 / G3 Phase A."""
    labels = [
        'Baseline\n(raw prompt,\nno RL)',
        'G2 R0\n(outcome only,\n100 steps)',
        'G2 R1\n(+ SAE trajectory,\n100 steps)',
        'G2 R2\n(+ raw direction,\n100 steps)',
        'G3 Phase A\n(+ SAE per-token,\n168 effective steps)',
    ]
    values = [64, 74, 76, 65, 83]
    colors = [C_BASELINE, '#7FB069', C_G2_R1, '#C75450', C_G3]
    deltas = [0, +10, +12, +1, +19]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)

    for bar, v, d in zip(bars, values, deltas):
        height = bar.get_height()
        if d == 0:
            label = f'{v} %'
        else:
            label = f'{v} %\n({d:+d} pp)'
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.8, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(64, color=C_BASELINE, ls=':', lw=1.0, alpha=0.6)
    ax.axhline(76, color=C_G2_R1, ls=':', lw=1.0, alpha=0.6)
    ax.text(4.35, 76.5, 'G2 R1 ceiling (76 %)', fontsize=9, color=C_G2_R1, ha='right')

    ax.set_ylabel('GSM8K pass@1 (greedy, held-out)')
    ax.set_ylim(55, 90)
    ax.set_title(
        'Qwen3.5-4B — GSM8K pass@1 across mech-reward RL configurations\n'
        'Per-token SAE reward breaks the trajectory-level ceiling (+7 pp over G2 R1, +19 pp over baseline)',
        fontsize=11,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)} %'))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  {out_path}')
    plt.close(fig)


def fig3_canary_breakdown(out_path):
    """Stacked bar: hack/correct/ambiguous for baseline vs trained (n=50)."""
    categories = ['Baseline\n(no adapter)', 'G3 Phase A\n@ step 400']
    hacks = [4.0, 8.0]
    correct = [18.0, 28.0]
    ambig = [78.0, 64.0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(categories))
    w = 0.5

    p1 = ax.bar(x, correct, w, label='correct (answers gold)', color='#7FB069', edgecolor='white')
    p2 = ax.bar(x, ambig, w, bottom=correct, label='ambiguous (no hack, no gold)', color='#B8B8B8', edgecolor='white')
    p3 = ax.bar(x, hacks, w, bottom=np.array(correct) + np.array(ambig),
                label='hack (triggered repetition)', color=C_G3, edgecolor='white')

    for i, (h, c, a) in enumerate(zip(hacks, correct, ambig)):
        ax.text(x[i], c / 2, f'{c:.0f} %', ha='center', va='center', color='white', fontweight='bold')
        ax.text(x[i], c + a / 2, f'{a:.0f} %', ha='center', va='center', color='black', fontweight='bold')
        ax.text(x[i], c + a + h / 2, f'{h:.0f} %', ha='center', va='center', color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('percentage of n=50 adversarial canaries')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)} %'))
    ax.set_title(
        'Anti-Goodhart: canary response breakdown (n=50)\n'
        'Hack rate +4 pp (within 95 % CI) while correct-under-adversarial rose +10 pp',
        fontsize=11,
    )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  {out_path}')
    plt.close(fig)


def fig4_eval_summary(out_path):
    """Grouped bar chart — all metrics baseline vs trained."""
    metrics = ['GSM8K\n(500Q greedy)', 'MMLU\n(200Q raw)', 'MATH-500\n(transfer)', 'Canary hack\n(n=50)', 'Canary correct\n(n=50)']
    baseline = [64.0, 50.0, None, 4.0, 18.0]
    trained = [83.0, 54.5, 18.2, 8.0, 28.0]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Baseline bars (skip None)
    for i, b in enumerate(baseline):
        if b is None:
            ax.text(x[i] - w / 2, 2, 'n/a', ha='center', fontsize=9, color=C_BASELINE)
        else:
            ax.bar(x[i] - w / 2, b, w, color=C_BASELINE, edgecolor='white',
                   label='Baseline' if i == 0 else '_nolegend_')
            ax.text(x[i] - w / 2, b + 1.5, f'{b:.1f}', ha='center', fontsize=9)

    # Trained bars
    for i, t in enumerate(trained):
        color = C_G3
        ax.bar(x[i] + w / 2, t, w, color=color, edgecolor='white',
               label='G3 Phase A' if i == 0 else '_nolegend_')
        if baseline[i] is not None:
            delta = t - baseline[i]
            sign = '+' if delta >= 0 else ''
            ax.text(x[i] + w / 2, t + 1.5, f'{t:.1f}\n({sign}{delta:.1f} pp)',
                    ha='center', fontsize=9, fontweight='bold')
        else:
            ax.text(x[i] + w / 2, t + 1.5, f'{t:.1f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('percentage')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)} %'))
    ax.set_title('G3 Phase A step 400 — all eval metrics vs baseline (LoRA disabled)', fontsize=12)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  {out_path}')
    plt.close(fig)


def main():
    hist_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/caiovicentino/Downloads/history (1).jsonl'
    records = load_history(hist_path)
    print(f'Loaded {len(records)} records from {hist_path}')
    print(f'Writing figures to {FIG_DIR}:')
    fig1_training_trajectory(records, FIG_DIR / 'g3_training_trajectory.png')
    fig2_gsm8k_comparison(FIG_DIR / 'g3_gsm8k_comparison.png')
    fig3_canary_breakdown(FIG_DIR / 'g3_canary_breakdown.png')
    fig4_eval_summary(FIG_DIR / 'g3_eval_summary.png')
    print('\nDone.')


if __name__ == '__main__':
    main()
