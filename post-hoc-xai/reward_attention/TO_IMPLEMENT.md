# TO_IMPLEMENT — Pre-Scale Validation & New Analyses

> Run these BEFORE launching the minimal 50-scenario experiment.
> All tasks are CPU-only (no GPU needed). All data already exists in pkl files.
> Estimated total time: 3–4 hours of coding + analysis.
> Last updated: 2026-02-23

---

## File to Create

```
reward_attention/
└── validate_and_extend.py    # Single script, 3 sections (run with flags)
```

Usage:
```bash
cd /home/med1e/post-hoc-xai
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

# Run all checks
python reward_attention/validate_and_extend.py --all

# Or run individually
python reward_attention/validate_and_extend.py --validate
python reward_attention/validate_and_extend.py --entropy
python reward_attention/validate_and_extend.py --leadlag
```

---

## Section 0: Data Loading (shared by all sections)

All three sections load the same pkl files. Here's the shared loading logic:

```python
import pickle
import numpy as np
from pathlib import Path
from scipy import stats

COMPLETE_PKL = Path('results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl')
MINIMAL_PKL  = Path('results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl')
BASIC_PKL    = Path('results/reward_attention/womd_sac_road_perceiver_basic_42/timestep_data.pkl')

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_scenario(records, scenario_id):
    """Return records for one scenario, sorted by timestep."""
    recs = [r for r in records if r.scenario_id == scenario_id]
    return sorted(recs, key=lambda r: r.timestep)

def get_timeseries(recs, field):
    """Extract a numpy array of one field from sorted records."""
    return np.array([getattr(r, field) for r in recs])
```

### TimestepRecord fields available (from config.py):
```
scenario_id, timestep
attn_sdc, attn_agents, attn_roadgraph, attn_lights, attn_gps
attn_to_nearest, attn_to_threat
collision_risk, safety_risk, navigation_risk, behavior_risk
min_ttc, accel, steering, ego_speed
num_valid_agents, is_collision_step, is_offroad_step
```

**Check if `attn_per_agent` exists** — the NEXT_STEPS.md says it's in the records but
this needs verification. If it's NOT a field on TimestepRecord, you'll need to either:
(a) add it to the extractor (requires GPU rerun — avoid), or
(b) compute entropy from the 5 category-level values instead (less granular but still useful).

```python
# Quick check:
recs = load_pkl(COMPLETE_PKL)
print(dir(recs[0]))  # Look for attn_per_agent
print(hasattr(recs[0], 'attn_per_agent'))
```

---

## Section 1: Validation Checks (MUST PASS before any 50-scenario run)

### Check 1A: Attention Budget Invariant

**What**: Verify that `attn_sdc + attn_agents + attn_roadgraph + attn_lights + attn_gps ≈ 1.0`
at every single timestep, for every model.

**Why**: If this doesn't hold, the attention extraction or aggregation has a bug. All downstream
findings (trade-offs, budget reallocation) would be invalid.

**Tolerance**: `|sum - 1.0| < 0.01` (1% tolerance for floating-point). Flag any timestep
that exceeds this.

```python
def check_attention_budget(records, model_name):
    """
    For every record, compute sum of 5 attention categories.
    Print: min, max, mean, std of the sum.
    Flag any record where |sum - 1.0| > 0.01.
    """
    violations = []
    sums = []
    for r in records:
        s = r.attn_sdc + r.attn_agents + r.attn_roadgraph + r.attn_lights + r.attn_gps
        sums.append(s)
        if abs(s - 1.0) > 0.01:
            violations.append((r.scenario_id, r.timestep, s))

    sums = np.array(sums)
    print(f"\n{'='*60}")
    print(f"ATTENTION BUDGET CHECK: {model_name}")
    print(f"  Records:    {len(records)}")
    print(f"  Sum min:    {sums.min():.6f}")
    print(f"  Sum max:    {sums.max():.6f}")
    print(f"  Sum mean:   {sums.mean():.6f}")
    print(f"  Sum std:    {sums.std():.6f}")
    print(f"  Violations: {len(violations)} / {len(records)}")
    if violations:
        print(f"  First 5 violations:")
        for sid, t, s in violations[:5]:
            print(f"    scenario={sid}, t={t}, sum={s:.6f}")
    print(f"  RESULT: {'PASS ✓' if len(violations) == 0 else 'FAIL ✗'}")
    return len(violations) == 0
```

**If this fails**: Trace back to `extractor.py` → check how TOKEN_RANGES are summed.
The 280 tokens should partition exhaustively into the 5 groups:
```
sdc:            [0, 5)      →  5 tokens
other_agents:   [5, 45)     → 40 tokens
roadgraph:      [45, 245)   → 200 tokens
traffic_lights: [245, 270)  → 25 tokens
gps_path:       [270, 280)  → 10 tokens
                              --------
                              280 tokens total
```
Verify: `5 + 40 + 200 + 25 + 10 = 280` ✓

---

### Check 1B: Cross-Model Risk Profile Comparison

**What**: For the 3 overlapping scenarios (s000, s001, s002), compare the `collision_risk`
timeseries between complete, minimal, and basic models.

**Why**: You're comparing attention patterns across models on "the same scenarios." But the
ego policy differs between models, so the ego follows different trajectories, which means
the ego's distance to other agents differs, which means collision_risk may differ.

If `collision_risk` timeseries are nearly identical → the comparison is clean (risk is
dominated by logged agent trajectories approaching, regardless of ego behavior).

If `collision_risk` timeseries diverge substantially → you're NOT comparing attention
responses to the same stimulus. The models face different risk profiles. This doesn't
invalidate the research, but it changes the claim: you'd be comparing "attention under
each model's self-generated risk experience" not "attention to the same external threat."

**Implementation**:

```python
def compare_risk_profiles(complete_recs, minimal_recs, basic_recs=None):
    """
    For each overlapping scenario (0, 1, 2):
    - Extract collision_risk timeseries from each model
    - Compute Spearman correlation between complete and minimal
    - Compute mean absolute difference
    - Plot overlaid timeseries
    - Print summary
    """
    overlapping = [0, 1, 2]  # scenarios present in all models

    for sid in overlapping:
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)

        # Truncate to common length (basic may have terminated early)
        min_len = min(len(c_recs), len(m_recs))
        if min_len == 0:
            print(f"  s{sid:03d}: no overlapping timesteps — SKIP")
            continue

        c_risk = get_timeseries(c_recs[:min_len], 'collision_risk')
        m_risk = get_timeseries(m_recs[:min_len], 'collision_risk')

        # Correlation
        if np.std(c_risk) > 0.01 and np.std(m_risk) > 0.01:
            rho, p = stats.spearmanr(c_risk, m_risk)
        else:
            rho, p = float('nan'), float('nan')

        # Mean absolute difference
        mad = np.mean(np.abs(c_risk - m_risk))

        print(f"\n  s{sid:03d} (n={min_len} common timesteps):")
        print(f"    Spearman ρ(complete_risk, minimal_risk) = {rho:+.3f}  p={p:.4f}")
        print(f"    Mean |Δrisk| = {mad:.4f}")
        print(f"    Complete risk: mean={c_risk.mean():.3f} std={c_risk.std():.3f}")
        print(f"    Minimal  risk: mean={m_risk.mean():.3f} std={m_risk.std():.3f}")

        # Interpretation
        if rho > 0.8 and mad < 0.1:
            print(f"    → GOOD: Risk profiles are very similar. Cross-model attention")
            print(f"      comparison is responding to approximately the same risk stimulus.")
        elif rho > 0.5:
            print(f"    → MODERATE: Risk profiles correlate but differ in magnitude.")
            print(f"      Cross-model comparison is qualitatively valid but quantitative")
            print(f"      differences in attention may partly reflect different risk levels.")
        else:
            print(f"    → WARNING: Risk profiles diverge substantially. Models face")
            print(f"      different risk experiences. Attention differences may reflect")
            print(f"      different stimuli, not just different attentional strategies.")

        # Also handle basic if provided
        if basic_recs is not None:
            b_recs = get_scenario(basic_recs, sid)
            if len(b_recs) > 0:
                b_min = min(min_len, len(b_recs))
                b_risk = get_timeseries(b_recs[:b_min], 'collision_risk')
                b_mad = np.mean(np.abs(c_risk[:b_min] - b_risk))
                print(f"    Basic risk  (n={b_min}): mean={b_risk.mean():.3f} MAD vs complete={b_mad:.4f}")

    # ALSO: generate overlay plot
    _plot_risk_comparison(complete_recs, minimal_recs, basic_recs, overlapping)
```

**Generate figure**: `fig_risk_profile_comparison.png` — 3 subplots (one per scenario),
each showing complete and minimal collision_risk overlaid. If they track closely, you're good.

```python
import matplotlib.pyplot as plt

def _plot_risk_comparison(complete_recs, minimal_recs, basic_recs, scenario_ids):
    fig, axes = plt.subplots(len(scenario_ids), 1, figsize=(12, 3*len(scenario_ids)),
                             sharex=False)
    if len(scenario_ids) == 1:
        axes = [axes]

    for ax, sid in zip(axes, scenario_ids):
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len == 0:
            ax.set_title(f's{sid:03d} — no data')
            continue

        c_risk = get_timeseries(c_recs[:min_len], 'collision_risk')
        m_risk = get_timeseries(m_recs[:min_len], 'collision_risk')
        t = np.arange(min_len)

        ax.plot(t, c_risk, 'b-', linewidth=1.5, label='complete', alpha=0.8)
        ax.plot(t, m_risk, 'r--', linewidth=1.5, label='minimal', alpha=0.8)

        if basic_recs is not None:
            b_recs = get_scenario(basic_recs, sid)
            if len(b_recs) > 0:
                b_len = min(min_len, len(b_recs))
                b_risk = get_timeseries(b_recs[:b_len], 'collision_risk')
                ax.plot(np.arange(b_len), b_risk, 'g:', linewidth=1.5, label='basic', alpha=0.8)

        ax.set_ylabel('collision_risk')
        ax.set_title(f's{sid:03d}')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel('timestep')
    fig.suptitle('Cross-Model Risk Profile Comparison\n(same scenarios, different policies)',
                 fontsize=12)
    plt.tight_layout()
    save_path = Path('results/reward_attention/fig_risk_profile_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()
```

---

### Check 1C: Vigilance Gap Replication on s000

**What**: Plot complete vs. minimal `attn_agents` timeseries for s000 (not just s002).
Check whether the vigilance gap (complete > minimal during calm phases) holds in a
second scenario.

**Why**: Currently the vigilance prior finding rests entirely on s002. If it doesn't appear in
s000, the finding may be scenario-specific rather than reward-driven.

**Implementation**:

```python
def check_vigilance_gap(complete_recs, minimal_recs, scenario_ids=[0, 2]):
    """
    For each scenario:
    - Plot complete and minimal attn_agents timeseries overlaid
    - Identify calm phases (collision_risk < 0.2) and dangerous phases (> 0.5)
    - Compute mean attn_agents during calm phases for each model
    - Report the gap
    """
    for sid in scenario_ids:
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len < 10:
            print(f"  s{sid:03d}: too few overlapping timesteps ({min_len})")
            continue

        c_attn = get_timeseries(c_recs[:min_len], 'attn_agents')
        m_attn = get_timeseries(m_recs[:min_len], 'attn_agents')
        c_risk = get_timeseries(c_recs[:min_len], 'collision_risk')
        m_risk = get_timeseries(m_recs[:min_len], 'collision_risk')

        # Use average of both models' risk to define phases
        # (or use complete's risk — either is defensible)
        avg_risk = (c_risk + m_risk) / 2.0

        calm_mask = avg_risk < 0.2
        danger_mask = avg_risk > 0.5

        print(f"\n  s{sid:03d} (n={min_len}):")
        print(f"    Calm timesteps (risk<0.2):   {calm_mask.sum()}")
        print(f"    Danger timesteps (risk>0.5): {danger_mask.sum()}")

        if calm_mask.sum() >= 3:
            c_calm = c_attn[calm_mask].mean()
            m_calm = m_attn[calm_mask].mean()
            gap = c_calm - m_calm
            gap_pct = (gap / m_calm * 100) if m_calm > 0.001 else float('nan')
            print(f"    Calm-phase attn_agents:")
            print(f"      complete = {c_calm:.4f}")
            print(f"      minimal  = {m_calm:.4f}")
            print(f"      gap      = {gap:+.4f}  ({gap_pct:+.1f}%)")
            if gap > 0.01:
                print(f"    → VIGILANCE GAP CONFIRMED: complete maintains higher agent")
                print(f"      attention during calm phases (TTC reward effect).")
            elif gap < -0.01:
                print(f"    → REVERSED: minimal has higher calm-phase agent attention.")
                print(f"      Vigilance prior hypothesis NOT supported in this scenario.")
            else:
                print(f"    → NEGLIGIBLE gap. No clear vigilance prior effect.")
        else:
            print(f"    Not enough calm timesteps to test vigilance gap.")

        if danger_mask.sum() >= 3:
            c_danger = c_attn[danger_mask].mean()
            m_danger = m_attn[danger_mask].mean()
            print(f"    Danger-phase attn_agents:")
            print(f"      complete = {c_danger:.4f}")
            print(f"      minimal  = {m_danger:.4f}")

        # Episode means
        print(f"    Episode mean attn_agents: complete={c_attn.mean():.4f}, minimal={m_attn.mean():.4f}")
        print(f"    Episode mean attn_gps:    complete={get_timeseries(c_recs[:min_len], 'attn_gps').mean():.4f}, "
              f"minimal={get_timeseries(m_recs[:min_len], 'attn_gps').mean():.4f}")

    # Generate figure
    _plot_vigilance_comparison(complete_recs, minimal_recs, scenario_ids)
```

**Figure**: `fig_vigilance_gap_s000_s002.png` — 2 rows (s000, s002), each with two panels:
left = attn_agents overlay with calm phases shaded, right = collision_risk overlay.

```python
def _plot_vigilance_comparison(complete_recs, minimal_recs, scenario_ids):
    fig, axes = plt.subplots(len(scenario_ids), 2, figsize=(14, 4*len(scenario_ids)))
    if len(scenario_ids) == 1:
        axes = axes.reshape(1, -1)

    for row, sid in enumerate(scenario_ids):
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len < 5:
            continue

        c_attn = get_timeseries(c_recs[:min_len], 'attn_agents')
        m_attn = get_timeseries(m_recs[:min_len], 'attn_agents')
        c_risk = get_timeseries(c_recs[:min_len], 'collision_risk')
        m_risk = get_timeseries(m_recs[:min_len], 'collision_risk')
        t = np.arange(min_len)

        # Left panel: attn_agents
        ax = axes[row, 0]
        ax.plot(t, c_attn, 'b-', linewidth=1.5, label='complete')
        ax.plot(t, m_attn, 'r--', linewidth=1.5, label='minimal')

        # Shade calm phases (using complete model's risk)
        calm = c_risk < 0.2
        for start, end in _contiguous_regions(calm):
            ax.axvspan(start, end, alpha=0.15, color='green', label='_' if start > 0 else 'calm phase')

        # Shade the gap between lines
        ax.fill_between(t, m_attn, c_attn, where=(c_attn > m_attn),
                        alpha=0.2, color='blue', label='vigilance gap')

        ax.set_ylabel('attn_agents')
        ax.set_title(f's{sid:03d} — Agent Attention')
        ax.legend(fontsize=7, loc='upper right')

        # Right panel: collision_risk
        ax = axes[row, 1]
        ax.plot(t, c_risk, 'b-', linewidth=1.5, label='complete')
        ax.plot(t, m_risk, 'r--', linewidth=1.5, label='minimal')
        ax.set_ylabel('collision_risk')
        ax.set_title(f's{sid:03d} — Collision Risk')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

    for ax in axes[-1]:
        ax.set_xlabel('timestep')

    fig.suptitle('Vigilance Gap: Complete vs Minimal Agent Attention\n'
                 '(blue shading = gap where complete > minimal)', fontsize=12)
    plt.tight_layout()
    save_path = Path('results/reward_attention/fig_vigilance_gap_s000_s002.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()


def _contiguous_regions(mask):
    """Yield (start, end) of contiguous True regions in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return zip(starts, ends)
```

### Section 1 Outcome: GO / NO-GO Decision

Print a summary at the end of all validation checks:

```python
def validation_summary(budget_ok, risk_ok, vigilance_ok):
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"  Attention budget sums to 1.0:    {'PASS ✓' if budget_ok else 'FAIL ✗'}")
    print(f"  Cross-model risk profiles:       {'reviewed' if risk_ok else 'needs investigation'}")
    print(f"  Vigilance gap replicates (s000): {'YES' if vigilance_ok else 'UNCLEAR — check figure'}")
    print()
    if budget_ok:
        print("  → Safe to proceed with 50-scenario runs.")
    else:
        print("  → FIX THE BUG before running anything else.")
    print("="*60)
```

---

## Section 2: Attention Entropy Analysis

### Concept

Instead of measuring *how much* attention goes to agents (mean of a category), measure
*how concentrated* the full attention distribution is across all 280 tokens.

**Shannon entropy**: H = −Σ p_i × log(p_i)

- High entropy → attention is spread evenly across many tokens (diffuse)
- Low entropy → attention is concentrated on a few tokens (focused)

**Hypothesis**: Entropy drops when collision risk rises (the model focuses on fewer,
more critical tokens under threat).

**Why this matters for the paper**: It's a qualitatively different claim from the
category-level analysis. You'd be saying:
- Finding A: The model shifts attention *toward* agents when risk rises (category mean)
- Finding B: The model *narrows* its attentional focus when risk rises (entropy)

These are complementary: you could shift attention toward agents while still attending to
many agents diffusely. If entropy ALSO drops, the model is zeroing in on specific threats.

### What data do we need?

The entropy of the full 280-token distribution. This is NOT directly available from
the 5 category-level values (attn_sdc, attn_agents, etc.) — those are sums, not
distributions.

**Option A (preferred)**: If the full `(n_queries, 280)` attention matrix is saved
somewhere, or if `attn_per_agent` (per-agent weights) is in the records, use that.

**Option B (fallback)**: Compute entropy from the 5 category-level values as a coarse
proxy. This gives max entropy = log(5) ≈ 1.61 bits. Less granular but still tests
whether the budget concentrates.

**Check which option is available:**

```python
# Check if per-token or per-agent attention is in the records
recs = load_pkl(COMPLETE_PKL)
r = recs[0]

# Option A check:
has_per_agent = hasattr(r, 'attn_per_agent')
print(f"Has attn_per_agent: {has_per_agent}")

# Also check if raw attention matrix is stored
has_raw = hasattr(r, 'attn_raw') or hasattr(r, 'attention_weights')
print(f"Has raw attention matrix: {has_raw}")
```

### Implementation (Option B — 5-category entropy, works with existing data)

```python
def compute_category_entropy(records):
    """
    Compute Shannon entropy of the 5-category attention distribution
    for every timestep.

    Returns: numpy array of entropy values, same length as records.
    """
    entropies = []
    for r in records:
        p = np.array([r.attn_sdc, r.attn_agents, r.attn_roadgraph,
                       r.attn_lights, r.attn_gps])
        # Clip to avoid log(0)
        p = np.clip(p, 1e-10, None)
        # Normalize (should already sum to 1, but safety)
        p = p / p.sum()
        H = -np.sum(p * np.log2(p))
        entropies.append(H)
    return np.array(entropies)
```

### Implementation (Option A — if attn_per_agent exists, 280-token entropy)

```python
def compute_token_entropy(records):
    """
    Compute Shannon entropy of the full 280-token attention distribution.
    Requires that the full distribution (or per-token weights) is stored.

    If attn_per_agent exists (shape: n_agents,), reconstruct approximate
    full distribution using category sums for non-agent tokens.
    """
    # This is approximate — see note below
    entropies = []
    for r in records:
        if hasattr(r, 'attn_per_agent') and r.attn_per_agent is not None:
            # Build approximate distribution:
            # - 5 ego tokens, each gets attn_sdc / 5
            # - 40 agent tokens with per-agent weights (if available)
            #   or attn_agents / 40
            # - 200 road tokens, each gets attn_roadgraph / 200
            # - 25 light tokens, each gets attn_lights / 25
            # - 10 gps tokens, each gets attn_gps / 10

            # NOTE: This assumes uniform distribution WITHIN each category
            # except agents. It's an approximation.
            p = []
            p.extend([r.attn_sdc / 5] * 5)

            # For agents: use per-agent if available
            per_agent = np.array(r.attn_per_agent)
            # Each agent has 5 tokens (5 timesteps), distribute evenly
            for agent_attn in per_agent:
                p.extend([agent_attn / 5] * 5)
            # Pad if fewer than 8 agents
            remaining = 40 - len(per_agent) * 5
            if remaining > 0:
                p.extend([0.0] * remaining)

            p.extend([r.attn_roadgraph / 200] * 200)
            p.extend([r.attn_lights / 25] * 25)
            p.extend([r.attn_gps / 10] * 10)

            p = np.array(p)
            p = np.clip(p, 1e-10, None)
            p = p / p.sum()
            H = -np.sum(p * np.log2(p))
        else:
            # Fallback: assume uniform within each category
            p = []
            p.extend([r.attn_sdc / 5] * 5)
            p.extend([r.attn_agents / 40] * 40)
            p.extend([r.attn_roadgraph / 200] * 200)
            p.extend([r.attn_lights / 25] * 25)
            p.extend([r.attn_gps / 10] * 10)
            p = np.array(p)
            p = np.clip(p, 1e-10, None)
            p = p / p.sum()
            H = -np.sum(p * np.log2(p))
        entropies.append(H)
    return np.array(entropies)
```

**IMPORTANT NOTE**: Both options have a limitation. The 5-category entropy (Option B)
has only 5 bins → max entropy = log2(5) = 2.32 bits. This is coarse. The 280-token
entropy (Option A fallback) assumes uniform distribution within each category, which
inflates road graph's contribution since it has 200 tokens.

**Recommendation**: Use Option B (5-category) for the initial test. It's honest about
what it measures. If the effect is clear at 5-category level, it will hold at finer
granularity too. If you want the 280-token version, you'd need to re-run the extractor
to save per-token weights — save that for after the paper draft.

### Analysis

```python
def entropy_analysis(records, model_name):
    """
    1. Compute entropy for every timestep
    2. Correlate entropy with collision_risk (within-episode)
    3. Compare entropy in calm vs danger phases
    4. Generate scatter plot and timeseries figure
    """
    entropies = compute_category_entropy(records)

    # Get scenario IDs
    scenario_ids = sorted(set(r.scenario_id for r in records))

    print(f"\n{'='*60}")
    print(f"ENTROPY ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"  Global entropy: mean={entropies.mean():.4f}, "
          f"std={entropies.std():.4f}, "
          f"range=[{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"  Max possible (5 categories): {np.log2(5):.4f} bits")

    # Within-episode correlation
    rhos = []
    for sid in scenario_ids:
        idx = [i for i, r in enumerate(records) if r.scenario_id == sid]
        if len(idx) < 10:
            continue
        risk = np.array([records[i].collision_risk for i in idx])
        ent = entropies[idx]
        if np.std(risk) < 0.2:  # skip low-variation scenarios
            continue
        rho, p = stats.spearmanr(risk, ent)
        rhos.append(rho)
        print(f"  s{sid:03d}: entropy×risk ρ={rho:+.3f} (p={p:.4f}) "
              f"{'**' if p < 0.05 else ''}")

    if rhos:
        mean_rho = np.mean(rhos)
        print(f"\n  Mean within-episode ρ(entropy, collision_risk) = {mean_rho:+.3f}")
        print(f"  n={len(rhos)} high-variation scenarios")
        if mean_rho < -0.1:
            print(f"  → HYPOTHESIS SUPPORTED: Entropy drops when risk rises.")
            print(f"    The model concentrates attention under threat.")
        elif mean_rho > 0.1:
            print(f"  → HYPOTHESIS REJECTED: Entropy RISES when risk rises.")
            print(f"    The model spreads attention more broadly under threat.")
        else:
            print(f"  → INCONCLUSIVE: No clear entropy-risk relationship.")

    # Calm vs danger phase comparison
    calm_ent = entropies[[i for i, r in enumerate(records) if r.collision_risk < 0.2]]
    danger_ent = entropies[[i for i, r in enumerate(records) if r.collision_risk > 0.5]]
    if len(calm_ent) > 5 and len(danger_ent) > 5:
        print(f"\n  Phase comparison:")
        print(f"    Calm (risk<0.2):   entropy={calm_ent.mean():.4f} ± {calm_ent.std():.4f}  (n={len(calm_ent)})")
        print(f"    Danger (risk>0.5): entropy={danger_ent.mean():.4f} ± {danger_ent.std():.4f}  (n={len(danger_ent)})")
        t_stat, t_p = stats.mannwhitneyu(calm_ent, danger_ent, alternative='two-sided')
        print(f"    Mann-Whitney U test: p={t_p:.4f} {'(significant)' if t_p < 0.05 else '(not significant)'}")

    # Generate figures
    _plot_entropy_scatter(records, entropies, model_name)
    _plot_entropy_timeseries(records, entropies, model_name)
```

### Figures

```python
def _plot_entropy_scatter(records, entropies, model_name):
    """Scatter plot: collision_risk vs entropy, colored by scenario."""
    fig, ax = plt.subplots(figsize=(8, 6))
    risks = np.array([r.collision_risk for r in records])
    sids = np.array([r.scenario_id for r in records])

    for sid in sorted(set(sids)):
        mask = sids == sid
        ax.scatter(risks[mask], entropies[mask], alpha=0.4, s=15,
                   label=f's{sid:03d}')

    # Overall trend line
    if np.std(risks) > 0.01:
        slope, intercept, r_val, p_val, _ = stats.linregress(risks, entropies)
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=2,
                label=f'trend: r={r_val:.3f}')

    ax.set_xlabel('Collision Risk')
    ax.set_ylabel('Attention Entropy (bits)')
    ax.set_title(f'Attention Concentration vs Risk — {model_name}')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    save_path = Path(f'results/reward_attention/fig_entropy_scatter_{model_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def _plot_entropy_timeseries(records, entropies, model_name, top_n=3):
    """
    Timeseries: entropy + collision_risk overlaid for the top_n most
    variable scenarios (same selection as main analysis).
    """
    scenario_ids = sorted(set(r.scenario_id for r in records))
    # Rank scenarios by risk variability
    ranked = []
    for sid in scenario_ids:
        idx = [i for i, r in enumerate(records) if r.scenario_id == sid]
        risk_std = np.std([records[i].collision_risk for i in idx])
        ranked.append((sid, risk_std, idx))
    ranked.sort(key=lambda x: -x[1])  # highest variability first

    n_plot = min(top_n, len(ranked))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3.5 * n_plot))
    if n_plot == 1:
        axes = [axes]

    for ax, (sid, risk_std, idx) in zip(axes, ranked[:n_plot]):
        t = np.array([records[i].timestep for i in idx])
        risk = np.array([records[i].collision_risk for i in idx])
        ent = entropies[idx]

        # Sort by timestep
        order = np.argsort(t)
        t, risk, ent = t[order], risk[order], ent[order]

        ax2 = ax.twinx()
        ax.plot(t, ent, 'b-', linewidth=1.5, label='entropy')
        ax2.plot(t, risk, 'r-', linewidth=1.5, alpha=0.7, label='collision_risk')

        ax.set_ylabel('Entropy (bits)', color='blue')
        ax2.set_ylabel('Collision Risk', color='red')
        ax.set_title(f's{sid:03d} (risk_std={risk_std:.3f})')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    axes[-1].set_xlabel('timestep')
    fig.suptitle(f'Attention Entropy vs Collision Risk — {model_name}', fontsize=12)
    plt.tight_layout()
    save_path = Path(f'results/reward_attention/fig_entropy_timeseries_{model_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
```

### What to report in the paper (if the finding holds)

If mean within-episode ρ(entropy, collision_risk) < −0.15 and significant:

> "Beyond shifting attention toward agents, the model also *concentrates* its attentional
> budget under threat: Shannon entropy of the 5-category attention distribution
> anticorrelates with collision risk within episodes (mean ρ = −X.XX, 95%CI [...]).
> This suggests the Perceiver encoder narrows its focus to fewer information sources
> when collision risk rises — a sharpening of attention consistent with threat
> prioritization."

If the finding does NOT hold (ρ ≈ 0):

> Don't include it. Move on. You've lost 1 hour of coding, not a paper.

---

## Section 3: Per-Scenario Lead-Lag Histogram

### Concept

The aggregate lead-lag analysis (from analyze_results.py) shows best ρ at lag=+2
(attention leads risk by 2 steps). But that's one number from one cross-correlation
across all HV scenarios concatenated (or averaged). The question:

**Is lag=+2 consistent across scenarios, or is it an outlier-driven artifact?**

### Method

For each high-variation scenario:
1. Extract `attn_agents` and `collision_risk` timeseries
2. Compute Spearman ρ at lags −8 to +8 (where lag=+k means attention at time t
   correlates with risk at time t+k → attention LEADS)
3. Record the lag with highest ρ for that scenario
4. Plot histogram of best lags across all HV scenarios

### Implementation

```python
def leadlag_per_scenario(records, lag_range=8, min_risk_std=0.2):
    """
    For each high-variation scenario, find the lag that maximizes
    Spearman correlation between attn_agents and collision_risk.

    lag > 0 means attention LEADS risk (attention predicts future risk)
    lag < 0 means attention LAGS risk (attention reacts to past risk)
    lag = 0 means simultaneous
    """
    scenario_ids = sorted(set(r.scenario_id for r in records))

    results = []  # list of (scenario_id, best_lag, best_rho, n_timesteps)
    all_lag_profiles = []  # for optional heatmap

    print(f"\n{'='*60}")
    print(f"LEAD-LAG PER-SCENARIO ANALYSIS")
    print(f"{'='*60}")

    for sid in scenario_ids:
        recs = get_scenario(records, sid)
        if len(recs) < 2 * lag_range + 5:  # need enough timesteps for shifted corr
            continue

        risk = get_timeseries(recs, 'collision_risk')
        attn = get_timeseries(recs, 'attn_agents')

        if np.std(risk) < min_risk_std:
            continue  # skip low-variation scenarios

        lag_rhos = []
        for lag in range(-lag_range, lag_range + 1):
            if lag > 0:
                # attention at t correlates with risk at t+lag
                # → attention LEADS: we shift risk forward (or attn back)
                a = attn[:len(attn)-lag]
                r = risk[lag:]
            elif lag < 0:
                # attention at t correlates with risk at t+lag (lag<0 → t-|lag|)
                # → attention LAGS: risk happened before attention shifted
                a = attn[-lag:]
                r = risk[:len(risk)+lag]
            else:
                a = attn
                r = risk

            if len(a) < 10:
                lag_rhos.append(float('nan'))
                continue

            rho, _ = stats.spearmanr(a, r)
            lag_rhos.append(rho)

        lag_rhos = np.array(lag_rhos)
        all_lag_profiles.append(lag_rhos)

        # Find best lag (highest positive rho)
        valid = ~np.isnan(lag_rhos)
        if valid.any():
            best_idx = np.nanargmax(lag_rhos)
            best_lag = best_idx - lag_range  # convert index to lag value
            best_rho = lag_rhos[best_idx]
        else:
            best_lag, best_rho = 0, float('nan')

        results.append((sid, best_lag, best_rho, len(recs)))
        print(f"  s{sid:03d}: best_lag={best_lag:+d}, ρ={best_rho:+.3f} (n={len(recs)})")

    if not results:
        print("  No high-variation scenarios found!")
        return results

    # Summary statistics
    best_lags = [r[1] for r in results]
    print(f"\n  Summary (n={len(results)} HV scenarios):")
    print(f"    Mean best lag: {np.mean(best_lags):+.2f}")
    print(f"    Median best lag: {np.median(best_lags):+.1f}")
    print(f"    Std: {np.std(best_lags):.2f}")

    # Count scenarios by lag bin
    positive = sum(1 for l in best_lags if l > 0)
    zero = sum(1 for l in best_lags if l == 0)
    negative = sum(1 for l in best_lags if l < 0)
    print(f"    Attention leads (lag>0): {positive}/{len(best_lags)} "
          f"({100*positive/len(best_lags):.0f}%)")
    print(f"    Simultaneous (lag=0):    {zero}/{len(best_lags)}")
    print(f"    Attention lags (lag<0):  {negative}/{len(best_lags)}")

    # Clustering check
    cluster_1_3 = sum(1 for l in best_lags if 1 <= l <= 3)
    print(f"    In lag +1 to +3 range:   {cluster_1_3}/{len(best_lags)} "
          f"({100*cluster_1_3/len(best_lags):.0f}%)")

    if cluster_1_3 / len(best_lags) > 0.4:
        print(f"    → ANTICIPATORY CLAIM SUPPORTED: majority of scenarios show")
        print(f"      attention leading risk by 1-3 steps.")
    elif positive / len(best_lags) > 0.5:
        print(f"    → ATTENTION LEADS in most scenarios but spread is wide.")
        print(f"      Anticipatory claim is directional, not precise.")
    else:
        print(f"    → ANTICIPATORY CLAIM WEAK: no clear clustering.")
        print(f"      The aggregate lag=+2 may be driven by outliers.")

    # Generate histogram
    _plot_leadlag_histogram(results, lag_range)
    # Generate heatmap of all lag profiles (optional, nice for appendix)
    if all_lag_profiles:
        _plot_leadlag_heatmap(all_lag_profiles, results, lag_range)

    return results


def _plot_leadlag_histogram(results, lag_range):
    """Histogram of per-scenario best lags."""
    best_lags = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-lag_range - 0.5, lag_range + 1.5, 1)
    counts, _, patches = ax.hist(best_lags, bins=bins, edgecolor='black',
                                  color='steelblue', alpha=0.8)

    # Color the bars: green for positive (leads), red for negative (lags)
    for patch, left_edge in zip(patches, bins[:-1]):
        center = left_edge + 0.5
        if center > 0:
            patch.set_facecolor('#4CAF50')  # green = leads
        elif center < 0:
            patch.set_facecolor('#F44336')  # red = lags
        else:
            patch.set_facecolor('#9E9E9E')  # gray = simultaneous

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Best Lag (positive = attention LEADS risk)')
    ax.set_ylabel('Number of scenarios')
    ax.set_title(f'Per-Scenario Best Lead-Lag\n(n={len(best_lags)} high-variation scenarios)')
    ax.set_xticks(range(-lag_range, lag_range + 1))

    # Annotate
    median_lag = np.median(best_lags)
    ax.axvline(x=median_lag, color='orange', linestyle='-', linewidth=2,
               label=f'median={median_lag:+.1f}')
    ax.legend()

    plt.tight_layout()
    save_path = Path('results/reward_attention/fig_leadlag_histogram.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()


def _plot_leadlag_heatmap(lag_profiles, results, lag_range):
    """
    Heatmap: rows = scenarios (sorted by best lag), columns = lags,
    color = rho at that lag. Nice appendix figure.
    """
    # Sort by best lag
    sorted_idx = sorted(range(len(results)), key=lambda i: results[i][1])

    matrix = np.array([lag_profiles[i] for i in sorted_idx])
    labels = [f"s{results[i][0]:03d}" for i in sorted_idx]
    lags = list(range(-lag_range, lag_range + 1))

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(lags, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Lag (positive = attention leads)')
    ax.set_ylabel('Scenario')
    ax.set_title('Lead-Lag Correlation Profile per Scenario')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    # Mark best lag per scenario
    for row, i in enumerate(sorted_idx):
        best_lag_idx = results[i][1] + lag_range
        ax.plot(best_lag_idx, row, 'k*', markersize=8)

    plt.tight_layout()
    save_path = Path('results/reward_attention/fig_leadlag_heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
```

---

## Main Script Structure

```python
#!/usr/bin/env python3
"""
validate_and_extend.py — Pre-scale validation + new analyses.
Run BEFORE launching 50-scenario experiments on minimal/basic models.

Usage:
    python reward_attention/validate_and_extend.py --all
    python reward_attention/validate_and_extend.py --validate
    python reward_attention/validate_and_extend.py --entropy
    python reward_attention/validate_and_extend.py --leadlag
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--leadlag', action='store_true')
    args = parser.parse_args()

    if args.all:
        args.validate = args.entropy = args.leadlag = True

    if not any([args.validate, args.entropy, args.leadlag]):
        print("Specify --all, --validate, --entropy, or --leadlag")
        return

    # Load data
    print("Loading complete model data...")
    complete = load_pkl(COMPLETE_PKL)
    print(f"  Loaded {len(complete)} records")

    minimal_exists = MINIMAL_PKL.exists()
    basic_exists = BASIC_PKL.exists()

    if minimal_exists:
        print("Loading minimal model data...")
        minimal = load_pkl(MINIMAL_PKL)
        print(f"  Loaded {len(minimal)} records")
    else:
        print("  ⚠ Minimal pkl not found — cross-model checks will be skipped")
        minimal = None

    if basic_exists:
        basic = load_pkl(BASIC_PKL)
        print(f"  Loaded basic: {len(basic)} records")
    else:
        basic = None

    # ── Section 1: Validation ──
    if args.validate:
        print("\n" + "="*60)
        print("SECTION 1: VALIDATION CHECKS")
        print("="*60)

        budget_ok = check_attention_budget(complete, 'complete')
        if minimal is not None:
            check_attention_budget(minimal, 'minimal')
        if basic is not None:
            check_attention_budget(basic, 'basic')

        risk_ok = True  # set to False if warnings appear
        if minimal is not None:
            print("\n--- Cross-Model Risk Profile Comparison ---")
            compare_risk_profiles(complete, minimal, basic)

        vigilance_ok = False
        if minimal is not None:
            print("\n--- Vigilance Gap Check (s000 + s002) ---")
            check_vigilance_gap(complete, minimal, scenario_ids=[0, 2])
            vigilance_ok = True  # user must inspect figure

        validation_summary(budget_ok, risk_ok, vigilance_ok)

    # ── Section 2: Entropy ──
    if args.entropy:
        print("\n" + "="*60)
        print("SECTION 2: ATTENTION ENTROPY ANALYSIS")
        print("="*60)
        entropy_analysis(complete, 'complete')
        if minimal is not None:
            entropy_analysis(minimal, 'minimal')

    # ── Section 3: Lead-Lag ──
    if args.leadlag:
        print("\n" + "="*60)
        print("SECTION 3: PER-SCENARIO LEAD-LAG")
        print("="*60)
        leadlag_per_scenario(complete, lag_range=8, min_risk_std=0.2)

if __name__ == '__main__':
    main()
```

---

## Expected Output Files

```
results/reward_attention/
  fig_risk_profile_comparison.png     ← Check 1B: do models face same risk?
  fig_vigilance_gap_s000_s002.png     ← Check 1C: does the gap replicate?
  fig_entropy_scatter_complete.png    ← Section 2: entropy vs risk
  fig_entropy_timeseries_complete.png ← Section 2: entropy over time
  fig_entropy_scatter_minimal.png     ← Section 2: minimal model (if pkl exists)
  fig_leadlag_histogram.png           ← Section 3: per-scenario best lag
  fig_leadlag_heatmap.png             ← Section 3: full lag profiles (appendix)
```

---

## Decision Matrix After Running This Script

| Check | Pass Condition | If Fails |
|-------|---------------|----------|
| Budget sums to 1.0 | All timesteps within 1% | **STOP. Fix extractor.py before anything else.** |
| Risk profiles similar | ρ > 0.5 on 2/3 scenarios | Not fatal. Acknowledge in paper that models face different risk experiences. Reframe comparison as "each model's response to its own risk." |
| Vigilance gap on s000 | complete > minimal calm-phase attn_agents by >10% | If reversed or absent on s000, the claim needs to be hedged ("observed in s002, requires 50-scenario validation") |
| Entropy drops with risk | mean within-episode ρ < −0.15 | If ρ ≈ 0: drop the entropy finding, don't include in paper. No harm done. |
| Lead-lag clusters at +1 to +3 | >40% of HV scenarios | If spread is wide: report aggregate lag=+2 with honest caveat about per-scenario heterogeneity. Do not make a strong anticipatory claim. |

---

## Time Estimate

| Task | Time |
|------|------|
| Write `validate_and_extend.py` | 1.5 hours |
| Run all checks + interpret output | 0.5 hours |
| Fix any bugs surfaced by validation | 0–1 hour |
| Decide on entropy/lead-lag inclusion | 0.5 hours |
| **Total** | **2.5–3.5 hours** |

---

## After This Script: What Next?

1. If all checks pass → launch minimal 50-scenario run (Priority 1a from NEXT_STEPS.md)
2. While it runs → start writing the paper introduction + method section
3. When minimal run finishes → run `validate_and_extend.py --validate` again with the
   new 50-scenario minimal pkl to check vigilance gap at scale
4. Run entropy + lead-lag on the 50-scenario minimal pkl too
5. Assemble figures → write results section
