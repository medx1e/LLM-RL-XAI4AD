// Synthetic data mimicking platform_cache artifacts
export type ModelMeta = {
  slug: string;
  name: string;
  family: "Perceiver" | "Wayformer" | "CBM";
  attention: boolean;
  scenarios: number[];
  notes: string;
};

export const MODELS: ModelMeta[] = [
  {
    slug: "sac_perceiver_s42",
    name: "SAC Perceiver — seed 42",
    family: "Perceiver",
    attention: true,
    scenarios: [0, 1, 2, 3, 5, 7, 9, 12, 18, 24],
    notes: "Primary policy. Cross-attention surfaced across 2 layers.",
  },
  {
    slug: "sac_perceiver_s17",
    name: "SAC Perceiver — seed 17",
    family: "Perceiver",
    attention: true,
    scenarios: [0, 1, 3, 5, 7, 12],
    notes: "Ablation seed for variance analysis.",
  },
  {
    slug: "wayformer_v2",
    name: "Wayformer v2",
    family: "Wayformer",
    attention: true,
    scenarios: [0, 3, 5, 12, 18],
    notes: "Transformer baseline with motion-prediction head.",
  },
  {
    slug: "cbm_v1",
    name: "CBM v1 — 15 concepts",
    family: "CBM",
    attention: false,
    scenarios: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    notes: "Concept Bottleneck Model. 15 interpretable concepts.",
  },
];

export const METHODS = [
  { id: "vanilla_grad", name: "Vanilla Gradient", color: "var(--a0)" },
  { id: "integrated_grad", name: "Integrated Gradients", color: "var(--a1)" },
  { id: "shap", name: "SHAP (Kernel)", color: "var(--a2)" },
  { id: "smooth_grad", name: "SmoothGrad", color: "var(--a3)" },
  { id: "occlusion", name: "Occlusion", color: "var(--a4)" },
];

export const CATEGORIES = [
  { id: "sdc", name: "SDC State", color: "var(--sdc)" },
  { id: "agents", name: "Other Agents", color: "var(--agents-c)" },
  { id: "roadgraph", name: "Road Graph", color: "var(--roadgraph)" },
  { id: "traffic_lights", name: "Traffic Lights", color: "var(--traffic-lights)" },
  { id: "gps", name: "GPS Path", color: "var(--gps)" },
];

export const CONCEPTS = [
  "ego_speed", "ego_acceleration", "ego_yaw_rate",
  "traffic_light_red", "traffic_light_green", "stop_sign_present",
  "ttc_lead_vehicle", "lead_vehicle_distance", "path_curvature_max",
  "lane_offset", "intersection_proximity", "pedestrian_nearby",
  "lane_change_intent", "speed_limit_compliance", "right_of_way",
];

export const ARCHETYPES = [
  { id: "red_light", label: "Red Light Stop", emoji: "🔴", color: "var(--traffic-lights)" },
  { id: "ttc", label: "TTC Success", emoji: "⚠️", color: "var(--warning)" },
  { id: "curve", label: "Curve Navigation", emoji: "〰", color: "var(--sdc)" },
  { id: "failure", label: "Concept Failure", emoji: "❌", color: "var(--destructive)" },
];

export const LLMS = [
  { key: "glm_51", name: "GLM-5.1" },
  { key: "deepseek", name: "DeepSeek V3" },
  { key: "gemma_4", name: "Gemma 4" },
  { key: "qwen_36", name: "Qwen 3.6" },
];

export const TOGGLES = [
  { id: "full", label: "Full" },
  { id: "no_grounding", label: "No grounding" },
  { id: "no_counterfactual", label: "No counterfactual" },
  { id: "minimal", label: "Minimal" },
];

// Deterministic PRNG
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const EPISODE_LEN = 80;

export function rewardSeries(seed: number) {
  const rnd = mulberry32(seed);
  const arr: number[] = [];
  let cum = 0;
  const series: { step: number; reward: number; cum: number }[] = [];
  for (let i = 0; i < EPISODE_LEN; i++) {
    const r = (rnd() - 0.45) * 1.4 + Math.sin(i / 8) * 0.4;
    arr.push(r);
    cum += r;
    series.push({ step: i, reward: +r.toFixed(3), cum: +cum.toFixed(3) });
  }
  return series;
}

export function attributionSeries(seed: number) {
  const rnd = mulberry32(seed);
  return Array.from({ length: EPISODE_LEN }, (_, i) => ({
    step: i,
    magnitude: +(0.3 + rnd() * 0.5 + Math.sin(i / 6 + seed) * 0.15).toFixed(3),
  }));
}

export function categoryBreakdown(seed: number) {
  const rnd = mulberry32(seed);
  const raw = CATEGORIES.map((c) => ({ ...c, value: rnd() * 0.4 + 0.1 }));
  const sum = raw.reduce((a, b) => a + b.value, 0);
  return raw.map((r) => ({ ...r, value: +(r.value / sum).toFixed(3) }));
}

export function topEntities(seed: number, n = 10) {
  const rnd = mulberry32(seed);
  const slots = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"];
  const items: { name: string; category: string; color: string; value: number }[] = [];
  for (let i = 0; i < 8; i++) {
    items.push({
      name: slots[i],
      category: "agents",
      color: `var(--a${i})`,
      value: rnd() * 0.9 + 0.1,
    });
  }
  for (let i = 0; i < 6; i++) {
    items.push({
      name: `RG-lane-${i}`,
      category: "roadgraph",
      color: "var(--roadgraph)",
      value: rnd() * 0.7,
    });
  }
  for (let i = 0; i < 4; i++) {
    items.push({
      name: `TL-${i}`,
      category: "traffic_lights",
      color: "var(--traffic-lights)",
      value: rnd() * 0.8,
    });
  }
  items.push({ name: "SDC-state", category: "sdc", color: "var(--sdc)", value: rnd() * 0.9 + 0.2 });
  items.push({ name: "GPS-wp", category: "gps", color: "var(--gps)", value: rnd() * 0.6 });

  return items.sort((a, b) => b.value - a.value).slice(0, n).map((x) => ({
    ...x,
    value: +x.value.toFixed(3),
  }));
}

export function attentionByEntity(seed: number) {
  const rnd = mulberry32(seed);
  return Array.from({ length: 8 }, (_, i) => ({
    slot: `A${i}`,
    color: `var(--a${i})`,
    value: +(rnd() * 0.9 + 0.05).toFixed(3),
  })).sort((a, b) => b.value - a.value);
}

export function methodAgreement() {
  const rnd = mulberry32(7);
  return METHODS.map((m1) =>
    METHODS.map((m2) =>
      m1.id === m2.id ? 1.0 : +(0.4 + rnd() * 0.5).toFixed(2)
    )
  );
}

export function alignmentMatrix() {
  const rnd = mulberry32(11);
  return Array.from({ length: 8 }, () =>
    METHODS.map(() => +(rnd() * 2 - 1).toFixed(2))
  );
}

export function methodProfiles() {
  const rnd = mulberry32(3);
  return METHODS.map((m) => ({
    ...m,
    sparsity: +(0.3 + rnd() * 0.6).toFixed(2),
    gini: +(0.4 + rnd() * 0.5).toFixed(2),
    top10: +(0.3 + rnd() * 0.5).toFixed(2),
    stability: +(0.4 + rnd() * 0.5).toFixed(2),
    dominantCategory: CATEGORIES[Math.floor(rnd() * CATEGORIES.length)].name,
    latencyMs: Math.round(40 + rnd() * 280),
  }));
}

export function conceptSeries(conceptIdx: number, seed: number) {
  const rnd = mulberry32(seed + conceptIdx);
  return Array.from({ length: EPISODE_LEN }, (_, i) => {
    const truth = Math.sin((i + conceptIdx * 5) / 10) * 0.5 + 0.5 + (rnd() - 0.5) * 0.05;
    const pred = truth + (rnd() - 0.5) * 0.2;
    const invalid = i < 4 || i > 75;
    return {
      step: i,
      truth: +truth.toFixed(3),
      pred: +pred.toFixed(3),
      invalid,
    };
  });
}

export const SCENARIOS_META: Record<number, { tag: string; outcome: string; description: string }> = {
  0: { tag: "Urban", outcome: "Success", description: "4-way intersection, right turn on green" },
  1: { tag: "Highway", outcome: "Success", description: "Lane change with lead vehicle" },
  2: { tag: "Urban", outcome: "Near-miss", description: "Pedestrian crossing mid-block" },
  3: { tag: "Urban", outcome: "Success", description: "Stop at red light, queue behind 2 cars" },
  5: { tag: "Suburban", outcome: "Success", description: "Curve navigation at posted speed" },
  7: { tag: "Highway", outcome: "Success", description: "Merge onto highway from on-ramp" },
  9: { tag: "Urban", outcome: "Failure", description: "Failed to yield at uncontrolled intersection" },
  12: { tag: "Urban", outcome: "Success", description: "Left turn across oncoming traffic" },
  18: { tag: "Highway", outcome: "Success", description: "Following distance compliance" },
  24: { tag: "Urban", outcome: "Near-miss", description: "Cyclist in adjacent lane" },
};

export function narrationFor(step: number, seed: number) {
  const rnd = mulberry32(seed * 100 + Math.floor(step / 8));
  const tones: Array<"detailed" | "brief" | "caveat" | "detailed-caveat"> = [
    "detailed", "brief", "caveat", "detailed-caveat",
  ];
  const tone = tones[Math.floor(rnd() * 4)];
  const samples = [
    "The ego vehicle is decelerating in anticipation of the red signal ahead. Attention is concentrated on the lead vehicle (A0) and the traffic light token, with secondary focus on the right-lane road graph segment.",
    "Maintaining lane and speed. No salient agents within 30m.",
    "Caveat: the prediction confidence dropped 22% over the last 3 steps. Lead vehicle behavior is inconsistent with prior trajectories in the training distribution.",
    "Initiating a left turn maneuver. The policy weights pedestrian-nearby concept highly despite no visible pedestrian — this may be a spurious correlation from training scenarios with crosswalks at intersections.",
  ];
  return {
    tone,
    text: samples[Math.floor(rnd() * samples.length)],
    responseTimeMs: Math.round(420 + rnd() * 1100),
    necessity: +(rnd() * 0.5 + 0.5).toFixed(2),
    grounding: +(rnd() * 0.4 + 0.55).toFixed(2),
    decisionClass: ["Stop", "Yield", "Proceed", "Slow"][Math.floor(rnd() * 4)],
    alternatives: ["Hold lane + brake", "Lane-change right", "Maintain"].slice(0, Math.floor(rnd() * 3) + 1),
  };
}
