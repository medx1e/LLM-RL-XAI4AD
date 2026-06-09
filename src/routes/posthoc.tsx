import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { ContextStrip } from "@/components/ContextStrip";
import { BevPlayer } from "@/components/BevPlayer";
import { MetricCard } from "@/components/MetricCard";
import { RewardChart } from "@/components/RewardChart";
import { MethodPanel } from "@/components/MethodPanel";
import { Heatmap } from "@/components/Heatmap";
import { ControlPanel, Field } from "@/components/ControlPanel";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  MODELS,
  METHODS,
  SCENARIOS_META,
  rewardSeries,
  attentionByEntity,
  methodAgreement,
  alignmentMatrix,
} from "@/lib/mockData";

export const Route = createFileRoute("/posthoc")({
  head: () => ({
    meta: [
      { title: "Post-hoc XAI · XAI4AD" },
      { name: "description", content: "Attribution maps and Perceiver cross-attention inspection." },
    ],
  }),
  component: PosthocPage,
});

function PosthocPage() {
  const [modelSlug, setModelSlug] = useState(MODELS[0].slug);
  const model = MODELS.find((m) => m.slug === modelSlug)!;
  const [scenario, setScenario] = useState(model.scenarios[0]);
  const [selectedMethods, setSelectedMethods] = useState<string[]>([METHODS[0].id, METHODS[1].id]);
  const [step, setStep] = useState(28);
  const [topN, setTopN] = useState(10);
  const [normalizeEnt, setNormalizeEnt] = useState(true);
  const [perToken, setPerToken] = useState(false);
  const [attentionOverlay, setAttentionOverlay] = useState(true);
  const [agents, setAgents] = useState<number[]>([0, 2]);
  const [layer, setLayer] = useState("cross_attn_avg");
  const [reportOpen, setReportOpen] = useState(true);

  const meta = SCENARIOS_META[scenario] || { tag: "Urban", outcome: "Success", description: "" };
  const rewards = rewardSeries(scenario);
  const totalReward = rewards.reduce((a, b) => a + b.reward, 0);
  const stepReward = rewards[step]?.reward ?? 0;

  const attn = useMemo(() => attentionByEntity(scenario * 13 + step), [scenario, step]);
  const agreement = methodAgreement();
  const align = alignmentMatrix();

  const toggleMethod = (id: string) =>
    setSelectedMethods((m) => (m.includes(id) ? m.filter((x) => x !== id) : [...m, id]));
  const toggleAgent = (i: number) =>
    setAgents((a) => (a.includes(i) ? a.filter((x) => x !== i) : [...a, i]));

  return (
    <>
      <ContextStrip
        title="Post-hoc XAI"
        subtitle={`${model.name} · scenario ${scenario} · ${meta.tag} · ${meta.outcome}`}
      />
      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          {/* Controls */}
          <aside className="space-y-4">
            <ControlPanel title="Selection">
              <Field label="Model">
                <Select value={modelSlug} onValueChange={(v) => {
                  setModelSlug(v);
                  const m = MODELS.find((x) => x.slug === v)!;
                  setScenario(m.scenarios[0]);
                }}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {MODELS.filter((m) => m.family !== "CBM").map((m) => (
                      <SelectItem key={m.slug} value={m.slug}>{m.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Scenario">
                <Select value={String(scenario)} onValueChange={(v) => setScenario(Number(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {model.scenarios.map((s) => (
                      <SelectItem key={s} value={String(s)}>
                        #{s} · {SCENARIOS_META[s]?.description ?? "scenario"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </ControlPanel>

            <ControlPanel title="Attribution methods">
              <div className="space-y-2">
                {METHODS.map((m) => {
                  const on = selectedMethods.includes(m.id);
                  return (
                    <button
                      key={m.id}
                      onClick={() => toggleMethod(m.id)}
                      className={`flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-xs transition-colors ${
                        on
                          ? "border-primary/50 bg-primary/10"
                          : "border-border bg-secondary/30 hover:bg-secondary/60"
                      }`}
                    >
                      <span className="h-2 w-2 rounded-full" style={{ background: m.color }} />
                      <span className="flex-1">{m.name}</span>
                      <span className={`h-3.5 w-3.5 rounded border ${on ? "bg-primary border-primary" : "border-border"}`} />
                    </button>
                  );
                })}
              </div>
            </ControlPanel>

            <ControlPanel title="Display">
              <Field label={`Top-N entities · ${topN}`}>
                <Slider value={[topN]} min={5} max={30} step={1} onValueChange={(v) => setTopN(v[0])} />
              </Field>
              <div className="flex items-center justify-between text-xs">
                <Label htmlFor="norm-ent" className="text-muted-foreground">Normalize entities</Label>
                <Switch id="norm-ent" checked={normalizeEnt} onCheckedChange={setNormalizeEnt} />
              </div>
              <div className="flex items-center justify-between text-xs">
                <Label htmlFor="per-tok" className="text-muted-foreground">Per-token normalize</Label>
                <Switch id="per-tok" checked={perToken} onCheckedChange={setPerToken} />
              </div>
            </ControlPanel>
          </aside>

          {/* Main */}
          <section className="space-y-4 min-w-0">
            <div className="grid gap-4 xl:grid-cols-2">
              {/* Left: BEV + metrics */}
              <div className="space-y-4 min-w-0">
                <BevPlayer
                  step={step}
                  setStep={setStep}
                  scenarioSeed={scenario}
                  highlights={[12, 28, 44, 60]}
                  agentOverlays={attentionOverlay ? agents : []}
                />
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard label="Total reward" value={totalReward.toFixed(2)} tone={totalReward >= 0 ? "success" : "error"} />
                  <MetricCard label="Reward @ step" value={stepReward.toFixed(2)} hint={`t = ${step}`} />
                  <MetricCard
                    label="Outcome"
                    value={meta.outcome}
                    tone={meta.outcome === "Success" ? "success" : meta.outcome === "Failure" ? "error" : "warning"}
                  />
                </div>
                <RewardChart seed={scenario} step={step} />
              </div>

              {/* Right: method panels */}
              <div className="space-y-3 min-w-0">
                {selectedMethods.length === 0 ? (
                  <EmptyState text="Pick at least one attribution method on the left to see analyses here." />
                ) : (
                  selectedMethods.map((id) => {
                    const m = METHODS.find((x) => x.id === id)!;
                    return (
                      <MethodPanel
                        key={id}
                        method={m}
                        step={step}
                        seedBase={scenario}
                        topN={topN}
                      />
                    );
                  })
                )}
              </div>
            </div>

            {/* Attention section */}
            {model.attention && (
              <div className="panel p-4">
                <div className="mb-3 flex flex-wrap items-center gap-3">
                  <h2 className="text-sm font-semibold tracking-tight">Perceiver cross-attention</h2>
                  <span className="chip mono normal-case tracking-normal">{layer}</span>
                  <div className="ml-auto flex items-center gap-3">
                    <Select value={layer} onValueChange={setLayer}>
                      <SelectTrigger className="h-8 w-44 text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {["cross_attn_avg", "cross_attn_l0", "cross_attn_l1"].map((l) => (
                          <SelectItem key={l} value={l}>{l}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <div className="flex items-center gap-2 text-xs">
                      <Label htmlFor="overlay" className="text-muted-foreground">Overlay on BEV</Label>
                      <Switch id="overlay" checked={attentionOverlay} onCheckedChange={setAttentionOverlay} />
                    </div>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-xs text-muted-foreground">Agent slots — click to highlight on BEV</p>
                    <div className="flex flex-wrap gap-1.5">
                      {Array.from({ length: 8 }, (_, i) => {
                        const on = agents.includes(i);
                        return (
                          <button
                            key={i}
                            onClick={() => toggleAgent(i)}
                            className="mono rounded-md border px-2 py-1 text-xs font-bold transition-all"
                            style={{
                              color: `var(--a${i})`,
                              borderColor: on ? `var(--a${i})` : "var(--color-border)",
                              background: on ? `color-mix(in oklab, var(--a${i}) 18%, transparent)` : "transparent",
                            }}
                          >
                            A{i}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div className="h-48">
                    <ResponsiveContainer>
                      <BarChart data={attn} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" stroke="var(--color-muted-foreground)" fontSize={11} />
                        <YAxis type="category" dataKey="slot" stroke="var(--color-muted-foreground)" fontSize={11} width={36} />
                        <Tooltip
                          contentStyle={{
                            background: "var(--color-popover)",
                            border: "1px solid var(--color-border)",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {attn.map((a, i) => (
                            <Cell key={i} fill={a.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}

            {/* Evaluation report */}
            <div className="panel">
              <button
                onClick={() => setReportOpen((o) => !o)}
                className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-secondary/30"
              >
                <h2 className="text-sm font-semibold tracking-tight">Evaluation report</h2>
                <span className="chip normal-case tracking-normal">{selectedMethods.length} methods</span>
                <ChevronDown className={`ml-auto h-4 w-4 transition-transform ${reportOpen ? "rotate-180" : ""}`} />
              </button>
              {reportOpen && (
                <div className="space-y-4 border-t border-border p-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    {selectedMethods.length >= 2 ? (
                      <Heatmap
                        caption="Method agreement (Spearman ρ)"
                        data={agreement.slice(0, METHODS.length).map((r) => r.slice(0, METHODS.length))}
                        rowLabels={METHODS.map((m) => m.name.split(" ")[0])}
                        colLabels={METHODS.map((m) => m.name.split(" ")[0])}
                        rowColors={METHODS.map((m) => m.color)}
                      />
                    ) : (
                      <EmptyState text="Select 2+ methods to compare agreement." />
                    )}
                    <Heatmap
                      caption="Attention × attribution alignment"
                      data={align}
                      rowLabels={Array.from({ length: 8 }, (_, i) => `A${i}`)}
                      colLabels={METHODS.map((m) => m.name.split(" ")[0])}
                      rowColors={Array.from({ length: 8 }, (_, i) => `var(--a${i})`)}
                    />
                  </div>
                  <div className="panel p-4 text-xs text-muted-foreground">
                    Faithfulness benchmark not yet computed for this scenario. Run{" "}
                    <code className="mono rounded bg-secondary px-1.5 py-0.5 text-foreground">
                      scripts/precompute_faithfulness.py --scenario {scenario}
                    </code>{" "}
                    to populate this section.
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="panel flex h-full min-h-32 flex-col items-center justify-center p-6 text-center">
      <div className="mb-2 h-2 w-2 rounded-full bg-muted-foreground/50" />
      <p className="max-w-sm text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
