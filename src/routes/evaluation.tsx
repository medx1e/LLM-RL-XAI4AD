import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Sparkles, AlertTriangle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { ContextStrip } from "@/components/ContextStrip";
import { MetricCard } from "@/components/MetricCard";
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
import {
  MODELS,
  METHODS,
  CATEGORIES,
  SCENARIOS_META,
  methodProfiles,
  methodAgreement,
  alignmentMatrix,
  attributionSeries,
} from "@/lib/mockData";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Evaluation · XAI4AD" },
      { name: "description", content: "Attribution method quality, agreement, and alignment report." },
    ],
  }),
  component: EvalPage,
});

function EvalPage() {
  const [modelSlug, setModelSlug] = useState(MODELS[0].slug);
  const model = MODELS.find((m) => m.slug === modelSlug)!;
  const [scenario, setScenario] = useState(model.scenarios[0]);
  const [refStep, setRefStep] = useState(40);

  const profiles = methodProfiles();
  const agreement = methodAgreement();
  const align = alignmentMatrix();

  const focusData = CATEGORIES.map((c) => {
    const row: Record<string, number | string> = { name: c.name };
    METHODS.forEach((m, mi) => {
      row[m.name] = +(((mi + 1) * 0.07 + Math.abs(Math.sin(c.id.length + mi)) * 0.6).toFixed(2));
    });
    return row;
  });

  const stabilityData = Array.from({ length: 80 }, (_, step) => {
    const row: Record<string, number> = { step };
    METHODS.forEach((m, mi) => {
      row[m.name] = attributionSeries(scenario * 17 + mi)[step].magnitude;
    });
    return row;
  });

  return (
    <>
      <ContextStrip title="Evaluation" subtitle={`${model.name} · scenario ${scenario}`} />
      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <ControlPanel title="Selection">
              <Field label="Model">
                <Select value={modelSlug} onValueChange={(v) => {
                  setModelSlug(v);
                  setScenario(MODELS.find((m) => m.slug === v)!.scenarios[0]);
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
                        #{s} · {SCENARIOS_META[s]?.description ?? ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label={`Reference step · ${refStep}`}>
                <Slider value={[refStep]} min={0} max={79} step={1} onValueChange={(v) => setRefStep(v[0])} />
              </Field>
            </ControlPanel>

            <button
              disabled
              className="panel flex w-full items-center justify-center gap-2 px-4 py-3 text-sm font-medium text-muted-foreground opacity-60"
              title="Requires LLM API key"
            >
              <Sparkles className="h-4 w-4" />
              Generate LLM analysis
            </button>
            <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/5 p-3 text-[11px] text-warning">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span>Enable an LLM in Settings to unlock automated narrative summaries.</span>
            </div>
          </aside>

          <section className="space-y-6 min-w-0">
            {/* A. Method profiles */}
            <Section letter="A" title="Method profiles">
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {profiles.map((p) => (
                  <div
                    key={p.id}
                    className="panel p-4"
                    style={{ borderTop: `2px solid ${p.color}` }}
                  >
                    <div className="text-xs font-semibold">{p.name}</div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <Metric label="Sparsity" value={p.sparsity} />
                      <Metric label="Gini" value={p.gini} />
                      <Metric label="Top-10%" value={p.top10} />
                      <Metric label="Stability" value={p.stability} />
                    </div>
                    <div className="mt-3 text-[10px] text-muted-foreground">
                      Dominates: <span className="text-foreground/80">{p.dominantCategory}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* B. Category focus */}
            <Section letter="B" title="Category focus">
              <div className="panel p-4">
                <div className="h-72">
                  <ResponsiveContainer>
                    <BarChart data={focusData}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                      <XAxis dataKey="name" stroke="var(--color-muted-foreground)" fontSize={11} />
                      <YAxis stroke="var(--color-muted-foreground)" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          background: "var(--color-popover)",
                          border: "1px solid var(--color-border)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {METHODS.map((m) => (
                        <Bar key={m.id} dataKey={m.name} fill={m.color} radius={[3, 3, 0, 0]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Section>

            {/* C + D heatmaps */}
            <div className="grid gap-4 xl:grid-cols-2">
              <Section letter="C" title="Method agreement">
                <Heatmap
                  data={agreement}
                  rowLabels={METHODS.map((m) => m.name.split(" ")[0])}
                  colLabels={METHODS.map((m) => m.name.split(" ")[0])}
                  rowColors={METHODS.map((m) => m.color)}
                />
              </Section>
              <Section letter="D" title="Attention × attribution">
                <Heatmap
                  data={align}
                  rowLabels={Array.from({ length: 8 }, (_, i) => `A${i}`)}
                  colLabels={METHODS.map((m) => m.name.split(" ")[0])}
                  rowColors={Array.from({ length: 8 }, (_, i) => `var(--a${i})`)}
                />
              </Section>
            </div>

            {/* E. Temporal stability */}
            <Section letter="E" title="Temporal stability">
              <div className="panel p-4">
                <div className="h-64">
                  <ResponsiveContainer>
                    <LineChart data={stabilityData}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                      <XAxis dataKey="step" stroke="var(--color-muted-foreground)" fontSize={11} />
                      <YAxis stroke="var(--color-muted-foreground)" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          background: "var(--color-popover)",
                          border: "1px solid var(--color-border)",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {METHODS.map((m) => (
                        <Line key={m.id} type="monotone" dataKey={m.name} stroke={m.color} strokeWidth={1.6} dot={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </Section>

            {/* F + G placeholders */}
            <div className="grid gap-4 xl:grid-cols-2">
              <Section letter="F" title="Faithfulness">
                <div className="panel p-6">
                  <MetricCard label="Status" value="Not computed" tone="warning" hint="Run precompute script" />
                  <p className="mt-3 text-xs text-muted-foreground">
                    Faithfulness benchmark (deletion + insertion AUC) requires re-running the policy on
                    perturbed observations. Schedule via{" "}
                    <code className="mono rounded bg-secondary px-1.5 py-0.5">scripts/precompute_faithfulness.py</code>.
                  </p>
                </div>
              </Section>
              <Section letter="G" title="LLM narrative summary">
                <div className="panel p-6 text-center">
                  <Sparkles className="mx-auto h-6 w-6 text-muted-foreground" />
                  <p className="mt-2 text-sm font-medium">Generate an AI-written report</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Summarises method profiles, agreement, and alignment into a paragraph suitable for
                    paper appendices.
                  </p>
                  <button
                    disabled
                    className="mt-3 inline-flex items-center gap-2 rounded-md border border-border bg-secondary px-3 py-1.5 text-xs font-medium opacity-60"
                  >
                    Generate analysis
                  </button>
                </div>
              </Section>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

function Section({ letter, title, children }: { letter: string; title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-2 flex items-center gap-2">
        <span className="mono flex h-6 w-6 items-center justify-center rounded-md bg-primary/15 text-[11px] font-bold text-primary">
          {letter}
        </span>
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  const tone = value > 0.7 ? "text-success" : value > 0.45 ? "text-warning" : "text-muted-foreground";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={`text-base font-bold tabular-nums ${tone}`}>{value.toFixed(2)}</div>
    </div>
  );
}
