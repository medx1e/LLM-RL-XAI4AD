import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import { ContextStrip } from "@/components/ContextStrip";
import { BevPlayer } from "@/components/BevPlayer";
import { MetricCard } from "@/components/MetricCard";
import { RewardChart } from "@/components/RewardChart";
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
  ARCHETYPES,
  CONCEPTS,
  conceptSeries,
  rewardSeries,
  SCENARIOS_META,
} from "@/lib/mockData";

export const Route = createFileRoute("/cbm")({
  head: () => ({
    meta: [
      { title: "CBM Explorer · XAI4AD" },
      { name: "description", content: "Concept Bottleneck Model concept timelines." },
    ],
  }),
  component: CbmPage,
});

function CbmPage() {
  const [archetype, setArchetype] = useState(ARCHETYPES[0].id);
  const [rank, setRank] = useState(1);
  const [selected, setSelected] = useState<string[]>(CONCEPTS.slice(0, 6));
  const [step, setStep] = useState(34);
  const scenario = rank * 3;
  const meta = SCENARIOS_META[scenario] || { outcome: "Success", tag: "Urban", description: "Curated CBM scenario" };
  const rewards = rewardSeries(scenario);
  const totalReward = rewards.reduce((a, b) => a + b.reward, 0);

  const toggleConcept = (c: string) =>
    setSelected((s) => (s.includes(c) ? s.filter((x) => x !== c) : [...s, c]));

  return (
    <>
      <ContextStrip
        title="CBM Explorer"
        subtitle={`CBM v1 · archetype: ${ARCHETYPES.find((a) => a.id === archetype)?.label} · scenario #${scenario}`}
      />
      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <ControlPanel title="Archetype">
              <div className="grid grid-cols-2 gap-2">
                {ARCHETYPES.map((a) => {
                  const on = a.id === archetype;
                  return (
                    <button
                      key={a.id}
                      onClick={() => setArchetype(a.id)}
                      className="flex flex-col items-start gap-1 rounded-lg border px-3 py-2 text-left text-xs transition-all"
                      style={{
                        borderColor: on ? a.color : "var(--color-border)",
                        background: on ? `color-mix(in oklab, ${a.color} 12%, transparent)` : "var(--color-secondary)",
                      }}
                    >
                      <span className="text-lg">{a.emoji}</span>
                      <span className="font-medium">{a.label}</span>
                    </button>
                  );
                })}
              </div>
            </ControlPanel>

            <ControlPanel title="Selection">
              <Field label={`Scenario rank · ${rank}`}>
                <Slider value={[rank]} min={1} max={6} step={1} onValueChange={(v) => setRank(v[0])} />
              </Field>
            </ControlPanel>

            <ControlPanel title={`Concepts · ${selected.length}/${CONCEPTS.length}`}>
              <div className="max-h-72 space-y-1 overflow-y-auto pr-1">
                {CONCEPTS.map((c) => {
                  const on = selected.includes(c);
                  return (
                    <button
                      key={c}
                      onClick={() => toggleConcept(c)}
                      className={`mono flex w-full items-center justify-between rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors ${
                        on
                          ? "border-primary/50 bg-primary/10"
                          : "border-border bg-secondary/30 hover:bg-secondary/60"
                      }`}
                    >
                      <span>{c}</span>
                      <span className={`h-3 w-3 rounded ${on ? "bg-primary" : "border border-border"}`} />
                    </button>
                  );
                })}
              </div>
            </ControlPanel>
          </aside>

          <section className="space-y-4 min-w-0">
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="space-y-4 min-w-0">
                <BevPlayer step={step} setStep={setStep} scenarioSeed={scenario} highlights={[18, 34, 52]} />
                <div className="grid grid-cols-3 gap-3">
                  <MetricCard label="Total reward" value={totalReward.toFixed(2)} tone={totalReward >= 0 ? "success" : "error"} />
                  <MetricCard label="Reward @ step" value={(rewards[step]?.reward ?? 0).toFixed(2)} hint={`t = ${step}`} />
                  <MetricCard label="Outcome" value={meta.outcome} tone="success" />
                </div>
                <RewardChart seed={scenario} step={step} />
              </div>

              <div className="space-y-3 min-w-0">
                <div className="panel p-4">
                  <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                    Concept timeline
                  </h3>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Ground truth (blue) vs predicted (orange). Grey = invalid steps. Vertical line = current step.
                  </p>
                </div>
                <div className="grid gap-2">
                  {selected.length === 0 && (
                    <div className="panel p-6 text-center text-sm text-muted-foreground">
                      Select concepts on the left to plot timelines.
                    </div>
                  )}
                  {selected.map((c, idx) => {
                    const data = conceptSeries(idx, scenario);
                    const acc = (
                      data.reduce((a, d) => a + (1 - Math.abs(d.truth - d.pred)), 0) / data.length
                    ).toFixed(2);
                    return (
                      <div key={c} className="panel p-3">
                        <div className="mb-1 flex items-center justify-between">
                          <span className="mono text-xs font-semibold">{c}</span>
                          <span className="mono text-[10px] text-muted-foreground">
                            acc = {acc}
                          </span>
                        </div>
                        <div className="h-20">
                          <ResponsiveContainer>
                            <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
                              <XAxis dataKey="step" stroke="var(--color-muted-foreground)" fontSize={9} tick={false} />
                              <YAxis stroke="var(--color-muted-foreground)" fontSize={9} width={20} domain={[0, 1]} />
                              <Tooltip
                                contentStyle={{
                                  background: "var(--color-popover)",
                                  border: "1px solid var(--color-border)",
                                  borderRadius: 8,
                                  fontSize: 11,
                                }}
                              />
                              <ReferenceArea x1={0} x2={4} fill="var(--color-muted)" fillOpacity={0.4} />
                              <ReferenceArea x1={75} x2={79} fill="var(--color-muted)" fillOpacity={0.4} />
                              <Line type="monotone" dataKey="truth" stroke="var(--color-sdc)" strokeWidth={1.6} dot={false} />
                              <Line
                                type="monotone"
                                dataKey="pred"
                                stroke="var(--color-agents-c)"
                                strokeWidth={1.6}
                                strokeDasharray="4 3"
                                dot={false}
                              />
                              <ReferenceLine x={step} stroke="var(--color-warning)" strokeDasharray="2 2" />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Concept values table */}
            <div className="panel overflow-hidden">
              <div className="border-b border-border px-4 py-3">
                <h3 className="text-sm font-semibold">Concept values · step {step}</h3>
              </div>
              <table className="w-full text-sm">
                <thead className="bg-secondary/30 text-[11px] uppercase tracking-widest text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left">Concept</th>
                    <th className="px-4 py-2 text-right">Truth</th>
                    <th className="px-4 py-2 text-right">Pred</th>
                    <th className="px-4 py-2 text-right">Error</th>
                    <th className="px-4 py-2 text-left">Valid</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.slice(0, 8).map((c, i) => {
                    const d = conceptSeries(i, scenario)[step];
                    const err = Math.abs(d.truth - d.pred);
                    return (
                      <tr key={c} className="border-t border-border">
                        <td className="px-4 py-2 mono text-xs">{c}</td>
                        <td className="px-4 py-2 text-right tabular-nums">{d.truth.toFixed(3)}</td>
                        <td className="px-4 py-2 text-right tabular-nums">{d.pred.toFixed(3)}</td>
                        <td className="px-4 py-2 text-right tabular-nums">
                          <span className={err > 0.15 ? "text-warning" : "text-success"}>
                            {err.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-xs text-muted-foreground">
                          {d.invalid ? "—" : "✓"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
