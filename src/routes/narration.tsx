import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { ContextStrip } from "@/components/ContextStrip";
import { BevPlayer } from "@/components/BevPlayer";
import { ControlPanel, Field } from "@/components/ControlPanel";
import { ToneBadge, type Tone } from "@/components/ToneBadge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { LLMS, MODELS, TOGGLES, SCENARIOS_META, narrationFor } from "@/lib/mockData";

export const Route = createFileRoute("/narration")({
  head: () => ({
    meta: [
      { title: "LLM Narration · XAI4AD" },
      { name: "description", content: "Synchronized LLM narrations of RL driving policy decisions." },
    ],
  }),
  component: NarrationPage,
});

const TONE_COLORS: Record<Tone, string> = {
  detailed: "#DC2626",
  brief: "#10B981",
  caveat: "#F59E0B",
  "detailed-caveat": "#7C3AED",
};

function NarrationPage() {
  const scenarios = Object.keys(SCENARIOS_META).map(Number);
  const [scenario, setScenario] = useState(scenarios[0]);
  const [driver, setDriver] = useState(MODELS[0].slug);
  const [llmA, setLlmA] = useState(LLMS[0].key);
  const [toggleA, setToggleA] = useState(TOGGLES[0].id);
  const [compare, setCompare] = useState(true);
  const [llmB, setLlmB] = useState(LLMS[2].key);
  const [toggleB, setToggleB] = useState(TOGGLES[1].id);
  const [step, setStep] = useState(24);
  const [reportOpen, setReportOpen] = useState(false);

  const narrA = useMemo(() => narrationFor(step, scenario + llmA.length + toggleA.length), [step, scenario, llmA, toggleA]);
  const narrB = useMemo(() => narrationFor(step, scenario + llmB.length + toggleB.length + 1), [step, scenario, llmB, toggleB]);

  return (
    <>
      <ContextStrip
        title="LLM Narration"
        subtitle={`scenario #${scenario} · ${SCENARIOS_META[scenario]?.description ?? ""}`}
      />
      <main className="flex-1 px-4 py-6 md:px-6">
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <ControlPanel title="Scenario">
              <Field label="Curated scenario">
                <Select value={String(scenario)} onValueChange={(v) => setScenario(Number(v))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {scenarios.map((s) => (
                      <SelectItem key={s} value={String(s)}>
                        #{s} · {SCENARIOS_META[s]?.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Driving model">
                <Select value={driver} onValueChange={setDriver}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {MODELS.filter((m) => m.family !== "CBM").map((m) => (
                      <SelectItem key={m.slug} value={m.slug}>{m.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </ControlPanel>

            <ControlPanel title="Narration A">
              <Field label="LLM">
                <Select value={llmA} onValueChange={setLlmA}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {LLMS.map((l) => (
                      <SelectItem key={l.key} value={l.key}>{l.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Toggle combo">
                <Select value={toggleA} onValueChange={setToggleA}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {TOGGLES.map((t) => (
                      <SelectItem key={t.id} value={t.id}>{t.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            </ControlPanel>

            <ControlPanel title="Compare">
              <div className="flex items-center justify-between text-xs">
                <Label htmlFor="cmp" className="text-muted-foreground">Side-by-side</Label>
                <Switch id="cmp" checked={compare} onCheckedChange={setCompare} />
              </div>
              {compare && (
                <>
                  <Field label="LLM B">
                    <Select value={llmB} onValueChange={setLlmB}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {LLMS.map((l) => (
                          <SelectItem key={l.key} value={l.key}>{l.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Field label="Toggle B">
                    <Select value={toggleB} onValueChange={setToggleB}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {TOGGLES.map((t) => (
                          <SelectItem key={t.id} value={t.id}>{t.label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                </>
              )}
            </ControlPanel>
          </aside>

          <section className="space-y-4 min-w-0">
            <BevPlayer step={step} setStep={setStep} scenarioSeed={scenario} highlights={[16, 32, 48, 64]} />

            <div className={compare ? "grid gap-4 md:grid-cols-2" : ""}>
              <NarrationCard
                title={LLMS.find((l) => l.key === llmA)!.name}
                subtitle={TOGGLES.find((t) => t.id === toggleA)!.label}
                step={step}
                narr={narrA}
              />
              {compare && (
                <NarrationCard
                  title={LLMS.find((l) => l.key === llmB)!.name}
                  subtitle={TOGGLES.find((t) => t.id === toggleB)!.label}
                  step={step}
                  narr={narrB}
                />
              )}
            </div>

            <div className="panel">
              <button
                onClick={() => setReportOpen((o) => !o)}
                className="flex w-full items-center px-4 py-3 hover:bg-secondary/30"
              >
                <h3 className="text-sm font-semibold">Structured report</h3>
                <span className="ml-2 chip normal-case tracking-normal">step {step}</span>
                <ChevronDown className={`ml-auto h-4 w-4 transition-transform ${reportOpen ? "rotate-180" : ""}`} />
              </button>
              {reportOpen && (
                <div className="grid gap-4 border-t border-border p-4 md:grid-cols-2">
                  <StructuredReport title="Narration A" narr={narrA} />
                  {compare && <StructuredReport title="Narration B" narr={narrB} />}
                </div>
              )}
            </div>
          </section>
        </div>
      </main>
    </>
  );
}

function NarrationCard({
  title,
  subtitle,
  step,
  narr,
}: {
  title: string;
  subtitle: string;
  step: number;
  narr: ReturnType<typeof narrationFor>;
}) {
  const c = TONE_COLORS[narr.tone];
  return (
    <article
      className="panel overflow-hidden"
      style={{ borderLeft: `3px solid ${c}` }}
    >
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{title}</div>
          <div className="mono truncate text-[11px] text-muted-foreground">{subtitle}</div>
        </div>
        <div className="ml-auto"><ToneBadge tone={narr.tone} /></div>
      </div>
      <div className="px-4 py-4">
        <p className="text-[15px] leading-[1.65] text-foreground/95" style={{ fontFamily: "Source Serif 4, Georgia, serif" }}>
          {narr.text}
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
          <span className="mono">t = {step}</span>
          <span>·</span>
          <span className="mono">{narr.responseTimeMs} ms</span>
          <span>·</span>
          <span>decision: <span className="font-semibold text-foreground/80">{narr.decisionClass}</span></span>
        </div>
      </div>
    </article>
  );
}

function StructuredReport({ title, narr }: { title: string; narr: ReturnType<typeof narrationFor> }) {
  return (
    <div className="panel p-4">
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">{title}</h4>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="Necessity" value={narr.necessity} />
        <Stat label="Grounding" value={narr.grounding} />
        <div className="col-span-2">
          <div className="text-[11px] uppercase tracking-widest text-muted-foreground">Alternatives</div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {narr.alternatives.map((a) => (
              <span key={a} className="chip normal-case tracking-normal">{a}</span>
            ))}
          </div>
        </div>
      </dl>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-1 text-xl font-bold tabular-nums">{value.toFixed(2)}</div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div className="h-full bg-primary" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}
