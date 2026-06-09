import { createFileRoute, Link } from "@tanstack/react-router";
import { Activity, Brain, MessageSquare, BarChart3, ArrowRight, CircleCheck, CircleDashed } from "lucide-react";
import { ContextStrip } from "@/components/ContextStrip";
import { MODELS, SCENARIOS_META } from "@/lib/mockData";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "XAI4AD · Home" },
      { name: "description", content: "XAI4AD platform overview and curated scenario index." },
    ],
  }),
  component: Index,
});

const TAB_CARDS = [
  {
    to: "/posthoc",
    icon: Activity,
    color: "var(--a0)",
    title: "Post-hoc XAI",
    desc: "Attribution maps, Perceiver cross-attention, and entity-level inspection per timestep.",
  },
  {
    to: "/cbm",
    icon: Brain,
    color: "var(--a2)",
    title: "CBM Explorer",
    desc: "Concept Bottleneck Model timelines across 15 interpretable concepts.",
  },
  {
    to: "/narration",
    icon: MessageSquare,
    color: "var(--a1)",
    title: "LLM Narration",
    desc: "Natural-language narrations of policy decisions across LLM × toggle combinations.",
  },
  {
    to: "/evaluation",
    icon: BarChart3,
    color: "var(--a4)",
    title: "Evaluation",
    desc: "Method agreement, faithfulness, and attention–attribution alignment reports.",
  },
] as const;

function Index() {
  return (
    <>
      <ContextStrip
        title="XAI4AD"
        subtitle="Explainable AI for Autonomous Driving · Waymo Open Motion"
      />
      <main className="flex-1 px-6 py-8 md:px-10">
        {/* Hero */}
        <section className="relative overflow-hidden rounded-2xl border border-border bg-card grid-bg">
          <div className="relative scanline p-8 md:p-12">
            <div className="flex flex-wrap items-center gap-2">
              <span className="chip" style={{ background: "color-mix(in oklab, var(--color-primary) 18%, transparent)", color: "var(--color-primary)", borderColor: "color-mix(in oklab, var(--color-primary) 40%, transparent)" }}>
                Research preview
              </span>
              <span className="chip mono normal-case tracking-normal">v0.4 · llm</span>
              <span className="chip mono normal-case tracking-normal">{Object.keys(SCENARIOS_META).length} curated scenarios</span>
            </div>
            <h1 className="mt-5 max-w-3xl text-4xl font-bold leading-tight tracking-tight md:text-5xl">
              Audit <span className="text-primary">why</span> an RL driving agent did what it did —
              not just what.
            </h1>
            <p className="mt-4 max-w-2xl text-base text-muted-foreground md:text-lg">
              XAI4AD surfaces post-hoc attribution, Concept Bottleneck Model outputs, Perceiver
              cross-attention, and LLM-generated narrations — all synchronised to a Bird's-Eye View
              episode replay.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                to="/posthoc"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Open Post-hoc XAI <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                to="/narration"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-secondary px-5 py-2.5 text-sm font-medium hover:bg-secondary/70"
              >
                Compare LLM narrations
              </Link>
            </div>
          </div>
        </section>

        {/* Tab cards */}
        <section className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
          {TAB_CARDS.map((c) => (
            <Link
              key={c.to}
              to={c.to}
              className="group panel p-5 transition-all hover:-translate-y-0.5 hover:shadow-xl"
              style={{ borderLeft: `3px solid ${c.color}` }}
            >
              <div className="flex items-start gap-4">
                <div
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
                  style={{ background: `color-mix(in oklab, ${c.color} 18%, transparent)`, color: c.color }}
                >
                  <c.icon className="h-5 w-5" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">{c.title}</h3>
                    <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">{c.desc}</p>
                </div>
              </div>
            </Link>
          ))}
        </section>

        {/* Models table */}
        <section className="mt-10">
          <div className="mb-3 flex items-end justify-between">
            <div>
              <h2 className="text-lg font-semibold tracking-tight">Available models</h2>
              <p className="text-sm text-muted-foreground">
                Cached artifacts discovered on disk.
              </p>
            </div>
            <span className="mono text-[11px] text-muted-foreground">
              {MODELS.length} models · {MODELS.reduce((a, m) => a + m.scenarios.length, 0)} artifacts
            </span>
          </div>
          <div className="panel overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-secondary/40 text-[11px] uppercase tracking-widest text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold">Model</th>
                  <th className="px-4 py-3 text-left font-semibold">Family</th>
                  <th className="px-4 py-3 text-left font-semibold">Attention</th>
                  <th className="px-4 py-3 text-left font-semibold">Scenarios</th>
                  <th className="px-4 py-3 text-left font-semibold">Notes</th>
                </tr>
              </thead>
              <tbody>
                {MODELS.map((m) => (
                  <tr key={m.slug} className="border-t border-border">
                    <td className="px-4 py-3 font-medium">{m.name}</td>
                    <td className="px-4 py-3">
                      <span className="chip normal-case tracking-normal">{m.family}</span>
                    </td>
                    <td className="px-4 py-3">
                      {m.attention ? (
                        <span className="inline-flex items-center gap-1 text-success">
                          <CircleCheck className="h-4 w-4" /> yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-muted-foreground">
                          <CircleDashed className="h-4 w-4" /> n/a
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {m.scenarios.slice(0, 8).map((s) => (
                          <span key={s} className="mono rounded border border-border bg-secondary px-1.5 py-0.5 text-[11px]">
                            {s}
                          </span>
                        ))}
                        {m.scenarios.length > 8 && (
                          <span className="mono text-[11px] text-muted-foreground">
                            +{m.scenarios.length - 8}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{m.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </>
  );
}
