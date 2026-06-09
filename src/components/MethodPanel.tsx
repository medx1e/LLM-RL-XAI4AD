import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
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
  Cell,
} from "recharts";
import {
  attributionSeries,
  categoryBreakdown,
  topEntities,
  CATEGORIES,
} from "@/lib/mockData";

export function MethodPanel({
  method,
  step,
  seedBase,
  topN,
}: {
  method: { id: string; name: string; color: string };
  step: number;
  seedBase: number;
  topN: number;
}) {
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState<"category" | "entities" | "timeline">("entities");
  const seed = (seedBase * 17 + method.id.length * 13) | 0;

  const cats = categoryBreakdown(seed);
  const ents = topEntities(seed + step, topN);
  const timeline = attributionSeries(seed);
  const latency = 40 + ((seed * 7) % 240);
  const catMap: Record<string, string> = Object.fromEntries(
    CATEGORIES.map((c) => [c.id, c.color])
  );

  return (
    <div
      className="panel overflow-hidden"
      style={{ borderLeft: `3px solid ${method.color}` }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-secondary/30 transition-colors"
      >
        <span
          className="h-2 w-2 rounded-full"
          style={{ background: method.color }}
        />
        <span className="text-sm font-semibold">{method.name}</span>
        <span className="ml-auto flex items-center gap-2">
          <span className="chip mono normal-case tracking-normal">step {step}</span>
          <span className="chip mono normal-case tracking-normal">{latency}ms</span>
          <ChevronDown
            className={`h-4 w-4 text-muted-foreground transition-transform ${
              open ? "rotate-180" : ""
            }`}
          />
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            <div className="border-t border-border">
              <div className="flex border-b border-border">
                {(["category", "entities", "timeline"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`flex-1 border-b-2 px-3 py-2 text-xs font-medium capitalize transition-colors ${
                      tab === t
                        ? "border-primary text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>

              <div className="p-4">
                {tab === "category" && (
                  <div className="h-56">
                    <ResponsiveContainer>
                      <BarChart data={cats} layout="vertical" margin={{ left: 30 }}>
                        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" stroke="var(--color-muted-foreground)" fontSize={11} />
                        <YAxis type="category" dataKey="name" stroke="var(--color-muted-foreground)" fontSize={11} width={100} />
                        <Tooltip
                          contentStyle={{
                            background: "var(--color-popover)",
                            border: "1px solid var(--color-border)",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {cats.map((c, i) => (
                            <Cell key={i} fill={c.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {tab === "entities" && (
                  <div style={{ height: Math.max(220, topN * 22) }}>
                    <ResponsiveContainer>
                      <BarChart data={ents} layout="vertical" margin={{ left: 30 }}>
                        <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" stroke="var(--color-muted-foreground)" fontSize={11} />
                        <YAxis type="category" dataKey="name" stroke="var(--color-muted-foreground)" fontSize={10} width={90} />
                        <Tooltip
                          contentStyle={{
                            background: "var(--color-popover)",
                            border: "1px solid var(--color-border)",
                            borderRadius: 8,
                            fontSize: 12,
                          }}
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {ents.map((e, i) => (
                            <Cell key={i} fill={e.color || catMap[e.category]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {tab === "timeline" && (
                  <div className="h-56">
                    <ResponsiveContainer>
                      <LineChart data={timeline}>
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
                        <Line
                          type="monotone"
                          dataKey="magnitude"
                          stroke={method.color}
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
