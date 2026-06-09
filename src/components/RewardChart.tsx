import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { rewardSeries } from "@/lib/mockData";

export function RewardChart({ seed, step }: { seed: number; step: number }) {
  const data = rewardSeries(seed);
  return (
    <div className="panel p-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Episode reward
        </h3>
        <span className="mono text-[10px] text-muted-foreground">per-step + cumulative</span>
      </div>
      <div className="h-44">
        <ResponsiveContainer>
          <ComposedChart data={data}>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" />
            <XAxis dataKey="step" stroke="var(--color-muted-foreground)" fontSize={10} />
            <YAxis yAxisId="L" stroke="var(--color-muted-foreground)" fontSize={10} />
            <YAxis yAxisId="R" orientation="right" stroke="var(--color-muted-foreground)" fontSize={10} />
            <Tooltip
              contentStyle={{
                background: "var(--color-popover)",
                border: "1px solid var(--color-border)",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <ReferenceLine yAxisId="L" y={0} stroke="var(--color-border)" />
            <Area
              yAxisId="L"
              type="monotone"
              dataKey="reward"
              stroke="var(--color-agents-c)"
              fill="var(--color-agents-c)"
              fillOpacity={0.18}
            />
            <Line
              yAxisId="R"
              type="monotone"
              dataKey="cum"
              stroke="var(--color-primary)"
              strokeWidth={2}
              dot={false}
            />
            <ReferenceLine yAxisId="L" x={step} stroke="var(--color-warning)" strokeDasharray="4 4" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
