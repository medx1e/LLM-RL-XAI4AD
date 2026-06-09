import { useState } from "react";

export function Heatmap({
  data,
  rowLabels,
  colLabels,
  rowColors,
  diverging = true,
  caption,
}: {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  rowColors?: string[];
  diverging?: boolean;
  caption?: string;
}) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const cell = (v: number) => {
    if (diverging) {
      // -1..1 → blue..white..red
      const t = Math.max(-1, Math.min(1, v));
      if (t >= 0) {
        return `color-mix(in oklab, #EF4444 ${Math.round(t * 75)}%, transparent)`;
      }
      return `color-mix(in oklab, #5B6CF9 ${Math.round(-t * 75)}%, transparent)`;
    }
    // 0..1 → primary intensity
    const t = Math.max(0, Math.min(1, v));
    return `color-mix(in oklab, #5B6CF9 ${Math.round(t * 80 + 5)}%, transparent)`;
  };

  return (
    <div className="panel p-4">
      {caption && (
        <div className="mb-3 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          {caption}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5">
          <thead>
            <tr>
              <th className="w-32" />
              {colLabels.map((c, i) => (
                <th
                  key={i}
                  className="mono px-2 py-1 text-[10px] font-medium text-muted-foreground"
                  style={{ minWidth: 60 }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, r) => (
              <tr key={r}>
                <th className="px-2 py-1 text-right text-[11px] font-medium text-muted-foreground whitespace-nowrap">
                  <span className="inline-flex items-center gap-2">
                    {rowColors?.[r] && (
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ background: rowColors[r] }}
                      />
                    )}
                    {rowLabels[r]}
                  </span>
                </th>
                {row.map((v, c) => (
                  <td
                    key={c}
                    onMouseEnter={() => setHover({ r, c })}
                    onMouseLeave={() => setHover(null)}
                    className="relative h-10 cursor-default text-center text-[11px] font-medium tabular-nums transition-all"
                    style={{
                      background: cell(v),
                      outline:
                        hover?.r === r && hover?.c === c
                          ? "2px solid var(--color-primary)"
                          : "1px solid var(--color-border)",
                      borderRadius: 4,
                      minWidth: 60,
                    }}
                  >
                    <span className="text-foreground/90">{v.toFixed(2)}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hover && (
        <div className="mt-3 mono text-[11px] text-muted-foreground">
          ρ = {data[hover.r][hover.c].toFixed(3)}  ·  {rowLabels[hover.r]} × {colLabels[hover.c]}
        </div>
      )}
    </div>
  );
}
