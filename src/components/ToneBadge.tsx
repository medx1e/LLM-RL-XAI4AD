const TONES = {
  detailed: { bg: "#DC2626", label: "Detailed" },
  brief: { bg: "#10B981", label: "Brief" },
  caveat: { bg: "#F59E0B", label: "Caveat" },
  "detailed-caveat": { bg: "#7C3AED", label: "Detailed + Caveat" },
} as const;

export type Tone = keyof typeof TONES;

export function ToneBadge({ tone }: { tone: Tone }) {
  const t = TONES[tone];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.08em]"
      style={{
        background: `color-mix(in oklab, ${t.bg} 18%, transparent)`,
        color: t.bg,
        border: `1px solid color-mix(in oklab, ${t.bg} 40%, transparent)`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: t.bg }} />
      {t.label}
    </span>
  );
}
