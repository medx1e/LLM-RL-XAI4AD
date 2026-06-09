import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { EPISODE_LEN } from "@/lib/mockData";

export function BevPlayer({
  step,
  setStep,
  highlights = [],
  scenarioSeed = 1,
  agentOverlays = [],
}: {
  step: number;
  setStep: (s: number) => void;
  highlights?: number[];
  scenarioSeed?: number;
  agentOverlays?: number[];
}) {
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setStep(Math.min(EPISODE_LEN - 1, step + 1));
      if (step >= EPISODE_LEN - 1) setPlaying(false);
    }, 120);
    return () => clearInterval(id);
  }, [playing, step, setStep]);

  return (
    <div className="panel overflow-hidden">
      <BevCanvas step={step} scenarioSeed={scenarioSeed} agentOverlays={agentOverlays} />

      <div className="border-t border-border bg-card/60 px-3 py-2">
        <div className="flex items-center gap-2">
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8"
            onClick={() => setStep(Math.max(0, step - 1))}
            aria-label="Previous step"
          >
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            className="h-9 w-9 rounded-full"
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8"
            onClick={() => setStep(Math.min(EPISODE_LEN - 1, step + 1))}
            aria-label="Next step"
          >
            <SkipForward className="h-4 w-4" />
          </Button>

          <div className="relative flex-1 px-2">
            <Slider
              value={[step]}
              min={0}
              max={EPISODE_LEN - 1}
              step={1}
              onValueChange={(v) => setStep(v[0])}
            />
            {/* Highlight ticks */}
            <div className="pointer-events-none absolute inset-x-2 top-1/2 -translate-y-1/2">
              {highlights.map((h) => (
                <span
                  key={h}
                  className="absolute h-3 w-0.5 -translate-y-1/2 bg-warning/80"
                  style={{ left: `${(h / (EPISODE_LEN - 1)) * 100}%`, top: "50%" }}
                />
              ))}
            </div>
          </div>

          <div className="mono shrink-0 rounded-md border border-border bg-secondary px-2 py-1 text-[11px] tabular-nums">
            {String(step).padStart(2, "0")} / {EPISODE_LEN - 1}
          </div>
        </div>
      </div>
    </div>
  );
}

function BevCanvas({
  step,
  scenarioSeed,
  agentOverlays,
}: {
  step: number;
  scenarioSeed: number;
  agentOverlays: number[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const agents = useMemo(() => {
    const rng = (i: number) => {
      const x = Math.sin(scenarioSeed * 9999 + i * 31) * 10000;
      return x - Math.floor(x);
    };
    return Array.from({ length: 8 }, (_, i) => ({
      id: i,
      x0: rng(i) * 0.8 + 0.1,
      y0: rng(i + 100) * 0.8 + 0.1,
      vx: (rng(i + 200) - 0.5) * 0.006,
      vy: (rng(i + 300) - 0.5) * 0.006,
    }));
  }, [scenarioSeed]);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    const W = cvs.width;
    const H = cvs.height;

    // Dark BEV background
    ctx.fillStyle = "#0f1117";
    ctx.fillRect(0, 0, W, H);

    // Subtle grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i < W; i += 24) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, H);
      ctx.stroke();
    }
    for (let j = 0; j < H; j += 24) {
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(W, j);
      ctx.stroke();
    }

    // Roadgraph lanes
    ctx.strokeStyle = "rgba(85,168,104,0.55)";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(0, H * 0.5);
    ctx.bezierCurveTo(W * 0.3, H * 0.5, W * 0.6, H * 0.35, W, H * 0.4);
    ctx.stroke();
    ctx.strokeStyle = "rgba(85,168,104,0.35)";
    ctx.lineWidth = 18;
    ctx.beginPath();
    ctx.moveTo(W * 0.5, 0);
    ctx.bezierCurveTo(W * 0.5, H * 0.3, W * 0.55, H * 0.6, W * 0.6, H);
    ctx.stroke();

    // Centerlines (dashed)
    ctx.setLineDash([8, 10]);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, H * 0.5);
    ctx.bezierCurveTo(W * 0.3, H * 0.5, W * 0.6, H * 0.35, W, H * 0.4);
    ctx.stroke();
    ctx.setLineDash([]);

    // Traffic light
    const tl = { x: W * 0.62, y: H * 0.38 };
    ctx.fillStyle = step < 30 ? "#C44E52" : "#10B981";
    ctx.beginPath();
    ctx.arc(tl.x, tl.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.stroke();

    // GPS path (ego trajectory line)
    ctx.strokeStyle = "rgba(129,114,178,0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(W * 0.15, H * 0.78);
    ctx.bezierCurveTo(W * 0.35, H * 0.7, W * 0.5, H * 0.55, W * 0.85, H * 0.45);
    ctx.stroke();
    ctx.setLineDash([]);

    // Other agents
    const agentColors = [
      "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
      "#FF7F00", "#FFD92F", "#A65628", "#F781BF",
    ];
    agents.forEach((a) => {
      const x = (a.x0 + a.vx * step) * W;
      const y = (a.y0 + a.vy * step) * H;
      const highlighted = agentOverlays.includes(a.id);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.atan2(a.vy, a.vx));
      ctx.fillStyle = highlighted ? agentColors[a.id] : "rgba(221,132,82,0.85)";
      ctx.strokeStyle = highlighted ? agentColors[a.id] : "rgba(255,255,255,0.4)";
      ctx.lineWidth = highlighted ? 2.5 : 1;
      ctx.fillRect(-10, -5, 20, 10);
      ctx.strokeRect(-10, -5, 20, 10);
      if (highlighted) {
        ctx.shadowColor = agentColors[a.id];
        ctx.shadowBlur = 12;
        ctx.strokeRect(-10, -5, 20, 10);
      }
      ctx.restore();
      // Label
      if (highlighted) {
        ctx.fillStyle = agentColors[a.id];
        ctx.font = "bold 10px JetBrains Mono, monospace";
        ctx.fillText(`A${a.id}`, x + 12, y - 8);
      }
    });

    // SDC (ego) — animated along GPS
    const t = step / (EPISODE_LEN - 1);
    const ex = W * (0.15 + t * 0.7);
    const ey = H * (0.78 - t * 0.33);
    ctx.save();
    ctx.translate(ex, ey);
    ctx.rotate(-0.4);
    ctx.fillStyle = "#4C72B0";
    ctx.strokeStyle = "#7BA0DA";
    ctx.lineWidth = 2;
    ctx.fillRect(-12, -6, 24, 12);
    ctx.strokeRect(-12, -6, 24, 12);
    // Headlights
    ctx.fillStyle = "#FFD92F";
    ctx.fillRect(10, -5, 3, 3);
    ctx.fillRect(10, 2, 3, 3);
    ctx.restore();

    // SDC label
    ctx.fillStyle = "#4C72B0";
    ctx.font = "bold 10px JetBrains Mono, monospace";
    ctx.fillText("SDC", ex - 10, ey - 12);
  }, [step, agents, agentOverlays]);

  return (
    <div className="relative aspect-[16/10] w-full bg-[#0f1117]">
      <canvas
        ref={canvasRef}
        width={960}
        height={600}
        className="absolute inset-0 h-full w-full"
      />
      <div className="absolute left-3 top-3 flex gap-1.5">
        <span className="chip">BEV</span>
        <span className="chip mono normal-case tracking-normal">t={step}</span>
      </div>
      <div className="absolute right-3 top-3 flex gap-1.5">
        <LegendDot color="#4C72B0" label="SDC" />
        <LegendDot color="#DD8452" label="agents" />
        <LegendDot color="#55A868" label="road" />
        <LegendDot color="#C44E52" label="TL" />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1 rounded-full border border-white/10 bg-black/40 px-2 py-0.5 text-[10px] text-white/80">
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
