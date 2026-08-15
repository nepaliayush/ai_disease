import { Progress } from "@/components/ui/progress";
import { bandColor } from "@/lib/utils";

export default function RiskBar({ pct, level }) {
  const color = bandColor(level);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Risk score</span>
        <span className="font-semibold" style={{ color }}>
          {level} · {pct.toFixed(1)}%
        </span>
      </div>
      <Progress
        value={Math.min(100, Math.max(0, pct))}
        className="h-2"
        indicatorStyle={{ background: color }}
      />
    </div>
  );
}

export function BreakdownBars({ breakdown }) {
  const clinical = breakdown.clinical_share_pct;
  const symptom = breakdown.symptom_share_pct;
  return (
    <div className="space-y-1.5">
      <div className="flex h-4 w-full overflow-hidden rounded-md bg-muted">
        <div
          className="flex h-full items-center justify-start bg-primary text-[10px] font-semibold text-primary-foreground"
          style={{ width: `${clinical}%` }}
          title={`Clinical ${clinical}%`}
        >
          {clinical >= 12 ? `${clinical.toFixed(0)}%` : ""}
        </div>
        <div
          className="flex h-full items-center justify-end bg-secondary text-[10px] font-semibold text-secondary-foreground"
          style={{ width: `${symptom}%` }}
          title={`Symptoms ${symptom}%`}
        >
          {symptom >= 12 ? `${symptom.toFixed(0)}%` : ""}
        </div>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-primary" />
          Clinical data: {clinical.toFixed(0)}%
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-sm bg-secondary" />
          Symptoms: {symptom.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}