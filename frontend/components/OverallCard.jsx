import { Card, CardContent } from "@/components/ui/card";
import { healthColor, healthLabel, healthScore } from "@/lib/utils";

import RiskBadge from "./RiskBadge";
import RiskGauge from "./RiskGauge";

export default function OverallCard({ overall, diseases }) {
  const score = healthScore(overall.fused_avg_pct);
  const color = healthColor(score);

  return (
    <Card className="shadow-none">
      <CardContent className="grid gap-6 p-6 md:grid-cols-[auto_1fr] md:items-center">
        <div className="flex flex-col items-center gap-2">
          <RiskGauge pct={score} color={color} label="Overall health score" />
          <span
            className="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold"
            style={{ color, borderColor: `${color}55`, background: `${color}14` }}
          >
            {healthLabel(score)} · {overall.risk_level} risk
          </span>
          <p className="max-w-56 text-center text-xs text-muted-foreground">
            {overall.note}
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <h2 className="text-base font-semibold">Per-disease assessment</h2>
            <p className="text-xs text-muted-foreground">
              Fused clinical + symptom score for each condition.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {diseases.map((d) => (
              <div key={d.disease} className="rounded-lg border p-3">
                <p className="truncate text-xs font-medium text-muted-foreground">
                  {d.label}
                </p>
                <p className="mt-1 text-lg font-bold tabular-nums">
                  {d.fused_pct.toFixed(1)}%
                </p>
                <RiskBadge level={d.risk_level} className="mt-1" />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}