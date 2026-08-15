import { Activity, Droplet, FlaskConical, HeartPulse, Waves } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";

import RiskBadge from "./RiskBadge";
import RiskBar, { BreakdownBars } from "./RiskBar";
import ShapChart from "./ShapChart";

const DISEASE_ICONS = {
  diabetes: Droplet,
  heart_disease: HeartPulse,
  liver_disease: FlaskConical,
  ckd: Waves,
};

export default function DiseaseCard({ result, index }) {
  const lifestyle = result.lifestyle_adjustment;
  const Icon = DISEASE_ICONS[result.disease] || Activity;
  const prevalence = result.prevalence;
  const corrected = Math.abs(result.clinical_pct - result.clinical_pct_raw) > 0.5;

  return (
    <Card className="shadow-none">
      <CardContent className="space-y-4 p-5">
        <header className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-md border bg-muted text-muted-foreground">
            <Icon className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="text-base font-semibold leading-tight">
              {index}. {result.label}
            </h3>
            <p className="text-xs text-muted-foreground">
              Fused clinical + symptom risk
            </p>
          </div>
          <div className="flex flex-col items-end gap-1">
            <RiskBadge level={result.risk_level} />
            <span className="text-sm font-bold tabular-nums">
              {result.fused_pct.toFixed(1)}%
            </span>
          </div>
        </header>

        <RiskBar pct={result.fused_pct} level={result.risk_level} />

        <p className="text-xs text-muted-foreground">
          fused = 0.7 × clinical ({result.clinical_pct.toFixed(1)}%) + 0.3 ×
          symptoms ({result.symptom_pct.toFixed(1)}%)
        </p>

        <section className="space-y-2 border-t pt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Contribution
          </h4>
          <BreakdownBars breakdown={result.breakdown} />
          <p className="text-xs text-muted-foreground">
            Clinical model ({result.shap.model_name}):{" "}
            {result.clinical_pct.toFixed(1)}% · Symptom triage:{" "}
            {result.symptom_pct.toFixed(1)}%
          </p>
          {corrected && prevalence && (
            <p className="text-xs text-muted-foreground">
              Prevalence correction:{" "}
              <span className="font-medium text-foreground">
                {result.clinical_pct_raw.toFixed(1)}%
              </span>{" "}
              → {result.clinical_pct.toFixed(1)}% (training prevalence{" "}
              {(prevalence.source * 100).toFixed(0)}% → population{" "}
              {(prevalence.target * 100).toFixed(0)}%)
            </p>
          )}
          {lifestyle && (
            <p className="text-xs text-muted-foreground">
              Lifestyle adjustment (smoking + alcohol):{" "}
              <span className="font-medium text-foreground">
                {lifestyle.total > 0
                  ? `+${(lifestyle.total * 100).toFixed(1)}%`
                  : "0%"}
              </span>
            </p>
          )}
        </section>

        <section className="space-y-2 border-t pt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Top conditions (symptom model)
          </h4>
          <ul className="space-y-1.5">
            {result.top_conditions.map((c) => {
              const w = Math.max(2, c.probability * 100);
              return (
                <li key={c.disease} className="flex items-center gap-2 text-sm">
                  <span className="w-1/3 truncate">{c.disease}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${w}%` }}
                    />
                  </div>
                  <span className="w-10 text-right font-medium tabular-nums">
                    {(c.probability * 100).toFixed(0)}%
                  </span>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="space-y-2 border-t pt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Clinical model explanation (SHAP)
          </h4>
          <ShapChart shap={result.shap} />
        </section>
      </CardContent>
    </Card>
  );
}