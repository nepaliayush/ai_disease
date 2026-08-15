import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { buildAdvice } from "@/lib/advice";

const SEVERITY_STYLE = {
  high: { color: "#dc2626", bg: "#ef444414", border: "#ef444455" },
  medium: { color: "#b45309", bg: "#f59e0b14", border: "#f59e0b55" },
  good: { color: "#16a34a", bg: "#22c55e14", border: "#22c55e55" },
};

export default function HealthAdvice({ results }) {
  const tips = buildAdvice(results);
  if (!tips.length) return null;

  return (
    <Card className="shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">
          Personalized health advice
        </CardTitle>
        <CardDescription>
          Tailored to your fused risk scores, symptoms, and lifestyle factors.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {tips.map((t) => {
            const s = SEVERITY_STYLE[t.severity] || SEVERITY_STYLE.medium;
            return (
              <li key={t.id} className="flex items-start gap-3 rounded-lg border p-3">
                <span
                  className="mt-0.5 inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 text-xs font-semibold uppercase"
                  style={{ color: s.color, borderColor: s.border, background: s.bg }}
                >
                  {t.severity}
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-medium leading-snug">{t.title}</p>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {t.text}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}