import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { prettyLabel } from "@/lib/utils";

const METRIC_COLS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "balanced_accuracy", label: "Balanced Acc." },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1 Score" },
  { key: "roc_auc", label: "AUC" },
  { key: "pr_auc", label: "PR-AUC" },
];

export default function ModelMetrics({ diseases, models, metrics }) {
  if (!metrics) return null;
  const labels = Object.fromEntries((diseases || []).map((d) => [d.id, d.label]));

  const fmt = (m, key) => {
    const v = m[key] * 100;
    const std = m.holdout_std?.[key] * 100;
    return std != null ? `${v.toFixed(1)} ± ${std.toFixed(1)}%` : `${v.toFixed(1)}%`;
  };
  const pct = (v) => (v != null ? `${(v * 100).toFixed(1)}%` : "—");

  return (
    <Card className="shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">
          Model performance
        </CardTitle>
        <CardDescription>
          Holdout-test metrics (mean ± spread over {metrics ? "repeated random splits" : ""}) for
          each deployed clinical model. These are small public datasets with
          low majority-class baselines (diabetes 65%, heart 54%, liver 71%, CKD
          63%), so the accuracy shown is close to the achievable ceiling — and
          for the imbalanced liver set, accuracy can never rise far above its
          71% base rate. The "Overfit?" column shows the training-vs-holdout
          generalization gap; a small gap means the scores are real, not
          memorized.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="py-2 pr-4 font-semibold">Disease</th>
              <th className="py-2 pr-4 font-semibold">Model</th>
              {METRIC_COLS.map((c) => (
                <th key={c.key} className="py-2 pr-4 text-right font-semibold">
                  {c.label}
                </th>
              ))}
              <th className="py-2 text-right font-semibold">Baseline Acc.</th>
              <th className="py-2 text-right font-semibold">Overfit?</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([disease, m]) => (
              <tr
                key={disease}
                className="border-b text-muted-foreground last:border-0"
              >
                <td className="py-2 pr-4 font-medium text-foreground">
                  {labels[disease] || prettyLabel(disease)}
                </td>
                <td className="py-2 pr-4">{models?.[disease] || "—"}</td>
                {METRIC_COLS.map((c) => (
                  <td
                    key={c.key}
                    className="py-2 text-right tabular-nums text-foreground"
                  >
                    {fmt(m, c.key)}
                  </td>
                ))}
                <td className="py-2 text-right tabular-nums">
                  {pct(m.baseline_accuracy)}
                </td>
                <td className="py-2 text-right tabular-nums">
                  {m.generalization_gap != null ? (
                    m.generalization_gap < 0.05 ? (
                      <span className="text-emerald-600">
                        No ({pct(m.generalization_gap)})
                      </span>
                    ) : (
                      <span className="text-amber-600">
                        Yes ({pct(m.generalization_gap)})
                      </span>
                    )
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}