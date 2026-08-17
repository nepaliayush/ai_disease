import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { prettyLabel } from "@/lib/utils";

const COLS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "AUC" },
  { key: "cv_accuracy_mean", label: "CV Acc" },
  { key: "cv_roc_auc_mean", label: "CV AUC" },
];

export default function ModelComparison({
  diseases,
  deployedModels,
  comparison,
}) {
  if (!comparison) return null;
  const labels = Object.fromEntries((diseases || []).map((d) => [d.id, d.label]));

  return (
    <Card className="shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">
          Model family comparison
        </CardTitle>
        <CardDescription>
          All five model families compared per disease. The deployed model
          (marked with ✓) is selected by cross-validated ROC-AUC; accuracy / F1 /
          AUC are held-out test metrics on the stratified 80:20 split, CV Acc /
          CV AUC are the mean over repeated 10-fold stratified cross-validation.
        </CardDescription>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="py-2 pr-4 font-semibold">Disease</th>
              <th className="py-2 pr-4 font-semibold">Model</th>
              {COLS.map((c) => (
                <th key={c.key} className="py-2 pr-4 text-right font-semibold">
                  {c.label}
                </th>
              ))}
              <th className="py-2 text-right font-semibold">Deployed</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(comparison).map(([disease, models]) => (
              <>
                {Object.entries(models).map(([name, m], i) => {
                  const isDeployed = deployedModels?.[disease] === name;
                  const cv = (k) =>
                    m[k] != null ? `${(m[k] * 100).toFixed(1)}%` : "—";
                  return (
                    <tr
                      key={`${disease}-${name}`}
                      className={
                        "border-b text-muted-foreground last:border-0 " +
                        (isDeployed ? "bg-primary/5" : "")
                      }
                    >
                      {i === 0 && (
                        <td
                          rowSpan={Object.keys(models).length}
                          className="py-2 pr-4 align-top font-medium text-foreground"
                        >
                          {labels[disease] || prettyLabel(disease)}
                        </td>
                      )}
                      <td className="py-2 pr-4 font-medium text-foreground">
                        {name}
                      </td>
                      {COLS.map((c) => (
                        <td
                          key={c.key}
                          className="py-2 pr-4 text-right tabular-nums"
                        >
                          {c.key.startsWith("cv_")
                            ? cv(c.key)
                            : `${(m[c.key] * 100).toFixed(1)}%`}
                        </td>
                      ))}
                      <td className="py-2 text-right tabular-nums">
                        {isDeployed ? "✓" : ""}
                      </td>
                    </tr>
                  );
                })}
              </>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
