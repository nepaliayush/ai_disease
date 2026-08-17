"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

import ModelMetrics from "@/components/ModelMetrics";
import { useMetadata } from "@/hooks/useMetadata";
import { prettyLabel } from "@/lib/utils";

export default function Performance() {
  const { meta, error } = useMetadata();

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Cannot reach the prediction API</AlertTitle>
        <AlertDescription>
          {error}. Start the backend with{" "}
          <code>uvicorn app.main:app --port 8001</code> (from{" "}
          <code>backend/</code>) and refresh.
        </AlertDescription>
      </Alert>
    );
  }

  if (!meta) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-1/2" />
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-56 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Model Performance</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Holdout-test and cross-validation metrics for each deployed clinical
          model, computed during training on public, anonymized datasets.
        </p>
      </header>

      <ModelMetrics
        diseases={meta.diseases}
        models={meta.deployed_models}
        metrics={meta.model_metrics}
      />

      <Card className="shadow-none">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">
            About these numbers
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            These are <strong>holdout-test metrics</strong> on small, public,
            curated datasets — not real-world accuracy. They are reported as{" "}
            <strong className="text-foreground">mean ± spread across repeated
            random splits</strong>, so a single lucky hold-out never looks like a
            perfect 100%.
          </p>
          <p>
            <strong className="text-foreground">Why CKD looks near-100%:</strong>{" "}
            the UCI CKD dataset (400 rows) is near-perfectly separable. CKD
            patients have extreme lab values — low hemoglobin, low packed cell
            volume, abnormal specific gravity, high creatinine — that leave no
            overlap with healthy controls. The ~100% reflects that curated
            dataset, not an infallible model.
          </p>
          <p>
            The <strong className="text-foreground">cross-validation</strong>{" "}
            table below is the more honest generalization signal. Even so, early
            or mild disease in real patients will be far harder than any of
            these datasets, so treat these scores as indicative, not diagnostic.
          </p>
        </CardContent>
      </Card>

      <Card className="shadow-none">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">
            Cross-validation
          </CardTitle>
          <CardDescription>
            Mean ± standard deviation over repeated 10-fold stratified
            cross-validation (multiple shuffle seeds; not a single lucky
            split).
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-4 font-semibold">Disease</th>
                <th className="py-2 pr-4 text-right font-semibold">
                  CV Accuracy
                </th>
                <th className="py-2 pr-4 text-right font-semibold">CV AUC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(meta.model_metrics).map(([disease, m]) => (
                <tr
                  key={disease}
                  className="border-b text-muted-foreground last:border-0"
                >
                  <td className="py-2 pr-4 font-medium text-foreground">
                    {meta.diseases.find((d) => d.id === disease)?.label ||
                      prettyLabel(disease)}
                  </td>
                  <td className="py-2 pr-4 text-right tabular-nums text-foreground">
                    {(m.cv_acc_mean * 100).toFixed(1)}% ±{" "}
                    {(m.cv_acc_std * 100).toFixed(1)}
                  </td>
                  <td className="py-2 text-right tabular-nums text-foreground">
                    {(m.cv_auc_mean * 100).toFixed(1)}% ±{" "}
                    {(m.cv_auc_std * 100).toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-muted-foreground">
            Metrics are the mean ± standard deviation across 10-fold stratified
            CV repeated {(meta.model_metrics && Object.values(meta.model_metrics)[0]?.cv_repeats) ?? "multiple"}{" "}
            times. Even a single repeated-CV run keeps reporting ~100% for CKD:
            that dataset is genuinely near-perfectly separable, so the number
            reflects the curated data, not an infallible model.
          </p>
        </CardContent>
      </Card>

      <Card className="shadow-none">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">
            Confusion matrices (held-out test set)
          </CardTitle>
          <CardDescription>
            Each cell shows predictions vs. actuals on the stratified 80:20
            hold-out split (row = actual, column = predicted). 0 = no disease,
            1 = disease.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          {Object.entries(meta.model_metrics).map(([disease, m]) => (
            <div key={disease}>
              <p className="mb-2 text-sm font-semibold text-foreground">
                {meta.diseases.find((d) => d.id === disease)?.label ||
                  prettyLabel(disease)}
                <span className="ml-2 font-normal text-muted-foreground">
                  ({m.confusion_matrix?.[0]?.[0] + m.confusion_matrix?.[0]?.[1] + m.confusion_matrix?.[1]?.[0] + m.confusion_matrix?.[1]?.[1]} samples)
                </span>
              </p>
              {m.confusion_matrix ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="py-1 pr-4 font-semibold" />
                      <th className="py-1 pr-4 font-semibold">Pred 0</th>
                      <th className="py-1 font-semibold">Pred 1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {m.confusion_matrix.map((row, i) => (
                      <tr key={i} className="border-b text-muted-foreground last:border-0">
                        <td className="py-1 pr-4 font-medium text-foreground">
                          Actual {i}
                        </td>
                        {row.map((cell, j) => (
                          <td key={j} className="py-1 pr-4 tabular-nums">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-xs text-muted-foreground">n/a</p>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}