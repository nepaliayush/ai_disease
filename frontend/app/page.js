"use client";

import {
  Activity,
  ArrowRight,
  Droplet,
  FlaskConical,
  Gauge,
  HeartPulse,
  Waves,
} from "lucide-react";
import Link from "next/link";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

import { useMetadata } from "@/hooks/useMetadata";

const DISEASE_ICONS = {
  diabetes: Droplet,
  heart_disease: HeartPulse,
  liver_disease: FlaskConical,
  ckd: Waves,
};

export default function Dashboard() {
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
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Overview of the four deployed clinical ML models, how their outputs
          are fused, and where to run your own risk assessment.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {meta.diseases.map((d) => {
          const m = meta.model_metrics?.[d.id];
          const Icon = DISEASE_ICONS[d.id] || Activity;
          return (
            <Card key={d.id} className="shadow-none">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-8 w-8 items-center justify-center rounded-md border bg-muted text-muted-foreground">
                    <Icon className="h-4 w-4" />
                  </span>
                  <CardTitle className="text-base">{d.label}</CardTitle>
                </div>
                <CardDescription>
                  {meta.deployed_models?.[d.id] || "—"}
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2 text-sm">
                <div className="rounded-md border p-2">
                  <p className="text-xs text-muted-foreground">Accuracy</p>
                  <p className="font-semibold tabular-nums">
                    {m ? (m.accuracy * 100).toFixed(1) : "—"}%
                  </p>
                </div>
                <div className="rounded-md border p-2">
                  <p className="text-xs text-muted-foreground">AUC</p>
                  <p className="font-semibold tabular-nums">
                    {m ? (m.roc_auc * 100).toFixed(1) : "—"}%
                  </p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="shadow-none">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-semibold">
              How the fusion works
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted-foreground">
              <li>
                Each clinical model predicts a disease probability from lab
                values and vitals.
              </li>
              <li>
                A symptom triage model scores how well your symptoms align with
                each disease category.
              </li>
              <li>
                Fused risk = 0.7 × clinical + 0.3 × symptoms, plus a disclosed
                lifestyle adjustment and prevalence recalibration.
              </li>
              <li>Every prediction is explained with SHAP contributions.</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="shadow-none">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-semibold">
              Get started
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Enter your clinical parameters and symptoms to get a fused risk
              assessment for all four conditions, an overall health score, and
              personalized advice.
            </p>
            <div className="flex flex-wrap gap-3">
              <Button asChild>
                <Link href="/prediction" className="gap-2">
                  <Activity className="h-4 w-4" />
                  Go to Prediction
                </Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/performance" className="gap-2">
                  <Gauge className="h-4 w-4" />
                  Model Performance
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}