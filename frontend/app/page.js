"use client";

import {
  Activity,
  Droplet,
  FlaskConical,
  Gauge,
  HeartPulse,
  Waves,
} from "lucide-react";
import Link from "next/link";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
        <h1 className="text-2xl font-bold tracking-tight">
          An Intelligent e-Healthcare Web Application for Disease Prediction
          Using Machine Learning
        </h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          A multi-model fusion platform that assesses the risk of diabetes,
          heart disease, liver disease, and chronic kidney disease from clinical
          data and symptoms.
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {meta.diseases.map((d) => {
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
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Risk assessment for {d.label.toLowerCase()}.
                </p>
              </CardContent>
            </Card>
          );
        })}
      </section>

      <div className="flex flex-wrap gap-3">
        <Button asChild>
          <Link href="/prediction" className="gap-2">
            <Activity className="h-4 w-4" />
            Run Prediction
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/performance" className="gap-2">
            <Gauge className="h-4 w-4" />
            Model Performance
          </Link>
        </Button>
      </div>
    </div>
  );
}
