"use client";

import { ArrowLeft, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";

import DiseaseCard from "@/components/DiseaseCard";
import FormSection from "@/components/FormSection";
import HealthAdvice from "@/components/HealthAdvice";
import OverallCard from "@/components/OverallCard";
import SymptomChecklist from "@/components/SymptomChecklist";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";

import { useMetadata } from "@/hooks/useMetadata";
import { predict } from "@/lib/api";

export default function Prediction() {
  const { meta, error: metaError } = useMetadata();
  const [values, setValues] = useState({});
  const [symptoms, setSymptoms] = useState(new Set());
  const [errors, setErrors] = useState({});
  const [results, setResults] = useState(null);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const requiredFields = useMemo(() => {
    if (!meta) return [];
    return meta.sections.flatMap((s) => s.fields.map((f) => f));
  }, [meta]);

  function handleChange(name, value) {
    setValues((v) => ({ ...v, [name]: value }));
    setErrors((e) => {
      const next = { ...e };
      delete next[name];
      return next;
    });
  }

  function toggleSymptom(s) {
    setSymptoms((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  }

  function validate() {
    const errs = {};
    for (const f of requiredFields) {
      const v = values[f.name];
      if (v === undefined || v === null || v === "") {
        errs[f.name] = "Required";
        continue;
      }
      if (f.type === "number") {
        const n = Number(v);
        if (Number.isNaN(n)) errs[f.name] = "Must be a number";
        else if (f.min != null && n < f.min) errs[f.name] = `Min ${f.min}`;
        else if (f.max != null && n > f.max) errs[f.name] = `Max ${f.max}`;
      }
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitError(null);
    if (!validate()) {
      document.querySelector(".form-wrap")?.scrollIntoView({ behavior: "smooth" });
      return;
    }
    const payload = {
      clinical: Object.fromEntries(
        requiredFields.map((f) => [f.name, values[f.name]]),
      ),
      symptoms: Array.from(symptoms),
    };
    setLoading(true);
    try {
      const res = await predict(payload);
      setResults(res);
      setResultsOpen(true);
    } catch (err) {
      setSubmitError(String(err.message || err));
    } finally {
      setLoading(false);
    }
  }

  if (metaError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Cannot reach the prediction API</AlertTitle>
        <AlertDescription>
          {metaError}. Start the backend with{" "}
          <code>uvicorn app.main:app --port 8001</code> (from{" "}
          <code>backend/</code>) and refresh.
        </AlertDescription>
      </Alert>
    );
  }

  if (!meta) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-2/3" />
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Prediction</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Fuses four clinical ML models (diabetes, heart disease, liver disease,
          chronic kidney disease) with a symptom-based triage signal. Fused risk
          = 0.7 × clinical + 0.3 × symptoms, plus a disclosed lifestyle
          adjustment. All fields are required.
        </p>
      </header>

      <form className="form-wrap space-y-6" onSubmit={handleSubmit} noValidate>
        {meta.sections.map((section, i) => (
          <FormSection
            key={section.title}
            index={i + 1}
            title={section.title}
            fields={section.fields}
            values={values}
            onChange={handleChange}
            errors={errors}
          />
        ))}

        <Card className="shadow-none">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-semibold">
              {meta.sections.length + 1}. Symptom checklist
            </CardTitle>
            <CardDescription>
              Select any symptoms you are currently experiencing (optional, but
              improves the triage signal).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SymptomChecklist
              symptoms={meta.symptoms}
              selected={symptoms}
              onToggle={toggleSymptom}
            />
          </CardContent>
        </Card>

        <div className="flex justify-center pt-2">
          <Button
            type="submit"
            size="lg"
            disabled={loading}
            className="min-w-64 dark:shadow-[0_0_24px_-4px_hsl(var(--primary))]"
          >
            {loading && <Loader2 className="animate-spin" />}
            {loading ? "Assessing risk…" : "Run risk assessment"}
          </Button>
        </div>
      </form>

      {submitError && (
        <Alert variant="destructive">
          <AlertTitle>Prediction failed</AlertTitle>
          <AlertDescription>{submitError}</AlertDescription>
        </Alert>
      )}

      <Dialog open={resultsOpen && !!results} onOpenChange={setResultsOpen}>
        <DialogContent className="p-0">
          <div className="space-y-6 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <DialogTitle className="text-xl">
                  Risk Assessment Results
                </DialogTitle>
                <p className="mt-1 text-sm text-muted-foreground">
                  Fused 0.7 × clinical + 0.3 × symptom scores with a disclosed
                  lifestyle adjustment.
                </p>
              </div>
              <DialogClose asChild>
                <Button variant="ghost" size="sm" className="shrink-0 gap-2">
                  <ArrowLeft className="h-4 w-4" />
                  Back to form
                </Button>
              </DialogClose>
            </div>

            {results && (
              <>
                <OverallCard
                  overall={results.overall}
                  diseases={results.diseases}
                />
                <HealthAdvice results={results} />
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  {results.diseases.map((d, i) => (
                    <DiseaseCard key={d.disease} result={d} index={i + 1} />
                  ))}
                </div>

                <Card className="shadow-none">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base font-semibold">
                      Methodology
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted-foreground">
                      <li>
                        <strong>Formula:</strong>{" "}
                        {results.methodology.fusion_formula}
                      </li>
                      <li>
                        <strong>Weight rationale:</strong>{" "}
                        {results.methodology.weight_justification}
                      </li>
                      <li>
                        <strong>Overall assessment:</strong>{" "}
                        {results.methodology.overall_assessment}
                      </li>
                      <li>
                        <strong>Lifestyle adjustment:</strong>{" "}
                        {results.methodology.lifestyle_adjustment}
                      </li>
                      <li>
                        <strong>Prevalence recalibration:</strong>{" "}
                        {results.methodology.prevalence_recalibration}
                      </li>
                      <li>
                        <strong>Deployed clinical models:</strong>{" "}
                        {Object.entries(results.methodology.clinical_models)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(" · ")}
                      </li>
                      <li>
                        <strong>SHAP method:</strong>{" "}
                        {results.methodology.shap_method}
                      </li>
                    </ul>
                  </CardContent>
                </Card>

                <Card className="shadow-none">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base font-semibold">
                      Important
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      {results.disclaimer}
                    </p>
                  </CardContent>
                </Card>
              </>
            )}
          </div>

          <DialogFooter className="sticky bottom-0 border-t bg-background px-6 py-4">
            <DialogClose asChild>
              <Button variant="outline" className="gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back to form
              </Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}