"use client";

import { ArrowLeft, Loader2, UserRound, ShieldAlert, HeartPulse } from "lucide-react";
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

const SAMPLE_HEALTHY = {
  age: "28", sex: "female", blood_pressure: "118", max_heart_rate: "165", bmi: "22.5",
  pregnancies: "0", glucose: "85", insulin: "80", skin_thickness: "22", diabetes_pedigree_function: "0.25",
  cholesterol: "180", fasting_blood_sugar: "no", resting_ecg: "0", chest_pain_type: "0",
  exercise_angina: "no", oldpeak: "0.0", st_slope: "0", major_vessels: "0", thalassemia: "0",
  total_bilirubin: "0.8", direct_bilirubin: "0.2", alkaline_phosphatase: "70", alt: "25", ast: "22",
  total_proteins: "7.0", serum_albumin: "4.2", albumin_globulin_ratio: "1.5",
  specific_gravity: "1.020", urine_albumin: "0", urine_sugar: "0",
  urine_rbc: "normal", pus_cell: "normal", pus_cell_clumps: "notpresent", bacteria: "notpresent",
  blood_urea: "15", serum_creatinine: "0.8", sodium: "140", potassium: "4.2",
  hemoglobin: "13.5", packed_cell_volume: "42", wbc_count: "7000", rbc_count: "5.0",
  smoking_status: "never", alcohol_consumption: "none",
  has_hypertension: "no", has_diabetes: "no", has_cad: "no",
  appetite: "good", pedal_edema: "no", anemia: "no",
  _symptoms: [],
};

const SAMPLE_MODERATE = {
  age: "42", sex: "female", blood_pressure: "135", max_heart_rate: "145", bmi: "29.5",
  pregnancies: "2", glucose: "140", insulin: "160", skin_thickness: "30", diabetes_pedigree_function: "0.45",
  cholesterol: "230", fasting_blood_sugar: "no", resting_ecg: "1", chest_pain_type: "2",
  exercise_angina: "no", oldpeak: "1.0", st_slope: "1", major_vessels: "0", thalassemia: "0",
  total_bilirubin: "1.2", direct_bilirubin: "0.3", alkaline_phosphatase: "95", alt: "38", ast: "30",
  total_proteins: "6.8", serum_albumin: "3.8", albumin_globulin_ratio: "1.2",
  specific_gravity: "1.015", urine_albumin: "1", urine_sugar: "0",
  urine_rbc: "normal", pus_cell: "normal", pus_cell_clumps: "notpresent", bacteria: "notpresent",
  blood_urea: "25", serum_creatinine: "1.0", sodium: "139", potassium: "4.5",
  hemoglobin: "12.5", packed_cell_volume: "38", wbc_count: "8500", rbc_count: "4.5",
  smoking_status: "occasional", alcohol_consumption: "moderate",
  has_hypertension: "no", has_diabetes: "no", has_cad: "no",
  appetite: "good", pedal_edema: "no", anemia: "no",
  _symptoms: ["fatigue", "headache"],
};

const SAMPLE_HIGH_RISK = {
  age: "55", sex: "male", blood_pressure: "160", max_heart_rate: "120", bmi: "34.2",
  pregnancies: "0", glucose: "195", insulin: "300", skin_thickness: "38", diabetes_pedigree_function: "1.2",
  cholesterol: "280", fasting_blood_sugar: "yes", resting_ecg: "2", chest_pain_type: "3",
  exercise_angina: "yes", oldpeak: "3.5", st_slope: "2", major_vessels: "2", thalassemia: "2",
  total_bilirubin: "2.5", direct_bilirubin: "1.0", alkaline_phosphatase: "180", alt: "95", ast: "80",
  total_proteins: "6.2", serum_albumin: "3.0", albumin_globulin_ratio: "0.8",
  specific_gravity: "1.010", urine_albumin: "3", urine_sugar: "2",
  urine_rbc: "abnormal", pus_cell: "abnormal", pus_cell_clumps: "present", bacteria: "present",
  blood_urea: "55", serum_creatinine: "2.5", sodium: "135", potassium: "5.5",
  hemoglobin: "10.0", packed_cell_volume: "32", wbc_count: "12000", rbc_count: "3.8",
  smoking_status: "daily", alcohol_consumption: "heavy",
  has_hypertension: "yes", has_diabetes: "yes", has_cad: "yes",
  appetite: "poor", pedal_edema: "yes", anemia: "yes",
  _symptoms: ["fatigue", "breathlessness", "chest_pain", "dizziness", "obesity",
    "excessive_hunger", "polyuria", "family_history", "blurred_and_distorted_vision"],
};

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

  function loadSample(profile) {
    const samples = { healthy: SAMPLE_HEALTHY, moderate: SAMPLE_MODERATE, high_risk: SAMPLE_HIGH_RISK };
    const s = samples[profile];
    const { _symptoms, ...fieldValues } = s;
    setValues(fieldValues);
    setSymptoms(new Set(_symptoms));
    setErrors({});
    setResults(null);
    setSubmitError(null);
    document.querySelector(".form-wrap")?.scrollIntoView({ behavior: "smooth" });
  }

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
        {/* <p className="max-w-3xl text-sm text-muted-foreground">
          Fuses four clinical ML models (diabetes, heart disease, liver disease,
          chronic kidney disease) with a symptom-based triage signal. Fused risk
          = 0.7 × clinical + 0.3 × symptoms, plus a disclosed lifestyle
          adjustment. All fields are required.
        </p> */}
      </header>

      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm font-medium text-muted-foreground">Load sample:</span>
        <Button variant="outline" size="sm" className="gap-2" onClick={() => loadSample("healthy")}>
          <UserRound className="h-4 w-4" />
          Patient 1
        </Button>
        <Button variant="outline" size="sm" className="gap-2" onClick={() => loadSample("moderate")}>
          <UserRound className="h-4 w-4" />
          Patient 2
        </Button>
        <Button variant="outline" size="sm" className="gap-2" onClick={() => loadSample("high_risk")}>
          <UserRound className="h-4 w-4" />
          Patient 3
        </Button>
      </div>

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
                {/* <p className="mt-1 text-sm text-muted-foreground">
                  Fused 0.7 × clinical + 0.3 × symptom scores with a disclosed
                  lifestyle adjustment.
                </p> */}
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

                {/* <Card className="shadow-none">
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
                </Card> */}
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