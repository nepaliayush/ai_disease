const LEVEL_ORDER = ["Low", "Elevated", "Moderate", "High", "Very High"];
const ALERT_LEVELS = ["Elevated", "Moderate", "High", "Very High"];

const PER_LEVEL = {
  "Very High": {
    severity: "high",
    title: (label) => `${label} needs prompt attention`,
    text: (label, pct) =>
      `Your ${label.toLowerCase()} risk is very high at ${pct}%. Please see your clinician soon for screening.`,
  },
  "High": {
    severity: "high",
    title: (label) => `Screen for ${label.toLowerCase()}`,
    text: (label, pct) =>
      `Your ${label.toLowerCase()} risk is ${pct}% (high). Book a check-up with your doctor to investigate.`,
  },
  "Moderate": {
    severity: "medium",
    title: (label) => `Monitor ${label.toLowerCase()}`,
    text: (label, pct) =>
      `Your ${label.toLowerCase()} risk is moderate (${pct}%). Review the contributing factors and re-check at your next visit.`,
  },
  "Elevated": {
    severity: "medium",
    title: (label) => `Watch ${label.toLowerCase()}`,
    text: (label, pct) =>
      `Your ${label.toLowerCase()} risk is slightly elevated (${pct}%). Small lifestyle changes now can help keep it from rising.`,
  },
};

const SEVERITY_WEIGHT = { high: 3, medium: 2, good: 1 };

export function buildAdvice({ overall, diseases }) {
  const tips = [];
  let id = 0;
  const push = (severity, title, text) =>
    tips.push({ id: `tip-${id++}`, severity, title, text });

  const lifestyle = diseases.find(
    (d) => d.lifestyle_adjustment && d.lifestyle_adjustment.total > 0,
  );
  if (lifestyle) {
    const pct = (lifestyle.lifestyle_adjustment.total * 100).toFixed(1);
    push(
      "high",
      "Lifestyle factors are pushing your risk up",
      `Smoking and alcohol use add about +${pct}% to your fused scores. Cutting back on both is one of the most effective changes you can make.`,
    );
  }

  const sorted = [...diseases].sort(
    (a, b) => LEVEL_ORDER.indexOf(b.risk_level) - LEVEL_ORDER.indexOf(a.risk_level),
  );

  const elevated = sorted.filter((d) => ALERT_LEVELS.includes(d.risk_level));
  if (elevated.length) {
    for (const d of elevated) {
      const t = PER_LEVEL[d.risk_level];
      push(t.severity, t.title(d.label), t.text(d.label, d.fused_pct.toFixed(1)));
    }
  } else {
    const lowest = sorted[0];
    push(
      "good",
      "All disease risks are low",
      `Your highest fused score is ${lowest.fused_pct.toFixed(1)}% (${lowest.label}). No specific screening is indicated right now.`,
    );
  }

  const topSymptom = diseases
    .flatMap((d) => d.top_conditions || [])
    .filter((c) => c.probability >= 0.5)
    .sort((a, b) => b.probability - a.probability)[0];
  if (topSymptom) {
    push(
      "medium",
      `Symptoms point toward ${topSymptom.disease}`,
      `The symptom triage model most strongly relates your selection to ${topSymptom.disease} (${(topSymptom.probability * 100).toFixed(0)}%). Mention this when you talk to your doctor.`,
    );
  }

  return tips.sort(
    (a, b) => SEVERITY_WEIGHT[b.severity] - SEVERITY_WEIGHT[a.severity],
  );
}