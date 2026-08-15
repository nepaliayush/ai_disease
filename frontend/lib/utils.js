import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const BAND_COLORS = {
  "Low": "#22c55e",
  "Elevated": "#eab308",
  "Moderate": "#f97316",
  "High": "#ef4444",
  "Very High": "#dc2626",
};

export function bandColor(level) {
  return BAND_COLORS[level] || "#64748b";
}

export function pctColor(pct) {
  if (pct >= 80) return BAND_COLORS["Very High"];
  if (pct >= 60) return BAND_COLORS["High"];
  if (pct >= 40) return BAND_COLORS["Moderate"];
  if (pct >= 20) return BAND_COLORS["Elevated"];
  return BAND_COLORS["Low"];
}

export function prettyLabel(s) {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Health score (100 - fused risk) helpers
export function healthScore(riskPct) {
  return Math.round(100 - Math.min(100, Math.max(0, riskPct)));
}

export function healthColor(score) {
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#eab308";
  if (score >= 40) return "#f97316";
  if (score >= 20) return "#ef4444";
  return "#dc2626";
}

export function healthLabel(score) {
  if (score >= 80) return "Good";
  if (score >= 60) return "Fair";
  if (score >= 40) return "At risk";
  return "Needs attention";
}