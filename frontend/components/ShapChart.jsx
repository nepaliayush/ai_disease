export default function ShapChart({ shap }) {
  const entries = shap.entries || [];
  const max = Math.max(...entries.map((e) => Math.abs(e.shap)), 1e-6);
  const base = shap.base_value != null ? (shap.base_value * 100).toFixed(1) : "—";

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-1 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">
          SHAP — {shap.model_name}
        </span>
        <span>baseline risk: {base}%</span>
      </div>
      {entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">No SHAP values available.</p>
      ) : (
        <ul className="space-y-2">
          {entries.map((e) => {
            const width = (Math.abs(e.shap) / max) * 100;
            const positive = e.shap >= 0;
            return (
              <li key={e.feature} className="grid grid-cols-[140px_1fr_64px] items-center gap-3 sm:grid-cols-[180px_1fr_72px]">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{e.label}</p>
                  <p className="truncate text-xs text-muted-foreground">
                    value: {fmt(e.value)}
                  </p>
                </div>
                <div className="h-3.5 overflow-hidden rounded bg-muted">
                  <div
                    className="h-full rounded"
                    style={{
                      width: `${width}%`,
                      background: positive
                        ? "linear-gradient(90deg, #ef4444b3, #dc2626)"
                        : "linear-gradient(90deg, #22c55eb3, #16a34a)",
                    }}
                  />
                </div>
                <div
                  className={`text-right text-sm font-semibold ${
                    positive ? "text-rose-600 dark:text-rose-400" : "text-emerald-600 dark:text-emerald-400"
                  }`}
                >
                  {positive ? "+" : ""}
                  {(e.shap * 100).toFixed(1)}%
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <div className="flex gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-sm"
            style={{
              background: "linear-gradient(90deg, #ef4444b3, #dc2626)",
            }}
          />
          raises clinical risk
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-sm"
            style={{
              background: "linear-gradient(90deg, #22c55eb3, #16a34a)",
            }}
          />
          lowers clinical risk
        </span>
      </div>
    </div>
  );
}

function fmt(v) {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(2);
  return String(v);
}