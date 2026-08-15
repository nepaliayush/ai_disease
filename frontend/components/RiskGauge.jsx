export default function RiskGauge({
  pct,
  color,
  size = 168,
  label = "Overall risk",
}) {
  const clamped = Math.min(100, Math.max(0, pct));
  const radius = (size - 14) / 2;
  const circumference = 2 * Math.PI * radius;
  const filled = (circumference * clamped) / 100;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-90"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={10}
          className="stroke-muted"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={10}
          stroke={color}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference - filled}`}
          style={{ transition: "stroke-dasharray 0.9s ease" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-3xl font-bold tabular-nums" style={{ color }}>
          {clamped.toFixed(0)}
          <span className="text-lg font-semibold">%</span>
        </span>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
    </div>
  );
}