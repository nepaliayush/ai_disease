import { cn } from "@/lib/utils";
import { bandColor } from "@/lib/utils";

export default function RiskBadge({ level, className }) {
  const color = bandColor(level);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-semibold",
        className,
      )}
      style={{ color, borderColor: `${color}55`, background: `${color}14` }}
    >
      {level}
    </span>
  );
}