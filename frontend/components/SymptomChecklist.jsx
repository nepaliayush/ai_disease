import { Checkbox } from "@/components/ui/checkbox";
import { prettyLabel } from "@/lib/utils";

export default function SymptomChecklist({ symptoms, selected, onToggle }) {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4">
      {symptoms.map((s) => {
        const active = selected.has(s);
        return (
          <label
            key={s}
            className={`flex cursor-pointer items-center gap-2 rounded-md border p-2.5 text-sm transition-colors ${
              active
                ? "border-primary bg-accent"
                : "hover:bg-accent/50"
            }`}
          >
            <Checkbox
              checked={active}
              onCheckedChange={() => onToggle(s)}
              aria-label={prettyLabel(s)}
            />
            <span className="truncate">{prettyLabel(s)}</span>
          </label>
        );
      })}
    </div>
  );
}