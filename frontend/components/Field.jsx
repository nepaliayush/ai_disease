import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function Field({ field, value, onChange, error }) {
  const id = `field-${field.name}`;
  const numberVal = value === "" ? "" : value;

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>
        {field.label}
        {field.units && (
          <span className="text-muted-foreground"> ({field.units})</span>
        )}
      </Label>
      {field.type === "select" ? (
        <Select
          value={value || undefined}
          onValueChange={(v) => onChange(field.name, v)}
        >
          <SelectTrigger id={id} className={error ? "border-destructive" : ""}>
            <SelectValue placeholder="Select…" />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input
          id={id}
          type="number"
          value={numberVal}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(e) => onChange(field.name, e.target.value)}
          className={error ? "border-destructive" : ""}
          placeholder={`${field.min} – ${field.max}`}
        />
      )}
      {field.hint && <p className="text-xs text-muted-foreground">{field.hint}</p>}
      {error && <p className="text-xs font-medium text-destructive">{error}</p>}
    </div>
  );
}