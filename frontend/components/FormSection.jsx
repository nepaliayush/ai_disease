import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import Field from "./Field";

export default function FormSection({ index, title, fields, values, onChange, errors }) {
  return (
    <Card className="shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">
          {index}. {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {fields.map((f) => (
            <Field
              key={f.name}
              field={f}
              value={values[f.name] ?? ""}
              onChange={onChange}
              error={errors[f.name]}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}