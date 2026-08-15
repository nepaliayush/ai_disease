"use client";

import { useEffect, useState } from "react";

import { getMetadata } from "@/lib/api";

export function useMetadata() {
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetadata()
      .then(setMeta)
      .catch((e) => setError(String(e.message || e)));
  }, []);

  return { meta, error };
}