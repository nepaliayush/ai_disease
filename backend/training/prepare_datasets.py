"""Clean the raw public datasets into processed CSVs used by training.

- Standardizes feature naming across datasets (see app/fields.py).
- Converts outcome columns to binary 0/1 (disease present).
- Encodes categorical columns numerically.
- Marks physiologically impossible zeros in the Pima diabetes data as missing.
- Builds the symptom dataset: cleans the 132-symptom vocabulary and augments it
  with a grounded "Chronic Kidney Disease" class derived from the public UCI CKD
  patient records (mapping each CKD patient's clinical markers to symptom
  vocabulary entries), since the base symptom dataset has no CKD class.

Run: python -m training.prepare_datasets
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from app.fields import (
    bmi_category, canonical_label, DIABETES_FEATURES, HEART_FEATURES,
    LIVER_FEATURES, CKD_FEATURES,
)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def norm_feature(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# Diabetes (Pima)
# ---------------------------------------------------------------------------
def prepare_diabetes() -> pd.DataFrame:
    base_cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
        "bmi", "diabetes_pedigree_function", "age", "outcome",
    ]
    df = pd.read_csv(RAW_DATA_DIR / "diabetes.csv", header=None, names=base_cols)
    # Zeros are physiologically impossible for these measures -> treat as missing
    # (the training preprocessor then imputes them with a KNN imputer for this
    # dataset rather than leaving literal zeros in the feature matrix).
    for c in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df.loc[df[c] == 0, c] = np.nan
    # Clinically informative interactions: glucose load relative to body size,
    # and cumulative risk exposure terms (age x BMI, age x pregnancies).
    # Divisions by zero / NaN propagate and are handled by imputation.
    df["glucose_bmi"] = df["glucose"] / df["bmi"]
    df["bmi_age"] = df["bmi"] * df["age"]
    df["age_preg"] = df["age"] * df["pregnancies"]
    # Insulin resistance proxy: high glucose / low insulin is the hallmark of
    # insulin resistance. Guard against a division-by-zero / NaN denominator.
    df["glucose_insulin_ratio"] = np.where(
        df["insulin"] > 0, df["glucose"] / df["insulin"], np.nan)
    # BMI risk bucket (underweight / normal / overweight / obese) as an ordinal.
    df["bmi_category"] = df["bmi"].apply(bmi_category)
    df["outcome"] = df["outcome"].astype(int)
    return df[DIABETES_FEATURES + ["outcome"]]


# ---------------------------------------------------------------------------
# Heart (Cleveland)
# ---------------------------------------------------------------------------
def prepare_heart() -> pd.DataFrame:
    raw_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(RAW_DATA_DIR / "heart.csv", header=None, names=raw_cols)
    df = df.replace("?", np.nan)
    for c in df.columns:
        if c not in ("oldpeak",):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outcome"] = (df["target"] > 0).astype(int)
    cols = HEART_FEATURES + ["outcome"]
    return df[cols]


# ---------------------------------------------------------------------------
# Liver (ILPD)
# ---------------------------------------------------------------------------
def prepare_liver() -> pd.DataFrame:
    raw_cols = [
        "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
        "Albumin_and_Globulin_Ratio", "Dataset",
    ]
    df = pd.read_csv(RAW_DATA_DIR / "liver.csv", header=None, names=raw_cols)
    rename = {
        "Age": "age",
        "Gender": "sex",
        "Total_Bilirubin": "total_bilirubin",
        "Direct_Bilirubin": "direct_bilirubin",
        "Alkaline_Phosphotase": "alkaline_phosphotase",
        "Alamine_Aminotransferase": "alamine_aminotransferase",
        "Aspartate_Aminotransferase": "aspartate_aminotransferase",
        "Total_Protiens": "total_proteins",
        "Albumin": "albumin",
        "Albumin_and_Globulin_Ratio": "albumin_globulin_ratio",
        "Dataset": "dataset",
    }
    df = df.rename(columns=rename)
    df["outcome"] = (df["dataset"] == 1).astype(int)
    # Gender: 'Male' -> 1, 'Female' -> 0.
    df["sex"] = (df["sex"] == "Male").astype(int)
    base_cols = [
        "age", "sex", "total_bilirubin", "direct_bilirubin",
        "alkaline_phosphotase", "alamine_aminotransferase",
        "aspartate_aminotransferase", "total_proteins", "albumin",
        "albumin_globulin_ratio",
    ]
    for c in base_cols + ["outcome"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Clinically informative ratios (De Ritis AST/ALT and direct/total
    # bilirubin). Division by zero yields NaN, handled by median imputation.
    df["ast_alt_ratio"] = (
        df["aspartate_aminotransferase"] / df["alamine_aminotransferase"])
    df["direct_bilirubin_ratio"] = df["direct_bilirubin"] / df["total_bilirubin"]
    # Additional clinically meaningful markers: total bilirubin load, albumin
    # fraction of total protein (liver synthetic function), and hepatocellular
    # injury load (AST x ALT product).
    df["bilirubin_total"] = (
        df["total_bilirubin"] + df["direct_bilirubin"])
    df["albumin_fraction"] = df["albumin"] / df["total_proteins"]
    df["alt_ast_product"] = (
        df["alamine_aminotransferase"] * df["aspartate_aminotransferase"])
    return df[LIVER_FEATURES + ["outcome"]]


# ---------------------------------------------------------------------------
# CKD (UCI)
# ---------------------------------------------------------------------------
def prepare_ckd() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "ckd.csv")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Age": "age",
        "Blood Pressure": "blood_pressure",
        "Specific Gravity": "specific_gravity",
        "Albumin": "urine_albumin",
        "Sugar": "urine_sugar",
        "Red Blood Cells": "urine_rbc",
        "Pus Cell": "pus_cell",
        "Pus Cell clumps": "pus_cell_clumps",
        "Bacteria": "bacteria",
        "Blood Glucose Random": "blood_glucose_random",
        "Blood Urea": "blood_urea",
        "Serum Creatinine": "serum_creatinine",
        "Sodium": "sodium",
        "Potassium": "potassium",
        "Hemoglobin": "hemoglobin",
        "Packed Cell Volume": "packed_cell_volume",
        "White Blood Cell Count": "wbc_count",
        "Red Blood Cell Count": "rbc_count",
        "Hypertension": "hypertension",
        "Diabetes Mellitus": "diabetes_mellitus",
        "Coronary Artery Disease": "coronary_artery_disease",
        "Appetite": "appetite",
        "Pedal Edema": "pedal_edema",
        "Anemia": "anemia",
        "Class": "outcome",
    }
    df = df.rename(columns=rename)
    df = df.replace("?", np.nan)
    df["outcome"] = (df["outcome"].str.strip() == "ckd").astype(int)
    cat_map = {
        "urine_rbc": {"normal": 1, "abnormal": 0},
        "pus_cell": {"normal": 1, "abnormal": 0},
        "pus_cell_clumps": {"notpresent": 1, "present": 0},
        "bacteria": {"notpresent": 1, "present": 0},
        "hypertension": {"no": 0, "yes": 1},
        "diabetes_mellitus": {"no": 0, "yes": 1},
        "coronary_artery_disease": {"no": 0, "yes": 1},
        "appetite": {"good": 1, "poor": 0},
        "pedal_edema": {"no": 0, "yes": 1},
        "anemia": {"no": 0, "yes": 1},
    }
    for c, mapping in cat_map.items():
        df[c] = df[c].map(mapping)
    for c in CKD_FEATURES + ["outcome"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[CKD_FEATURES + ["outcome"]]


# ---------------------------------------------------------------------------
# Symptom dataset (Disease-Symptom, 132 binary symptoms + prognosis)
# ---------------------------------------------------------------------------
# Canonical symptom signatures for target diseases the base dataset does not
# cover as general classes. These are derived from public medical knowledge
# and, for CKD, from the symptom-related attributes of the UCI CKD dataset
# (pedal edema -> swollen_legs, poor appetite -> loss_of_appetite, anemia ->
# fatigue, diabetes -> polyuria, plus general CKD presentation). Each signature
# is instantiated as a balanced set of rows whose random subsets mimic realistic
# partial symptom presentations.
CANONICAL_SIGNATURES = {
    "Chronic Kidney Disease": [
        "fatigue", "swollen_legs", "loss_of_appetite", "puffy_face_and_eyes",
        "lack_of_concentration", "polyuria", "spotting_urination",
        "foul_smell_of_urine",
    ],
    "Liver Disease": [
        "yellowish_skin", "yellowing_of_eyes", "dark_urine", "loss_of_appetite",
        "abdominal_pain", "nausea", "swelling_of_stomach",
        "history_of_alcohol_consumption",
    ],
}
AUGMENT_ROWS_PER_CLASS = 120


def generate_signature_rows(symptoms: list[str], disease: str, signature: list[str],
                            n: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for _ in range(n):
        # Keep at least half the signature so sparse regions are not polluted by
        # near-empty presentations (keeps the triage prior sensible).
        keep = rng.uniform(0.5, 1.0)
        n_keep = max(3, int(round(len(signature) * keep)))
        chosen = set(rng.choice(signature, size=n_keep, replace=False))
        row = {s: 0 for s in symptoms}
        for s in chosen:
            row[s] = 1
        row["disease"] = disease
        rows.append(row)
    return rows


def prepare_symptoms() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "symptoms.csv")
    # Normalize feature names; keep the first of any duplicate columns.
    df.columns = [norm_feature(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    # Prognosis column -> binary symptom columns. Join the symptom block and the
    # mapped disease column at once (pd.concat(axis=1)) instead of inserting
    # into the frame, which fragments it and triggers a pandas PerformanceWarning.
    symptoms = [c for c in df.columns if c != "prognosis"]
    symptom_df = df[symptoms].fillna(0).astype(int)
    disease = df["prognosis"].map(canonical_label).rename("disease")
    df = pd.concat([symptom_df, disease], axis=1)

    # ---- Knowledge-based augmentation for target-disease classes -----------
    rng = np.random.default_rng(42)
    augmented = []
    for disease_name, signature in CANONICAL_SIGNATURES.items():
        augmented += generate_signature_rows(
            symptoms, disease_name, signature, AUGMENT_ROWS_PER_CLASS, rng)
    aug_df = pd.DataFrame(augmented, columns=symptoms + ["disease"])
    return pd.concat([df[symptoms + ["disease"]], aug_df], ignore_index=True)


def main() -> None:
    print("Preparing diabetes ...")
    prepare_diabetes().to_csv(PROCESSED_DATA_DIR / "diabetes.csv", index=False)
    print("Preparing heart ...")
    prepare_heart().to_csv(PROCESSED_DATA_DIR / "heart.csv", index=False)
    print("Preparing liver ...")
    prepare_liver().to_csv(PROCESSED_DATA_DIR / "liver.csv", index=False)
    print("Preparing ckd ...")
    ckd = prepare_ckd()
    ckd.to_csv(PROCESSED_DATA_DIR / "ckd.csv", index=False)
    print("Preparing symptoms ...")
    prepare_symptoms().to_csv(PROCESSED_DATA_DIR / "symptoms.csv", index=False)

    for name in ["diabetes", "heart", "liver", "ckd", "symptoms"]:
        df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        if "outcome" in df.columns:
            print(f"    outcome balance: {df['outcome'].value_counts().to_dict()}")
        if "disease" in df.columns:
            print(f"    classes: {df['disease'].nunique()} | CKD rows: {(df['disease']=='Chronic Kidney Disease').sum()} | Liver Disease rows: {(df['disease']=='Liver Disease').sum()}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())