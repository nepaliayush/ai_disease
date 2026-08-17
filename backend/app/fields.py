"""Unified form schema.

Defines the deduplicated clinical parameter set shared across the four disease
datasets, the curated symptom checklist, and how each unified field maps to the
raw feature name used by each clinical model. Also contains the mapping from the
symptom model's disease classes onto the four target disease categories.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-dataset clinical feature columns (as cleaned in `training/prepare_datasets.py`)
# ---------------------------------------------------------------------------
DIABETES_FEATURES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree_function", "age",
    # Engineered interactions (see training/prepare_datasets.py).
    "glucose_bmi", "bmi_age", "age_preg",
]

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]

LIVER_FEATURES = [
    "age", "sex", "total_bilirubin", "direct_bilirubin", "alkaline_phosphotase",
    "alamine_aminotransferase", "aspartate_aminotransferase", "total_proteins",
    "albumin", "albumin_globulin_ratio",
    "ast_alt_ratio", "direct_bilirubin_ratio",
    # Additional engineered markers (see training/prepare_datasets.py).
    "bilirubin_total", "albumin_fraction", "alt_ast_product",
]

CKD_FEATURES = [
    "age", "blood_pressure", "specific_gravity", "urine_albumin", "urine_sugar",
    "urine_rbc", "pus_cell", "pus_cell_clumps", "bacteria", "blood_glucose_random",
    "blood_urea", "serum_creatinine", "sodium", "potassium", "hemoglobin",
    "packed_cell_volume", "wbc_count", "rbc_count", "hypertension",
    "diabetes_mellitus", "coronary_artery_disease", "appetite", "pedal_edema",
    "anemia",
]

DATASET_FEATURES = {
    "diabetes": DIABETES_FEATURES,
    "heart_disease": HEART_FEATURES,
    "liver_disease": LIVER_FEATURES,
    "ckd": CKD_FEATURES,
}

# ---------------------------------------------------------------------------
# Unified clinical fields
# ---------------------------------------------------------------------------
# section -> list of field dicts. `type` is one of "number" | "select".
# `min`/`max` are used for form validation and outlier clipping hints.
UNIFIED_FIELDS = {
    "Demographics": [
        {"name": "age", "label": "Age", "units": "years", "type": "number",
         "min": 18, "max": 100, "step": 1},
        {"name": "sex", "label": "Sex (assigned at birth)", "type": "select",
         "options": [{"value": "male", "label": "Male"},
                     {"value": "female", "label": "Female"}]},
    ],
    "Vitals & Body Measures": [
        {"name": "blood_pressure", "label": "Blood pressure", "units": "mmHg",
         "type": "number", "min": 60, "max": 250, "step": 1,
         "hint": "Used as resting blood pressure for all four models"},
        {"name": "max_heart_rate", "label": "Max heart rate achieved", "units": "bpm",
         "type": "number", "min": 60, "max": 220, "step": 1},
        {"name": "bmi", "label": "BMI", "units": "kg/m²", "type": "number",
         "min": 10, "max": 60, "step": 0.1},
    ],
    "Diabetes & Metabolic Panel": [
        {"name": "pregnancies", "label": "Number of pregnancies",
         "type": "number", "min": 0, "max": 17, "step": 1},
        {"name": "glucose", "label": "Plasma glucose", "units": "mg/dL",
         "type": "number", "min": 40, "max": 300, "step": 1,
         "hint": "2-hour oral glucose tolerance / random glucose"},
        {"name": "insulin", "label": "2-hour serum insulin", "units": "µU/mL",
         "type": "number", "min": 0, "max": 850, "step": 1},
        {"name": "skin_thickness", "label": "Triceps skin fold thickness",
         "units": "mm", "type": "number", "min": 0, "max": 100, "step": 1},
        {"name": "diabetes_pedigree_function", "label": "Diabetes pedigree function",
         "type": "number", "min": 0.0, "max": 2.5, "step": 0.01},
    ],
    "Heart Disease Panel": [
        {"name": "cholesterol", "label": "Serum cholesterol", "units": "mg/dL",
         "type": "number", "min": 100, "max": 600, "step": 1},
        {"name": "fasting_blood_sugar", "label": "Fasting blood sugar > 120 mg/dL",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "resting_ecg", "label": "Resting ECG result", "type": "select",
         "options": [{"value": "0", "label": "0 — Normal"},
                     {"value": "1", "label": "1 — ST-T wave abnormality"},
                     {"value": "2", "label": "2 — Probable/definite LV hypertrophy"}]},
        {"name": "chest_pain_type", "label": "Chest pain type", "type": "select",
         "options": [{"value": "0", "label": "0 — Typical angina"},
                     {"value": "1", "label": "1 — Atypical angina"},
                     {"value": "2", "label": "2 — Non-anginal pain"},
                     {"value": "3", "label": "3 — Asymptomatic"}]},
        {"name": "exercise_angina", "label": "Exercise-induced angina",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "oldpeak", "label": "ST depression induced by exercise",
         "units": "mm", "type": "number", "min": 0.0, "max": 7.0, "step": 0.1},
        {"name": "st_slope", "label": "Slope of peak exercise ST segment",
         "type": "select", "options": [{"value": "0", "label": "0 — Upsloping"},
                                        {"value": "1", "label": "1 — Flat"},
                                        {"value": "2", "label": "2 — Downsloping"}]},
        {"name": "major_vessels", "label": "Major vessels colored by fluoroscopy",
         "type": "select", "options": [{"value": "0", "label": "0"},
                                        {"value": "1", "label": "1"},
                                        {"value": "2", "label": "2"},
                                        {"value": "3", "label": "3"}]},
        {"name": "thalassemia", "label": "Thalassemia", "type": "select",
         "options": [{"value": "0", "label": "0 — Normal"},
                     {"value": "1", "label": "1 — Fixed defect"},
                     {"value": "2", "label": "2 — Reversible defect"},
                     {"value": "3", "label": "3 — Unknown"}]},
    ],
    "Liver Panel": [
        {"name": "total_bilirubin", "label": "Total bilirubin", "units": "mg/dL",
         "type": "number", "min": 0.0, "max": 30.0, "step": 0.1},
        {"name": "direct_bilirubin", "label": "Direct bilirubin", "units": "mg/dL",
         "type": "number", "min": 0.0, "max": 20.0, "step": 0.1},
        {"name": "alkaline_phosphatase", "label": "Alkaline phosphatase", "units": "IU/L",
         "type": "number", "min": 0, "max": 2200, "step": 1},
        {"name": "alt", "label": "Alamine aminotransferase (SGPT)", "units": "IU/L",
         "type": "number", "min": 0, "max": 2000, "step": 1},
        {"name": "ast", "label": "Aspartate aminotransferase (SGOT)", "units": "IU/L",
         "type": "number", "min": 0, "max": 5000, "step": 1},
        {"name": "total_proteins", "label": "Total proteins", "units": "g/dL",
         "type": "number", "min": 1.0, "max": 12.0, "step": 0.1},
        {"name": "serum_albumin", "label": "Serum albumin", "units": "g/dL",
         "type": "number", "min": 0.0, "max": 10.0, "step": 0.1},
        {"name": "albumin_globulin_ratio", "label": "Albumin / globulin ratio",
         "type": "number", "min": 0.0, "max": 10.0, "step": 0.1},
    ],
    "Kidney / Renal Panel": [
        {"name": "specific_gravity", "label": "Urine specific gravity",
         "type": "number", "min": 1.000, "max": 1.030, "step": 0.001},
        {"name": "urine_albumin", "label": "Urine albumin", "type": "select",
         "options": [{"value": "0", "label": "0 (none)"}, {"value": "1", "label": "1"},
                     {"value": "2", "label": "2"}, {"value": "3", "label": "3"},
                     {"value": "4", "label": "4"}, {"value": "5", "label": "5"}]},
        {"name": "urine_sugar", "label": "Urine sugar", "type": "select",
         "options": [{"value": "0", "label": "0 (none)"}, {"value": "1", "label": "1"},
                     {"value": "2", "label": "2"}, {"value": "3", "label": "3"},
                     {"value": "4", "label": "4"}, {"value": "5", "label": "5"}]},
        {"name": "urine_rbc", "label": "Red blood cells in urine", "type": "select",
         "options": [{"value": "normal", "label": "Normal"},
                     {"value": "abnormal", "label": "Abnormal"}]},
        {"name": "pus_cell", "label": "Pus cells in urine", "type": "select",
         "options": [{"value": "normal", "label": "Normal"},
                     {"value": "abnormal", "label": "Abnormal"}]},
        {"name": "pus_cell_clumps", "label": "Pus cell clumps", "type": "select",
         "options": [{"value": "notpresent", "label": "Not present"},
                     {"value": "present", "label": "Present"}]},
        {"name": "bacteria", "label": "Bacteria in urine", "type": "select",
         "options": [{"value": "notpresent", "label": "Not present"},
                     {"value": "present", "label": "Present"}]},
        {"name": "blood_urea", "label": "Blood urea", "units": "mg/dL",
         "type": "number", "min": 0, "max": 400, "step": 1},
        {"name": "serum_creatinine", "label": "Serum creatinine", "units": "mg/dL",
         "type": "number", "min": 0.0, "max": 20.0, "step": 0.1},
        {"name": "sodium", "label": "Serum sodium", "units": "mEq/L",
         "type": "number", "min": 100, "max": 180, "step": 1},
        {"name": "potassium", "label": "Serum potassium", "units": "mEq/L",
         "type": "number", "min": 1.0, "max": 9.0, "step": 0.1},
        {"name": "hemoglobin", "label": "Hemoglobin", "units": "g/dL",
         "type": "number", "min": 3.0, "max": 20.0, "step": 0.1},
        {"name": "packed_cell_volume", "label": "Packed cell volume", "units": "%",
         "type": "number", "min": 10, "max": 60, "step": 1},
        {"name": "wbc_count", "label": "White blood cell count", "units": "cells/cumm",
         "type": "number", "min": 2000, "max": 30000, "step": 100},
        {"name": "rbc_count", "label": "Red blood cell count", "units": "millions/cmm",
         "type": "number", "min": 1.0, "max": 9.0, "step": 0.1},
    ],
    "Lifestyle": [
        {"name": "smoking_status", "label": "Smoking status", "type": "select",
         "options": [{"value": "never", "label": "Never smoked"},
                     {"value": "occasional", "label": "Occasional"},
                     {"value": "daily", "label": "Daily"}]},
        {"name": "alcohol_consumption", "label": "Alcohol consumption",
         "type": "select", "options": [{"value": "none", "label": "None"},
                                       {"value": "light", "label": "Light"},
                                       {"value": "moderate", "label": "Moderate"},
                                       {"value": "heavy", "label": "Heavy"}]},
    ],
    "Comorbidities & History": [
        {"name": "has_hypertension", "label": "Hypertension history",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "has_diabetes", "label": "Diabetes mellitus history",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "has_cad", "label": "Coronary artery disease history",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "appetite", "label": "Appetite", "type": "select",
         "options": [{"value": "good", "label": "Good"},
                     {"value": "poor", "label": "Poor"}]},
        {"name": "pedal_edema", "label": "Pedal edema (leg swelling)",
         "type": "select", "options": [{"value": "no", "label": "No"},
                                        {"value": "yes", "label": "Yes"}]},
        {"name": "anemia", "label": "Anemia", "type": "select",
         "options": [{"value": "no", "label": "No"},
                     {"value": "yes", "label": "Yes"}]},
    ],
}

# Map unified form field -> per-disease model feature name.
FIELD_TO_MODEL_FEATURE = {
    "age": {"diabetes": "age", "heart_disease": "age", "liver_disease": "age", "ckd": "age"},
    "sex": {"heart_disease": "sex", "liver_disease": "sex"},
    "blood_pressure": {
        "diabetes": "blood_pressure", "heart_disease": "trestbps", "ckd": "blood_pressure",
    },
    "max_heart_rate": {"heart_disease": "thalach"},
    "bmi": {"diabetes": "bmi"},
    "pregnancies": {"diabetes": "pregnancies"},
    "glucose": {"diabetes": "glucose", "ckd": "blood_glucose_random"},
    "insulin": {"diabetes": "insulin"},
    "skin_thickness": {"diabetes": "skin_thickness"},
    "diabetes_pedigree_function": {"diabetes": "diabetes_pedigree_function"},
    "cholesterol": {"heart_disease": "chol"},
    "fasting_blood_sugar": {"heart_disease": "fbs"},
    "resting_ecg": {"heart_disease": "restecg"},
    "chest_pain_type": {"heart_disease": "cp"},
    "exercise_angina": {"heart_disease": "exang"},
    "oldpeak": {"heart_disease": "oldpeak"},
    "st_slope": {"heart_disease": "slope"},
    "major_vessels": {"heart_disease": "ca"},
    "thalassemia": {"heart_disease": "thal"},
    "total_bilirubin": {"liver_disease": "total_bilirubin"},
    "direct_bilirubin": {"liver_disease": "direct_bilirubin"},
    "alkaline_phosphatase": {"liver_disease": "alkaline_phosphotase"},
    "alt": {"liver_disease": "alamine_aminotransferase"},
    "ast": {"liver_disease": "aspartate_aminotransferase"},
    "total_proteins": {"liver_disease": "total_proteins"},
    "serum_albumin": {"liver_disease": "albumin"},
    "albumin_globulin_ratio": {"liver_disease": "albumin_globulin_ratio"},
    "specific_gravity": {"ckd": "specific_gravity"},
    "urine_albumin": {"ckd": "urine_albumin"},
    "urine_sugar": {"ckd": "urine_sugar"},
    "urine_rbc": {"ckd": "urine_rbc"},
    "pus_cell": {"ckd": "pus_cell"},
    "pus_cell_clumps": {"ckd": "pus_cell_clumps"},
    "bacteria": {"ckd": "bacteria"},
    "blood_urea": {"ckd": "blood_urea"},
    "serum_creatinine": {"ckd": "serum_creatinine"},
    "sodium": {"ckd": "sodium"},
    "potassium": {"ckd": "potassium"},
    "hemoglobin": {"ckd": "hemoglobin"},
    "packed_cell_volume": {"ckd": "packed_cell_volume"},
    "wbc_count": {"ckd": "wbc_count"},
    "rbc_count": {"ckd": "rbc_count"},
    "has_hypertension": {"ckd": "hypertension"},
    "has_diabetes": {"ckd": "diabetes_mellitus"},
    "has_cad": {"ckd": "coronary_artery_disease"},
    "appetite": {"ckd": "appetite"},
    "pedal_edema": {"ckd": "pedal_edema"},
    "anemia": {"ckd": "anemia"},
}

# Human-readable value -> numeric encoding used by the models.
ENCODERS = {
    "sex": {"male": 1, "female": 0},
    "fasting_blood_sugar": {"no": 0, "yes": 1},
    "exercise_angina": {"no": 0, "yes": 1},
    "has_hypertension": {"no": 0, "yes": 1},
    "has_diabetes": {"no": 0, "yes": 1},
    "has_cad": {"no": 0, "yes": 1},
    "pedal_edema": {"no": 0, "yes": 1},
    "anemia": {"no": 0, "yes": 1},
    "urine_rbc": {"normal": 1, "abnormal": 0},
    "pus_cell": {"normal": 1, "abnormal": 0},
    "pus_cell_clumps": {"notpresent": 1, "present": 0},
    "bacteria": {"notpresent": 1, "present": 0},
    "appetite": {"good": 1, "poor": 0},
    "smoking_status": {"never": 0, "occasional": 1, "daily": 2},
    "alcohol_consumption": {"none": 0, "light": 1, "moderate": 2, "heavy": 3},
    "resting_ecg": {"0": 0, "1": 1, "2": 2},
    "chest_pain_type": {"0": 0, "1": 1, "2": 2, "3": 3},
    "st_slope": {"0": 0, "1": 1, "2": 2},
    "major_vessels": {"0": 0, "1": 1, "2": 2, "3": 3},
    "thalassemia": {"0": 0, "1": 1, "2": 2, "3": 3},
    "urine_albumin": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
    "urine_sugar": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
}

# ---------------------------------------------------------------------------
# Symptom checklist (curated subset of the general symptom-disease vocabulary).
# Only these symptoms are collected in the unified form; all other symptoms are
# treated as absent (0) by the triage model.
# ---------------------------------------------------------------------------
SYMPTOM_CHECKLIST = [
    "fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level",
    "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating",
    "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine",
    "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "redness_of_eyes", "runny_nose",
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "dizziness", "cramps", "bruising", "obesity", "swollen_legs",
    "puffy_face_and_eyes", "swollen_extremeties", "excessive_hunger",
    "knee_pain", "muscle_weakness", "swelling_joints", "movement_stiffness",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
    "passage_of_gases", "depression", "irritability", "muscle_pain",
    "altered_sensorium", "belly_pain", "increased_appetite", "polyuria",
    "family_history", "lack_of_concentration", "visual_disturbances",
    "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption",
    "palpitations", "skin_peeling",
]

# ---------------------------------------------------------------------------
# Symptom model disease classes -> target disease categories.
# Each target disease category is a weighted set of symptom-model classes; the
# per-disease symptom relevance score is the weighted sum of the triage model's
# class probabilities over the mapped classes.
# ---------------------------------------------------------------------------
SYMPTOM_DISEASE_MAP = {
    "diabetes": {"Diabetes": 1.0, "Hypoglycemia": 0.5},
    "heart_disease": {"Heart attack": 1.0, "Hypertension": 0.5},
    "liver_disease": {
        "Alcoholic hepatitis": 1.0, "Jaundice": 0.8, "Liver Disease": 1.0,
        "Hepatitis A": 0.9, "Hepatitis B": 0.9, "Hepatitis C": 0.9,
        "Hepatitis D": 0.9, "Hepatitis E": 0.9, "Chronic cholestasis": 0.9,
    },
    "ckd": {"Chronic Kidney Disease": 1.0, "Urinary tract infection": 0.3},
}

SYMPTOM_DISCLAIMER = (
    "The symptom component is a triage-style signal from self-reported symptoms "
    "and is not a diagnosis. Clinical values dominate the fused score."
)


def canonical_label(s: str) -> str:
    """Normalize a disease/symptom label to a canonical form."""
    s = str(s).strip()
    words = []
    for word in s.split():
        if not word:
            continue
        if word == word.upper() and word.isalpha() and len(word) > 1:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


# Symptom-model classes mapped onto each target category, canonicalized so they
# match the cleaned symptom dataset labels exactly.
SYMPTOM_DISEASE_MAP_CANONICAL = {
    disease: {canonical_label(cls): w for cls, w in classes.items()}
    for disease, classes in SYMPTOM_DISEASE_MAP.items()
}


def encode_field(name: str, value):
    """Encode a unified-form field value into the model numeric representation."""
    if name in ENCODERS:
        return ENCODERS[name][value]
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported value for field '{name}': {value!r}")