"""API tests using FastAPI TestClient (no live server needed)."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app
from app.fusion import prevalence_recalibrate

client = TestClient(app)


def realistic_payload() -> dict:
    return {
        "clinical": {
            "age": 62, "sex": "male",
            "blood_pressure": 140, "max_heart_rate": 172, "bmi": 29.1,
            "pregnancies": 2, "glucose": 158, "insulin": 175,
            "skin_thickness": 32, "diabetes_pedigree_function": 0.61,
            "cholesterol": 260, "fasting_blood_sugar": "yes",
            "resting_ecg": "1", "chest_pain_type": "1",
            "exercise_angina": "yes", "oldpeak": 2.1, "st_slope": "1",
            "major_vessels": "2", "thalassemia": "1",
            "total_bilirubin": 1.2, "direct_bilirubin": 0.4,
            "alkaline_phosphatase": 120, "alt": 60, "ast": 45,
            "total_proteins": 6.8, "serum_albumin": 3.4,
            "albumin_globulin_ratio": 1.0,
            "specific_gravity": 1.020, "urine_albumin": "2",
            "urine_sugar": "1", "urine_rbc": "abnormal",
            "pus_cell": "abnormal", "pus_cell_clumps": "present",
            "bacteria": "notpresent", "blood_urea": 55,
            "serum_creatinine": 1.4, "sodium": 135, "potassium": 4.5,
            "hemoglobin": 12.5, "packed_cell_volume": 38,
            "wbc_count": 9000, "rbc_count": 4.5,
            "has_hypertension": "yes", "has_diabetes": "yes",
            "has_cad": "no", "appetite": "good",
            "pedal_edema": "no", "anemia": "no",
            "smoking_status": "daily", "alcohol_consumption": "moderate",
        },
        "symptoms": [
            "fatigue", "chest_pain", "excessive_hunger", "polyuria",
            "irregular_sugar_level", "loss_of_appetite", "swollen_legs",
        ],
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata_shape():
    r = client.get("/metadata")
    assert r.status_code == 200
    meta = r.json()
    assert len(meta["diseases"]) == 4
    assert len(meta["sections"]) == 8
    assert len(meta["symptoms"]) > 40
    assert meta["fusion_weights"] == {"clinical": 0.7, "symptom": 0.3}
    field_names = {f["name"] for s in meta["sections"] for f in s["fields"]}
    assert "age" in field_names and "serum_creatinine" in field_names
    assert "smoking_status" in field_names and "alcohol_consumption" in field_names


def test_predict_runs_all_models_and_fusion():
    t0 = time.time()
    r = client.post("/predict", json=realistic_payload())
    elapsed = time.time() - t0
    assert r.status_code == 200, r.text
    body = r.json()

    assert len(body["diseases"]) == 4
    for d in body["diseases"]:
        expected_fused = min(
            1.0, 0.7 * d["clinical_risk"] + 0.3 * d["symptom_relevance"]
            + d["lifestyle_adjustment"]["total"])
        assert abs(d["fused_risk"] - round(expected_fused, 4)) < 1e-3
        assert 0.0 <= d["lifestyle_adjustment"]["total"] <= 1.0
        assert 0.0 <= d["clinical_risk"] <= d["clinical_risk_raw"] <= 1.0
        assert 0 < d["prevalence"]["target"] < d["prevalence"]["source"] < 1
        bd = d["breakdown"]
        assert abs(bd["clinical_share_pct"] + bd["symptom_share_pct"] - 100.0) < 0.2
        assert d["shap"]["entries"] and d["shap"]["model_name"]
        assert d["top_conditions"]

    overall = body["overall"]
    expected_avg = sum(d["fused_risk"] for d in body["diseases"]) / 4
    assert abs(overall["fused_avg"] - round(expected_avg, 4)) < 1e-3
    print(f"  predict latency: {elapsed:.2f}s")


def test_missing_fields_rejected():
    payload = realistic_payload()
    del payload["clinical"]["age"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "age" in r.json()["detail"]


def test_invalid_value_rejected():
    payload = realistic_payload()
    payload["clinical"]["sex"] = "other"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_unknown_symptom_rejected():
    payload = realistic_payload()
    payload["symptoms"].append("not_a_real_symptom")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_healthy_vs_high_risk_discriminates():
    low = realistic_payload()
    low["clinical"].update({
        "age": 42, "blood_pressure": 110, "max_heart_rate": 160, "bmi": 24.0,
        "pregnancies": 1, "glucose": 82, "insulin": 85, "skin_thickness": 18,
        "diabetes_pedigree_function": 0.3,
        "cholesterol": 175, "fasting_blood_sugar": "no", "resting_ecg": "0",
        "chest_pain_type": "0", "exercise_angina": "no", "oldpeak": 0.1,
        "st_slope": "0", "major_vessels": "0", "thalassemia": "0",
        "total_bilirubin": 0.5, "direct_bilirubin": 0.1, "alkaline_phosphatase": 85,
        "alt": 20, "ast": 18, "total_proteins": 7.2, "serum_albumin": 4.2,
        "albumin_globulin_ratio": 1.4,
        "specific_gravity": 1.010, "urine_albumin": "0", "urine_sugar": "0",
        "urine_rbc": "normal", "pus_cell": "normal", "pus_cell_clumps": "notpresent",
        "blood_urea": 24, "serum_creatinine": 0.9, "sodium": 138, "potassium": 4.2,
        "hemoglobin": 15.5, "packed_cell_volume": 46, "wbc_count": 7000,
        "rbc_count": 4.9,
        "has_hypertension": "no", "has_diabetes": "no", "has_cad": "no",
        "appetite": "good", "pedal_edema": "no", "anemia": "no",
        "smoking_status": "never", "alcohol_consumption": "none",
    })
    low["symptoms"] = []
    r_high = client.post("/predict", json=realistic_payload()).json()
    r_low = client.post("/predict", json=low).json()
    for hd, ld in zip(r_high["diseases"], r_low["diseases"]):
        assert hd["fused_risk"] > ld["fused_risk"], hd["disease"]
    assert r_high["overall"]["fused_avg"] > r_low["overall"]["fused_avg"]


def test_prevalence_recalibration_unit_math():
    # liver: source 0.7136, target 0.10; a 72.5% raw score should drop to ~10%
    corrected = prevalence_recalibrate(0.725, 0.7136, 0.10)
    assert 0.09 < corrected < 0.12
    # monotonic: higher raw score => higher corrected score
    assert prevalence_recalibrate(0.9, 0.7136, 0.10) > corrected
    # extremes pass through
    assert prevalence_recalibrate(0.0, 0.7136, 0.10) == 0.0
    assert prevalence_recalibrate(1.0, 0.7136, 0.10) == 1.0


def test_healthy_profile_is_low_after_recalibration():
    healthy = realistic_payload()
    healthy["clinical"].update({
        "age": 35, "blood_pressure": 115, "max_heart_rate": 160, "bmi": 22.5,
        "pregnancies": 0, "glucose": 88, "insulin": 80, "skin_thickness": 16,
        "diabetes_pedigree_function": 0.25,
        "cholesterol": 180, "fasting_blood_sugar": "no", "resting_ecg": "0",
        "chest_pain_type": "0", "exercise_angina": "no", "oldpeak": 0.0,
        "st_slope": "0", "major_vessels": "0", "thalassemia": "0",
        "total_bilirubin": 0.5, "direct_bilirubin": 0.1, "alkaline_phosphatase": 80,
        "alt": 18, "ast": 15, "total_proteins": 7.5, "serum_albumin": 4.4,
        "albumin_globulin_ratio": 1.5,
        "specific_gravity": 1.015, "urine_albumin": "0", "urine_sugar": "0",
        "urine_rbc": "normal", "pus_cell": "normal", "pus_cell_clumps": "notpresent",
        "bacteria": "notpresent", "blood_urea": 22, "serum_creatinine": 0.8,
        "sodium": 139, "potassium": 4.1, "hemoglobin": 15.0, "packed_cell_volume": 45,
        "wbc_count": 6800, "rbc_count": 4.8,
        "has_hypertension": "no", "has_diabetes": "no", "has_cad": "no",
        "appetite": "good", "pedal_edema": "no", "anemia": "no",
        "smoking_status": "never", "alcohol_consumption": "none",
    })
    healthy["symptoms"] = []
    r = client.post("/predict", json=healthy)
    assert r.status_code == 200, r.text
    body = r.json()
    # Every disease should now land in the Low band for a healthy profile
    for d in body["diseases"]:
        assert d["risk_level"] == "Low", f"{d['label']}: {d['fused_pct']}%"
        assert d["clinical_risk"] < d["clinical_risk_raw"]
    assert body["overall"]["risk_level"] in ("Low", "Elevated")