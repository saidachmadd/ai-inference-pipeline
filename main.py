"""
End-to-End AI Inference Micro-Pipeline Starter Kit (BPJS Healthkathon 2026)
Fokus Teknis: Cross-Modal Feature Engineering, Low-Latency Ensemble Inference (GradientBoosting + IsolationForest),
Local SHAP Feature Attribution, dan Fault-Tolerant Async LLM Reasoning di atas FastAPI Lifespan.
Jalankan server: uvicorn main:app --reload --port 8000
"""

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Memuat variabel dari file .env ke os.environ
import os

import httpx
import numpy as np
from fastapi import FastAPI, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest

# Opsional: Gunakan pustaka shap untuk exact TreeSHAP attribution jika terinstal
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai-inference-pipeline")

MODEL_PATH = Path("jkn_upcoding_model.pkl")


# =====================================================================
# 1. SKEMA KONTRAK DATA (PYDANTIC V2 GATEKEEPER) - INPUT & XAI OUTPUT
# =====================================================================
class VitalSigns(BaseModel):
    systolic_bp_mmhg: int = Field(..., ge=40, le=300)
    diastolic_bp_mmhg: int = Field(..., ge=20, le=200)
    heart_rate_bpm: int = Field(..., ge=20, le=250)
    respiratory_rate_cpm: int = Field(..., ge=5, le=80)
    temperature_celsius: float = Field(..., ge=30.0, le=45.0)
    spo2_percent: float = Field(..., ge=50.0, le=100.0)


class LabResults(BaseModel):
    leukocytes_per_ul: float = Field(..., ge=0.0)
    creatinine_mg_dl: float = Field(..., ge=0.0)
    blood_glucose_mg_dl: float = Field(..., ge=0.0)
    troponin_positive: bool = False
    widal_titer_score: int = Field(default=0, ge=0)


class ClinicalRME(BaseModel):
    resume_medis_text: str = Field(..., min_length=10)
    cppt_summary: str = Field(..., min_length=10)
    vital_signs: VitalSigns
    lab_results: LabResults


class ClaimPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., examples=["CLM-2026-9942"])
    faskes_id: str = Field(..., examples=["FKRTL-3275-RS09"])
    faskes_class: str = Field(..., pattern="^(A|B|C|D)$")
    patient_age: int = Field(..., ge=0, le=120)
    admission_type: str = Field(..., pattern="^(INPATIENT|OUTPATIENT)$")
    room_class: int = Field(..., ge=1, le=3)
    length_of_stay_days: int = Field(..., ge=1, le=365)
    ina_cbg_code: str = Field(..., examples=["A-4-13-III"])
    claimed_ina_cbg_tariff_idr: float = Field(..., gt=0)
    real_hospital_cost_idr: float = Field(..., gt=0)
    primary_diagnosis_icd10: str
    secondary_diagnoses_icd10: list[str] = Field(default_factory=list)
    procedures_icd9cm: list[str] = Field(default_factory=list)
    clinical_rme: ClinicalRME

    @field_validator("ina_cbg_code")
    @classmethod
    def validate_ina_cbg_format(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 4 or parts[3] not in {"0", "I", "II", "III"}:
            raise ValueError(
                "Invalid hierarchical code structure. Expected 4 segments: CMG-CaseType-Group-Severity (e.g., A-4-13-III)"
            )
        return v


class FeatureAttribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_contribution: float
    impact_direction: str
    clinical_interpretation: str


class XAIFraudResponse(BaseModel):
    claim_id: str
    faskes_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float
    risk_tier: str
    detected_modus_operandi: list[str]
    engineered_features: dict[str, float]
    clinical_rule_discrepancies: list[str]
    xai_shap_attributions: list[FeatureAttribution]
    xai_llm_reasoning: str
    recommended_verifier_action: str


# =====================================================================
# 2. PRE-PROCESSING BLOCK ("KACAMATA AI" - CROSS-MODAL FEATURE ENGINE)
# =====================================================================
class ClinicalFeatureExtractor:
    """
    Cross-Modal Feature Engineering Engine:
    Mentransformasi payload JSON multimoda (label tag terstruktur, rasio finansial,
    sensor numerik laboratorium, dan teks bebas) menjadi vektor fitur padat R^6
    beserta sinyal kontradiksi lintas-modalitas (cross-modal discrepancies).
    """

    SEVERITY_MAP = {"0": 0, "I": 1, "II": 2, "III": 3}
    MCC_CODES = {
        "A40.0",
        "A40.9",
        "A41.9",
        "J96.0",
        "J96.2",
        "N17.0",
        "N17.9",
        "E11.1",
        "I63.9",
    }
    CC_CODES = {"E11.2", "E11.9", "I10", "D62", "J44.1"}
    HIGH_IMPACT_PROCEDURES = {"93.90", "96.71", "96.72"}

    FEATURE_NAMES = [
        "severity_numeric",
        "tariff_to_real_cost_ratio",
        "los_severity_anomaly_index",
        "unverified_mcc_count",
        "unverified_procedure_flag",
        "coding_redundancy_flag",
    ]

    @classmethod
    def transform(cls, payload: ClaimPayload) -> dict[str, Any]:
        severity_str = payload.ina_cbg_code.split("-")[3]
        severity_num = float(cls.SEVERITY_MAP.get(severity_str, 1))

        # Fitur 1: Financial Outlier Ratio (Claimed Tariff vs Real Operational Cost)
        tariff_ratio = round(
            payload.claimed_ina_cbg_tariff_idr
            / max(payload.real_hospital_cost_idr, 1.0),
            4,
        )

        # Fitur 2: Temporal-Severity Divergence Index (High Severity vs Short LOS)
        if severity_num == 3 and payload.length_of_stay_days <= 3:
            los_anomaly_index = round((4.0 - payload.length_of_stay_days) * 1.5, 4)
        elif severity_num == 2 and payload.length_of_stay_days <= 1:
            los_anomaly_index = 1.2
        else:
            los_anomaly_index = round(
                severity_num / max(payload.length_of_stay_days, 1), 4
            )

        # Fitur 3: Cross-Modal Entity & Telemetry Verification (Secondary MCC Tags vs Lab Sensors & Free Text)
        discrepancies: list[str] = []
        unverified_mcc = 0.0
        rme_text_lower = (
            f"{payload.clinical_rme.resume_medis_text} {payload.clinical_rme.cppt_summary}"
        ).lower()
        labs = payload.clinical_rme.lab_results
        vitals = payload.clinical_rme.vital_signs

        for code in payload.secondary_diagnoses_icd10:
            # Cross-check tag A40/A41 terhadap sensor leukosit, suhu, HR, dan negasi leksikal
            if code.startswith(("A40", "A41")):
                sepsis_lab_valid = (
                    labs.leukocytes_per_ul > 12000 or labs.leukocytes_per_ul < 4000
                ) and (vitals.temperature_celsius >= 38.3 or vitals.heart_rate_bpm > 90)
                explicit_sepsis_doc = (
                    "sepsis" in rme_text_lower and "sepsis-like" not in rme_text_lower
                )
                if not (sepsis_lab_valid and explicit_sepsis_doc):
                    unverified_mcc += 1.0
                    discrepancies.append(
                        f"Cross-modal mismatch pada tag MCC ({code}): telemetri leukosit ({labs.leukocytes_per_ul}/uL) "
                        f"dan tanda vital (Suhu {vitals.temperature_celsius}C, HR {vitals.heart_rate_bpm} bpm) berada di rentang baseline, "
                        f"serta tidak ditemukan entitas eksplisit 'sepsis' pada korpus teks."
                    )

            # Cross-check tag N17.x terhadap sensor kreatinin serum
            if code.startswith("N17"):
                if labs.creatinine_mg_dl < 1.5:
                    unverified_mcc += 1.0
                    discrepancies.append(
                        f"Cross-modal mismatch pada tag MCC ({code}): sensor kreatinin serum ({labs.creatinine_mg_dl} mg/dL) "
                        f"di bawah ambang batas anomali (< 1.5 mg/dL)."
                    )

            # Cross-check tag E11.1 terhadap sensor glukosa darah
            if code == "E11.1" and labs.blood_glucose_mg_dl < 250.0:
                unverified_mcc += 1.0
                discrepancies.append(
                    f"Cross-modal mismatch pada tag MCC ({code}): sensor glukosa darah ({labs.blood_glucose_mg_dl} mg/dL < 250 mg/dL) tidak mendukung."
                )

        # Fitur 4: High-Impact Procedure Telemetry Verification (Tag 93.90 / 96.7x vs SpO2/RR & Text)
        unverified_proc = 0.0
        for proc in payload.procedures_icd9cm:
            if proc in cls.HIGH_IMPACT_PROCEDURES:
                resp_distress = (
                    vitals.spo2_percent < 92.0 or vitals.respiratory_rate_cpm > 28
                )
                vent_mentioned = any(
                    k in rme_text_lower
                    for k in [
                        "ventilator",
                        "intubasi",
                        "cpap",
                        "bipap",
                        "ventilasi mekanik",
                    ]
                )
                if not (resp_distress and vent_mentioned):
                    unverified_proc = 1.0
                    discrepancies.append(
                        f"Unverified procedure tag ({proc}): telemetri SpO2 {vitals.spo2_percent}% dan laju napas "
                        f"{vitals.respiratory_rate_cpm}x/menit stabil, serta tidak ada token tindakan ventilasi pada log teks."
                    )

        # Fitur 5: Hierarchical Tag Unbundling / Redundancy Detection (A01.0 + A09)
        coding_redundancy = 0.0
        if (
            payload.primary_diagnosis_icd10 == "A01.0"
            and "A09" in payload.secondary_diagnoses_icd10
        ):
            coding_redundancy = 1.0
            discrepancies.append(
                "Hierarchical tag redundancy (Unbundling): kode sekunder A09 dikirim terpisah padahal secara semantik merupakan subset dari kode primer A01.0."
            )

        if tariff_ratio > 2.5:
            discrepancies.append(
                f"Extreme financial ratio outlier: nilai klaim (Rp{payload.claimed_ina_cbg_tariff_idr:,.0f}) mencapai "
                f"{tariff_ratio:.2f}x dari biaya riil operasional (Rp{payload.real_hospital_cost_idr:,.0f})."
            )

        feature_dict = {
            "severity_numeric": severity_num,
            "tariff_to_real_cost_ratio": tariff_ratio,
            "los_severity_anomaly_index": los_anomaly_index,
            "unverified_mcc_count": unverified_mcc,
            "unverified_procedure_flag": unverified_proc,
            "coding_redundancy_flag": coding_redundancy,
        }

        feature_vector = np.array(
            [[feature_dict[name] for name in cls.FEATURE_NAMES]], dtype=np.float64
        )
        return {
            "feature_dict": feature_dict,
            "feature_vector": feature_vector,
            "discrepancies": discrepancies,
        }


# =====================================================================
# 3. PLUG-AND-PLAY MODEL BLOCK (LOCAL .PKL + ASYNC LLM XAI SYNTHESIZER)
# =====================================================================
def bootstrap_or_load_local_model(model_path: Path) -> dict[str, Any]:
    """
    Singleton Model Hydration & Self-Bootstrapping:
    Memuat artefak model .pkl ke memori RAM satu kali saat startup server, atau
    mengkalibrasi decision boundary secara otomatis jika file .pkl belum tersedia.
    """
    if model_path.exists():
        logger.info(f"Hydrating local ML model artifact into RAM from {model_path}...")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
    else:
        logger.info("Bootstrapping and calibrating baseline ensemble .pkl artifact...")
        rng = np.random.default_rng(2026)

        # Sintesis distribusi payload konsisten (n=400) vs adversarial payload (n=400)
        legit_X = np.column_stack(
            [
                rng.choice([1.0, 2.0, 3.0], size=400, p=[0.6, 0.3, 0.1]),
                rng.uniform(0.75, 1.45, size=400),
                rng.uniform(0.15, 0.70, size=400),
                np.zeros(400),
                np.zeros(400),
                np.zeros(400),
            ]
        )
        fraud_X = np.column_stack(
            [
                rng.choice([2.0, 3.0], size=400, p=[0.2, 0.8]),
                rng.uniform(2.10, 6.50, size=400),
                rng.uniform(1.20, 4.50, size=400),
                rng.choice([1.0, 2.0, 3.0], size=400, p=[0.4, 0.4, 0.2]),
                rng.choice([0.0, 1.0], size=400, p=[0.3, 0.7]),
                rng.choice([0.0, 1.0], size=400, p=[0.4, 0.6]),
            ]
        )

        X_train = np.vstack([legit_X, fraud_X])
        y_train = np.hstack([np.zeros(400), np.ones(400)])

        clf = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3, random_state=2026
        )
        clf.fit(X_train, y_train)

        iso = IsolationForest(contamination=0.15, random_state=2026)
        iso.fit(legit_X)

        bundle = {
            "classifier": clf,
            "anomaly_detector": iso,
            "background_data": legit_X[:50],
        }
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)

    if SHAP_AVAILABLE:
        bundle["shap_explainer"] = shap.TreeExplainer(bundle["classifier"])
    else:
        bundle["shap_explainer"] = None

    return bundle


class HybridFraudInferenceEngine:
    """
    Dual-Track Hybrid Inference Orchestrator:
    1. Low-Latency Local Track: GradientBoosting probability + IsolationForest outlier score + TreeSHAP attribution.
    2. Grounded Async LLM Track: Non-blocking causal synthesis dengan Deterministic Circuit-Breaker Fallback.
    """

    INTERPRETATION_TEMPLATES = {
        "severity_numeric": "Ordinal encoding dari tier keparahan (Severity Level) pada payload",
        "tariff_to_real_cost_ratio": "Rasio disparitas antara nilai tarif paket terhadap biaya riil operasional",
        "los_severity_anomaly_index": "Indeks divergensi antara klaim tier keparahan tinggi dengan durasi tinggal (LOS) pendek",
        "unverified_mcc_count": "Jumlah tag komorbiditas mayor (MCC) yang kontradiktif terhadap sensor lab & korpus teks",
        "unverified_procedure_flag": "Sinyal biner klaim prosedur berbiaya tinggi tanpa dukungan telemetri fisiologis/teks",
        "coding_redundancy_flag": "Sinyal biner duplikasi/pemecahan tag hierarkis yang bersifat inklusif (unbundling)",
    }

    def __init__(self, model_bundle: dict[str, Any], http_client: httpx.AsyncClient):
        self.clf = model_bundle["classifier"]
        self.iso = model_bundle["anomaly_detector"]
        self.shap_explainer = model_bundle.get("shap_explainer")
        self.http_client = http_client

    def run_local_tabular_inference(
        self, feature_vector: np.ndarray, feature_dict: dict[str, float]
    ) -> dict[str, Any]:
        fraud_prob = float(self.clf.predict_proba(feature_vector)[0][1])
        raw_iso_score = float(self.iso.decision_function(feature_vector)[0])
        anomaly_score = round(max(0.0, min(1.0, 0.5 - raw_iso_score)), 4)

        # Ekstraksi exact TreeSHAP values atau local feature importance projection
        if self.shap_explainer is not None:
            shap_vals = self.shap_explainer.shap_values(feature_vector)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            contributions = np.array(shap_vals).flatten()
        else:
            importances = self.clf.feature_importances_
            baselines = np.array([1.2, 1.1, 0.35, 0.0, 0.0, 0.0])
            deviations = (feature_vector.flatten() - baselines) / np.maximum(
                baselines, 0.5
            )
            contributions = importances * deviations

        attributions: list[FeatureAttribution] = []
        for idx, feat_name in enumerate(ClinicalFeatureExtractor.FEATURE_NAMES):
            val = float(feature_dict[feat_name])
            contrib = round(float(contributions[idx]), 4)
            direction = (
                "INCREASES_FRAUD_RISK"
                if contrib > 0.01
                else ("DECREASES_FRAUD_RISK" if contrib < -0.01 else "NEUTRAL")
            )
            attributions.append(
                FeatureAttribution(
                    feature_name=feat_name,
                    feature_value=val,
                    shap_contribution=contrib,
                    impact_direction=direction,
                    clinical_interpretation=self.INTERPRETATION_TEMPLATES[feat_name],
                )
            )

        attributions.sort(key=lambda x: abs(x.shap_contribution), reverse=True)
        return {
            "fraud_probability": round(fraud_prob, 4),
            "anomaly_score": anomaly_score,
            "attributions": attributions,
        }

    async def generate_llm_xai_reasoning(
        self,
        payload: ClaimPayload,
        fraud_prob: float,
        attributions: list[FeatureAttribution],
        discrepancies: list[str],
    ) -> str:
        """
        Grounded Async LLM Reasoning Synthesizer:
        Mengirimkan Top 3 SHAP Drivers dan daftar cross-modal discrepancies secara non-blocking.
        Dilengkapi Circuit-Breaker Fallback deterministik jika API key kosong atau network timeout.
        """
        api_key = os.getenv("LLM_API_KEY")
        api_url = os.getenv("LLM_API_BASE_URL")
        model_name = os.getenv("LLM_MODEL_NAME")

        top_drivers = [
            f"{a.feature_name}={a.feature_value} (SHAP={a.shap_contribution:+.3f})"
            for a in attributions[:3]
        ]

        system_prompt = (
            "Anda adalah AI Explainability (XAI) Engine untuk sistem deteksi anomali payload multimoda. "
            "Tugas Anda adalah menyintesis atribusi fitur matematis SHAP dan sinyal diskrepansi lintas-modalitas "
            "(antara tag klaim terstruktur, rasio biaya, sensor laboratorium, dan teks bebas) menjadi satu paragraf "
            "penjelasan teknis yang tajam, objektif, dan bebas halusinasi."
        )
        user_prompt = (
            f"Payload ID: {payload.claim_id} | Group Tag: {payload.ina_cbg_code} | "
            f"Ensemble Anomaly Probability: {fraud_prob:.1%}\n"
            f"Primary Tag: {payload.primary_diagnosis_icd10} | Secondary Tags: {payload.secondary_diagnoses_icd10}\n"
            f"Procedure Tags: {payload.procedures_icd9cm} | Duration (LOS): {payload.length_of_stay_days} days\n"
            f"Top 3 Mathematical SHAP Drivers: {', '.join(top_drivers)}\n"
            f"Cross-Modal Discrepancy Signals: {discrepancies}\n"
            f"Unstructured Text Corpus: {payload.clinical_rme.resume_medis_text}\n"
            "Jelaskan secara kausal mengapa vektor fitur payload ini diklasifikasikan sebagai konsisten atau terindikasi "
            "manipulasi eskalasi tag (upcoding) berdasarkan bukti kontribusi SHAP dan telemetri data."
        )

        if api_key and api_url and model_name:
            try:
                # Merangkai URL native Google secara dinamis dari .env
                url = f"{api_url.rstrip('/')}/{model_name}:generateContent"

                payload_data = {
                    "contents": [
                        {"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                    ]
                }

                async with httpx.AsyncClient(timeout=90.0) as temp_client:
                    response = await temp_client.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "X-goog-api-key": api_key,
                        },
                        json=payload_data,
                    )

                    # Tangkap detail alasan error dari body Google jika gagal
                    if response.status_code != 200:
                        print(f"GOOGLE API ERROR REASON: {response.text}")

                    response.raise_for_status()
                    data = response.json()

                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

            except Exception as exc:
                logger.warning(
                    f"External LLM endpoint failed with reason: {type(exc).__name__} - {exc}; triggering local deterministic XAI circuit-breaker."
                )
        else:
            logger.warning(
                "LLM environment variables (KEY, URL, or MODEL) are missing; triggering circuit-breaker."
            )

        # Deterministic Circuit-Breaker Fallback Synthesizer
        if fraud_prob >= 0.50:
            return (
                f"[XAI SYNTHESIS - ADVERSARIAL PAYLOAD DETECTED] Payload {payload.claim_id} menghasilkan probabilitas "
                f"anomali {fraud_prob:.1%} dengan kontributor matematis tertinggi pada {', '.join(top_drivers)}. "
                f"Terdeteksi kontradiksi lintas-modalitas antara eskalasi tag '{payload.ina_cbg_code}' "
                f"(Rp{payload.claimed_ina_cbg_tariff_idr:,.0f}) dengan telemetri dasar: {' '.join(discrepancies)} "
                f"Secara struktural, injeksi tag komorbiditas/prosedur tanpa dukungan sinyal sensor laboratorium "
                f"dan entitas teks memenuhi pola manipulasi eskalasi nilai (Upcoding)."
            )
        return (
            f"[XAI SYNTHESIS - MULTIMODAL PAYLOAD VERIFIED] Payload {payload.claim_id} memiliki probabilitas anomali "
            f"sangat rendah ({fraud_prob:.1%}). Konfigurasi tag '{payload.ina_cbg_code}' (primer: "
            f"{payload.primary_diagnosis_icd10}, sekunder: {payload.secondary_diagnoses_icd10}) terkonfirmasi koheren "
            f"terhadap parameter sensor laboratorium, tanda vital, log teks selama {payload.length_of_stay_days} hari, "
            f"serta rasio tarif terhadap biaya riil yang ekuilibrium ({payload.claimed_ina_cbg_tariff_idr / payload.real_hospital_cost_idr:.2f}x)."
        )


# =====================================================================
# 4. FASTAPI LIFESPAN & ENDPOINT INFERENSI UTAMA
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ASGI Lifespan Context Manager:
    Menjamin model ML (.pkl), TreeSHAP Explainer, dan Async HTTP Connection Pool
    diinisialisasi satu kali di memori sebelum server menerima request.
    """
    model_bundle = bootstrap_or_load_local_model(MODEL_PATH)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    app.state.inference_engine = HybridFraudInferenceEngine(
        model_bundle=model_bundle, http_client=http_client
    )
    logger.info("FastAPI Hybrid AI Inference Pipeline is online and ready.")
    yield
    await http_client.aclose()
    logger.info("Async HTTP connection pool closed and memory resources released.")


app = FastAPI(
    title="Hybrid AI Inference & XAI Micro-Pipeline",
    description="High-Throughput Multimodal Anomaly Detection with TreeSHAP & Async LLM Reasoning",
    version="2.1.0",
    lifespan=lifespan,
)


@app.post(
    "/api/v1/verify-claim",
    response_model=XAIFraudResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Multimodal Feature Extraction, Ensemble Inference, and XAI Synthesis",
)
async def verify_medical_claim(
    payload: ClaimPayload, request: Request
) -> XAIFraudResponse:
    engine: HybridFraudInferenceEngine = request.app.state.inference_engine

    # Stage 1: Deterministic Cross-Modal Feature Extraction ("Kacamata AI")
    extracted = ClinicalFeatureExtractor.transform(payload)
    feature_dict = extracted["feature_dict"]
    feature_vector = extracted["feature_vector"]
    discrepancies = extracted["discrepancies"]

    # Stage 2: Low-Latency Local Tabular Inference & TreeSHAP Attribution
    local_result = engine.run_local_tabular_inference(feature_vector, feature_dict)
    fraud_prob = local_result["fraud_probability"]
    anomaly_score = local_result["anomaly_score"]
    attributions = local_result["attributions"]

    # Stage 3: Automated Risk Tiering & Pattern Classification
    detected_modus: list[str] = []
    if fraud_prob >= 0.75:
        risk_tier = "CRITICAL_HIGH_RISK"
        if (
            feature_dict["unverified_mcc_count"] > 0
            or feature_dict["los_severity_anomaly_index"] > 1.0
        ):
            detected_modus.append(
                "Manipulasi diagnosis dan/atau tindakan (Upcoding Severity Level)"
            )
        if feature_dict["unverified_procedure_flag"] > 0:
            detected_modus.append(
                "Menagihkan tindakan yang tidak dilakukan (Phantom Procedure)"
            )
        if feature_dict["coding_redundancy_flag"] > 0:
            detected_modus.append("Services unbundling / fragmentation")
        action = "HOLD_FOR_DEEP_INSPECTION: Route payload to human-in-the-loop queue due to high-magnitude cross-modal discrepancies."
    elif fraud_prob >= 0.40:
        risk_tier = "MODERATE_REVIEW_REQUIRED"
        detected_modus.append(
            "Moderate cross-modal divergence between secondary tags and telemetry logs"
        )
        action = "FLAG_FOR_SECONDARY_CHECK: Verify raw laboratory sensor logs and daily text notes."
    else:
        risk_tier = "LOW_RISK_AUTO_APPROVE"
        detected_modus.append(
            "Multimodal telemetry consistent (No adversarial manipulation detected)"
        )
        action = (
            "AUTO_COMMIT_PIPELINE: Proceed with automated straight-through processing."
        )

    # Stage 4: Non-Blocking Async LLM XAI Synthesis (with Circuit-Breaker Fallback)
    llm_reasoning = await engine.generate_llm_xai_reasoning(
        payload=payload,
        fraud_prob=fraud_prob,
        attributions=attributions,
        discrepancies=discrepancies,
    )

    return XAIFraudResponse(
        claim_id=payload.claim_id,
        faskes_id=payload.faskes_id,
        fraud_probability=fraud_prob,
        anomaly_score=anomaly_score,
        risk_tier=risk_tier,
        detected_modus_operandi=detected_modus,
        engineered_features=feature_dict,
        clinical_rule_discrepancies=discrepancies,
        xai_shap_attributions=attributions,
        xai_llm_reasoning=llm_reasoning,
        recommended_verifier_action=action,
    )
