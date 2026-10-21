# ⚡ Practical AI Inference Micro-Pipeline: From Multimodal Data to XAI API

> **Production-Grade Reference Architecture for Real-Time Anomaly Detection & Explainable AI (XAI)**  
> Dirancang untuk workshop teknis **BPJS Healthkathon 2026** (*"Building a Practical AI Inference Pipeline: From Data to API"*). Repositori ini mendemonstrasikan cara merancang sistem inferensi hibrida berkecepatan tinggi yang menggabungkan **Cross-Modal Feature Engineering**, **Local Ensemble Models + SHAP Attribution**, dan **Fault-Tolerant Async LLM Reasoning** di atas **FastAPI & Pydantic V2**.

---

## 🏗️ Sorotan Arsitektur & Prinsip MLOps

1. **Rust-Backed Schema Gatekeeper (Pydantic V2):**
   Memvalidasi kontrak data bersarang (*nested JSON*), rentang batas sensor numerik, dan pola *regex* kode hierarkis secara ketat (`extra="forbid"`) sebelum payload menyentuh lapisan komputasi model.
2. **Zero-Blocking ASGI Lifecycle (`@asynccontextmanager lifespan`):**
   Menghindari *anti-pattern* pemuatan model berulang di dalam *route handler*. Artefak `.pkl` (`GradientBoostingClassifier` & `IsolationForest`), matriks `shap.TreeExplainer`, dan *connection pool* `httpx.AsyncClient` diinisialisasi satu kali di memori RAM saat *startup* (dilengkapi mekanisme *self-bootstrapping* otomatis).
3. **Cross-Modal Feature Engineering ("The AI Lens"):**
   Modul deterministik (`ClinicalFeatureExtractor`) yang merekonsiliasi ketidaksinkronan antara **label klaim terstruktur** dengan **telemetri numerik (sensor lab/tanda vital)** dan **entitas teks bebas** menjadi vektor fitur padat $\mathbb{R}^6$.
4. **Dual-Layer Explainable AI (SHAP + Circuit-Breaker Async LLM):**
   Menggabungkan atribusi fitur matematis lokal dari `SHAP` dengan sintesis kausalitas berbasis LLM secara asinkron (*non-blocking I/O*). Dilengkapi **Deterministic Local Fallback Synthesizer** yang menjamin ketersediaan layanan 100% (*zero downtime*) meskipun tanpa koneksi API eksternal.

---

## 📂 Struktur Direktori

```text
ai-inference-pipeline-starter/
├── main.py                        # FastAPI Hybrid Inference Server (POST /api/v1/verify-claim)
├── pivot_employer_challenge.py    # Modul Strategy Pattern: Pivot ke deteksi anomali distribusi gaji
├── mock_payloads.json             # 2 variasi payload uji (Clean Payload vs Adversarial Payload)
├── requirements.txt               # Daftar pustaka Python produksi
└── README.md                      # Dokumentasi arsitektur & panduan eksekusi
```

---

## 🚀 Instalasi & Eksekusi Cepat (Codespaces / Lokal)

Pastikan lingkungan Anda menggunakan **Python 3.10+**.

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

*(Opsional)* Untuk mengaktifkan *Async LLM Reasoning Engine* eksternal, Salin file contoh konfigurasi dan sesuaikan isinya:
```bash
cp .env.example .env
```
Atau buat file .env di direktori root proyek Anda dengan konfigurasi berikut:
LLM_API_KEY=sk-proj-isi-api-key-anda-disini
LLM_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/models
LLM_MODEL_NAME=gemini-3.5-flash

Jika tidak disetel, *Circuit-Breaker Fallback* akan otomatis menggunakan *Local Deterministic XAI Synthesizer* 

### 2. Menjalankan Server ASGI (Uvicorn)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Saat pertama kali dieksekusi, fungsi `lifespan` akan secara otomatis melakukan *self-bootstrapping* (mengkalibrasi dan menyimpan artefak `jkn_upcoding_model.pkl` ke direktori lokal) sebelum membuka trafik HTTP.

* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **OpenAPI Schema:** `http://localhost:8000/openapi.json`

---

## 🧪 Pengujian Pipeline via cURL

Jalankan perintah `cURL` berikut di terminal terpisah untuk menguji respons *pipeline* terhadap dua karakteristik payload dari `mock_payloads.json`.

### 1. Clean & Consistent Multimodal Payload (`CLM-2026-8801`)
Payload di mana tag hierarkis (`I-4-10-II`), durasi (6 hari), rasio biaya (`1.06x`), serta telemetri sensor laboratorium dan teks bebas saling mendukung secara koheren:
```bash
curl -X POST "http://localhost:8000/api/v1/verify-claim" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-2026-8801",
    "faskes_id": "FKRTL-3275-RS01",
    "faskes_class": "B",
    "patient_age": 58,
    "admission_type": "INPATIENT",
    "room_class": 1,
    "length_of_stay_days": 6,
    "ina_cbg_code": "I-4-10-II",
    "claimed_ina_cbg_tariff_idr": 15100000.0,
    "real_hospital_cost_idr": 14200000.0,
    "primary_diagnosis_icd10": "I21.0",
    "secondary_diagnoses_icd10": ["E11.9", "I10"],
    "procedures_icd9cm": ["89.52", "99.10"],
    "clinical_rme": {
      "resume_medis_text": "Pasien datang dengan nyeri dada kiri menjalar ke lengan sejak 3 jam SMRS. EKG menunjukkan elevasi segmen ST anterior. Diberikan terapi trombolitik sesuai clinical pathway. Selama perawatan 6 hari, Diabetes Mellitus tipe 2 dan Hipertensi esensial aktif dikontrol dengan insulin basal dan amlodipin.",
      "cppt_summary": "Hari 1-3 observasi intensif ICCU, pemberian agen trombolitik, monitoring glukosa darah harian dan tekanan darah. Hari 4-6 mobilisasi bertahap, kondisi stabil.",
      "vital_signs": {
        "systolic_bp_mmhg": 165,
        "diastolic_bp_mmhg": 100,
        "heart_rate_bpm": 104,
        "respiratory_rate_cpm": 22,
        "temperature_celsius": 36.8,
        "spo2_percent": 96.0
      },
      "lab_results": {
        "leukocytes_per_ul": 10200.0,
        "creatinine_mg_dl": 1.1,
        "blood_glucose_mg_dl": 245.0,
        "troponin_positive": true,
        "widal_titer_score": 0
      }
    }
  }'
```

### 2. Adversarial / Manipulated Payload (`CLM-2026-9942`)
Payload dengan injeksi tag *Severity III* (`A-4-13-III`), rasio disparitas biaya ekstrem (`5.86x`), durasi anomalis (2 hari), serta kontradiksi langsung antara klaim kode sekunder/prosedur dengan angka sensor laboratorium dan catatan teks:
```bash
curl -X POST "http://localhost:8000/api/v1/verify-claim" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-2026-9942",
    "faskes_id": "FKRTL-3275-RS09",
    "faskes_class": "C",
    "patient_age": 29,
    "admission_type": "INPATIENT",
    "room_class": 1,
    "length_of_stay_days": 2,
    "ina_cbg_code": "A-4-13-III",
    "claimed_ina_cbg_tariff_idr": 18450000.0,
    "real_hospital_cost_idr": 3150000.0,
    "primary_diagnosis_icd10": "A01.0",
    "secondary_diagnoses_icd10": ["A41.9", "N17.0", "A09"],
    "procedures_icd9cm": ["93.90", "99.21"],
    "clinical_rme": {
      "resume_medis_text": "Pasien demam 3 hari disertai mual dan diare ringan. Kondisi tampak infeksi berat dan dehidrasi. Diberikan infus RL dan injeksi ceftriaxone. Hari kedua demam turun, pasien minta pulang atas permintaan sendiri dalam keadaan membaik.",
      "cppt_summary": "Hari 1: Pasien sadar penuh, kompos mentis, makan minum baik, terpasang infus RL 20 tpm. Hari 2: Bebas demam, keluhan mual hilang, rencana rawat jalan.",
      "vital_signs": {
        "systolic_bp_mmhg": 120,
        "diastolic_bp_mmhg": 80,
        "heart_rate_bpm": 82,
        "respiratory_rate_cpm": 18,
        "temperature_celsius": 37.6,
        "spo2_percent": 99.0
      },
      "lab_results": {
        "leukocytes_per_ul": 7800.0,
        "creatinine_mg_dl": 0.9,
        "blood_glucose_mg_dl": 98.0,
        "troponin_positive": false,
        "widal_titer_score": 160
      }
    }
  }'
```

---

## 🔄 5-Minute Architecture Pivot (Strategy Pattern Demo)
Untuk menguji bagaimana pola ekstraksi fitur yang sama diadaptasi ke kasus **Deteksi Anomali Distribusi Gaji & Rekonsiliasi Entitas Korporat**:
```bash
python pivot_employer_challenge.py
```