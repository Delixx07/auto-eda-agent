# AUDA — Automated EDA Agent

**AUDA** (Automated EDA & Feature Engineering Agent) adalah library Python untuk analisis data otomatis dengan rekomendasi fitur berbasis aturan dan LLM (opsional).

---

## Daftar Isi

1. [Instalasi](#1-instalasi)
2. [Struktur Proyek](#2-struktur-proyek)
3. [Cara Menjalankan Contoh](#3-cara-menjalankan-contoh)
4. [Cara Pakai di Script Sendiri](#4-cara-pakai-di-script-sendiri)
5. [Cara Pakai dengan Groq LLM](#5-cara-pakai-dengan-groq-llm)
6. [Evaluasi ML (Baseline vs AUDA)](#6-evaluasi-ml-baseline-vs-auda)
7. [Penjelasan Laporan](#7-penjelasan-laporan)
8. [Komponen Library](#8-komponen-library)

---

## 1. Instalasi

### Prasyarat
- Python 3.10 atau lebih baru
- Git (untuk clone repo)

### Langkah 1 — Clone repository

```bash
git clone https://github.com/USERNAME/auto-eda-agent.git
cd auto-eda-agent
```

### Langkah 2 — (Opsional) Buat virtual environment

```powershell
# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### Langkah 3 — Install semua dependency

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
pyarrow>=12.0.0
groq>=0.11.0
```

### Langkah 4 — Verifikasi instalasi

```bash
python -m examples.basic_usage
```

Kalau muncul laporan EDA tanpa error, instalasi sukses.

---

## 2. Struktur Proyek

```
final-project/
├── README.md
├── requirements.txt
├── .env.example
├── auto_eda_agent/              <- library utama
│   ├── __init__.py
│   ├── core.py                  <- class AUDA (orchestrator)
│   ├── profiler.py              <- deteksi tipe kolom
│   ├── missing_handler.py       <- imputasi missing value
│   ├── anomaly_detector.py      <- deteksi outlier
│   ├── feature_recommender.py   <- rekomendasi fitur (rule-based)
│   ├── llm_providers.py         <- koneksi ke Groq API
│   ├── llm_recommender.py       <- rekomendasi fitur (LLM)
│   ├── utils.py
│   └── exceptions.py
└── examples/
    ├── basic_usage.py           <- contoh tanpa LLM
    ├── groq_usage.py            <- contoh dengan Groq LLM
    └── ml_evaluation.py         <- evaluasi perbandingan ML
```

---

## 3. Cara Menjalankan Contoh

Semua perintah dijalankan dari folder proyek:

```powershell
cd "c:\Kuliah\SEM 6\BIGDATA\final-project"
```

### Contoh 1 — Rule-based saja (tanpa API key)

```powershell
python -m examples.basic_usage
```

Menjalankan pipeline EDA lengkap pada dataset sintetis. Tidak butuh internet atau API key.

### Contoh 2 — Dengan Groq LLM

```powershell
# Set API key terlebih dahulu
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxx"

python -m examples.groq_usage
```

Menjalankan pipeline EDA + rekomendasi fitur dari LLM Llama 3.3 70B.

### Contoh 3 — Evaluasi ML (Baseline vs AUDA)

```powershell
python -m examples.ml_evaluation
```

Membuktikan bahwa data yang dibersihkan AUDA menghasilkan model ML lebih baik dibanding preprocessing biasa.

---

## 4. Cara Pakai di Script Sendiri

Buat file Python baru di folder proyek, lalu isi dengan kode berikut:

### Penggunaan paling sederhana

```python
import pandas as pd
from auto_eda_agent import AUDA

# Muat dataset
df = pd.read_csv("train.csv")

# Buat agent
agent = AUDA(df)

# Jalankan semua tahap sekaligus
agent.run_full_pipeline()

# Tampilkan laporan
agent.print_report()
```

### Ambil hasil pipeline

```python
# DataFrame yang sudah dibersihkan (missing values sudah diimputasi)
cleaned = agent.cleaned_df_

# Laporan dalam bentuk dict
report = agent.get_report()

# Simpan laporan ke file JSON
agent.export_report("laporan.json")
```

### Format dataset yang didukung

```python
# Dari file CSV
agent = AUDA("data.csv")

# Dari file Excel
agent = AUDA("data.xlsx")

# Dari file Parquet
agent = AUDA("data.parquet")

# Dari DataFrame langsung
agent = AUDA(df)

# Dari dict
agent = AUDA({"kolom1": [1, 2, 3], "kolom2": ["a", "b", "c"]})
```

### Jalankan tahap satu per satu

```python
agent = AUDA(df)

# Tahap 1: Deteksi tipe kolom dan statistik
agent.profile_data()
print(agent.profile_report_)

# Tahap 2: Bersihkan missing values
agent.handle_missing()
print(agent.cleaned_df_.shape)

# Tahap 3: Deteksi outlier
agent.detect_anomalies()
print(agent.anomaly_report_)

# Tahap 4: Rekomendasi fitur
agent.recommend_features()
print(agent.feature_report_)
```

---

## 5. Cara Pakai dengan Groq LLM

Groq memberikan rekomendasi fitur yang lebih kontekstual dari LLM Llama 3.3 70B.

### Langkah 1 — Daftar API key gratis

Daftar di [console.groq.com](https://console.groq.com) — tidak perlu kartu kredit.
Kuota gratis: **1.000 request/hari, 30 request/menit**.

### Langkah 2 — Set API key

```powershell
# PowerShell (Windows)
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
```

### Langkah 3 — Gunakan di kode

```python
import pandas as pd
from auto_eda_agent import AUDA, GroqProvider

df = pd.read_csv("train.csv")

agent = AUDA(
    df,
    llm_provider=GroqProvider(model="llama-3.3-70b-versatile")
)

agent.run_full_pipeline()
agent.print_report()
```

Rekomendasi dari LLM ditandai dengan `[LLM]` di laporan:

```
Column: 'income'  [numeric]
* Apply log1p transform to 'income'       <- rule-based
* Scale 'income' - StandardScaler         <- rule-based
[LLM] income_to_age_ratio                 <- dari LLM
[LLM] is_high_income                      <- dari LLM
```

### Catatan penting

Jika Groq tidak tersedia atau terjadi error, pipeline **tetap berjalan** menggunakan rule-based saja. Program tidak akan crash.

---

## 6. Evaluasi ML (Baseline vs AUDA)

Script ini membuktikan bahwa data yang dibersihkan AUDA menghasilkan model ML lebih baik.

### Langkah 1 — Buka file evaluasi

Buka [examples/ml_evaluation.py](examples/ml_evaluation.py) dan cari bagian ini di bawah fungsi `main()`:

```python
# Pilihan A: pakai dataset sintetis bawaan (default)
df = build_dataset_with_target(n=800, seed=42)
target_col = "will_churn"

# Pilihan B: pakai file CSV sendiri -> hapus tanda # di bawah ini
# df = pd.read_csv("nama_file.csv")
# target_col = "nama_kolom_target"
```

### Langkah 2 — Ganti dengan dataset sendiri

Contoh pakai Titanic (`train.csv`):

```python
df = pd.read_csv("train.csv")
target_col = "Survived"
```

### Langkah 3 — Jalankan

```powershell
python -m examples.ml_evaluation
```

### Contoh output

```
=================================================================
  ML PERFORMANCE COMPARISON: Baseline vs AUDA
=================================================================
  Metric                   Baseline       AUDA      Delta
-----------------------------------------------------------------
  Accuracy                   0.8075     0.7950   -0.0124
  F1-Score                   0.7520     0.7273   -0.0247
  ROC-AUC                    0.8908     0.8939  +0.0031
-----------------------------------------------------------------
  Training samples              644        583
=================================================================

  Pemenang (ROC-AUC): AUDA
  AUDA meningkatkan ROC-AUC sebesar 0.34%
```

**Cara membaca:**
- **Baseline** = preprocessing sederhana (SimpleImputer mean + OrdinalEncoder)
- **AUDA** = pipeline otomatis (KNN imputation + drop strategi cerdas)
- **ROC-AUC** = metrik utama — semakin tinggi semakin baik
- **Delta** = selisih AUDA dikurangi Baseline (positif = AUDA lebih baik)

---

## 7. Penjelasan Laporan

Laporan AUDA terdiri dari 4 bagian:

### [1] DATA PROFILE

```
Rows: 205 | Columns: 8 | Duplicate rows: 5
```

Menampilkan jumlah baris, kolom, duplikat, dan tipe setiap kolom:

| Tipe | Artinya |
|---|---|
| `numeric` | Kolom angka kontinu (age, income) |
| `categorical` | Kolom dengan nilai terbatas (city, gender) |
| `datetime` | Kolom tanggal/waktu (signup_date) |
| `text` | Kolom teks panjang (komentar, deskripsi) |
| `boolean` | Kolom benar/salah (is_premium) |

### [2] MISSING VALUE HANDLING

Strategi yang dipilih otomatis per kolom:

| Strategi | Kondisi | Contoh |
|---|---|---|
| `drop_column` | Missing >= 60% | Kolom hampir kosong |
| `drop_rows` | Missing < 5% | Sedikit baris bermasalah |
| `impute_knn` | Numeric, data <= 10.000 baris | Isi dengan KNN |
| `impute_median` | Numeric, data > 10.000 baris | Isi dengan nilai tengah |
| `impute_mode` | Categorical / boolean | Isi dengan nilai terbanyak |
| `impute_interpolate` | Datetime | Interpolasi linear |
| `impute_constant` | Text | Isi dengan "Unknown" |

### [3] ANOMALY DETECTION

Tiga metode deteksi outlier dijalankan sekaligus:

| Metode | Cara Kerja |
|---|---|
| **IQR** | Nilai di luar `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` |
| **Z-Score** | Nilai dengan `|z| > 3.0` |
| **Isolation Forest** | Baris yang "mudah diisolasi" oleh pohon keputusan acak |

### [4] FEATURE RECOMMENDATIONS

Saran transformasi fitur per kolom:

| Tipe Kolom | Contoh Saran |
|---|---|
| Numeric skewed | log1p, sqrt, Box-Cox transform |
| Numeric range besar | StandardScaler, MinMaxScaler |
| Categorical kardinality rendah | One-hot encoding |
| Categorical kardinality tinggi | Target encoding, frequency encoding |
| Datetime | Ekstrak tahun, bulan, hari, sin/cos cyclical |
| Text | Panjang teks, TF-IDF, word count |

---

## 8. Komponen Library

| Class | Fungsi |
|---|---|
| `AUDA` | Orchestrator utama — jalankan semua tahap pipeline |
| `DataProfiler` | Deteksi tipe kolom dan statistik dataset |
| `MissingValueHandler` | Pilih dan terapkan strategi imputasi per kolom |
| `AnomalyDetector` | Deteksi outlier dengan IQR, Z-Score, Isolation Forest |
| `FeatureRecommender` | Rekomendasi fitur berbasis aturan statistik |
| `LLMFeatureRecommender` | Rekomendasi fitur dengan augmentasi LLM |
| `GroqProvider` | Koneksi ke Groq API (Llama 3.3 70B) |
| `MockProvider` | Provider palsu untuk testing tanpa API |
