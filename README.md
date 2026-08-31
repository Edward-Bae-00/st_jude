# SCOGS: Sickle Cell Outcome Grading System & MedGemma Extraction

This repository contains the deterministic rule engine for the **Sickle Cell Outcome Grading System (SCOGS)**, the schema definition for clinical features, and the test harness for evaluating **MedGemma** (4B and 27B) standalone clinical feature extraction against the §2 verbatim quote verification contract.

---

## 📦 Datasets & Data Acquisition

### 1. Pre-Packaged Data (Included in Git Repository)

The repository comes pre-packaged with all data required to run the test suite and feature extraction experiments immediately upon cloning:

- **[`PMC-Patients/scd_cache.json`](PMC-Patients/scd_cache.json) (3.6 MB):** 978 curated Sickle Cell Disease (SCD) patient case reports filtered from the PubMed Central patient dataset. This is the primary dataset read by `scripts/experiments/medgemma_extraction.py`.
- **[`data/clincal_notes.csv`](data/clincal_notes.csv) & [`data/clincal_notes_org.csv`](data/clincal_notes_org.csv):** Clinical note datasets with multi-outcome labels.
- **[`data/scogs_feature_schema.json`](data/scogs_feature_schema.json):** The 137 clinical feature schema definitions for the 53 SCOGS health outcomes.

---

### 2. Downloading Full Raw Datasets (~1.3 GB)

If you need the entire raw 250,000-patient case report corpus from PubMed Central (e.g. for broader cohort exploration or training):

#### Automated Download Script (Cross-Platform: Windows, Mac, Linux)
Run the built-in downloader script:

```bash
# Downloads full PMC-Patients-V2.json (~800MB) & PMC-Patients.csv (~520MB) from Hugging Face
python scripts/download_data.py
```

Options:
```bash
python scripts/download_data.py --files v2    # Download only PMC-Patients-V2.json
python scripts/download_data.py --files csv   # Download only PMC-Patients.csv
python scripts/download_data.py --force       # Re-download even if already present
```

#### Manual Download Links (Hugging Face)
You can also download the files directly from the official [Hugging Face dataset repo (`zhengyun21/PMC-Patients`)](https://huggingface.co/datasets/zhengyun21/PMC-Patients):

- **PMC-Patients-V2.json (250,294 patients):**
  [`https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json`](https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json)
- **PMC-Patients.csv (167k patient summaries):**
  [`https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv`](https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv)

Place the downloaded files inside the `PMC-Patients/` directory at the project root.

---

## Quick Start & Installation

### 1. Clone & Set Up Python Environment

```bash
# Clone repository
git clone https://github.com/Edward-Bae-00/st_jude.git
cd st_jude

# Set up Python virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install testing dependencies
pip install pytest
```

### 2. Verify Rule Engine & Unit Tests

Run the full test suite (264 unit tests covering table evaluation, schema logic, predicates, and extraction verification):

```bash
pytest
```

---

## Running MedGemma Feature Extraction

The test harness evaluates whether MedGemma can extract clinical findings from patient notes and pass the verbatim quote verification gate.

### Tier Overview

| Tier | Model ID | Recommended Hardware | Purpose |
| :--- | :--- | :--- | :--- |
| **`mock`** | N/A (Deterministic dummy) | Any CPU | Validates the harness and verifier pipeline end-to-end |
| **`local`** | `google/medgemma-1.5-4b-it` | Apple Silicon / CPU / GPU | Fast dev iteration loop |
| **`full`** | `google/medgemma-27b-text-it` | NVIDIA GPU (CUDA / Windows / Linux) | Reporting tier for high-capacity extraction |

---

## 🖥️ Running on an NVIDIA Windows PC (MedGemma 27B)

For running the full 27B model tier on Windows with an NVIDIA GPU (e.g. RTX 3090, RTX 4090, RTX 5000/6000, A5000, A6000, A100):

### Hardware & VRAM Guide

- **Full precision / bfloat16:** Requires $\ge 56\text{ GB}$ VRAM (dual RTX 3090/4090, A6000, A100).
- **8-bit Quantization:** Requires $\ge 30\text{ GB}$ VRAM.
- **4-bit Quantization (`bitsandbytes` or GGUF Q4):** Requires $\ge 18\text{ GB}$ VRAM (runs smoothly on a **single 24GB RTX 3090 / RTX 4090**).

---

### Option A: Using Ollama for Windows (Recommended)

1. **Install Ollama for Windows:**
   - Download and install from [ollama.com/download/windows](https://ollama.com/download/windows).
   - Ensure NVIDIA CUDA drivers are installed (Ollama automatically detects NVIDIA GPUs).

2. **Pull the 27B Model:**
   Open PowerShell and run:
   ```powershell
   ollama pull medgemma-27b-text-it
   ```

3. **Run Extraction:**
   - **Using the PowerShell Runner Script:**
     ```powershell
     .\scripts\run_windows_27b.ps1 -Backend ollama -Notes 20 -Repeat 2 -Out results\full.json
     ```
   - **Using the Batch Runner Script (Command Prompt):**
     ```cmd
     scripts\run_windows_27b.bat --notes 20 --repeat 2 --out results\full.json
     ```
   - **Using Direct Python Command:**
     ```powershell
     python scripts\experiments\medgemma_extraction.py --tier full --backend ollama --notes 20 --repeat 2 --out results\full.json
     ```

---

### Option B: Using PyTorch + Hugging Face with CUDA & 4-bit / 8-bit Quantization

If you want to run directly via Python without Ollama:

1. **Install PyTorch with CUDA 12.1+:**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install Transformers & Quantization packages:**
   ```powershell
   pip install transformers accelerate bitsandbytes sentencepiece
   ```

3. **Run with 4-bit quantization (Fits on a single 24GB RTX 3090/4090):**
   ```powershell
   .\scripts\run_windows_27b.ps1 -Backend hf -Quant 4bit -Notes 20 -Repeat 2 -Out results\full.json
   ```
   Or via direct Python command:
   ```powershell
   python scripts\experiments\medgemma_extraction.py --tier full --backend hf --quant 4bit --notes 20 --repeat 2 --out results\full.json
   ```

---

### Option C: Using vLLM or OpenAI-Compatible Local Server

If serving MedGemma 27B via vLLM, SGLang, or llama.cpp:

```powershell
python scripts\experiments\medgemma_extraction.py --tier full --backend openai --host http://localhost:8000 --notes 20 --repeat 2 --out results\full.json
```

---

## 🍎 Running on Mac / Apple Silicon (MedGemma 4B)

1. **Install and Start Ollama:**
   ```bash
   ollama serve
   ollama pull medgemma-1.5-4b-it
   ```

2. **Run Extraction (20 notes, 2 repeats):**
   ```bash
   python3 scripts/experiments/medgemma_extraction.py --tier local --notes 20 --repeat 2 --out results/local.json
   ```

3. **Run Mock Mode (No GPU / No Model Needed):**
   ```bash
   python3 scripts/experiments/medgemma_extraction.py --backend mock --notes 2
   ```

---

## 📊 Understanding the Output & Metrics

The extraction harness automatically calculates:

- **Quote-verified %:** Measures whether proposed features carry a verbatim quote directly present in the note (catches hallucinations). **This is a grounding check, not precision.** It asks whether the quoted words are in the note; it cannot ask whether they *support* the value. `fio2_pct = 21` quoting *"the patient needed increasing oxygen by nasal cannula"* verifies at 100%. Precision is a human judgement and comes only from the hand-check sheet's `supports_value` column.
- **Unit guard:** For a numeric feature whose schema declares a convertible unit, the number is read out of its own verified quote, in the unit the note wrote it in, and converted. A note using mg/L throughout otherwise puts `7.0` into an mg/dL creatinine field — a 10× error no type check can see. A value matching neither its quote's number nor that number's conversion is rejected and counted.
- **Value conflicts:** One `(note, outcome)` routinely yields several verified values for one feature — a note carries five creatinines across twelve years and one of them belongs to the transplant donor. Where the schema states an aggregation ("Highest level of care this event actually reached") the values collapse by it; where it does not, the disagreement is reported and the feature is withheld from grading rather than resolved by emission order.
- **Run-to-run consistency:** Measures output determinism at temperature 0 across repeated runs. At `--concurrency 1` with greedy decoding, 100% is the *expected* result — it is a smoke test for a nondeterministic serving stack, not evidence about the model.
- **Invalid-value rate:** Detects values outside declared types or enums.
- **Unparseable replies:** Tracks any malformed JSON generations.
- **SCOGS Decision Table Grading:** Evaluates extracted features through the official SCOGS logic (outcomes: `graded`, `absent`, `refuted`, `grade_set`, or `cannot_grade`). `refuted` is split out of `absent`: the model *did* evidence the outcome and the decision tables overruled the call (a 36.5 °C "fever"). Pooled with `absent` it sends a reviewer to confirm an absence the rule engine, not the model, produced.
- **Provenance & `run_id`:** Every result file carries a `run_id`, and every review sheet generated from it stamps that id on each row. A filled-in sheet that has drifted apart from its results file is worse than no sheet, because the review hours land on the wrong run and nothing says so.
- **Throughput & Profiling:** Wall-clock time, tokens/second, and tokens per note.

All detailed extractions and clinical note text are saved to the JSON file specified by `--out` (e.g. `results/full.json` or `results/local.json`).

---

## Repository Structure

```
├── docs/
│   └── windows_gpu_setup.md       # Detailed Windows NVIDIA GPU setup guide
├── scripts/
│   ├── download_data.py           # Automated dataset downloader & cache builder
│   ├── audit/                     # Booklet extraction and comparison scripts
│   ├── experiments/
│   │   └── medgemma_extraction.py # Main MedGemma extraction test harness
│   ├── run_windows_27b.ps1        # PowerShell runner for Windows
│   ├── run_windows_27b.bat        # Batch runner for Windows
│   └── scogs/                     # Core SCOGS rule engine and tables
│       ├── evaluate.py            # Rule evaluator
│       ├── features.py            # Feature definitions
│       ├── predicates.py          # Predicate parsing
│       └── tables.py              # Decision table logic
├── tasks/
│   ├── medgemma_extraction_test.md# P11 test protocol & metrics definitions
│   └── plan.md                    # Project roadmap
├── tests/                         # Pytest test suite (264 tests)
├── data/                          # Feature schemas & dataset metadata
├── PMC-Patients/
│   ├── scd_cache.json             # 978 pre-filtered sickle cell patient summaries
│   ├── PMC-Patients-V2.json       # Full raw patient corpus (downloaded via script)
│   └── PMC-Patients.csv           # Full patient summary table (downloaded via script)
└── results/                       # Generated experiment results & artifacts
```
