# SCOGS: Sickle Cell Outcome Grading System & MedGemma Extraction

This repository contains the deterministic rule engine for the **Sickle Cell Outcome Grading System (SCOGS)**, the schema definition for clinical features, and the test harness for evaluating **MedGemma** (4B and 27B) standalone clinical feature extraction against the §2 verbatim quote verification contract.

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

- **Quote-verified %:** Measures whether proposed features carry a verbatim quote directly present in the note (catches hallucinations).
- **Run-to-run consistency:** Measures output determinism at temperature 0 across repeated runs.
- **Invalid-value rate:** Detects values outside declared types or enums.
- **Unparseable replies:** Tracks any malformed JSON generations.
- **SCOGS Decision Table Grading:** Evaluates extracted features through the official SCOGS logic (outcomes: `graded`, `absent`, `grade_set`, or `cannot_grade`).
- **Throughput & Profiling:** Wall-clock time, tokens/second, and tokens per note.

All detailed extractions and clinical note text are saved to the JSON file specified by `--out` (e.g. `results/full.json` or `results/local.json`).

---

## Repository Structure

```
├── docs/
│   └── windows_gpu_setup.md       # Detailed Windows NVIDIA GPU setup guide
├── scripts/
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
│   └── scd_cache.json             # 978 pre-filtered sickle cell patient summaries
└── results/                       # Generated experiment results & artifacts
```
