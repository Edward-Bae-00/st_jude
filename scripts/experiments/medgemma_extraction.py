"""P11 - Does MedGemma actually extract features well enough?

Everything in the current plan assumes MedGemma is good at reading a note and
filling in feature values. Nothing has checked that. This script checks it.

It runs MedGemma ALONE - note in, features out, straight into the real decision
tables - so what it measures is Version B of the plan (§3). Version A cannot be
measured until the lexicon (P5) and a trained BERT exist. If the numbers here
are bad, delete this file and the whole MedGemma direction with it. If they are
good, this grows into Version B.

Grades always come from the tables. Extraction quality is what is measured;
grading never is.

The contract from §2 is enforced, not assumed: every proposed value must carry a
quote, and the quote must appear VERBATIM in the note. A value whose quote
cannot be found is rejected, not down-weighted - that turns hallucination into a
counted failure rather than a silent one.

Usage examples:
    # 1. Local 4B tier (Apple Silicon / CPU / GPU via Ollama):
    python3 scripts/experiments/medgemma_extraction.py --tier local --notes 20

    # 2. Full 27B tier on NVIDIA Windows / Linux via Ollama:
    python scripts/experiments/medgemma_extraction.py --tier full --notes 20 --repeat 2 --out results/full.json

    # 3. Full 27B tier on NVIDIA Windows / Linux directly via PyTorch / HuggingFace (4-bit quant for 16-24GB GPUs):
    python scripts/experiments/medgemma_extraction.py --tier full --backend hf --quant 4bit --notes 20 --out results/full.json

    # 4. Mock backend (no GPU/model required):
    python3 scripts/experiments/medgemma_extraction.py --backend mock --notes 2

    # 5. Throughput run on a GPU with VRAM to spare (needs OLLAMA_NUM_PARALLEL>=4).
    #    Measure run-to-run consistency separately, at --concurrency 1:
    python scripts/experiments/medgemma_extraction.py --tier full --notes 20 --concurrency 4 --out results/full.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Ensure UTF-8 output on Windows consoles (prevents charmap / cp1252 encode errors on °, µ, ×, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scogs.evaluate import grade
from scogs.features import FEATURES
from scogs.predicates import parse
from scogs.tables import TABLES

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Model ids are configuration, never hardcoded logic (§3).
TIERS = {
    "local": {
        "hf": "google/medgemma-1.5-4b-it",
        "ollama": "medgemma-1.5-4b-it",
        "note": "dev tier, Apple Silicon; never quote its numbers (§3)",
    },
    "full": {
        "hf": "google/medgemma-27b-text-it",
        "ollama": "medgemma-27b-text-it",
        "note": "reporting tier; needs a real GPU (NVIDIA CUDA / A100 / RTX 3090/4090)",
    },
}


# ------------------------------------------------------------------- prompting

def feature_brief(name: str, outcome: str) -> str:
    spec = FEATURES[name]
    bits = [f'"{name}" ({spec["type"]}']
    if spec["values"]: bits.append(f', one of: {", ".join(spec["values"])}')
    if spec["unit"]:   bits.append(f', in {spec["unit"]}')
    bits.append(")")
    line = "".join(bits) + " - " + spec["definition"]
    clarifier = spec["per_outcome"].get(outcome)
    if clarifier: line += f" For this outcome specifically: {clarifier}"
    return line


def expand_derived(names: set[str]) -> set[str]:
    """Replace derived features with the features they are computed from.

    A table row says `life_support`, but nothing in a note says that - it is
    resolved from ventilation, vasopressors, renal replacement and other life
    support. Asking the model for the derived name yields nothing, and every
    outcome that uses it would silently cap one grade below its true ceiling.
    """
    out, seen = set(), set()
    stack = list(names)
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        spec = FEATURES[n]
        if not spec["derived"]:
            out.add(n); continue
        inputs = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec["derived"])
                  if w in FEATURES and w != n]
        if inputs: stack += inputs
        else: out.add(n)
    return out


def build_prompt(note: str, outcome: str) -> str:
    table = TABLES[outcome]
    needed = {n for _, pred in table.all_rows() for n in parse(pred).names()}
    if table.on: needed.add(table.on)
    needed = sorted(expand_derived(needed))
    lines = "\n".join(f"  - {feature_brief(n, outcome)}" for n in needed)
    return f"""You are reading a clinical note and extracting specific findings. \
You are NOT assigning a severity grade - that is done separately by a rule engine.

Health outcome under consideration: {outcome} - {table.name}

Extract only these findings:
{lines}

Rules:
- Report a finding ONLY if the note explicitly supports it. Omit anything the note does not address.
  Omitting is correct and expected; guessing is not.
- Extract each finding at most ONCE. Do NOT repeat findings.
- Every finding MUST include "quote": a concise sentence or phrase copied EXACTLY from the note,
  character for character. If you cannot copy an exact supporting quote, do not report the finding.
  Do NOT include findings with null, empty, or unquoted text.
- "present" is whether the note evidences this health outcome at all.

Reply with JSON only, no prose:
{{"present": true|false,
  "findings": [{{"feature": "<name>", "value": <value>, "quote": "<exact text from the note>"}}]}}

NOTE:
\"\"\"{note}\"\"\""""


# -------------------------------------------------------------------- backends

def get_model_info(model: str, host: str) -> dict:
    try:
        body = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(f"{host}/api/show", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


def call_ollama(prompt: str, model: str, host: str, stats: dict | None = None,
                timeout: int = 300, **kwargs) -> str:
    # keep_alive -1 pins the model in VRAM. Without it Ollama unloads after 5 idle
    # minutes and the next call silently pays a full reload - 50+ GB at the F16 tier.
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "keep_alive": -1,
                       "options": {"temperature": 0, "seed": 0, "num_predict": 1024, "repeat_penalty": 1.1}}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
                if stats is not None:
                    stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + resp.get("prompt_eval_count", 0)
                    stats["eval_count"] = stats.get("eval_count", 0) + resp.get("eval_count", 0)
                    stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + resp.get("eval_duration", 0) / 1e9
                    stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + (time.time() - t0)
                return resp.get("response", "")
        except Exception as e:
            if attempt == 2:
                print(f"    [warning] ollama call failed after 3 attempts: {e}", flush=True)
                return ""
            time.sleep(2)
    return ""


_HF_PIPELINE = None
_HF_MODEL_NAME = None

def get_hf_pipeline(model_name: str, quant: str = "none"):
    global _HF_PIPELINE, _HF_MODEL_NAME
    if _HF_PIPELINE is not None and _HF_MODEL_NAME == model_name:
        return _HF_PIPELINE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        raise SystemExit(
            "Hugging Face backend requires PyTorch and transformers:\n"
            "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "    pip install transformers accelerate bitsandbytes\n"
        )

    print(f"\n[HF Backend] Loading model {model_name} on CUDA/GPU (quant={quant})...", flush=True)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32

    if quant == "4bit":
        kwargs["load_in_4bit"] = True
    elif quant == "8bit":
        kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    _HF_PIPELINE = pipeline("text-generation", model=model, tokenizer=tokenizer)
    _HF_MODEL_NAME = model_name
    print(f"[HF Backend] Model {model_name} loaded successfully.\n", flush=True)
    return _HF_PIPELINE


def call_hf(prompt: str, model: str, host: str, stats: dict | None = None,
            quant: str = "none", **kwargs) -> str:
    pipe = get_hf_pipeline(model, quant)
    t0 = time.time()
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt_formatted, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)
    gen_text = out[0]["generated_text"][len(prompt_formatted):].strip()
    dur = time.time() - t0
    if stats is not None:
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + dur
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + dur
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + len(prompt_formatted.split())
        stats["eval_count"] = stats.get("eval_count", 0) + len(gen_text.split())

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", gen_text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", gen_text, re.S)
    return m.group(1) if m else gen_text


def call_openai_compatible(prompt: str, model: str, host: str, stats: dict | None = None,
                           timeout: int = 300, **kwargs) -> str:
    url = f"{host.rstrip('/')}/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
            usage = resp.get("usage", {})
            if stats is not None:
                stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + usage.get("prompt_tokens", 0)
                stats["eval_count"] = stats.get("eval_count", 0) + usage.get("completion_tokens", 0)
                stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + (time.time() - t0)
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"    [warning] openai-compatible call failed: {e}", flush=True)
        return ""


def call_mock(prompt: str, model: str, host: str, stats: dict | None = None, **kwargs) -> str:
    """Deterministic stand-in so the harness itself can be tested and reviewed
    without a GPU. Emits one findable quote and one deliberate hallucination, so
    the verifier must report exactly 50%.

    `present` is flipped deterministically per (note, outcome) so the ABSENT path
    is exercised too - otherwise the absence audit has nothing to run on and a
    broken absent branch ships unnoticed. It does not touch the quote tally,
    which counts proposals regardless of presence."""
    m = re.search(r'NOTE:\n"""(.*)"""', prompt, re.S)
    note = m.group(1) if m else ""
    outcome = (re.search(r"Health outcome under consideration: (\S+)", prompt) or [None, "0"])[1]
    present = zlib.crc32(f"{outcome}:{note[:120]}".encode()) % 3 != 0
    first = next((w for w in re.findall(r"[A-Za-z]{6,}", note)), "unknown")
    if stats is not None:
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + 100
        stats["eval_count"] = stats.get("eval_count", 0) + 50
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + 0.01
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + 0.01
    return json.dumps({"present": present, "findings": [
        {"feature": "death_attributed", "value": False, "quote": first},
        {"feature": "death_attributed", "value": True,
         "quote": "a sentence that is definitely not in this note"},
    ]})


BACKENDS = {
    "ollama": call_ollama,
    "hf": call_hf,
    "openai": call_openai_compatible,
    "vllm": call_openai_compatible,
    "mock": call_mock,
}


# ------------------------------------------------------- verification & scoring

# SentencePiece byte-token wreckage from a badly converted GGUF. Its presence
# means the served weights are corrupt, not that the model hallucinated - so it
# is counted and reported, never quietly normalized into a passing quote.
ARTIFACT = re.compile(r"\[UNK_BYTE_|\u2581")


def normalize(s: str) -> str:
    # Strip community GGUF SentencePiece byte token artifacts (e.g. [UNK_BYTE_0xe29681▁...])
    s = re.sub(r"\[UNK_BYTE_[^\]]+\]", " ", s)
    s = s.replace("\u2581", " ")
    s = re.sub(r"\s+", " ", s)
    # Strip whitespace before closing punctuation from dataset scraping artifacts (e.g. '(Figure )' -> '(Figure)')
    s = re.sub(r"\s+([\)\]\.,;:])", r"\1", s)
    return s.strip().lower()


@dataclass
class Tally:
    proposed: int = 0
    quote_ok: int = 0
    quote_missing: int = 0      # no quote field at all
    quote_unfound: int = 0      # quote not verbatim in the note -> hallucination
    value_bad: int = 0          # value not of the declared type / not a declared enum
    unknown_feature: int = 0
    accepted: int = 0
    bad_json: int = 0
    tokenizer_artifacts: int = 0   # quotes carrying corrupt-GGUF byte tokens
    per_feature: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_sec: float = 0.0

    def report(self):
        p = self.proposed or 1
        # A null-quote placeholder is a prompt-compliance failure, not a grounding
        # result. Reporting one grounding number lets the denominator be chosen to
        # flatter the model, so both are always emitted together.
        quoted = self.quote_ok + self.quote_unfound
        q = quoted or 1
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "quoted": quoted,
            "quote_verified": self.quote_ok,
            "quote_unfound": self.quote_unfound,
            "null_placeholder": self.quote_missing,
            "null_placeholder_pct": round(100 * self.quote_missing / p, 1),
            "quote_verified_pct": round(100 * self.quote_ok / p, 1),
            "quote_verified_pct_of_quoted": round(100 * self.quote_ok / q, 1),
            "hallucinated_quote_pct": round(100 * self.quote_unfound / p, 1),
            "hallucinated_pct_of_quoted": round(100 * self.quote_unfound / q, 1),
            "tokenizer_artifacts": self.tokenizer_artifacts,
            "missing_quote": self.quote_missing,
            "invalid_value": self.value_bad,
            "unknown_feature": self.unknown_feature,
            "unparseable_replies": self.bad_json,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_clock_sec": round(self.wall_clock_sec, 2),
        }


def coerce(name: str, value):
    """-> (ok, coerced). Enforces the declared type; never widens it."""
    spec = FEATURES.get(name)
    if spec is None: return False, None
    t = spec["type"]
    if t == "bool":
        if isinstance(value, bool): return True, value
        if str(value).lower() in {"true", "yes"}:  return True, True
        if str(value).lower() in {"false", "no"}:  return True, False
        return False, None
    if t == "num":
        try: return True, float(value)
        except (TypeError, ValueError): return False, None
    v = str(value).strip().lower()
    return (True, v) if v in (spec["values"] or []) else (False, None)


def verify(reply: str, note: str, tally: Tally) -> tuple[dict, bool | None]:
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        tally.bad_json += 1
        return {}, None
    hay = normalize(note)
    out = {}
    for f in data.get("findings", []) or []:
        tally.proposed += 1
        name = f.get("feature")
        if name not in FEATURES:
            tally.unknown_feature += 1
            continue
        quote = f.get("quote")
        if not quote:
            tally.quote_missing += 1
            continue
        if ARTIFACT.search(quote):
            tally.tokenizer_artifacts += 1
        if normalize(quote) not in hay:
            tally.quote_unfound += 1          # the §2 rule doing its job
            continue
        tally.quote_ok += 1
        ok, val = coerce(name, f.get("value"))
        if not ok:
            tally.value_bad += 1
            continue
        out[name] = val
        tally.accepted += 1
        tally.per_feature[name] += 1
    return out, data.get("present")


# ------------------------------------------------------------------------ main

# Cohort gate. Three traps live in this corpus and each one silently poisons the
# eval set with a note that can only ever score `absent`:
#   1. "sickle cell trait" is the heterozygous carrier state, not the disease.
#   2. In cardiology, SCD means *sudden cardiac death* - so the bare abbreviation
#      never qualifies a note on its own.
#   3. The mention may be negated, or belong to the mother rather than the patient.
SICKLE_EXPLICIT = re.compile(r"sickle[- ]cell(?!\s+trait)|HbSS|HbSC|\bHb\s?S\b", re.I)
SCD_TERM = re.compile(r"sickle[- ]cell(?!\s+trait)|HbSS|HbSC|\bHb\s?S\b|\bSCD\b", re.I)
NEG_SCD = re.compile(r"(?:denie[sd]|den(?:y|ying)|no|without|negative for|ruled out|"
                     r"family history|maternal|paternal|mother|father|sibling|"
                     r"brother|sister|cousin)\b[^.]{0,70}?"
                     r"(?:sickle[- ]cell|\bSCD\b|HbSS|HbSC)", re.I)
TRAIT = re.compile(r"sickle[- ]cell\s+trait", re.I)


def _clean(text: str) -> str:
    """Drop trait mentions, then mentions that are negated or somebody else's."""
    return NEG_SCD.sub(" ", TRAIT.sub(" ", text or ""))


def scd_mentions(text: str) -> int:
    """SCD mentions that are the patient's own and not negated."""
    return len(SCD_TERM.findall(_clean(text)))


def is_scd_primary(rec: dict) -> bool:
    """Is this note ABOUT sickle cell disease, or does it merely say the words?

    The loose mention regex admits notes whose only SCD reference is a denial
    ("denied a family history of SCD"), a carrier state, the mother's diagnosis,
    or a cardiology note using SCD for sudden cardiac death. Those land in the
    eval set as guaranteed `absent`, inflating the absent count and deflating
    every rate computed over the sample - a measurement artifact indistinguishable
    from a model that simply extracts nothing.
    """
    raw_title = rec.get("title", "") or ""
    if TRAIT.search(raw_title) and not SICKLE_EXPLICIT.search(TRAIT.sub(" ", raw_title)):
        return False                       # a paper titled "Sickle Cell Trait: ..." is about trait
    if SICKLE_EXPLICIT.search(_clean(raw_title)):
        return True                        # the paper names the disease in its title
    body = _clean(rec.get("patient", "") or "")
    if not SICKLE_EXPLICIT.search(body):
        return False                       # "SCD" alone is not evidence of sickle cell
    return len(SCD_TERM.findall(body)) >= 2


def load_notes(cohort: str = "loose") -> list[dict]:
    """The candidate pool. Selection happens in select_notes()."""
    cache = ROOT / "PMC-Patients" / "scd_cache.json"
    if cache.exists():
        scd = json.loads(cache.read_text(encoding="utf-8"))
    else:
        pat = re.compile(r"sickle cell|\bSCD\b|HbSS|HbSC", re.I)
        data = json.loads((ROOT / "PMC-Patients" / "PMC-Patients-V2.json").read_text(encoding="utf-8"))
        scd = [r for r in data if pat.search(r.get("patient", ""))]
        scd.sort(key=lambda r: r["patient_uid"])          # deterministic before sampling
        try:
            cache.write_text(json.dumps(scd), encoding="utf-8")
        except Exception:
            pass
    if cohort == "scd_primary":
        kept = [r for r in scd if is_scd_primary(r)]
        print(f"cohort=scd_primary: {len(kept)}/{len(scd)} kept "
              f"({len(scd) - len(kept)} dropped as mention-only)")
        scd = kept
    return scd


# ------------------------------------------------------------- note selection
#
# The unit of evaluation is the (note, outcome) PAIR, not the note. `absent` is a
# first-class answer (plan §1), so an eval set needs outcomes that are genuinely
# present AND outcomes that are genuinely not - otherwise the absent decision,
# which is what actually gates whether anything gets graded, goes unmeasured.
#
# Seeds pick notes only. They never touch extraction, features or grading, so
# they cannot bias a grade - but they DO bias which cases get seen, toward the
# lexically obvious ones. That is what the unstratified holdout is for: it is
# drawn at random from the same pool, so the size of the bias is measurable
# rather than merely disclosed.

OUTCOME_SEEDS = {
    "10": r"chronic pain|daily pain|persistent pain",
    "19": r"acute kidney injur|\bAKI\b|renal failure|rising creatinine|creatinine",
    "24": r"priapism",
    "28": r"pain(ful)? (crisis|crises|episode)|vaso-?occlusive|\bVOC\b|sickle cell crisis|pain control",
    "29": r"splenic sequestration|sequestration crisis",
    "30": r"alloimmuni|delayed h(a)?emolytic|\bDHTR\b",
    "34": r"iron overload|h(a)?emochromatosis|h(a)?emosiderosis|ferritin|chelat",
    "35": r"aplastic crisis|parvovirus",
    "36": r"\bfever|febrile|pyrexia|temperature of \d",
    "37": r"sepsis|septic|bacter(a)?emia",
    "40": r"leg ulcer|ankle ulcer|venous ulcer",
    "43": r"multiorgan failure|multi-organ failure|\bMOF\b",
    "48": r"acute chest|chest syndrome|\bACS\b",
    "49": r"asthma|wheez|bronchodilator",
    "04": r"heart failure|\bCHF\b|cardiac decompensation",
    "05": r"myocardial infarction|\bMI\b|troponin",
    "06": r"hypertension|hypertensive|elevated blood pressure",
    "11": r"cognitive|neurocognitive|memory (loss|impairment)",
    "15": r"stroke|infarct|h(a)?emorrhage|\bCVA\b",
    "18": r"cholecyst|cholelith|gallstone|gallbladder",
    "21": r"chronic kidney disease|\bCKD\b|nephropathy|proteinuria",
    "31": r"hypersplenism|splenomegaly",
    "32": r"hepatopathy|hepatic|liver (failure|dysfunction)|transaminas",
    "39": r"avascular necrosis|osteonecrosis|\bAVN\b",
    "42": r"osteoporo|osteopeni|bone mineral density",
    "47": r"depress|\bPHQ",
    "52": r"pulmonary hypertension|\bPAH\b|elevated TRV",
    "53": r"sleep apn(o)?ea|\bOSA\b|polysomnograph",
}


def outcome_seed(num: str) -> re.Pattern:
    """A lexical prior for 'this note probably discusses outcome `num`'.

    Falls back to the outcome's own name, which is usually enough ("Priapism",
    "Leg Ulcer", "Osteomyelitis"); OUTCOME_SEEDS covers the ones where the
    rubric's phrasing is not what a clinician writes.
    """
    if num in OUTCOME_SEEDS:
        return re.compile(OUTCOME_SEEDS[num], re.I)
    name = TABLES[num].name
    alts = []
    paren = re.search(r"\(([^)]*)\)", name)
    base = re.sub(r"\s*\([^)]*\)", "", name).strip()
    alts += [re.escape(x.strip()) for x in base.split("/") if x.strip()]
    if paren and re.fullmatch(r"[A-Z]{2,6}", paren.group(1).strip()):
        alts.append(r"\b" + re.escape(paren.group(1).strip()) + r"\b")
    return re.compile("|".join(alts), re.I)


def select_notes(pool: list[dict], n: int, outcomes: list[str], *, seed: int = 20260828,
                 holdout_frac: float = 0.25, stratify: bool = True):
    """-> (notes, selection) where selection maps uid -> 'seeded:<outcome>' | 'holdout'."""
    import random
    rng = random.Random(seed)
    if not stratify:
        picked = rng.sample(pool, min(n, len(pool)))
        return picked, {r["patient_uid"]: "random" for r in picked}

    n_hold = max(1, round(n * holdout_frac))
    n_strat = max(0, n - n_hold)
    per = [n_strat // len(outcomes)] * len(outcomes)
    for i in range(n_strat - sum(per)):
        per[i] += 1

    picked, selection = [], {}
    taken = set()
    for num, want in zip(outcomes, per):
        pat = outcome_seed(num)
        cands = [r for r in pool
                 if r["patient_uid"] not in taken and pat.search(r.get("patient", "") or "")]
        got = rng.sample(cands, min(want, len(cands)))
        if len(got) < want:
            print(f"  seeded {num:>3s} ({TABLES[num].name[:28]}): only {len(got)}/{want} "
                  f"candidates in the pool")
        for r in got:
            taken.add(r["patient_uid"])
            selection[r["patient_uid"]] = f"seeded:{num}"
        picked += got

    rest = [r for r in pool if r["patient_uid"] not in taken]
    hold = rng.sample(rest, min(n_hold, len(rest)))
    for r in hold:
        selection[r["patient_uid"]] = "holdout"
    picked += hold

    seeded_n = len(picked) - len(hold)
    print(f"selection: {seeded_n} seeded across {len(outcomes)} outcomes + "
          f"{len(hold)} unstratified holdout = {len(picked)} notes")
    return picked, selection


def run(notes, outcomes, backend, model, host, tally, quant="none", timeout=300,
        concurrency=1):
    """Model calls first (optionally in parallel), then verification - always serial.

    Verification stays on the main thread in the original note-major order, so the
    Tally accumulates identically at any concurrency and `Tally` itself never needs
    a lock. Only the backend calls fan out.

    Each task carries its own stats dict; `call_ollama` does an unlocked
    read-modify-write on that dict, which is only safe because no two tasks share
    one. They are summed here.
    """
    tasks = [(rec, num) for rec in notes for num in outcomes]
    total, done = len(tasks), 0
    replies, stats_parts = [None] * total, [None] * total
    print_lock = threading.Lock()
    t0 = time.time()

    def call(i):
        nonlocal done
        rec, num = tasks[i]
        local = {"prompt_eval_count": 0, "eval_count": 0,
                 "eval_duration_sec": 0.0, "total_duration_sec": 0.0}
        reply = BACKENDS[backend](build_prompt(rec["patient"], num), model, host,
                                  stats=local, quant=quant, timeout=timeout)
        replies[i], stats_parts[i] = reply, local
        with print_lock:
            done += 1
            print(f"  [{done}/{total}] UID {rec['patient_uid']} outcome {num} "
                  f"({TABLES[num].name})", flush=True)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(call, range(total)))
    else:
        for i in range(total):
            call(i)

    results = {}
    for i, (rec, num) in enumerate(tasks):
        reply = replies[i] or ""
        feats, present = verify(reply, rec["patient"], tally)
        results.setdefault(rec["patient_uid"], {})[num] = (feats, present, reply)

    for part in stats_parts:
        if part:
            tally.prompt_tokens += part["prompt_eval_count"]
            tally.completion_tokens += part["eval_count"]
    tally.wall_clock_sec += (time.time() - t0)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), default="local")
    ap.add_argument("--model", help="override the tier's model id")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ollama",
                    help="inference backend (ollama, hf, openai/vllm, mock)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama host or OpenAI-compatible server URL")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="quantization for Hugging Face backend (4bit/8bit recommended for 24GB GPUs)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-request timeout in seconds (default 300)")
    ap.add_argument("--notes", type=int, default=20)
    ap.add_argument("--cohort", choices=["scd_primary", "loose"], default="loose",
                    help="pool definition. loose (default): any SCD mention - keeps notes "
                         "where outcomes are genuinely absent, which the absence audit needs. "
                         "scd_primary: SCD-primary notes only")
    ap.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True,
                    help="pick notes to hit every target outcome, plus a random holdout")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="fraction of notes drawn at random, to measure the seeds' bias")
    ap.add_argument("--outcomes", default="28,48,36,19",
                    help="comma-separated; default is a common-outcome sample")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and report run-to-run consistency at temperature 0")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight backend requests (default 1 = sequential). Needs a "
                         "server that batches, e.g. OLLAMA_NUM_PARALLEL>=N. See the "
                         "warning printed when this is combined with --repeat")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.concurrency > 1 and a.backend == "hf":
        print("!! --backend hf holds one lazily-initialised global pipeline and is not "
              "safe to call\n   concurrently. Forcing --concurrency 1.")
        a.concurrency = 1
    if a.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    tier = TIERS[a.tier]
    if a.model:
        model = a.model
    elif a.backend in {"hf", "openai", "vllm"}:
        model = tier["hf"]
    else:
        model = tier["ollama"]

    outcomes = [o.strip() for o in a.outcomes.split(",") if o.strip()]
    for o in outcomes:
        if o not in TABLES: raise SystemExit(f"unknown outcome {o!r}")

    pool = load_notes(cohort=a.cohort)
    notes, selection = select_notes(pool, a.notes, outcomes,
                                    holdout_frac=a.holdout_frac, stratify=a.stratify)
    model_info = get_model_info(model, a.host) if a.backend == "ollama" else {}
    model_digest = model_info.get("details", {}).get("parent_model", "") or model_info.get("model_info", {}).get("general.file_type", "")

    print("=" * 70)
    print(f"P11 MedGemma Extraction Test")
    print(f"tier={a.tier}  weights={tier['hf']}  served-as={model}  backend={a.backend}")
    if a.backend == "hf":
        print(f"hf_quant={a.quant}")
    print(f"notes={len(notes)}  outcomes={','.join(outcomes)}  repeat={a.repeat}  "
          f"concurrency={a.concurrency}")
    print(f"tier_note={tier['note']}")
    print("=" * 70)

    if a.concurrency > 1 and a.repeat > 1:
        print()
        print("!! CONCURRENCY WARNING - run-to-run consistency is confounded.")
        print(f"   At --concurrency {a.concurrency} the server batches requests, and a batch's")
        print("   composition depends on timing, so it differs between repeats. Batched float")
        print("   reductions are not bit-identical, so a token can flip at temperature 0 for")
        print("   reasons that have nothing to do with the model. Mismatches below are then")
        print("   'model nondeterminism OR batching', and you cannot tell which.")
        print("   Measure consistency with --concurrency 1. Use >1 for throughput and cost.")
        print()

    runs, tallies = [], []
    for i in range(a.repeat):
        t = Tally()
        t_start = time.time()
        try:
            runs.append(run(notes, outcomes, a.backend, model, a.host, t,
                            quant=a.quant, timeout=a.timeout, concurrency=a.concurrency))
        except (urllib.error.URLError, TimeoutError) as e:
            raise SystemExit(f"backend unreachable at {a.host}: {e}\n"
                             f"start it, or use --backend mock to exercise the harness")
        t.wall_clock_sec = time.time() - t_start
        tallies.append(t)
        rep = t.report()
        sec_per_note = t.wall_clock_sec / len(notes) if notes else 0
        tok_per_sec = t.completion_tokens / t.wall_clock_sec if t.wall_clock_sec > 0 else 0
        print(f"\nRun {i+1} completed in {t.wall_clock_sec:.1f}s ({sec_per_note:.2f}s/note, {tok_per_sec:.1f} tok/s):")
        print(f"  Proposed findings:    {rep['proposed']}")
        print(f"    null placeholders:  {rep['null_placeholder']:4d}  {rep['null_placeholder_pct']:5.1f}%  (no quote -> prompt not followed)")
        print(f"    quote verified:     {t.quote_ok:4d}  {rep['quote_verified_pct_of_quoted']:5.1f}% of quoted | {rep['quote_verified_pct']:.1f}% of all")
        print(f"    quote not in note:  {t.quote_unfound:4d}  {rep['hallucinated_pct_of_quoted']:5.1f}% of quoted | {rep['hallucinated_quote_pct']:.1f}% of all")
        print(f"  Accepted findings:    {rep['accepted']}")
        print(f"  Invalid values:       {rep['invalid_value']}")
        if rep['tokenizer_artifacts']:
            print(f"  !! TOKENIZER ARTIFACTS: {rep['tokenizer_artifacts']} quotes carry corrupt GGUF byte tokens.")
            print(f"     The served weights are broken; these numbers are not a clean measurement.")
        print(f"  Unparseable replies:  {rep['unparseable_replies']}")
        print(f"  Prompt tokens:        {rep['prompt_tokens']} (~{rep['prompt_tokens']//len(notes)} tok/note)")
        print(f"  Completion tokens:    {rep['completion_tokens']} (~{rep['completion_tokens']//len(notes)} tok/note)")

    # grades, from the run-1 features through the real decision tables
    statuses = Counter()
    by_outcome = {num: Counter() for num in outcomes}
    by_selection = {"seeded": Counter(), "holdout": Counter(), "random": Counter()}
    grade_results_detail = {}
    for uid, per in runs[0].items():
        grade_results_detail[uid] = {}
        for num, (feats, present, *_) in per.items():
            res = grade(num, feats, present=bool(present))
            statuses[res.status] += 1
            by_outcome[num][res.status] += 1
            by_selection[selection[uid].split(":")[0]][res.status] += 1
            grade_results_detail[uid][num] = {
                "status": res.status,
                "grade": res.grade,
                "features": feats,
                "present": present,
                "reason": res.reason,
            }

    print(f"\nGrade status over {len(notes)}x{len(outcomes)} note-outcome pairs:")
    for k, v in statuses.most_common():
        print(f"   {k:14s} {v:3d} ({100*v/(len(notes)*len(outcomes)):.1f}%)")

    cols = ["graded", "grade_set", "cannot_grade", "absent", "not_applicable"]
    print(f"\nPer outcome (n={len(notes)} each) - a pooled number hides this shape:")
    print(f"   {'':>3s} {'outcome':30s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for num in outcomes:
        c = by_outcome[num]
        print(f"   {num:>3s} {TABLES[num].name[:30]:30s} "
              + " ".join(f"{c.get(col, 0):>12d}" for col in cols))

    print("\nSeeded vs unstratified holdout - the size of the selection bias:")
    for k, c in by_selection.items():
        if sum(c.values()):
            print(f"   {k:8s} n={sum(c.values()):3d}  " + "  ".join(
                f"{col}={c.get(col, 0)}" for col in cols if c.get(col)))

    consistency_pct = 100.0
    if a.repeat > 1:
        same = tot = 0
        for uid in runs[0]:
            for num in outcomes:
                tot += 1
                same += runs[0][uid][num][0] == runs[1][uid][num][0]
        consistency_pct = round(100 * same / tot, 1)
        print(f"\nTemperature-0 consistency across runs 1-{a.repeat}: {consistency_pct}% "
              f"({same}/{tot} note-outcome pairs identical)")

    # Threshold assessment
    rep0 = tallies[0].report()
    print("\n" + "=" * 70)
    print("Automated Metrics Evaluation (Tasks/medgemma_extraction_test.md §Automated metrics):")
    def band(v, good, workable):
        return f"GOOD (≥{good}%)" if v >= good else (f"WORKABLE ({workable}-{good}%)" if v >= workable else f"CONCERNING (<{workable}%)")
    qv_quoted = rep0["quote_verified_pct_of_quoted"]
    qv_status = band(qv_quoted, 95, 85)
    cs_status = "GOOD (≥98%)" if consistency_pct >= 98 else ("WORKABLE (90-98%)" if consistency_pct >= 90 else "CONCERNING (<90%)")
    iv_pct = 100 * rep0["invalid_value"] / (rep0["proposed"] or 1)
    iv_status = "GOOD (≤2%)" if iv_pct <= 2 else ("WORKABLE (2-10%)" if iv_pct <= 10 else "CONCERNING (>10%)")
    print(f"  - Quote-verified % (of quoted proposals):  {qv_quoted}% -> {qv_status}")
    print(f"  - Quote-verified % (of ALL proposals):     {rep0['quote_verified_pct']}%")
    print(f"  - Null-placeholder rate:   {rep0['null_placeholder_pct']}% -> "
          f"{'GOOD (≤5%)' if rep0['null_placeholder_pct'] <= 5 else 'CONCERNING - the prompt omission rule is being ignored'}")
    print(f"  - Run-to-run consistency:  {consistency_pct}% -> {cs_status}")
    print(f"  - Invalid-value rate:      {iv_pct:.1f}% -> {iv_status}")
    print(f"  - Unparseable replies:     {rep0['unparseable_replies']} -> {'GOOD (0)' if rep0['unparseable_replies']==0 else 'CONCERNING'}")
    print("=" * 70)

    if a.out:
        out_path = pathlib.Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_records = []
        for rec in notes:
            uid = rec["patient_uid"]
            per_outcome_details = {}
            for num in outcomes:
                feats, present, reply = runs[0][uid][num]
                per_outcome_details[num] = {
                    "outcome_name": TABLES[num].name,
                    "present": present,
                    "extracted_features": feats,
                    "grade_result": grade_results_detail[uid][num],
                    "raw_reply": reply,
                }
            detailed_records.append({
                "patient_uid": uid,
                "selection": selection[uid],          # seeded:<outcome> | holdout | random
                "scd_primary": is_scd_primary(rec),   # a label now, not a filter
                "title": rec.get("title", ""),
                "age": rec.get("age"),
                "gender": rec.get("gender"),
                "patient_note": rec["patient"],
                "outcomes": per_outcome_details,
            })

        out_data = {
            "provenance": {
                "tier": a.tier,
                "weights": tier["hf"],
                "served_as": model,
                "backend": a.backend,
                "quant": a.quant if a.backend == "hf" else None,
                "model_digest": model_digest,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "notes_count": len(notes),
                "cohort": a.cohort,
                "stratified": a.stratify,
                "holdout_frac": a.holdout_frac if a.stratify else None,
                "outcomes": outcomes,
                "repeat": a.repeat,
                "concurrency": a.concurrency,
                # True when consistency was measured under batching, which can flip a
                # token at temperature 0 for reasons unrelated to the model.
                "consistency_confounded_by_batching": a.concurrency > 1 and a.repeat > 1,
            },
            "profiling": {
                "total_wall_clock_sec": round(tallies[0].wall_clock_sec, 2),
                "sec_per_note": round(tallies[0].wall_clock_sec / len(notes), 2),
                "total_prompt_tokens": tallies[0].prompt_tokens,
                "total_completion_tokens": tallies[0].completion_tokens,
                "prompt_tokens_per_note": tallies[0].prompt_tokens // len(notes),
                "completion_tokens_per_note": tallies[0].completion_tokens // len(notes),
                "completion_tokens_per_sec": round(tallies[0].completion_tokens / tallies[0].wall_clock_sec, 1) if tallies[0].wall_clock_sec > 0 else 0,
            },
            "automated_metrics": {
                "proposed": rep0["proposed"],
                "quoted": rep0["quoted"],
                "quote_verified": rep0["quote_verified"],
                "quote_unfound": rep0["quote_unfound"],
                "null_placeholder": rep0["null_placeholder"],
                "null_placeholder_pct": rep0["null_placeholder_pct"],
                "quote_verified_pct": rep0["quote_verified_pct"],
                "quote_verified_pct_of_quoted": rep0["quote_verified_pct_of_quoted"],
                "hallucinated_quote_pct": rep0["hallucinated_quote_pct"],
                "hallucinated_pct_of_quoted": rep0["hallucinated_pct_of_quoted"],
                "tokenizer_artifacts": rep0["tokenizer_artifacts"],
                "invalid_value_count": rep0["invalid_value"],
                "invalid_value_pct": round(iv_pct, 1),
                "unparseable_replies": rep0["unparseable_replies"],
                "run_to_run_consistency_pct": consistency_pct,
            },
            "runs": [t.report() for t in tallies],
            "grade_status": dict(statuses),
            "grade_status_by_outcome": {k: dict(v) for k, v in by_outcome.items()},
            "grade_status_by_selection": {k: dict(v) for k, v in by_selection.items() if sum(v.values())},
            "features_extracted": dict(tallies[0].per_feature),
            "detailed_records": detailed_records,
        }
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nWrote full test results and note texts to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
