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
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
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
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json",
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
    without a GPU. Emits one findable quote and one deliberate hallucination."""
    m = re.search(r'NOTE:\n"""(.*)"""', prompt, re.S)
    note = m.group(1) if m else ""
    first = next((w for w in re.findall(r"[A-Za-z]{6,}", note)), "unknown")
    if stats is not None:
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + 100
        stats["eval_count"] = stats.get("eval_count", 0) + 50
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + 0.01
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + 0.01
    return json.dumps({"present": True, "findings": [
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

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


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
    per_feature: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_sec: float = 0.0

    def report(self):
        p = self.proposed or 1
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "quote_verified_pct": round(100 * self.quote_ok / p, 1),
            "hallucinated_quote_pct": round(100 * self.quote_unfound / p, 1),
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

def load_notes(n: int, seed: int = 20260828) -> list[dict]:
    import random
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
    return random.Random(seed).sample(scd, min(n, len(scd)))


def run(notes, outcomes, backend, model, host, tally, quant="none", timeout=300):
    results = {}
    stats = {"prompt_eval_count": 0, "eval_count": 0, "eval_duration_sec": 0.0, "total_duration_sec": 0.0}
    t0 = time.time()
    total_notes = len(notes)
    for idx, rec in enumerate(notes):
        note = rec["patient"]
        per_outcome = {}
        for num in outcomes:
            print(f"  [{idx+1}/{total_notes}] processing UID {rec['patient_uid']} outcome {num} ({TABLES[num].name})...", flush=True)
            prompt = build_prompt(note, num)
            reply = BACKENDS[backend](prompt, model, host, stats=stats, quant=quant, timeout=timeout)
            feats, present = verify(reply, note, tally)
            per_outcome[num] = (feats, present, reply)
        results[rec["patient_uid"]] = per_outcome
    tally.prompt_tokens += stats["prompt_eval_count"]
    tally.completion_tokens += stats["eval_count"]
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
    ap.add_argument("--outcomes", default="28,48,36,19",
                    help="comma-separated; default is a common-outcome sample")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and report run-to-run consistency at temperature 0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

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

    notes = load_notes(a.notes)
    model_info = get_model_info(model, a.host) if a.backend == "ollama" else {}
    model_digest = model_info.get("details", {}).get("parent_model", "") or model_info.get("model_info", {}).get("general.file_type", "")

    print("=" * 70)
    print(f"P11 MedGemma Extraction Test")
    print(f"tier={a.tier}  weights={tier['hf']}  served-as={model}  backend={a.backend}")
    if a.backend == "hf":
        print(f"hf_quant={a.quant}")
    print(f"notes={len(notes)}  outcomes={','.join(outcomes)}  repeat={a.repeat}")
    print(f"tier_note={tier['note']}")
    print("=" * 70)

    runs, tallies = [], []
    for i in range(a.repeat):
        t = Tally()
        t_start = time.time()
        try:
            runs.append(run(notes, outcomes, a.backend, model, a.host, t, quant=a.quant, timeout=a.timeout))
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
        print(f"  Accepted findings:    {rep['accepted']} ({rep['quote_verified_pct']}% quote-verified)")
        print(f"  Hallucinated quotes:  {rep['hallucinated_quote_pct']}% ({t.quote_unfound})")
        print(f"  Invalid values:       {rep['invalid_value']}")
        print(f"  Unparseable replies:  {rep['unparseable_replies']}")
        print(f"  Prompt tokens:        {rep['prompt_tokens']} (~{rep['prompt_tokens']//len(notes)} tok/note)")
        print(f"  Completion tokens:    {rep['completion_tokens']} (~{rep['completion_tokens']//len(notes)} tok/note)")

    # grades, from the run-1 features through the real decision tables
    statuses = Counter()
    grade_results_detail = {}
    for uid, per in runs[0].items():
        grade_results_detail[uid] = {}
        for num, (feats, present, *_) in per.items():
            res = grade(num, feats, present=bool(present))
            statuses[res.status] += 1
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
    qv_status = "GOOD (≥95%)" if rep0["quote_verified_pct"] >= 95 else ("WORKABLE (85-95%)" if rep0["quote_verified_pct"] >= 85 else "CONCERNING (<85%)")
    cs_status = "GOOD (≥98%)" if consistency_pct >= 98 else ("WORKABLE (90-98%)" if consistency_pct >= 90 else "CONCERNING (<90%)")
    iv_pct = 100 * rep0["invalid_value"] / (rep0["proposed"] or 1)
    iv_status = "GOOD (≤2%)" if iv_pct <= 2 else ("WORKABLE (2-10%)" if iv_pct <= 10 else "CONCERNING (>10%)")
    print(f"  - Quote-verified %:        {rep0['quote_verified_pct']}% -> {qv_status}")
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
                "outcomes": outcomes,
                "repeat": a.repeat,
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
                "quote_verified_pct": rep0["quote_verified_pct"],
                "hallucinated_quote_pct": rep0["hallucinated_quote_pct"],
                "invalid_value_count": rep0["invalid_value"],
                "invalid_value_pct": round(iv_pct, 1),
                "unparseable_replies": rep0["unparseable_replies"],
                "run_to_run_consistency_pct": consistency_pct,
            },
            "runs": [t.report() for t in tallies],
            "grade_status": dict(statuses),
            "features_extracted": dict(tallies[0].per_feature),
            "detailed_records": detailed_records,
        }
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nWrote full test results and note texts to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
