"""The MedGemma extraction check's verification layer - the §2 rule that turns a hallucinated quote
into a counted rejection rather than a silent wrong answer."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "experiments"))
from medgemma_extraction import (  # noqa: E402
    Tally, build_prompt, coerce, is_scd_primary, normalize, outcome_seed,
    call_mock, scd_mentions, select_notes, verify,
)

NOTE = ("A 14-year-old with HbSS presented with chest pain. FiO2 was escalated to 60%. "
        "He received a simple transfusion of 2 units and was started on norepinephrine.")


def run_verify(findings, present=True, note=NOTE):
    t = Tally()
    feats, p = verify(json.dumps({"present": present, "findings": findings}), note, t)
    return feats, p, t


# ------------------------------------------------------------ quote enforcement

def test_a_verbatim_quote_is_accepted():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                               "quote": "FiO2 was escalated to 60%"}])
    assert feats == {"fio2_pct": 60.0} and t.accepted == 1 and t.quote_ok == 1


def test_a_quote_not_in_the_note_is_rejected_not_downweighted():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 90,
                               "quote": "FiO2 was escalated to 90%"}])
    assert feats == {} and t.quote_unfound == 1 and t.accepted == 0


def test_a_finding_with_no_quote_is_rejected():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 60}])
    assert feats == {} and t.quote_missing == 1


def test_quote_matching_tolerates_whitespace_but_not_paraphrase():
    ok, _, _ = run_verify([{"feature": "fio2_pct", "value": 60,
                            "quote": "FiO2   was\n escalated to 60%"}])
    assert ok == {"fio2_pct": 60.0}
    bad, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                             "quote": "the FiO2 was raised to 60 percent"}])
    assert bad == {} and t.quote_unfound == 1


# --------------------------------------------------------------- value coercion

@pytest.mark.parametrize("value,expected", [
    (True, True), ("true", True), ("yes", True), (False, False), ("no", False),
])
def test_bool_coercion(value, expected):
    assert coerce("death_attributed", value) == (True, expected)


def test_bool_rejects_a_non_boolean():
    assert coerce("death_attributed", "probably") == (False, None)


def test_numeric_coercion_and_rejection():
    assert coerce("fio2_pct", "60") == (True, 60.0)
    assert coerce("fio2_pct", "sixty") == (False, None)


def test_enum_value_must_be_declared():
    assert coerce("transfusion_type", "exchange") == (True, "exchange")
    assert coerce("transfusion_type", "EXCHANGE") == (True, "exchange")
    assert coerce("transfusion_type", "double volume") == (False, None)


def test_an_invented_feature_name_is_rejected():
    feats, _, t = run_verify([{"feature": "vibes_score", "value": 3, "quote": "chest pain"}])
    assert feats == {} and t.unknown_feature == 1


def test_a_valid_quote_with_an_invalid_value_is_still_rejected():
    feats, _, t = run_verify([{"feature": "transfusion_type", "value": "megatransfusion",
                               "quote": "received a simple transfusion"}])
    assert feats == {} and t.quote_ok == 1 and t.value_bad == 1


def test_unparseable_reply_is_counted_not_raised():
    t = Tally()
    feats, present = verify("here is my answer, thanks!", NOTE, t)
    assert feats == {} and present is None and t.bad_json == 1


# ------------------------------------------------------------------ prompt shape

def test_prompt_asks_only_for_features_the_outcome_grades_on():
    prompt = build_prompt(NOTE, "36")            # Fever: temperature only
    assert '"temperature"' in prompt
    assert '"fio2_pct"' not in prompt


def test_prompt_omits_derived_features_the_model_cannot_observe():
    prompt = build_prompt(NOTE, "48")
    assert '"life_support"' not in prompt        # derived from its components
    assert '"vasopressors"' in prompt or "vasopressor" in prompt


def test_prompt_carries_the_outcome_specific_definition_of_treated():
    prompt = build_prompt(NOTE, "29")
    assert "For this outcome specifically" in prompt
    assert "splenectomy" in prompt


def test_prompt_forbids_grading():
    assert "NOT assigning a severity grade" in build_prompt(NOTE, "48")


# ------------------------------------------------------- reporting denominators

def _rec(title, body):
    return {"title": title, "patient": body}


def test_null_placeholders_are_kept_out_of_the_grounding_denominator():
    """A finding with no quote is a prompt-compliance failure, not a grounding
    result. Leaving it in the denominator lets a clean extractor look like a
    hallucinating one - which is exactly how the A100 run read as 7.4%."""
    feats, _, t = run_verify([
        {"feature": "fio2_pct", "value": 60, "quote": "FiO2 was escalated to 60%"},
        {"feature": "vasopressors", "value": None, "quote": None},
        {"feature": "renal_replacement", "value": None, "quote": None},
        {"feature": "temperature", "value": None, "quote": None},
    ])
    rep = t.report()
    assert rep["proposed"] == 4 and rep["quoted"] == 1
    assert rep["null_placeholder"] == 3 and rep["null_placeholder_pct"] == 75.0
    assert rep["quote_verified_pct_of_quoted"] == 100.0   # the model grounded what it quoted
    assert rep["quote_verified_pct"] == 25.0              # ...and ignored the omission rule


def test_both_grounding_denominators_are_always_reported():
    _, _, t = run_verify([{"feature": "fio2_pct", "value": 90,
                           "quote": "FiO2 was escalated to 90%"}])
    rep = t.report()
    assert rep["hallucinated_pct_of_quoted"] == 100.0 and rep["hallucinated_quote_pct"] == 100.0


def test_corrupt_gguf_byte_tokens_are_counted_not_silently_passed():
    _, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                           "quote": "FiO2[UNK_BYTE_0xe29681\u2581was]was escalated to 60%"}])
    assert t.tokenizer_artifacts == 1


# --------------------------------------------------------------- cohort gate

def test_a_negated_mention_is_not_a_case():
    assert not is_scd_primary(_rec("Cardiac arrest in an athlete",
                                   "The family denied a family history of SCD."))


def test_sickle_cell_trait_is_not_sickle_cell_disease():
    assert not is_scd_primary(_rec("Cerebellar edema after opioid ingestion",
                                   "A 25-month-old male with sickle cell trait was noted "
                                   "to be breathing heavily. Sickle cell trait is benign."))


def test_scd_alone_does_not_qualify_because_cardiology_means_sudden_cardiac_death():
    assert not is_scd_primary(_rec("Electrical storm in hypertrophic cardiomyopathy",
                                   "The ESC HCM Risk-SCD score was low. Risk of SCD at 5 "
                                   "years was 3%. An ICD prevents SCD."))


def test_a_mention_belonging_to_the_mother_is_not_the_patients_diagnosis():
    assert not is_scd_primary(_rec("Novel dystrophin variant",
                                   "The pregnancy was complicated by maternal sickle cell "
                                   "disease. Family history was significant for sickle cell disease."))


def test_a_title_match_qualifies():
    assert is_scd_primary(_rec("Intracardiac Thrombosis in Sickle Cell Disease", "..."))


def test_a_without_title_does_not_qualify():
    assert not is_scd_primary(_rec("Salmonella Osteomyelitis in a Child without Sickle Cell",
                                   "The child had osteomyelitis."))


def test_the_patients_own_repeated_diagnosis_qualifies_without_a_title_match():
    assert is_scd_primary(_rec("Rapid development of seizures and PRES in a COVID-19 patient",
                               "A 43-year-old female with sickle cell disease (SCD) and chronic "
                               "opioid use became gradually unresponsive. Her SCD was managed with "
                               "hydroxyurea."))


def test_trait_mentions_do_not_count_toward_the_threshold():
    assert scd_mentions("sickle cell trait was noted; sickle cell trait is benign") == 0


# ---------------------------------------------- note selection (present AND absent)

def _pool(n=200):
    """A synthetic pool where outcome prevalence is known by construction."""
    out = []
    for i in range(n):
        body = "A patient with sickle cell disease was admitted. "
        if i % 4 == 0:  body += "He was treated for a vaso-occlusive pain crisis. "
        if i % 5 == 0:  body += "Acute chest syndrome was diagnosed. "
        if i % 3 == 0:  body += "The patient was febrile on admission. "
        out.append({"patient_uid": f"u{i}", "title": "Case report", "patient": body})
    return out


def test_the_seed_falls_back_to_the_outcome_name():
    assert outcome_seed("24").search("presented with priapism")          # Priapism
    assert outcome_seed("40").search("a chronic leg ulcer")              # Leg Ulcer


def test_a_slash_separated_name_matches_either_side():
    pat = outcome_seed("18")   # Cholecystitis/Cholelithiasis (gallstones)
    assert pat.search("acute cholecystitis") and pat.search("cholelithiasis on ultrasound")


def test_an_abbreviation_in_the_name_is_usable_on_its_own():
    assert outcome_seed("13").search("developed PRES")   # ...Encephalopathy Syndrome (PRES)


def test_selection_reaches_every_target_outcome():
    """Random sampling yields 40 pain crises and zero leg ulcers (plan §7)."""
    _, sel = select_notes(_pool(), 20, ["28", "48", "36"])
    seeded = {v.split(":")[1] for v in sel.values() if v.startswith("seeded:")}
    assert seeded == {"28", "48", "36"}


def test_selection_keeps_an_unstratified_holdout_so_the_bias_is_measurable():
    _, sel = select_notes(_pool(), 20, ["28", "48"], holdout_frac=0.25)
    assert sum(1 for v in sel.values() if v == "holdout") == 5


def test_a_note_is_never_selected_twice():
    notes, sel = select_notes(_pool(), 24, ["28", "48", "36"])
    uids = [r["patient_uid"] for r in notes]
    assert len(uids) == len(set(uids)) == len(sel)


def test_stratification_can_be_turned_off_for_an_unbiased_sample():
    _, sel = select_notes(_pool(), 10, ["28"], stratify=False)
    assert set(sel.values()) == {"random"}


def test_the_pool_keeps_notes_where_outcomes_are_genuinely_absent():
    """`absent` is a first-class answer (plan §1) and the absence audit needs
    negatives, so selection must not quietly drop notes that lack an outcome."""
    pool = _pool()
    notes, _ = select_notes(pool, 20, ["28"])
    assert any("vaso-occlusive" not in r["patient"] for r in notes)


# ------------------------------------------------------------------ the mock

def test_the_mock_still_reports_exactly_half_its_quotes_verified():
    """The P11 protocol uses this as the harness's own smoke test."""
    t = Tally()
    verify(call_mock(build_prompt(NOTE, "48"), "", ""), NOTE, t)
    assert t.quote_ok == 1 and t.quote_unfound == 1


def test_the_mock_exercises_the_absent_path_too():
    """Without an absent branch the absence audit has nothing to run on, and a
    broken absent path ships unnoticed."""
    seen = set()
    for i in range(40):
        note = f"Case {i}: a patient with sickle cell disease was admitted."
        for num in ("28", "48", "36", "19"):
            seen.add(json.loads(call_mock(build_prompt(note, num), "", ""))["present"])
    assert seen == {True, False}
