"""The MedGemma extraction check's verification layer - the §2 rule that turns a hallucinated quote
into a counted rejection rather than a silent wrong answer."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "experiments"))
from medgemma_extraction import Tally, build_prompt, coerce, normalize, verify  # noqa: E402

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
