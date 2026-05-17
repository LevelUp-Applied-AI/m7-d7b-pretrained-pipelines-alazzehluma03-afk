"""
Autograder tests for Module 7 Week B — Core Skills Drill.

Uses tiny test models for pipeline tests (avoids 250 MB download in CI for
each run). Tasks 2/3/5 run against deterministic string fixtures — no network.
"""

import math
import os
import sys

import pytest

# Add starter dir to sys.path (tests run from repo root after Classroom flatten)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import drill  # noqa: E402


# -- Task 2: normalize_answer + exact_match -----------------------------------

def test_normalize_answer_lowercases_strips_articles_punct():
    assert drill.normalize_answer("The Eiffel Tower") == "eiffel tower"


def test_normalize_answer_strips_articles_with_word_boundaries():
    # "thereby" should NOT have "the" stripped from inside it
    out = drill.normalize_answer("Thereby a fact")
    assert "thereby" in out
    assert " a " not in f" {out} "  # standalone "a" is stripped


def test_normalize_answer_collapses_whitespace():
    assert drill.normalize_answer("  hello   world  ") == "hello world"


def test_normalize_answer_strips_punctuation():
    assert drill.normalize_answer("Hello, world!") == "hello world"


def test_exact_match_handles_punctuation_and_articles():
    assert drill.exact_match("The Eiffel Tower!", "eiffel tower") == 1
    assert drill.exact_match("Eiffel Tower", "Statue of Liberty") == 0


# -- Task 3: token_f1 --------------------------------------------------------

def test_token_f1_perfect_match_is_one():
    assert drill.token_f1("the quick brown fox", "the quick brown fox") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert drill.token_f1("apple banana", "carrot date") == 0.0


def test_token_f1_partial_overlap_known_value():
    # "cat sat mat" (3 tokens) vs "cat sat on mat" (4 tokens) after normalization.
    # "on" is not an article, so it stays.
    # overlap = {cat, sat, mat} = 3
    # precision = 3/3 = 1.0, recall = 3/4 = 0.75
    # f1 = 2 * 1.0 * 0.75 / (1.0 + 0.75) = 6/7
    f1 = drill.token_f1("cat sat mat", "cat sat on mat")
    assert math.isclose(f1, 6 / 7, rel_tol=1e-6)


def test_token_f1_handles_empty_prediction():
    out = drill.token_f1("", "the answer")
    assert out == 0.0
    assert not math.isnan(out)


def test_token_f1_both_empty_is_one():
    assert drill.token_f1("", "") == 1.0


# -- Tasks 1, 4: pipeline construction (uses small public model) -------------
# These tests rely on pipeline construction succeeding. They DO download the
# model on first CI run; subsequent runs use the pip cache. Pinned to small
# models for CI speed.

@pytest.fixture(scope="module")
def qa_pipeline():
    """Construct a real QA pipeline once per test module."""
    return drill.build_qa_pipeline(drill._qa_model_name())


def test_build_qa_pipeline_returns_callable(qa_pipeline):
    assert callable(qa_pipeline)


def test_answer_one_returns_dict_with_required_keys(qa_pipeline):
    result = drill.answer_one(
        qa_pipeline,
        question="Where was Alan Turing born?",
        context="Alan Turing was born in Maida Vale, London.",
    )
    assert isinstance(result, dict)
    for key in ("answer", "score", "start", "end"):
        assert key in result, f"missing key: {key}"
    # The answer should be a non-empty string (the pipeline always returns a span)
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    # start/end are character offsets — required for downstream highlighting
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)


@pytest.fixture(scope="module")
def summ_pipeline():
    return drill.build_summarizer(drill._summ_model_name())


def test_build_summarizer_returns_callable(summ_pipeline):
    assert callable(summ_pipeline)


def test_summarize_one_returns_string(summ_pipeline):
    text = (
        "Researchers at a major university reported a new battery technology "
        "achieving 1,000 charge cycles with under 5% capacity loss. The team "
        "plans to license the technology for commercial trials next year. "
        "The breakthrough could enable longer-range and faster-charging "
        "electric vehicles."
    )
    out = drill.summarize_one(summ_pipeline, text, max_length=40, min_length=10)
    assert isinstance(out, str)
    assert len(out) > 0


# -- Task 5: compute_rouge ---------------------------------------------------

def test_compute_rouge_returns_dict_with_three_keys():
    out = drill.compute_rouge("hello world", "hello world")
    assert set(out.keys()) == {"rouge1", "rouge2", "rougeL"}


def test_compute_rouge_values_in_unit_interval():
    out = drill.compute_rouge("the quick brown fox jumps", "a brown fox jumps high")
    for k, v in out.items():
        assert 0.0 <= v <= 1.0, f"{k} = {v} not in [0, 1]"


def test_compute_rouge_perfect_match_is_one():
    out = drill.compute_rouge("the quick brown fox", "the quick brown fox")
    assert math.isclose(out["rouge1"], 1.0, rel_tol=1e-6)
    assert math.isclose(out["rouge2"], 1.0, rel_tol=1e-6)
    assert math.isclose(out["rougeL"], 1.0, rel_tol=1e-6)
