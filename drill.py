"""
Module 7 Week B — Core Skills Drill: Pre-Trained Pipelines & Metrics.

Implement the functions below. See the drill guide for full task descriptions.
"""

import os
import re
import string
from transformers import pipeline
from rouge_score.rouge_scorer import RougeScorer


# -- Helpers (provided — do NOT modify) --------------------------------------

def _qa_model_name() -> str:
    """Return env override (CI smoke) or the default QA model."""
    return os.environ.get("QA_MODEL_FOR_CI", "distilbert-base-cased-distilled-squad")


def _summ_model_name() -> str:
    """Return env override (CI smoke) or the default summarization model."""
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


# -- Task 1: QA pipeline ------------------------------------------------------

def build_qa_pipeline(model_name: str):
    """
    Construct a Hugging Face question-answering pipeline.

    Returns the pipeline object (callable).
    """
    #  build a question-answering pipeline using the given model name (see reading § 3)
    return pipeline("question-answering", model=model_name)


def answer_one(qa, question: str, context: str) -> dict:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the pipeline output dict with keys "answer", "score", "start", "end".
    """
    # call qa(question=..., context=...) and return the result
    result = qa(question=question, context=context)
    return result


# -- Task 2: Normalization + EM ----------------------------------------------

def normalize_answer(s: str) -> str:
    """
    SQuAD-style normalization.

    Apply (in order):
      - lowercase
      - strip standalone articles (a, an, the) using word-boundary regex
      - strip all string.punctuation
      - collapse whitespace
    """
    #  apply the four normalization steps in order; word-boundary regex is required for the article strip
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> int:
    """
    Return 1 if normalized prediction equals normalized gold, else 0.
    """
    # normalize both, compare, return int
    return int(normalize_answer(pred) == normalize_answer(gold))
    


# -- Task 3: Token-F1 --------------------------------------------------------

def token_f1(pred: str, gold: str) -> float:
    """
    Compute token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty (after normalization) -> 1.0
      - one empty -> 0.0
    Returns a float in [0.0, 1.0]. Never returns NaN.
    """
    #  normalize both, split on whitespace
    #  handle empty cases
    #  compute multiset overlap, precision, recall, harmonic mean
    p_norm = normalize_answer(pred)
    g_norm = normalize_answer(gold)
    
    if not p_norm and not g_norm:
        return 1.0
    if not p_norm or not g_norm:
        return 0.0
    
    pred_tokens = p_norm.split()
    gold_tokens = g_norm.split()
    
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) 
                 for t in set(pred_tokens))
    
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# -- Task 4: Summarization pipeline ------------------------------------------

def build_summarizer(model_name: str):
    """
    Construct a Hugging Face summarization pipeline.

    Returns the pipeline object (callable).
    """
    # build a summarization pipeline using the given model name (see reading § 6)
    return pipeline("summarization", model=model_name)

def summarize_one(summ, text: str, max_length: int, min_length: int) -> str:
    """
    Run the summarization pipeline on one document.

    Use do_sample=False, num_beams=4. Return the summary_text string from the
    first output element (the pipeline returns a list-of-dicts).
    """
    #  invoke the pipeline with deterministic generation parameters and return the summary string
    #       (the pipeline returns a list-of-dicts — see reading § 6 for the output shape)
    out = summ(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3   
    )
    
    return out[0]["summary_text"]


# -- Task 5: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 between predicted and reference.

    Use rouge_score.rouge_scorer.RougeScorer with use_stemmer=True.
    Argument order is scorer.score(reference, predicted) — reference FIRST.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float}, all F1.
    """
    #  build a stemming-enabled ROUGE scorer over the three metric variants
    # score the (reference, predicted) pair (mind the argument order) and return F1 measures only    
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)          
    
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


if __name__ == "__main__":
    # Minimal smoke when run directly: tasks 2/3/5 don't need network.
    sample_norm = normalize_answer("The Eiffel Tower")
    print("normalize_answer('The Eiffel Tower') ->", repr(sample_norm))
    print("exact_match('The Eiffel Tower', 'eiffel tower') ->", exact_match("The Eiffel Tower", "eiffel tower"))
    print("token_f1('the cat sat on the mat', 'cat sat on mat') ->", token_f1("the cat sat on the mat", "cat sat on mat"))
    print("compute_rouge('a b c d', 'a b c d') ->", compute_rouge("a b c d", "a b c d"))
