"""Produce the (N, 4) probability matrix P for MCQ datasets.

Three routes:
  - hf / letter: one forward pass per question; read next-token logits over
    the option letters A-D. Fast and exact, but requires each letter to be a
    single token for the model's tokeniser (asserted loudly).
  - hf / cloze: length-normalised log-likelihood of each full option text.
    Slower, model-agnostic fallback.
  - api: OpenAI-compatible endpoint with logprobs over letter tokens.

All results (P, logits, labels, qids) are cached to outputs/cache/ as .npz;
re-runs never touch the model.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import MCQItem

_LETTERS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = (
    "The following is a multiple choice question. "
    "Answer with the letter of the correct option.\n\n"
    "Question: {question}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    "Answer:"
)


def build_prompt(item: MCQItem) -> str:
    """Render one question into the fixed A-D prompt ending in 'Answer:'."""
    a, b, c, d = item.options
    return PROMPT_TEMPLATE.format(question=item.question, a=a, b=b, c=c, d=d)


def cache_path(cache_dir: str | Path, model: str, dataset: str, split: str, mode: str) -> Path:
    """Deterministic cache file for a (model, dataset, split, mode) combo."""
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "-", model)
    return Path(cache_dir) / f"{safe_model}__{dataset}__{split}__{mode}.npz"


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {"P": z["P"], "logits": z["logits"], "labels": z["labels"], "qids": z["qids"]}


def _save_cache(path: Path, P: np.ndarray, logits: np.ndarray, labels: np.ndarray, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, P=P, logits=logits, labels=labels, qids=np.array(qids))


# ------------------------------------------------------------ HF route -----

def _resolve_letter_ids(tokenizer) -> list[int]:
    """Token ids for the four letters as they appear after 'Answer:'.

    Prefers the leading-space variant (' A'), falling back to bare 'A'.
    Fails loudly if a letter is not a single token, suggesting cloze mode.
    """
    ids = []
    for letter in _LETTERS:
        for variant in (f" {letter}", letter):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
                break
        else:
            raise ValueError(
                f"Neither ' {letter}' nor '{letter}' is a single token for this "
                "tokeniser; letter-logit scoring is unsound here. "
                "Set score_mode: cloze in config.yaml instead."
            )
    assert len(set(ids)) == 4, "letter token ids must be distinct"
    return ids


def _hf_load(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _score_hf_letter(items: list[MCQItem], model_name: str, batch_size: int = 8) -> np.ndarray:
    """(N, 4) raw logits over the letter tokens, one forward pass per batch."""
    import torch

    model, tokenizer, device = _hf_load(model_name)
    letter_ids = _resolve_letter_ids(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keep 'Answer:' adjacent to the next token

    logits_out = np.zeros((len(items), 4), dtype=np.float64)
    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc=f"scoring ({model_name})"):
            batch = items[start : start + batch_size]
            enc = tokenizer(
                [build_prompt(it) for it in batch], return_tensors="pt", padding=True
            ).to(device)
            out = model(**enc)
            next_logits = out.logits[:, -1, :].float().cpu().numpy()
            logits_out[start : start + len(batch)] = next_logits[:, letter_ids]
    return logits_out


def _score_hf_cloze(items: list[MCQItem], model_name: str) -> np.ndarray:
    """(N, 4) length-normalised option log-likelihoods (act as logits)."""
    import torch

    model, tokenizer, device = _hf_load(model_name)
    scores = np.zeros((len(items), 4), dtype=np.float64)
    with torch.no_grad():
        for i, item in enumerate(tqdm(items, desc=f"cloze scoring ({model_name})")):
            prompt_ids = tokenizer.encode(build_prompt(item), add_special_tokens=False)
            for k, option in enumerate(item.options):
                opt_ids = tokenizer.encode(" " + option, add_special_tokens=False)
                input_ids = torch.tensor([prompt_ids + opt_ids], device=device)
                logits = model(input_ids).logits[0].float()
                logprobs = torch.log_softmax(logits, dim=-1)
                total = sum(
                    logprobs[len(prompt_ids) + j - 1, tok].item()
                    for j, tok in enumerate(opt_ids)
                )
                scores[i, k] = total / len(opt_ids)
    return scores


# ----------------------------------------------------------- API route -----

def _score_api(items: list[MCQItem], api_model: str) -> np.ndarray:
    """(N, 4) letter logprobs from an OpenAI-compatible endpoint.

    Reads OPENAI_API_KEY (required) and OPENAI_BASE_URL (optional) from the
    environment. Letters missing from top_logprobs get a floor logprob.
    """
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment for the api route.")
    client = OpenAI()  # honours OPENAI_BASE_URL if set

    logits = np.full((len(items), 4), -20.0, dtype=np.float64)
    for i, item in enumerate(tqdm(items, desc=f"scoring ({api_model})")):
        resp = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": build_prompt(item)}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            temperature=0,
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        for entry in top:
            token = entry.token.strip()
            if token in _LETTERS:
                k = _LETTERS.index(token)
                logits[i, k] = max(logits[i, k], entry.logprob)
    return logits


# ----------------------------------------------------------- public API ----

def score_split(
    items: list[MCQItem],
    dataset: str,
    split: str,
    model_route: str = "hf",
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct",
    api_model: str | None = None,
    score_mode: str = "letter",
    cache_dir: str | Path = "outputs/cache",
) -> dict:
    """Score one split, cache-aware. Returns {P, logits, labels, qids}.

    P is the softmax over the four per-option logits; logits are kept raw
    for temperature scaling.
    """
    from .calibration import softmax

    model_name = api_model if model_route == "api" else hf_model
    mode = "api" if model_route == "api" else score_mode
    path = cache_path(cache_dir, model_name, dataset, split, mode)

    cached = _load_cache(path)
    if cached is not None:
        return cached

    if model_route == "api":
        logits = _score_api(items, api_model)
    elif score_mode == "letter":
        logits = _score_hf_letter(items, hf_model)
    elif score_mode == "cloze":
        logits = _score_hf_cloze(items, hf_model)
    else:
        raise ValueError(f"unknown score_mode: {score_mode!r}")

    labels = np.array([it.label_idx for it in items], dtype=np.int64)
    qids = [it.qid for it in items]
    P = softmax(logits)
    assert P.shape == (len(items), 4)
    _save_cache(path, P, logits, labels, qids)
    return {"P": P, "logits": logits, "labels": labels, "qids": np.array(qids)}
