"""Produce the (N, K) probability matrix P for MCQ datasets.

Three routes:
  - hf / letter: one forward pass per question; read next-token logits over
    the option letters A, B, ... Fast and exact, but requires each letter to
    be a single token for the model's tokeniser (verified, with automatic
    fallback to cloze when it is not — mandatory check for K > 4).
  - hf / cloze: length-normalised log-likelihood of each full option text.
    Slower, model-agnostic fallback.
  - api: OpenAI-compatible endpoint with logprobs over letter tokens.

K (the number of options) is fixed per dataset and comes from the loader,
never inferred here. All results (P, logits, labels, qids) are cached to
outputs/cache/ as .npz; re-runs never touch the model.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import MCQItem


def option_letters(k: int) -> list[str]:
    """['A', 'B', ...] for K options (K <= 26 by construction)."""
    assert 2 <= k <= 26, f"K must be in [2, 26], got {k}"
    return [chr(ord("A") + i) for i in range(k)]


def build_prompt(item: MCQItem) -> str:
    """Render one question into the lettered-options prompt ending in 'Answer:'."""
    letters = option_letters(len(item.options))
    lines = "\n".join(f"{l}. {opt}" for l, opt in zip(letters, item.options))
    return (
        "The following is a multiple choice question. "
        "Answer with the letter of the correct option.\n\n"
        f"Question: {item.question}\n{lines}\nAnswer:"
    )


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

def _resolve_letter_ids(tokenizer, k: int) -> list[int]:
    """Token ids for the K option letters as they appear after 'Answer:'.

    Prefers the leading-space variant (' A'), falling back to bare 'A'.
    Fails loudly if a letter is not a single token, suggesting cloze mode.
    """
    ids = []
    for letter in option_letters(k):
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
    assert len(set(ids)) == k, "letter token ids must be distinct"
    return ids


def letters_are_single_token(tokenizer, k: int) -> bool:
    """True iff every option letter resolves to a single token."""
    try:
        _resolve_letter_ids(tokenizer, k)
        return True
    except ValueError:
        return False


def choose_score_mode(tokenizer, k: int, requested: str = "letter") -> str:
    """Pick the effective scoring mode for this tokeniser and K.

    Letter-logit scoring is only sound if every option letter is a single
    token. That is checked against the *actual* tokeniser — mandatory for
    K > 4 (letters beyond D are less likely to be single tokens), and applied
    to all K so a broken tokeniser degrades to cloze instead of erroring.
    """
    if requested == "cloze":
        return "cloze"
    if requested != "letter":
        raise ValueError(f"unknown score_mode: {requested!r} (use 'letter' or 'cloze')")
    if letters_are_single_token(tokenizer, k):
        return "letter"
    print(
        f"[scoring] letter mode requested but not all of {option_letters(k)} are "
        "single tokens for this tokeniser; falling back to cloze scoring."
    )
    return "cloze"


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


def _score_hf_letter(items: list[MCQItem], model_name: str, k: int, batch_size: int = 8) -> np.ndarray:
    """(N, K) raw logits over the letter tokens, one forward pass per batch."""
    import torch

    model, tokenizer, device = _hf_load(model_name)
    letter_ids = _resolve_letter_ids(tokenizer, k)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keep 'Answer:' adjacent to the next token

    logits_out = np.zeros((len(items), k), dtype=np.float64)
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


def _score_hf_cloze(items: list[MCQItem], model_name: str, k: int) -> np.ndarray:
    """(N, K) length-normalised option log-likelihoods (act as logits)."""
    import torch

    model, tokenizer, device = _hf_load(model_name)
    scores = np.zeros((len(items), k), dtype=np.float64)
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

def _score_api(items: list[MCQItem], api_model: str, k: int) -> np.ndarray:
    """(N, K) letter logprobs from an OpenAI-compatible endpoint.

    Reads OPENAI_API_KEY (required) and OPENAI_BASE_URL (optional) from the
    environment. Letters missing from top_logprobs get a floor logprob.
    """
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the environment for the api route.")
    client = OpenAI()  # honours OPENAI_BASE_URL if set

    letters = option_letters(k)
    logits = np.full((len(items), k), -20.0, dtype=np.float64)
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
            if token in letters:
                j = letters.index(token)
                logits[i, j] = max(logits[i, j], entry.logprob)
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
    n_options: int = 4,
    cache_dir: str | Path = "outputs/cache",
) -> dict:
    """Score one split, cache-aware. Returns {P, logits, labels, qids}.

    P is the softmax over the K per-option logits; logits are kept raw for
    temperature scaling. `n_options` (K) comes from the loader — it is fixed
    per dataset by contract and enforced here, never inferred.
    """
    from .calibration import softmax

    bad = [it.qid for it in items if len(it.options) != n_options]
    assert not bad, (
        f"K is fixed per dataset: expected {n_options} options, but items "
        f"{bad[:5]} differ (dataset {dataset!r})"
    )

    model_name = api_model if model_route == "api" else hf_model
    mode = "api" if model_route == "api" else score_mode
    path = cache_path(cache_dir, model_name, dataset, split, mode)

    cached = _load_cache(path)
    if cached is not None:
        return cached

    if model_route == "api":
        logits = _score_api(items, api_model, n_options)
    else:
        # verify letter single-token-ness against the actual tokeniser before
        # committing to a mode (falls back to cloze; mandatory check for K > 4)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        mode = choose_score_mode(tokenizer, n_options, score_mode)
        if mode != score_mode:
            path = cache_path(cache_dir, model_name, dataset, split, mode)
            cached = _load_cache(path)
            if cached is not None:
                return cached
        if mode == "letter":
            logits = _score_hf_letter(items, hf_model, n_options)
        else:
            logits = _score_hf_cloze(items, hf_model, n_options)

    labels = np.array([it.label_idx for it in items], dtype=np.int64)
    qids = [it.qid for it in items]
    P = softmax(logits)
    assert P.shape == (len(items), n_options)
    _save_cache(path, P, logits, labels, qids)
    return {"P": P, "logits": logits, "labels": labels, "qids": np.array(qids)}
