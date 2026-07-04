# uncertainty-gate

**Claim:** a model that attaches *calibrated, guarantee-backed* uncertainty to its answers — and abstains when unsure — becomes dramatically more reliable on the answers it does give. This repo proves it with one chart: the risk–coverage curve for Qwen2.5-3B-Instruct on ARC-Challenge, where **at 70% coverage, error drops from 16.6% (answer everything) to 4.7% (calibrated confidence gate)** — a 3.5× reduction — backed by a distribution-free conformal coverage guarantee that is verified empirically in-repo. Temperature scaling cuts the model's ECE from 0.120 to 0.032 (T = 2.69: the raw model is heavily overconfident).

![risk-coverage curve](outputs/risk_coverage.png)

![reliability diagram](outputs/reliability.png)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-min.txt for the LLM-free tabular spine
python -m src.pipeline                 # figures + JSON land in outputs/
pytest                                 # the guarantee, tested
```

First run on `dataset: arc` downloads Qwen2.5-3B-Instruct and scores ~1.5k questions (cached to `outputs/cache/`; re-runs never touch the model). With `dataset: tabular` the whole pipeline runs in under a minute with no torch/transformers at all.

Artefacts written to `outputs/`:

| file | what it shows |
|---|---|
| `reliability.png` | reliability diagram + ECE, before/after temperature scaling |
| `coverage_check.json` | empirical conformal coverage ≈ 1 − α for LAC and APS |
| `risk_coverage.png` | risk vs coverage, raw vs calibrated confidence, AURC |
| `caught_examples.md` | confidently-wrong answers the layer flagged |
| `metrics.json` | every number above, machine-readable |

## How the guarantee works

Split conformal prediction: score every calibration point by how "nonconforming" the true label is, take the finite-sample-corrected (1 − α) quantile of those scores, and include in each test prediction set every label that scores below it — if calibration and test data are exchangeable, the set contains the true label with probability ≥ 1 − α, *no matter how wrong the model's probabilities are*. The entire construction is ~30 readable lines in [`src/conformal.py`](src/conformal.py), and [`tests/test_conformal.py`](tests/test_conformal.py) is the proof: over hundreds of random splits of deliberately miscalibrated synthetic data, mean empirical coverage lands on 1 − α to within Monte-Carlo noise.

## Config (`config.yaml`)

| key | default | meaning |
|---|---|---|
| `dataset` | `arc` | `arc` \| `mmlu` \| `tabular` (Covertype spine, no LLM) \| `csv` (bring your own) |
| `csv_path` | `null` | path to a contract-conforming CSV, required when `dataset: csv` |
| `answer_format` | `auto` | CSV answer encoding: `auto` \| `letter` \| `index0` \| `index1` |
| `model_route` | `hf` | `hf` (local transformers) \| `api` (OpenAI-compatible logprobs endpoint) |
| `hf_model` | `Qwen/Qwen2.5-3B-Instruct` | any causal LM (letter scoring needs single-token option letters) |
| `score_mode` | `letter` | `letter` (next-token logits over the option letters) \| `cloze` (length-normalised option log-likelihood). Letter mode is verified against the tokeniser and falls back to cloze if any letter is not a single token |
| `alpha` | `[0.05, 0.10, 0.20]` | conformal miscoverage levels to sweep |
| `cal_frac` | `0.30` | calibration fraction when a dataset has no native split |
| `crosscheck` | `false` | verify the from-scratch conformal code against MAPIE |

## Bring your own data (CSV)

Any labelled multiple-choice dataset can be calibrated, not just ARC. The contract (full details in [`src/custom_csv.py`](src/custom_csv.py)): a header row with `question`, contiguous option columns `option_a, option_b, ...` (or `option_1, option_2, ...` — K ≥ 2, fixed for the whole file), and `answer` as a letter, 0-based, or 1-based index. The answer encoding is auto-detected but never guessed when ambiguous — 1-based-looking integer answers are rejected unless you pass `answer_format` explicitly.

```csv
question,option_a,option_b,option_c,option_d,answer
"Which gas do plants absorb?","oxygen","carbon dioxide","nitrogen","hydrogen",b
```

Validate a file standalone (all problems reported at once) before spending GPU time on it:

```bash
python -m src.custom_csv path/to/data.csv --answer-format auto
```

Then set `dataset: csv` and `csv_path` in `config.yaml` and run the pipeline as usual. Everything downstream of the probability matrix — temperature scaling, conformal sets, risk–coverage — is already K-agnostic, so K > 4 files work end-to-end (with automatic cloze fallback if the model's tokeniser doesn't encode the extra option letters as single tokens).

## Repo map

```
src/
  data.py         # ARC / MMLU / Covertype / custom-CSV loaders
  custom_csv.py   # bring-your-own CSV contract: validation + normalised records
  scoring.py      # (N, K) probability matrix from letter logits, cloze, or API — cached
  calibration.py  # temperature scaling, ECE, reliability data — from scratch
  conformal.py    # split conformal (LAC + APS), the whole guarantee
  selective.py    # risk-coverage, AURC, accuracy@coverage
  plots.py        # the two figures
  crosscheck.py   # optional MAPIE agreement check
  pipeline.py     # end-to-end orchestration
notebooks/demo.ipynb   # the same story, narrated
tests/                 # the claims, encoded as pytest
```

## Honest limitations

- The conformal guarantee is **marginal**: coverage ≥ 1 − α on average over the whole distribution, not per-topic or per-difficulty-slice (conditional coverage is impossible in general without assumptions).
- It rests on **exchangeability** of calibration and test data; under distribution shift the guarantee degrades.
- Correctness is **exact-match multiple choice** only — no free-form generation, no LLM judges. That is what makes every number in this repo unambiguous.
- With ARC's fixed calibration split of only 295 questions, per-run empirical coverage can fluctuate a few points around 1 − α (e.g. 0.76 at α = 0.20 in one run); the guarantee is marginal over calibration draws, which is exactly what `tests/test_conformal.py` verifies over hundreds of random splits.
