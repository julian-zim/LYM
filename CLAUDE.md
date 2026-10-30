# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

`LYM/` is the project root. Two upstream clones sit next to it in the parent folder `YGO/`, which is only needed by `convert.py` and `read.py`. Only `LYM/` is our own code:

- `CardScripts/` is a clone of `ProjectIgnis/CardScripts`, the EDOPro Lua card scripts. `official/c<id>.lua` holds one script per card; the `proc_*.lua` and `utility.lua` files are the shared libraries those scripts call.
- `BabelCDB/` is a clone of `ProjectIgnis/BabelCDB`, the SQLite card databases (`*.cdb`). The `texts` table has `id`, `name` and `desc`.
- `LYM/` is a Python project that fine-tunes Qwen2.5-Coder-1.5B-Instruct with QLoRA so it writes a card's EDOPro Lua script from the card's text.

Treat `CardScripts/` and `BabelCDB/` as read-only upstream data.

`LYM/` is a git repository (branch `main`, remote `origin` = private GitHub repo `julian-zim/LYM`). The venv `lym/`, `models/`, `data/`, `splits/` and `runs/` are gitignored. Split changes into logical commits with proper messages; push only when asked. The `pipeline-monitor` subagent lives in `.claude/agents/`.

## LYM commands

The virtualenv is `.venv/lym/` (Python 3.14). Run the scripts from `LYM/` or anywhere else. Every path comes from `LYM_DIR = Path(__file__).resolve().parents[1]` in `common.py`, and the upstream clones are found at `UPSTREAM_DIR = LYM_DIR.parent`, so `CardScripts/` and `BabelCDB/` must stay next to `LYM/`. Scripts import each other as `from common import ...`, which only works when they are run as files, not as modules.

```bash
source .venv/lym/bin/activate
pip install -r requirements.txt

python src/convert.py       # CardScripts/official + BabelCDB -> data/{id}_text.txt / {id}_code.txt
python src/split.py --k 5 --seed 42   # -> splits/folds.json
python src/fetch_model.py   # downloads the base model into models/
python src/train.py --fold 0          # train and evaluate one cross-validation round (cluster profile)
python src/train.py --profile local --fold 0   # the same with the RTX 2070 settings
python src/train.py --fold all        # full cross-validation, then writes runs/cv_summary.json
python src/train.py --fold 0 --eval-only
# Quick test of the whole pipeline (there is no test suite):
python src/train.py --fold 0 --limit-train 64 --limit-test 16 --gen-samples 4 --run-name smoke
python src/read.py <id|c<id>.lua>     # print one card's name and text from cards.cdb
```

`scripts/run.sh` holds the user's local training launch command (`--profile local`; run it from `LYM/` with the venv active).

On the VSC-5 cluster `scripts/train.slrm` (training, `sbatch scripts/train.slrm [fold|all]`) and `scripts/check_gpu.slrm` (GPU check) are submitted from `LYM/` and use paths relative to it. The cluster needs `CardScripts/` and `BabelCDB/` next to `LYM/` only for `convert.py`/`read.py`; copying `data/` and `splits/` over avoids that.

`train.py` needs a CUDA GPU. Run outputs go to `runs/[<run-name>_]fold_<i>/`: the adapter, `config.json`, `log.jsonl`, `metrics.json` and `predictions/`. On the RTX 2070 (`--profile local`) a fold takes 1330 optimizer steps at ~9-10 s each (~3.4 h), plus evaluation.

`convert.py` and `split.py` canonicalize every script (~0.05 s each) and therefore use all CPU cores via `multiprocessing`. Python 3.14 starts workers with forkserver, so multiprocessing code must live in a file with an `if __name__ == "__main__"` guard (it fails from stdin or `-c`).

## LYM architecture

The pipeline is convert → split → fetch_model → train. Each stage reads the previous stage's files on disk.

- **`common.py` is the single source of truth** for paths, the model ID and the prompt format (`SYSTEM_PROMPT`, `build_prompt_ids`, `build_example`), so change prompt formatting only here. It also has the text-based `strip_lua_comments`/`normalize_lua`, which are only a fallback now: `train.py` uses them to score generated scripts that cannot be parsed.
- **`canonical.py`** turns a script into a canonical form via its AST (`luaparser`):
  - Comments are removed, and the layout is fixed by a custom printer: tab indentation, short tables on one line.
  - Runs of consecutive Effect setter calls on the same effect are sorted into `EFFECT_SETTERS` order. Pairs in `SETTER_CONFLICTS` (SetType vs. SetRange/SetCode, SetTargetRange vs. SetAbsoluteRange) keep their order, because the EDOPro core (`libeffect.cpp`) makes them order-dependent. Setters whose arguments are impure or read the receiver are not moved.
  - Runs of consecutive top-level `function s.xxx` definitions are sorted by first use, starting from `initial_effect`.
  - With `rename=True` (only for comparisons, never for labels), locals become `v1, v2, ...` and the script's functions become `f1, f2, ...`.
  - Every result is parsed again and must give the same AST. This guard catches luaparser printer bugs (e.g. `- -x` printed as `--x`), so keep it. Any change to the rules must stay semantics-preserving.
- **Data format** (from `convert.py`):
  - A `_text` file has 3 parts. Line 1 is the Japanese name, taken from the original script's first `--` header comment, which is read before canonicalization (empty if the comment is a placeholder). Line 2 is the English name. The rest is the card description.
  - A `_code` file is the **canonical** script (`canonicalize(code)`, no renaming). Scripts that fail to canonicalize are skipped, and stale pair files from earlier runs are deleted.
  - Card text comes from `cards.cdb` first, then from `release-*.cdb`; when a card appears in several databases, the earlier one wins.
- **Grouped folds** (`split.py`): cards whose description (without the name lines) or whose renamed canonical script is identical are joined with union-find, and each group stays in a single fold. This prevents leakage between train and test. Keep this guarantee whenever you change the splitting.
- **Training** (`train.py`) is a hand-written loop, not the HF `Trainer`. `--profile` picks the hardware defaults (`PROFILES` in `train.py`); single options like `--dtype`, `--quantize`, `--grad-checkpointing`, `--batch-size` override them:
  - `cluster` (default, A100): plain LoRA on bfloat16 weights, no gradient checkpointing, batch 8 x grad-accum 2, generation batch 32. The `GradScaler` is disabled for bfloat16. Without checkpointing a token costs ~6 MB of activations (measured), so `--max-batch-tokens 4096` also caps the padded tokens (cards x longest card) per forward pass (~31 GB peak on the 40 GB A100); the longest cards (2688 tokens) run alone.
  - `local` (8 GB RTX 2070, Turing; `scripts/run.sh` uses it): the settings described below, batch 1 x grad-accum 16, generation batch 8.
  - Within every optimizer step's window of `batch-size * grad-accum` cards, the cards are sorted by length before being split into micro-batches (`split_micro_batches`: at most `batch-size` cards and `max-batch-tokens` padded tokens each), which saves padding without changing the gradient.
  - Local profile: the base model is loaded in 4-bit NF4, with LoRA on all linear projections.
  - Local profile: it computes in float16 with a `GradScaler`, because bfloat16 is slow on Turing and FlashAttention 2 is unsupported, so attention uses SDPA.
  - The prompt tokens are masked out of the loss (`-100`).
  - The loss runs the inner model without `lm_head`, then computes the logits in checkpointed chunks of `LOSS_CHUNK`. This brings peak VRAM from ~6.4 GB down to ~2.2 GB.
  - With gradient accumulation, the loss is normalized by the token count of the whole window.
  - Generation uses greedy decoding with an explicitly set `GenerationConfig`, overriding Qwen's default sampling and repetition penalty, and stops at `<|im_end|>`.
- **Metrics:**
  - Teacher-forced: loss, perplexity and token accuracy.
  - From generation:
    - `exact_match`: identical to the canonical label.
    - `exact_match_canonical`: equal after renamed canonicalization.
    - `similarity`: on the renamed canonical forms.
    - `finished`
    - `syntax_valid`: compiles in Lua 5.4 via `lupa`. No system `luac` is needed.

## Code style

The code is heavily commented: comments explain *why* and give hardware or engine background. Lines are up to ~120 characters, and section dividers are `# ----` lines. Match this style.

## Verifying your work (global rule)

Always check an implementation before calling it done. Testing is crucial, and for coding tasks it is almost always
possible, so "it should work" is never enough:

- **Execute the code.** Run what you wrote, or the affected part of it. Scale the run to the cost of the real job: for
  expensive pipelines do a ghost run instead of the full one (e.g. a training run on a few examples with a tiny step
  count, like the smoke run under "LYM commands"), not a multi-hour training.
- **Use dummy examples.** Feed small hand-made inputs, including edge cases, whose correct output you know in advance,
  and compare. Where it makes sense, add a negative control (an input that must fail or differ) to prove the check can
  catch errors.
- **Inspect the results.** Open the files the code writes and check their content, format and count; check that
  scores and metrics are plausible (ranges, signs, orders of magnitude, consistency with each other).
- **Analyse compiled output.** When the source is compiled or rendered (e.g. LaTeX), build it and analyse the result
  itself (e.g. read the PDF: layout, missing references, overfull boxes, rendered figures), not just the compiler's exit
  code.

Report what you ran and what came out. If something could not be checked, say so explicitly instead of assuming it works.
