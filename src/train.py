"""Fine-tune Qwen2.5-Coder-1.5B-Instruct to write EDOPro Lua card scripts from card texts, then measure its accuracy.

Method: LoRA. The base model is frozen, and small trainable LoRA matrices are added to all of its linear layers. Only
those (~18M parameters, ~1% of the model) are trained. Two profiles (--profile) set the hardware-dependent defaults:
    cluster (default)  for a big GPU such as the A100: plain LoRA on bfloat16 weights, no gradient checkpointing, big
                       batches capped at --max-batch-tokens. Fastest, but needs a 40 GB GPU (~31 GB peak).
    local              for the 8 GB RTX 2070 that also drives the desktop: QLoRA (frozen 4-bit weights), gradient
                       checkpointing, one card per forward pass and a chunked loss computation, which together need
                       ~2.5 GB of VRAM even for the longest card. The RTX 2070 (Turing) has no fast bfloat16, so all
                       computation runs in float16, with a gradient scaler to keep small gradients from underflowing.
Single options (--dtype, --quantize, --grad-checkpointing, --batch-size, ...) override the profile.

Each cross-validation round trains a fresh adapter on K-1 folds and tests it on the remaining fold (see split.py).
Results land in runs/fold_<i>/:
    adapter/            the trained LoRA adapter (load on top of the base model to use it)
    config.json         the arguments of the run
    log.jsonl           training loss, learning rate and validation loss over time
    metrics.json        test metrics (see below)
    predictions/        generated scripts of the test cards scored by generation, as <card_id>.lua

Test metrics:
    teacher_forced.token_accuracy   for every script token of every test card: does the model predict the correct next
                                    token when given the true script so far? Cheap, smooth, good for comparing runs.
    teacher_forced.loss/perplexity  the corresponding cross-entropy (lower is better)
    generation.*                    the model writes complete scripts on its own (greedy decoding) for a random subset
                                    of the test cards, which are compared to the real scripts. The real scripts are in
                                    canonical form (see canonical.py), which the model learns to write as well:
        exact_match                 identical to the canonical script, character for character
        exact_match_canonical       identical up to statement order and naming, i.e. equal once both are canonicalized
                                    with renaming (correct code that merely uses other names counts here)
        similarity                  mean character-level similarity (0..1) of both scripts canonicalized with renaming
                                    (if the generated script does not parse: of both with comments and formatting
                                    removed)
        finished                    share of scripts that ended properly instead of hitting --max-new-tokens
        syntax_valid                share of scripts that compile in Lua 5.4 (the Lua version of EDOPro)

Usage:
    python src/train.py --fold 0                     train and evaluate one round (fold 0 is the test set)
    python src/train.py --profile local --fold 0     the same with the settings for the RTX 2070
    python src/train.py --fold all                   full cross-validation: all rounds, then mean/std over folds
    python src/train.py --fold 0 --eval-only         evaluate an adapter trained earlier
    python src/train.py --fold 0 --limit-train 64 --limit-test 16 --gen-samples 4 --run-name smoke
                                                     quick smoke test of the whole pipeline

Measured on an RTX 2070 (local profile): ~1.6 h per epoch per fold for training (~3.2 h per fold with the default 2
epochs), plus the evaluation time; the progress bars show the exact speed.
"""
import argparse
import difflib
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path

import lupa.lua54
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
)

from canonical import CanonicalizationError, canonicalize
from common import (
    MODEL_DIR,
    RUNS_DIR,
    SPLITS_PATH,
    build_example,
    build_prompt_ids,
    end_of_turn_id,
    load_pair,
    normalize_lua,
)

# Hardware-dependent defaults, chosen with --profile; every entry can be overridden by its own option.
PROFILES = {
    # A100 class GPU: plain LoRA in bfloat16 (no dequantization overhead, no gradient scaler needed), no activation
    # recomputation, and batches large enough to keep the GPU busy. 8 x 2 keeps the effective batch size of 16.
    # Without checkpointing every token costs ~6 MB of activations (measured), so a micro-batch is also capped to
    # max_batch_tokens padded tokens (~25 GB + 3 GB weights): short cards share a batch, the longest ones go alone.
    "cluster": dict(dtype="bfloat16", quantize=False, grad_checkpointing=False, batch_size=8, grad_accum=2,
                    max_batch_tokens=4096, gen_batch_size=32),
    # 8 GB RTX 2070: QLoRA in float16 with gradient checkpointing, one card per forward pass.
    "local": dict(dtype="float16", quantize=True, grad_checkpointing=True, batch_size=1, grad_accum=16,
                  max_batch_tokens=0, gen_batch_size=8),
}
IGNORE_INDEX = -100  # label value for tokens that must not count towards the loss (the prompt, padding)
DTYPE = None  # compute dtype, set in main() from --dtype (float16 on Turing GPUs, which only emulate bfloat16)
LOSS_CHUNK = 512  # positions whose logits are computed at once (see label_loss_sum)

# ---------------------------------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", default="0", help="test fold: an index, a comma separated list, or 'all' (default: 0)")
    parser.add_argument("--run-name", default="", help="prefix for the run folders, to keep experiments apart")
    parser.add_argument("--eval-only", action="store_true", help="skip training and evaluate the saved adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=list(PROFILES), default="cluster",
                        help="hardware-dependent defaults: 'cluster' for an A100 class GPU, 'local' for the 8 GB "
                             "RTX 2070 (default: cluster)")

    data = parser.add_argument_group("data")
    data.add_argument("--max-len", type=int, default=2688,
                      help="max tokens of prompt + script; longer training examples are skipped (default: 2688, "
                           "which covers every card)")
    data.add_argument("--val-size", type=int, default=200,
                      help="cards held out of the training folds to monitor the validation loss (default: 200)")
    data.add_argument("--limit-train", type=int, default=0, help="use only this many training cards (for testing)")
    data.add_argument("--limit-test", type=int, default=0, help="use only this many test cards (for testing)")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--lr", type=float, default=2e-4, help="peak learning rate (default: 2e-4, usual for LoRA)")
    train.add_argument("--batch-size", type=int, help="cards per forward pass (default: set by the profile)")
    train.add_argument("--grad-accum", type=int,
                       help="forward passes per optimizer step; effective batch = batch-size * grad-accum "
                            "(default: set by the profile, 16 in both)")
    train.add_argument("--max-batch-tokens", type=int,
                       help="upper limit of padded tokens (cards x longest card) per forward pass; 0 = no limit "
                            "(default: set by the profile)")
    train.add_argument("--dtype", choices=["float16", "bfloat16"],
                       help="compute dtype (default: set by the profile)")
    train.add_argument("--quantize", action=argparse.BooleanOptionalAction,
                       help="freeze the base model in 4-bit NF4 (QLoRA) instead of full precision (default: set by "
                            "the profile)")
    train.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction,
                       help="recompute activations in the backward pass to save VRAM at ~30%% speed (default: set by "
                            "the profile)")
    train.add_argument("--warmup-ratio", type=float, default=0.03, help="share of steps with linear LR warmup")
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping threshold")
    train.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    train.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling (alpha / r multiplies the update)")
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--eval-every", type=int, default=100,
                       help="optimizer steps between validation loss measurements (default: 100)")

    gen = parser.add_argument_group("generation evaluation")
    gen.add_argument("--gen-samples", type=int, default=200,
                     help="test cards whose scripts are fully generated and compared (default: 200, 0 to skip)")
    gen.add_argument("--gen-batch-size", type=int,
                     help="scripts generated in parallel (default: set by the profile)")
    gen.add_argument("--max-new-tokens", type=int, default=2560, help="max length of a generated script in tokens")

    args = parser.parse_args()
    for name, value in PROFILES[args.profile].items():  # fill in everything the command line left open
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.batch_size < 1 or args.grad_accum < 1 or args.gen_batch_size < 1:
        parser.error("batch sizes and --grad-accum must be at least 1")
    return args


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:  # Qwen defines <|endoftext|> as pad token, but be safe with other models
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(args):
    """The frozen base model, with 4-bit NF4 weights (QLoRA, --quantize) or in full DTYPE precision.

    4-bit takes ~1.2 GB of VRAM instead of ~3 GB, but dequantizing the weights for every matrix multiplication costs
    time, so it is only worth it on a small GPU.
    """
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 4-bit format designed for normally distributed weights (best for LLMs)
        bnb_4bit_use_double_quant=True,  # also quantize the quantization constants, saves another ~0.4 bit/weight
        bnb_4bit_compute_dtype=DTYPE,  # weights are dequantized to DTYPE on the fly for each matrix multiplication
    ) if args.quantize else None
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quantization,
        dtype=DTYPE,  # dtype of the unquantized weights (everything, if not quantizing)
        device_map={"": 0},  # everything on the GPU
        attn_implementation="sdpa",  # PyTorch's fused attention (FlashAttention 2 does not support Turing)
    )


def make_trainable(model, args):
    """Add fresh LoRA adapters to the frozen base model and prepare it for training."""
    # The KV cache only speeds up generation; during training it would just waste memory.
    model.config.use_cache = False
    if args.grad_checkpointing:
        # Gradient checkpointing: instead of keeping every intermediate activation for the backward pass, keep only
        # each layer's input and recompute the rest during backward. Costs ~30% speed, saves most of the activation
        # memory. The non-reentrant variant also works when the inputs themselves need no gradient (frozen embeddings).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # All linear layers of attention and MLP, which works clearly better than attention-only LoRA.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    # PEFT keeps the adapter weights in float32 (autocast_adapter_dtype), which the gradient scaler requires (float16).
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------


def encode(tokenizer, card_ids: list[int], max_len: int, skip_long: bool) -> tuple[list[dict], int]:
    """Tokenize cards into training examples: dicts of input_ids and labels (the prompt part masked out of the loss).

    Examples longer than max_len are skipped (skip_long, for training) or cut at max_len (for evaluation, so every test
    card still counts). Returns the examples and how many were skipped or cut.
    """
    examples, n_long = [], 0
    for card_id in card_ids:
        ids, prompt_len = build_example(tokenizer, *load_pair(card_id))
        if len(ids) > max_len:
            n_long += 1
            if skip_long:
                continue
            ids = ids[:max_len]
        input_ids = torch.tensor(ids)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        examples.append({"card_id": card_id, "input_ids": input_ids, "labels": labels})
    return examples, n_long


def collate(examples: list[dict], pad_id: int) -> dict:
    """Stack examples into one right-padded batch on the GPU. Padding is masked from attention and loss."""
    length = max(len(ex["input_ids"]) for ex in examples)
    input_ids = torch.full((len(examples), length), pad_id)
    labels = torch.full((len(examples), length), IGNORE_INDEX)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    for row, ex in enumerate(examples):
        n = len(ex["input_ids"])
        input_ids[row, :n] = ex["input_ids"]
        labels[row, :n] = ex["labels"]
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids.cuda(), "labels": labels.cuda(), "attention_mask": attention_mask.cuda()}


def split_micro_batches(cards: list[dict], max_cards: int, max_tokens: int) -> list[list[dict]]:
    """Split the cards of one optimizer step into micro-batches of similar length.

    A micro-batch holds at most max_cards cards, and (if max_tokens > 0) at most max_tokens padded tokens, i.e. number
    of cards x length of its longest card, because activation memory grows with that product. A card longer than the
    limit on its own still gets its own micro-batch.
    """
    batches, current = [], []
    for ex in sorted(cards, key=lambda ex: len(ex["input_ids"])):  # ascending: the newest card is the longest
        too_many = len(current) == max_cards
        too_long = max_tokens and len(ex["input_ids"]) * (len(current) + 1) > max_tokens
        if current and (too_many or too_long):
            batches.append(current)
            current = []
        current.append(ex)
    return batches + ([current] if current else [])


def n_label_tokens(examples: list[dict]) -> int:
    """Number of tokens the loss is computed on (labels shifted by one, as in forward_label_logits)."""
    return sum(int((ex["labels"][1:] != IGNORE_INDEX).sum()) for ex in examples)


def forward_label_hidden(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the transformer without its output layer; return the hidden states at the script positions and the tokens
    they should predict.

    The output at position t predicts token t+1, hence the shift by one. Only positions with a real label are kept.
    The output layer (lm_head) is left out on purpose: its logits (one score per vocabulary entry, i.e. 151k numbers per
    position) would take several GB for a long script in one piece, so they are computed chunk-wise in label_loss_sum.
    The LoRA adapters live inside the transformer layers, so calling the inner model still applies them.
    """
    with torch.autocast("cuda", dtype=DTYPE):
        hidden = model.get_base_model().model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
    targets = batch["labels"][:, 1:]
    mask = targets != IGNORE_INDEX
    return hidden[:, :-1][mask], targets[mask]


def chunk_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Float32 logits of a chunk of hidden states (float32 for a numerically safe softmax in the loss)."""
    with torch.autocast("cuda", dtype=DTYPE):
        return model.get_base_model().lm_head(hidden).float()


def chunk_loss_sum(model, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(chunk_logits(model, hidden), targets, reduction="sum")


def label_loss_sum(model, batch: dict) -> torch.Tensor:
    """Summed cross-entropy over all script tokens of a batch, computed in chunks of LOSS_CHUNK positions.

    Each chunk is checkpointed: its logits are freed right after its loss is computed and recomputed during backward,
    so at most one chunk of logits is in memory at a time. This lowers peak memory for the longest cards by ~3.5 GB.
    """
    hidden, targets = forward_label_hidden(model, batch)
    return sum(
        checkpoint(chunk_loss_sum, model, h, t, use_reentrant=False)
        for h, t in zip(hidden.split(LOSS_CHUNK), targets.split(LOSS_CHUNK))
    )


# ---------------------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------------------


def train(model, train_examples: list[dict], val_examples: list[dict], args, run_dir: Path, pad_id: int) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    window_size = args.batch_size * args.grad_accum  # cards per optimizer step
    steps_per_epoch = math.ceil(len(train_examples) / window_size)
    total_steps = steps_per_epoch * args.epochs
    # Linear warmup (avoids large, destructive updates while Adam's statistics are still unreliable), then a cosine
    # decay to zero, so the last steps fine-tune with small, stable updates.
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(args.warmup_ratio * total_steps), total_steps)
    # Multiplies the loss before backward so that small float16 gradients do not underflow to zero; it unscales the
    # gradients before the optimizer step and skips steps whose gradients overflowed (inf/nan), lowering its factor.
    # bfloat16 has the range of float32, so it needs none of this (a disabled scaler passes everything through).
    scaler = torch.amp.GradScaler("cuda", enabled=DTYPE == torch.float16)

    rng = random.Random(args.seed)
    log_path = run_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)
    step, skipped_steps = 0, 0
    print(f"Training on {len(train_examples)} cards: {args.epochs} epochs x {steps_per_epoch} steps "
          f"(effective batch size {args.batch_size * args.grad_accum})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        rng.shuffle(order)  # new random order every epoch
        # Each optimizer step covers a window of batch_size * grad_accum random cards. Inside the window the cards are
        # sorted by length before they are split into micro-batches, so a batch holds cards of similar length and
        # little compute is wasted on padding. The window itself (and thus the gradient) stays the same.
        windows = [split_micro_batches([train_examples[j] for j in order[k:k + window_size]], args.batch_size,
                                       args.max_batch_tokens)
                   for k in range(0, len(order), window_size)]
        progress = tqdm(windows, desc=f"epoch {epoch}/{args.epochs}", unit="step", dynamic_ncols=True)

        for window in progress:
            # Gradient accumulation: the gradients of all micro-batches in the window are summed before one optimizer
            # step. Each micro-batch's summed token loss is divided by the token count of the *whole window*, so every
            # script token weighs the same, no matter how long its script is or which micro-batch it is in.
            window_tokens = sum(n_label_tokens(batch) for batch in window)
            window_loss = 0.0
            for batch in window:
                loss_sum = label_loss_sum(model, collate(batch, pad_id))
                scaler.scale(loss_sum / window_tokens).backward()
                window_loss += loss_sum.item()

            scaler.unscale_(optimizer)  # real gradient values, so that clipping uses the true norm
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)  # skipped internally if the gradients contain inf/nan
            scaler.update()
            if scaler.get_scale() < scale_before:  # the scale only drops when a step was skipped
                skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            record = {"step": step, "epoch": epoch, "loss": window_loss / window_tokens,
                      "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm)}
            progress.set_postfix(loss=f"{record['loss']:.4f}", lr=f"{record['lr']:.1e}")
            if not math.isfinite(record["loss"]):
                raise RuntimeError(f"Loss became {record['loss']} at step {step}: overflow? Try a lower --lr.")
            if step % args.eval_every == 0 or step == total_steps:
                record["val_loss"] = evaluate_teacher_forced(model, val_examples, pad_id, "validation")["loss"]
                model.train()  # evaluation switched to eval mode (no dropout)
                tqdm.write(f"step {step}/{total_steps}: train loss {record['loss']:.4f}, "
                           f"val loss {record['val_loss']:.4f}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")

        # Save after every epoch, so an interrupted run still leaves the last finished epoch behind.
        model.save_pretrained(run_dir / "adapter")
        tqdm.write(f"Epoch {epoch} done, adapter saved to {run_dir / 'adapter'}")

    if skipped_steps:
        # A few skipped steps at the start are normal (the scaler searches its factor); many mean instability.
        print(f"Note: {skipped_steps} of {total_steps} steps were skipped because of float16 overflow")


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate_teacher_forced(model, examples: list[dict], pad_id: int, desc: str) -> dict:
    """Loss, perplexity and next-token accuracy on the script tokens, given the true script so far."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for ex in tqdm(examples, desc=desc, leave=False, dynamic_ncols=True):
        hidden, targets = forward_label_hidden(model, collate([ex], pad_id))
        for h, t in zip(hidden.split(LOSS_CHUNK), targets.split(LOSS_CHUNK)):  # chunked as in label_loss_sum
            logits = chunk_logits(model, h)
            loss_sum += F.cross_entropy(logits, t, reduction="sum").item()
            correct += int((logits.argmax(dim=-1) == t).sum())
        total += targets.numel()
    return {"loss": loss_sum / total, "perplexity": math.exp(loss_sum / total), "token_accuracy": correct / total}


def canonical_or_none(code: str) -> str | None:
    """The renamed canonical form of a script, or None if it cannot be parsed."""
    try:
        return canonicalize(code, rename=True)
    except CanonicalizationError:
        return None


@torch.no_grad()
def evaluate_generation(model, tokenizer, card_ids: list[int], args, out_dir: Path) -> dict:
    """Let the model write complete scripts for the given cards and compare them to the real scripts."""
    model.eval()
    model.config.use_cache = True  # the KV cache makes generation many times faster
    end_id = end_of_turn_id(tokenizer)
    # Replace the model's default generation settings entirely: they sample randomly (temperature 0.7) and apply a
    # repetition penalty of 1.1, which would punish code for its naturally repetitive structure. Greedy decoding is
    # deterministic, so the scores are reproducible.
    model.get_base_model().generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=[end_id, tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
    )
    # A real Lua 5.4 interpreter; load() only compiles the code without running it, returning nil on syntax errors.
    lua_compiles = lupa.lua54.LuaRuntime().eval("function(code) return load(code) ~= nil end")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {"exact_match": [], "exact_match_canonical": [], "similarity": [], "finished": [], "syntax_valid": []}
    for start in tqdm(range(0, len(card_ids), args.gen_batch_size), desc="generating", dynamic_ncols=True):
        batch_ids = card_ids[start:start + args.gen_batch_size]
        pairs = [load_pair(card_id) for card_id in batch_ids]
        prompts = [build_prompt_ids(tokenizer, text) for text, _ in pairs]

        # Left padding: generation appends tokens on the right, so all prompts must end at the same position.
        width = max(map(len, prompts))
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prompts]).cuda()
        attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts]).cuda()
        with torch.autocast("cuda", dtype=DTYPE):
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        for card_id, (_, true_code), new_tokens in zip(batch_ids, pairs, output[:, width:].tolist()):
            finished = end_id in new_tokens or tokenizer.eos_token_id in new_tokens
            predicted = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            (out_dir / f"{card_id}.lua").write_text(predicted + "\n", encoding="utf-8")

            pred_canon, true_canon = canonical_or_none(predicted), canonical_or_none(true_code)
            if pred_canon is None or true_canon is None:  # unparseable: fall back to comparing the plain code
                pred_canon, true_canon = normalize_lua(predicted), normalize_lua(true_code)
            scores["exact_match"].append(predicted == true_code.strip())
            scores["exact_match_canonical"].append(pred_canon == true_canon)
            scores["similarity"].append(difflib.SequenceMatcher(None, pred_canon, true_canon, autojunk=False).ratio())
            scores["finished"].append(finished)
            scores["syntax_valid"].append(bool(lua_compiles(predicted)))

    model.config.use_cache = False  # back to the training setting
    metrics = {name: sum(values) / len(values) for name, values in scores.items()}
    metrics["n_cards"] = len(card_ids)
    return metrics


# ---------------------------------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------------------------------


def run_fold(fold: int, folds: list[list[int]], tokenizer, args) -> dict:
    """Train on all folds except `fold` (unless --eval-only), then evaluate on `fold`. Returns the test metrics."""
    rng = random.Random(args.seed)
    test_ids = folds[fold]
    train_ids = sorted(card_id for i, fold_ids in enumerate(folds) if i != fold for card_id in fold_ids)
    if args.limit_train:
        train_ids = sorted(rng.sample(train_ids, min(args.limit_train, len(train_ids))))
    if args.limit_test:
        test_ids = sorted(rng.sample(test_ids, min(args.limit_test, len(test_ids))))
    # The validation cards come out of the training folds, never out of the test fold, so the test fold stays unseen
    # until the very end. They only monitor training (overfitting shows as a rising validation loss).
    val_ids = set(rng.sample(train_ids, min(args.val_size, len(train_ids) // 10)))
    train_ids = [card_id for card_id in train_ids if card_id not in val_ids]
    gen_ids = sorted(rng.sample(test_ids, min(args.gen_samples, len(test_ids))))

    run_dir = RUNS_DIR / f"{args.run_name + '_' if args.run_name else ''}fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Fold {fold}: {len(train_ids)} train / {len(val_ids)} validation / {len(test_ids)} test cards "
          f"-> {run_dir} ===")

    model = load_base_model(args)
    if args.eval_only:
        adapter_dir = run_dir / "adapter"
        if not adapter_dir.is_dir():
            raise SystemExit(f"No trained adapter in {adapter_dir}; train this fold first.")
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        train_examples, n_skipped = encode(tokenizer, train_ids, args.max_len, skip_long=True)
        if n_skipped:
            print(f"Skipped {n_skipped} training cards longer than {args.max_len} tokens")
        val_examples, _ = encode(tokenizer, sorted(val_ids), args.max_len, skip_long=False)
        model = make_trainable(model, args)
        start = time.time()
        train(model, train_examples, val_examples, args, run_dir, tokenizer.pad_token_id)
        print(f"Training took {(time.time() - start) / 3600:.2f} h")
        del train_examples, val_examples

    test_examples, n_cut = encode(tokenizer, test_ids, args.max_len, skip_long=False)
    if n_cut:
        print(f"Note: {n_cut} test cards were cut at {args.max_len} tokens for the teacher-forced metrics")
    metrics = {"fold": fold, "n_test_cards": len(test_ids)}
    metrics["teacher_forced"] = evaluate_teacher_forced(model, test_examples, tokenizer.pad_token_id, "testing")
    if gen_ids:
        metrics["generation"] = evaluate_generation(model, tokenizer, gen_ids, args, run_dir / "predictions")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Fold {fold} results: {json.dumps(metrics, indent=2)}")

    # Free the GPU completely before the next fold loads a fresh model.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def summarize(all_metrics: list[dict]) -> dict:
    """Mean and standard deviation of every metric over the folds (the cross-validated result)."""
    summary = {"folds": [m["fold"] for m in all_metrics]}
    for section in ("teacher_forced", "generation"):
        sections = [m[section] for m in all_metrics if section in m]
        if not sections:
            continue
        summary[section] = {}
        for name in sections[0]:
            if name == "n_cards":  # a count, not a metric
                continue
            values = [s[name] for s in sections if name in s]
            summary[section][name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available.")
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Base model not found in {MODEL_DIR}; run fetch_model.py first.")
    if not SPLITS_PATH.is_file():
        raise SystemExit(f"No folds in {SPLITS_PATH}; run split.py first.")

    folds = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["folds"]
    if args.fold == "all":
        fold_indices = list(range(len(folds)))
    else:
        fold_indices = [int(i) for i in args.fold.split(",")]
        if not all(0 <= i < len(folds) for i in fold_indices):
            raise SystemExit(f"Folds must be between 0 and {len(folds) - 1}")

    global DTYPE
    DTYPE = getattr(torch, args.dtype)
    print(f"Profile {args.profile}: {args.dtype}, {'4-bit' if args.quantize else 'unquantized'} base model, "
          f"gradient checkpointing {'on' if args.grad_checkpointing else 'off'}, batch {args.batch_size} x "
          f"{args.grad_accum} (max {args.max_batch_tokens or 'unlimited'} padded tokens per forward pass), "
          f"generation batch {args.gen_batch_size}")
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer()
    all_metrics = [run_fold(fold, folds, tokenizer, args) for fold in fold_indices]

    if len(all_metrics) > 1:
        summary = summarize(all_metrics)
        summary_path = RUNS_DIR / f"{args.run_name + '_' if args.run_name else ''}cv_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nCross-validation summary (saved to {summary_path}):\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
