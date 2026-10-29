"""Shared paths, data loading, prompt formatting and Lua helpers.

Everything that must be *identical* between fetching, splitting, training and evaluating lives here, most importantly
the prompt format: if training and inference built the prompt even slightly differently, the model would see inputs at
test time that it was never trained on.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------------
# Paths (LYM_DIR is the repository root, one level above this file: LYM/src/common.py)
# ---------------------------------------------------------------------------------------------------------------------
LYM_DIR = Path(__file__).resolve().parents[1]
# CardScripts/ and BabelCDB/ are separate upstream clones that sit next to LYM/; only convert.py and read.py need them
UPSTREAM_DIR = LYM_DIR.parent
DATA_DIR = LYM_DIR / "data"  # {card_id}_text.txt / {card_id}_code.txt pairs written by convert.py
SPLITS_PATH = LYM_DIR / "splits" / "folds.json"  # cross-validation folds written by split.py
RUNS_DIR = LYM_DIR / "runs"  # adapters, logs, predictions and metrics written by train.py

# Qwen2.5-Coder-1.5B-Instruct: a code-specialised model (Lua is part of its training data) that also understands the
# Japanese card names. In 4-bit with LoRA it trains comfortably within the ~7.4 GB of free VRAM on an RTX 2070.
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_DIR = LYM_DIR / "models" / MODEL_ID.split("/")[-1]

# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------


def list_card_ids(data_dir: Path = DATA_DIR) -> list[int]:
    """All card ids that have both a _text and a _code file, sorted ascending."""
    code_ids = {int(p.name.removesuffix("_code.txt")) for p in data_dir.glob("*_code.txt")}
    text_ids = {int(p.name.removesuffix("_text.txt")) for p in data_dir.glob("*_text.txt")}
    return sorted(code_ids & text_ids)


def load_pair(card_id: int, data_dir: Path = DATA_DIR) -> tuple[str, str]:
    """Return (card text, Lua script) of one card. The text is the feature, the script the label."""
    text = (data_dir / f"{card_id}_text.txt").read_text(encoding="utf-8")
    code = (data_dir / f"{card_id}_code.txt").read_text(encoding="utf-8")
    return text, code


# ---------------------------------------------------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert EDOPro card script developer. Given a Yu-Gi-Oh! card's Japanese name, English name and card "
    "text, write the card's complete Lua script."
)
# Qwen's chat format ends every turn with this special token. The model learns to emit it after the script, and
# generation stops as soon as it does.
END_OF_TURN = "<|im_end|>"


def build_prompt_ids(tokenizer, card_text: str) -> list[int]:
    """Token ids of the chat prompt (system + user turn + the opening of the assistant turn) for one card text.

    The template is rendered to a string first and tokenized afterwards, because what apply_chat_template returns when
    asked to tokenize directly differs between transformers versions. Qwen's tokenizer adds no BOS token, so
    add_special_tokens=False reproduces the template exactly.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card_text.strip()},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def build_example(tokenizer, card_text: str, code: str) -> tuple[list[int], int]:
    """Token ids of a full training example (prompt + script + end-of-turn), and the length of its prompt part.

    The prompt part is later masked out of the loss, so the model is only trained to produce the script, not to repeat
    the card text.
    """
    prompt_ids = build_prompt_ids(tokenizer, card_text)
    code_ids = tokenizer(code.strip(), add_special_tokens=False)["input_ids"]
    return prompt_ids + code_ids + [end_of_turn_id(tokenizer)], len(prompt_ids)


def end_of_turn_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer has no {END_OF_TURN} token; is this a Qwen chat model?")
    return token_id


# ---------------------------------------------------------------------------------------------------------------------
# Lua helpers working on the text alone. They also handle code that does not parse, so train.py falls back on them to
# score generated scripts that canonical.py (which needs a valid AST) cannot handle.
# ---------------------------------------------------------------------------------------------------------------------
# Opening long bracket of a Lua long string or long comment: [[, [=[, [==[, ... The number of '=' must match the closing
# bracket (]], ]=], ]==], ...), which has the same length as the opening one.
_LONG_BRACKET = re.compile(r"\[(=*)\[")


def strip_lua_comments(code: str) -> str:
    """Remove all comments ('-- line' and '--[[ block ]]') from Lua code, leaving string literals untouched.

    A plain regex is not enough because '--' can also appear inside strings, so this walks the code once and skips over
    strings, the only place where '--' does not start a comment. Newlines that end line comments are kept.
    """
    out = []
    i, n = 0, len(code)
    while i < n:
        if code.startswith("--", i):
            long_bracket = _LONG_BRACKET.match(code, i + 2)
            if long_bracket:  # block comment: skip up to and including the matching closing bracket
                end = code.find("]" + long_bracket.group(1) + "]", long_bracket.end())
                i = n if end == -1 else end + len(long_bracket.group(0))
            else:  # line comment: skip up to (not including) the newline
                end = code.find("\n", i)
                i = n if end == -1 else end
            continue
        if code[i] in "\"'":  # short string: copy up to the closing quote, honouring backslash escapes
            quote, j = code[i], i + 1
            while j < n and code[j] != quote and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(code[i:j])
            i = j
            continue
        long_bracket = _LONG_BRACKET.match(code, i)
        if long_bracket:  # long string: copy up to and including the matching closing bracket
            end = code.find("]" + long_bracket.group(1) + "]", long_bracket.end())
            j = n if end == -1 else end + len(long_bracket.group(0))
            out.append(code[i:j])
            i = j
            continue
        out.append(code[i])
        i += 1
    return "".join(out)


def normalize_lua(code: str) -> str:
    """Lua code without comments, blank lines and formatting differences (indentation, repeated spaces).

    Two scripts with the same normalized form behave identically. This is a weaker, text-based variant of the canonical
    form in canonical.py, for code that cannot be parsed.
    """
    lines = (" ".join(line.split()) for line in strip_lua_comments(code).split("\n"))
    return "\n".join(line for line in lines if line)
