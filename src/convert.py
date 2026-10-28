import multiprocessing
import sqlite3
from pathlib import Path

from canonical import CanonicalizationError, canonicalize

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "CardScripts" / "official"
CDB_DIR = ROOT / "BabelCDB"
# cards.cdb holds the official cards; release-*.cdb hold freshly released sets not yet merged into it
CDB_PATHS = [CDB_DIR / "cards.cdb", *sorted(CDB_DIR.glob("release-*.cdb"))]
DATA_DIR = ROOT / "LYM" / "data"
JP_NAME_PLACEHOLDERS = {"JP name", "Japanese name"}


def load_card_texts(cdb_paths: list[Path] = CDB_PATHS) -> dict[int, str]:
    """Map card id -> card text (English name on the first line, then the description). Earlier cdbs take precedence."""
    texts = {}
    for cdb_path in cdb_paths:
        with sqlite3.connect(cdb_path) as conn:
            for card_id, name, desc in conn.execute("SELECT id, name, desc FROM texts"):
                if card_id not in texts:
                    desc = (desc or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                    texts[card_id] = f"{name.strip()}\n{desc}\n"
    return texts


def get_jp_name(code: str) -> str:
    """Return the Japanese name from the script header ('--<JP name>' then '--<EN name>'), or '' if it has none."""
    lines = [line.strip() for line in code.lstrip("﻿").split("\n", 2)[:2]]
    if len(lines) < 2 or not all(line.startswith("--") for line in lines):
        return ""
    jp_name = lines[0][2:].strip()
    return "" if jp_name in JP_NAME_PLACEHOLDERS else jp_name


def read_script(script_path: Path) -> tuple[str, str | None, str | None]:
    """Return (Japanese name, canonical script, error) of a script file; the script is None if it failed."""
    code = script_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    jp_name = get_jp_name(code)  # from the header comments, so before canonicalizing removes them
    try:
        return jp_name, canonicalize(code), None
    except CanonicalizationError as error:
        return jp_name, None, str(error)


def main() -> None:
    texts = load_card_texts()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    script_paths, missing_text = {}, []
    for script_path in sorted(SCRIPTS_DIR.glob("c*.lua")):
        stem = script_path.stem[1:]
        if not stem.isdigit():
            continue
        card_id = int(stem)
        if card_id in texts:
            script_paths[card_id] = script_path
        else:
            missing_text.append(card_id)

    # Canonicalizing takes ~0.05 s per script, so it runs on all CPU cores.
    with multiprocessing.Pool() as pool:
        results = pool.map(read_script, script_paths.values(), chunksize=16)

    written, failed = set(), {}
    for card_id, (jp_name, code, error) in zip(script_paths, results):
        if code is None:
            failed[card_id] = error
            continue
        (DATA_DIR / f"{card_id}_text.txt").write_text(f"{jp_name}\n{texts[card_id]}", encoding="utf-8", newline="\n")
        (DATA_DIR / f"{card_id}_code.txt").write_text(code + "\n", encoding="utf-8", newline="\n")
        written.add(card_id)

    # Remove pairs left over from earlier runs whose card is no longer written, so they cannot sneak into training.
    for path in [*DATA_DIR.glob("*_text.txt"), *DATA_DIR.glob("*_code.txt")]:
        if int(path.name.split("_")[0]) not in written:
            path.unlink()

    print(f"Wrote {len(written)} text/code pairs with canonical scripts to {DATA_DIR}")
    if missing_text:
        print(f"Skipped {len(missing_text)} scripts without card text: {missing_text}")
    for card_id, error in failed.items():
        print(f"Skipped card {card_id}, its script could not be canonicalized: {error}")


if __name__ == "__main__":
    main()
