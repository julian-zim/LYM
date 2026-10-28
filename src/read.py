import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "CardScripts" / "official"
CDB_PATH = ROOT / "BabelCDB" / "cards.cdb"


def card_id_from_filename(filename: str) -> int:
    """Turn '10000.lua', 'c10000.lua' or '10000' into the card id 10000."""
    stem = Path(filename).stem
    if stem.startswith("c"):
        stem = stem[1:]
    return int(stem)


def get_card_text(card_id: int, cdb_path: Path = CDB_PATH) -> tuple[str, str] | None:
    with sqlite3.connect(cdb_path) as conn:
        return conn.execute("SELECT name, desc FROM texts WHERE id = ?", (card_id,)).fetchone()


def main() -> None:
    if len(sys.argv) != 2:
        card_id = card_id_from_filename(input('Card ID: '))
    else:
        card_id = card_id_from_filename(sys.argv[1])
    if not (SCRIPTS_DIR / f"c{card_id}.lua").is_file():
        print(f"Warning: no script c{card_id}.lua in {SCRIPTS_DIR}", file=sys.stderr)

    row = get_card_text(card_id)
    if row is None:
        sys.exit(f"No card with id {card_id} in {CDB_PATH}")

    name, desc = row
    print(name)
    print(desc)


if __name__ == "__main__":
    main()
