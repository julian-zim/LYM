"""Split the text/code pairs into K cross-validation folds and save them to LYM/splits/folds.json.

Fold i is the test set of cross-validation round i; the other K-1 folds are its training set.

The folds are *grouped*: some cards share the same card text (reprints under another name, e.g. anime/OCG variants) or
the same script (identical up to order and naming, see canonical.py). If one such card landed in the training set
and its twin in the test set, the model could score by remembering the twin's script instead of having learned to write
scripts, inflating the measured accuracy. So such cards are merged into a group, and each group goes into one fold as
a whole.

Usage:  python LYM/src/split.py [--k 5] [--seed 42]
"""
import argparse
import json
import multiprocessing
import random

from canonical import canonicalize
from common import SPLITS_PATH, list_card_ids, load_pair


def duplicate_keys(card_id: int) -> tuple[str, str]:
    """The two keys under which cards count as duplicates: their card text and their script.

    The text is taken without the two name lines, so the same effect under two names still counts as a duplicate. The
    script is compared in its renamed canonical form, so scripts that differ only in order or naming count as well.
    """
    text, code = load_pair(card_id)
    return "\n".join(text.split("\n")[2:]).strip(), canonicalize(code, rename=True)


def find_groups(card_ids: list[int]) -> list[list[int]]:
    """Group card ids that are connected through an identical card text or an identical script.

    Uses union-find, so grouping is transitive: if A shares its text with B and B shares its script with C, then A, B
    and C all end up in one group.
    """
    parent = {card_id: card_id for card_id in card_ids}

    def find(card_id: int) -> int:
        while parent[card_id] != card_id:
            parent[card_id] = parent[parent[card_id]]  # path halving keeps the trees flat
            card_id = parent[card_id]
        return card_id

    with multiprocessing.Pool() as pool:  # canonicalizing takes ~0.05 s per script, so use all CPU cores
        keys = pool.map(duplicate_keys, card_ids, chunksize=16)

    first_with_key = {}  # duplicate key -> first card id seen with it
    for card_id, (description, code) in zip(card_ids, keys):
        for key in (("text", description), ("code", code)):
            if key in first_with_key:
                parent[find(card_id)] = find(first_with_key[key])
            else:
                first_with_key[key] = card_id

    groups = {}
    for card_id in card_ids:
        groups.setdefault(find(card_id), []).append(card_id)
    return list(groups.values())


def assign_folds(groups: list[list[int]], k: int, seed: int) -> list[list[int]]:
    """Distribute groups over k folds so that all folds end up (nearly) the same size.

    Groups are shuffled, then placed largest first, each into the currently smallest fold. Placing the big groups first
    leaves the many single-card groups to even out the fold sizes at the end. Sorting is stable, so groups of equal size
    keep their shuffled (random, but seed-reproducible) order.
    """
    groups = [sorted(group) for group in groups]
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    folds = [[] for _ in range(k)]
    for group in groups:
        min(folds, key=len).extend(group)
    return [sorted(fold) for fold in folds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the shuffling (default: 42)")
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be at least 2")

    card_ids = list_card_ids()
    if not card_ids:
        raise SystemExit("No text/code pairs found; run convert.py first.")
    groups = find_groups(card_ids)
    folds = assign_folds(groups, args.k, args.seed)

    # Sanity checks: every card is in exactly one fold, and no duplicate group is split across folds.
    assert sorted(card_id for fold in folds for card_id in fold) == card_ids
    fold_of = {card_id: i for i, fold in enumerate(folds) for card_id in fold}
    assert all(len({fold_of[card_id] for card_id in group}) == 1 for group in groups)

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps({"k": args.k, "seed": args.seed, "folds": folds}), encoding="utf-8")

    multi = [group for group in groups if len(group) > 1]
    print(f"{len(card_ids)} cards in {len(groups)} groups ({len(multi)} groups of duplicates covering "
          f"{sum(map(len, multi))} cards, largest: {max(map(len, groups))} cards)")
    print(f"Fold sizes: {[len(fold) for fold in folds]}")
    print(f"Saved to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
