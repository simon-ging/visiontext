from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def analyze_list_of_text(
    text_list: List[str], verbose=True
) -> Tuple[Dict[str, int], Dict[str, int], List[int]]:
    uniques = defaultdict(int)
    vocabs = defaultdict(int)
    lens = []
    for i, item in enumerate(text_list):
        if verbose and (i + 1) % 100000 == 0:
            print(f"{i + 1}/{len(text_list)}", end=" ")

        uniques[item] += 1
        words = item.split(" ")
        lens.append(len(words))
        for word in words:
            vocabs[word] += 1

    # uniques: dict unique_entry -> count
    # vocabs: dict word -> count
    # lens: list of int, length of each datapoint
    return uniques, vocabs, lens


def show_top_k(values_counts, k):
    if isinstance(values_counts, tuple):
        values, counts = values_counts
    else:
        values = list(values_counts.keys())
        counts = list(values_counts.values())

    top_used_idx = np.argsort(counts)
    for n in range(min(k, len(top_used_idx))):
        idx = top_used_idx[-n - 1]
        print("    ", end="")
        print(f"({counts[idx]}) {values[idx]} ", end="")
    print()
