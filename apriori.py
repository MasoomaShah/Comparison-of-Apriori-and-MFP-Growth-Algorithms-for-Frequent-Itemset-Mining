

import time
import tracemalloc
from itertools import combinations
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = "trimmed_chess_1.dat"   # change to your file
MIN_SUP      = 0.80                    # minimum support threshold (0.0 – 1.0)
MAX_ROWS     = 1000                    # use half the dataset (set None for all)
# ─────────────────────────────────────────────────────────────────────────────


def load_transactions(path, max_rows=None):
    transactions = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            items = frozenset(line.strip().split())
            if items:
                transactions.append(items)
    return transactions


def get_frequent_1_itemsets(transactions, min_sup_count):
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[frozenset([item])] += 1
    return {k: v for k, v in counts.items() if v >= min_sup_count}


def apriori_gen(freq_itemsets, k):
    """Generate candidate k-itemsets from frequent (k-1)-itemsets."""
    freq_list = list(freq_itemsets.keys())
    candidates = set()
    for i in range(len(freq_list)):
        for j in range(i + 1, len(freq_list)):
            union = freq_list[i] | freq_list[j]
            if len(union) == k:
                # Prune: all (k-1) subsets must be frequent
                if all(
                    frozenset(s) in freq_itemsets
                    for s in combinations(union, k - 1)
                ):
                    candidates.add(union)
    return candidates


def apriori(transactions, min_sup):
    n = len(transactions)
    min_sup_count = int(min_sup * n)

    all_frequent = {}
    total_candidates = 0

    # Pass 1: frequent 1-itemsets
    freq_k = get_frequent_1_itemsets(transactions, min_sup_count)
    total_candidates += len(freq_k)
    all_frequent.update(freq_k)

    k = 2
    while freq_k:
        candidates = apriori_gen(freq_k, k)
        total_candidates += len(candidates)

        # Count support
        counts = defaultdict(int)
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[c] += 1

        freq_k = {c: v for c, v in counts.items() if v >= min_sup_count}
        all_frequent.update(freq_k)
        k += 1

    return all_frequent, total_candidates


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  APRIORI — Frequent Itemset Mining")
    print("=" * 60)

    transactions = load_transactions(DATASET_PATH, MAX_ROWS)
    print(f"  Dataset      : {DATASET_PATH}")
    print(f"  Transactions : {len(transactions)}")
    print(f"  Min Support  : {MIN_SUP:.0%}")
    print("-" * 60)

    tracemalloc.start()
    t_start = time.perf_counter()

    freq_itemsets, total_candidates = apriori(transactions, MIN_SUP)

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Frequent Itemsets : {len(freq_itemsets)}")
    print(f"  Candidates Gen.   : {total_candidates}")
    print(f"  Execution Time    : {t_end - t_start:.4f} s")
    print(f"  Peak Memory       : {peak_mem / 1024 / 1024:.2f} MB")
    print("=" * 60)

    # Show top-10 by support
    print("\nTop-10 Frequent Itemsets by Support Count:")
    sorted_fi = sorted(freq_itemsets.items(), key=lambda x: -x[1])
    for itemset, count in sorted_fi[:10]:
        sup = count / len(transactions)
        print(f"  {set(itemset)}  →  support={sup:.4f}  (count={count})")
