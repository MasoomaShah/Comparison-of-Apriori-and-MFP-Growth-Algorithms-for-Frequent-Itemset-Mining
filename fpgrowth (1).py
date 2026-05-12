"""
fpgrowth.py  —  FP-Growth algorithm for Frequent Itemset Mining
DAA Semester Project CS-478

Based on: Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns
without candidate generation. ACM SIGMOD Record, 29(2), 1–12.
Optimized variant inspired by: "An optimized FP-growth algorithm for
discovery of association rules" (the paper cited in the project).

Upload your .dat file to Colab, then set DATASET_PATH and MIN_SUP below.
"""

import time
import tracemalloc
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
            items = line.strip().split()
            if items:
                transactions.append(items)
    return transactions


# ── FP-Tree Node ──────────────────────────────────────────────────────────────
class FPNode:
    def __init__(self, item, count, parent):
        self.item     = item
        self.count    = count
        self.parent   = parent
        self.children = {}
        self.link     = None   # header table link


# ── FP-Tree ───────────────────────────────────────────────────────────────────
class FPTree:
    def __init__(self, transactions, min_sup_count, root_item=None, root_count=None):
        self.root          = FPNode(None, 1, None)
        self.header_table  = {}   # item -> [count, first_node]
        self.min_sup_count = min_sup_count

        # Count item frequencies
        freq = defaultdict(int)
        for t in transactions:
            for item in t:
                freq[item] += 1

        self.freq_items = {k for k, v in freq.items() if v >= min_sup_count}

        # Build the tree
        for t in transactions:
            # Filter and sort by frequency descending (optimized ordering)
            filtered = sorted(
                [i for i in t if i in self.freq_items],
                key=lambda x: freq[x],
                reverse=True
            )
            if filtered:
                self._insert(filtered, self.root, freq)

    def _insert(self, items, node, freq):
        if not items:
            return
        item = items[0]
        if item in node.children:
            node.children[item].count += 1
        else:
            child = FPNode(item, 1, node)
            node.children[item] = child
            # Update header table
            if item not in self.header_table:
                self.header_table[item] = [0, None]
            # Append to linked list
            if self.header_table[item][1] is None:
                self.header_table[item][1] = child
            else:
                cur = self.header_table[item][1]
                while cur.link:
                    cur = cur.link
                cur.link = child
        self.header_table[item][0] = self.header_table[item][0] + 1 if item in self.header_table else 1
        self._insert(items[1:], node.children[item], freq)

    def nodes(self, item):
        """Traverse linked list for an item."""
        node = self.header_table.get(item, [0, None])[1]
        while node:
            yield node
            node = node.link

    def is_single_path(self):
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()))
        return True


# ── FP-Growth ─────────────────────────────────────────────────────────────────
def fpgrowth_mine(tree, min_sup_count, prefix, freq_itemsets):
    items = sorted(
        tree.header_table.keys(),
        key=lambda x: tree.header_table[x][0]
    )
    for item in items:
        new_prefix = prefix + [item]
        support    = tree.header_table[item][0]
        freq_itemsets[frozenset(new_prefix)] = support

        # Build conditional pattern base
        cond_pat_base = []
        for node in tree.nodes(item):
            path   = []
            parent = node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                cond_pat_base.append((path, node.count))

        # Build conditional FP-tree
        cond_transactions = []
        for path, count in cond_pat_base:
            cond_transactions.extend([path] * count)

        if cond_transactions:
            cond_tree = FPTree(cond_transactions, min_sup_count)
            if cond_tree.header_table:
                fpgrowth_mine(cond_tree, min_sup_count, new_prefix, freq_itemsets)


def fpgrowth(transactions, min_sup):
    n             = len(transactions)
    min_sup_count = int(min_sup * n)
    freq_itemsets = {}

    tree = FPTree(transactions, min_sup_count)
    fpgrowth_mine(tree, min_sup_count, [], freq_itemsets)
    return freq_itemsets


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FP-GROWTH — Frequent Itemset Mining")
    print("=" * 60)

    transactions = load_transactions(DATASET_PATH, MAX_ROWS)
    print(f"  Dataset      : {DATASET_PATH}")
    print(f"  Transactions : {len(transactions)}")
    print(f"  Min Support  : {MIN_SUP:.0%}")
    print("-" * 60)

    tracemalloc.start()
    t_start = time.perf_counter()

    freq_itemsets = fpgrowth(transactions, MIN_SUP)

    t_end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Frequent Itemsets : {len(freq_itemsets)}")
    print(f"  Execution Time    : {t_end - t_start:.4f} s")
    print(f"  Peak Memory       : {peak_mem / 1024 / 1024:.2f} MB")
    print("=" * 60)

    # Show top-10
    print("\nTop-10 Frequent Itemsets by Support Count:")
    sorted_fi = sorted(freq_itemsets.items(), key=lambda x: -x[1])
    for itemset, count in sorted_fi[:10]:
        sup = count / len(transactions)
        print(f"  {set(itemset)}  →  support={sup:.4f}  (count={count})")
