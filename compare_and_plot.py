

import time
import tracemalloc
from itertools import combinations
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASETS = {
    "Chess":   ("trimmed_chess_1.dat",   [0.80, 0.85, 0.90, 0.95]),
    "Connect": ("trimmed_connect_1.dat", [0.90, 0.92, 0.95, 0.97]),
}
MAX_ROWS = 1000       # half of 2000-row files
OUT_DIR  = "results"  # PNGs saved here
os.makedirs(OUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  APRIORI
# ══════════════════════════════════════════════════════════════════════════════
def load_transactions(path, max_rows=None):
    txs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            items = frozenset(line.strip().split())
            if items:
                txs.append(items)
    return txs


def _get_freq1(transactions, msc):
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[frozenset([item])] += 1
    return {k: v for k, v in counts.items() if v >= msc}


def _apriori_gen(freq_k, k):
    freq_list = list(freq_k.keys())
    candidates = set()
    for i in range(len(freq_list)):
        for j in range(i + 1, len(freq_list)):
            union = freq_list[i] | freq_list[j]
            if len(union) == k:
                if all(frozenset(s) in freq_k for s in combinations(union, k - 1)):
                    candidates.add(union)
    return candidates


def run_apriori(transactions, min_sup):
    msc = int(min_sup * len(transactions))
    all_freq = {}
    total_cands = 0

    freq_k = _get_freq1(transactions, msc)
    total_cands += len(freq_k)
    all_freq.update(freq_k)

    k = 2
    while freq_k:
        cands = _apriori_gen(freq_k, k)
        total_cands += len(cands)
        counts = defaultdict(int)
        for t in transactions:
            for c in cands:
                if c.issubset(t):
                    counts[c] += 1
        freq_k = {c: v for c, v in counts.items() if v >= msc}
        all_freq.update(freq_k)
        k += 1

    return len(all_freq), total_cands


# ══════════════════════════════════════════════════════════════════════════════
#  FP-GROWTH
# ══════════════════════════════════════════════════════════════════════════════
class FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")
    def __init__(self, item, count, parent):
        self.item = item; self.count = count; self.parent = parent
        self.children = {}; self.link = None


class FPTree:
    def __init__(self, transactions, msc):
        self.root = FPNode(None, 1, None)
        self.header = {}   # item -> [support, first_node]
        freq = defaultdict(int)
        for t in transactions:
            for item in t: freq[item] += 1
        self.freq_items = {k for k, v in freq.items() if v >= msc}
        for t in transactions:
            filtered = sorted([i for i in t if i in self.freq_items],
                              key=lambda x: freq[x], reverse=True)
            if filtered:
                self._insert(filtered, self.root)

    def _insert(self, items, node):
        if not items: return
        item = items[0]
        if item in node.children:
            node.children[item].count += 1
        else:
            child = FPNode(item, 1, node)
            node.children[item] = child
            if item not in self.header:
                self.header[item] = [0, None]
            cur = self.header[item][1]
            if cur is None:
                self.header[item][1] = child
            else:
                while cur.link: cur = cur.link
                cur.link = child
        self.header[item][0] += 1
        self._insert(items[1:], node.children[item])

    def nodes(self, item):
        node = self.header.get(item, [0, None])[1]
        while node:
            yield node
            node = node.link


def _fpg_mine(tree, msc, prefix, result):
    for item in sorted(tree.header, key=lambda x: tree.header[x][0]):
        new_prefix = prefix + [item]
        support    = tree.header[item][0]
        result[frozenset(new_prefix)] = support
        # Conditional pattern base
        cond_txs = []
        for node in tree.nodes(item):
            path = []
            p = node.parent
            while p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                cond_txs.extend([path] * node.count)
        if cond_txs:
            cond_tree = FPTree(cond_txs, msc)
            if cond_tree.header:
                _fpg_mine(cond_tree, msc, new_prefix, result)


def run_fpgrowth(transactions, min_sup):
    # Convert frozensets to lists for FP-tree
    raw = []
    for t in transactions:
        raw.append(list(t))
    msc    = int(min_sup * len(raw))
    result = {}
    tree   = FPTree(raw, msc)
    _fpg_mine(tree, msc, [], result)
    return len(result)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
def measure(fn, *args):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024 / 1024, out   # time, MB, return value


all_results = {}   # ds -> sup -> {algo -> (time, mem, itemsets, cands)}

for ds_name, (path, sups) in DATASETS.items():
    transactions = load_transactions(path, MAX_ROWS)
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name}  ({len(transactions)} transactions)")
    print(f"{'='*60}")
    all_results[ds_name] = {}

    for sup in sups:
        all_results[ds_name][sup] = {}

        # Apriori
        t, m, (n_fi, n_cands) = measure(run_apriori, transactions, sup)
        all_results[ds_name][sup]["Apriori"] = (t, m, n_fi, n_cands)
        print(f"  sup={sup:.0%}  Apriori    time={t:.4f}s  mem={m:.2f}MB  "
              f"itemsets={n_fi}  candidates={n_cands}")

        # FP-Growth
        t, m, n_fi = measure(run_fpgrowth, transactions, sup)
        all_results[ds_name][sup]["FP-Growth"] = (t, m, n_fi, 0)
        print(f"  sup={sup:.0%}  FP-Growth  time={t:.4f}s  mem={m:.2f}MB  "
              f"itemsets={n_fi}  candidates=N/A")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
CA = "#E74C3C"   # Apriori red
CF = "#2ECC71"   # FP-Growth green
DS = list(DATASETS.keys())

def x_labels(ds):
    return [f"{s:.0%}" for s in sorted(all_results[ds].keys())]

def metric(ds, algo, idx):
    return [all_results[ds][s][algo][idx] for s in sorted(all_results[ds].keys())]


# ── Fig 1: Execution Time ─────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 1 — Execution Time: Apriori vs FP-Growth", fontsize=13, fontweight="bold")
for ax, ds in zip(axs, DS):
    sups = sorted(all_results[ds])
    x = range(len(sups)); w = 0.35
    import numpy as np
    x = np.arange(len(sups))
    for i, (algo, col) in enumerate([("Apriori", CA), ("FP-Growth", CF)]):
        vals = metric(ds, algo, 0)
        bars = ax.bar(x + i*w - w/2, vals, w, label=algo, color=col,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                    f"{v:.3f}s", ha="center", va="bottom", fontsize=9)
    ax.set_title(ds, fontsize=12, fontweight="bold")
    ax.set_xlabel("Min Support Threshold")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xticks(x); ax.set_xticklabels(x_labels(ds))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_execution_time.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"\n[Saved] fig1_execution_time.png")


# ── Fig 2: Peak Memory ────────────────────────────────────────────────────────
import numpy as np
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 2 — Peak Memory Usage: Apriori vs FP-Growth", fontsize=13, fontweight="bold")
for ax, ds in zip(axs, DS):
    sups = sorted(all_results[ds])
    x = np.arange(len(sups))
    for i, (algo, col) in enumerate([("Apriori", CA), ("FP-Growth", CF)]):
        vals = metric(ds, algo, 1)
        bars = ax.bar(x + i*w - w/2, vals, w, label=algo, color=col,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(ds, fontsize=12, fontweight="bold")
    ax.set_xlabel("Min Support Threshold")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xticks(x); ax.set_xticklabels(x_labels(ds))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_memory_usage.png", dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] fig2_memory_usage.png")


# ── Fig 3: Frequent Itemsets ──────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 3 — Frequent Itemsets Discovered", fontsize=13, fontweight="bold")
for ax, ds in zip(axs, DS):
    sups = sorted(all_results[ds])
    x = np.arange(len(sups))
    for i, (algo, col) in enumerate([("Apriori", CA), ("FP-Growth", CF)]):
        vals = metric(ds, algo, 2)
        bars = ax.bar(x + i*w - w/2, vals, w, label=algo, color=col,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                    str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(ds, fontsize=12, fontweight="bold")
    ax.set_xlabel("Min Support Threshold")
    ax.set_ylabel("Number of Frequent Itemsets")
    ax.set_xticks(x); ax.set_xticklabels(x_labels(ds))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_itemsets.png", dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] fig3_itemsets.png")


# ── Fig 4: Speedup ────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 4 — Speedup of FP-Growth over Apriori", fontsize=13, fontweight="bold")
for ax, ds in zip(axs, DS):
    sups = sorted(all_results[ds])
    t_ap = metric(ds, "Apriori",   0)
    t_fp = metric(ds, "FP-Growth", 0)
    su   = [a / f if f > 0 else 1.0 for a, f in zip(t_ap, t_fp)]
    bars = ax.bar(x_labels(ds), su, color="#3498DB", alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Baseline (1×)")
    for bar, v in zip(bars, su):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(ds, fontsize=12, fontweight="bold")
    ax.set_xlabel("Min Support Threshold")
    ax.set_ylabel("Speedup (×)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_speedup.png", dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] fig4_speedup.png")


# ── Fig 5: Scalability Lines ──────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 5 — Scalability: Time vs Min Support", fontsize=13, fontweight="bold")
for ax, ds in zip(axs, DS):
    for algo, col, mk in [("Apriori", CA, "o"), ("FP-Growth", CF, "s")]:
        ax.plot(x_labels(ds), metric(ds, algo, 0),
                marker=mk, color=col, linewidth=2, markersize=8, label=algo)
    ax.set_title(ds, fontsize=12, fontweight="bold")
    ax.set_xlabel("Min Support Threshold")
    ax.set_ylabel("Time (seconds)")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_scalability.png", dpi=150, bbox_inches="tight")
plt.close(); print("[Saved] fig5_scalability.png")


# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Dataset':<10} {'Sup':>5} {'Algorithm':<12} {'Time(s)':>10} "
      f"{'Mem(MB)':>10} {'Itemsets':>10} {'Candidates':>12} {'Speedup':>9}")
print("-" * 80)
for ds in DS:
    for sup in sorted(all_results[ds]):
        t_ap, m_ap, fi_ap, c_ap = all_results[ds][sup]["Apriori"]
        t_fp, m_fp, fi_fp, _    = all_results[ds][sup]["FP-Growth"]
        speedup = t_ap / t_fp if t_fp > 0 else 0
        print(f"{ds:<10} {sup:>5.0%} {'Apriori':<12} {t_ap:>10.4f} {m_ap:>10.2f} "
              f"{fi_ap:>10} {c_ap:>12} {'—':>9}")
        print(f"{'':>15} {'FP-Growth':<12} {t_fp:>10.4f} {m_fp:>10.2f} "
              f"{fi_fp:>10} {'N/A':>12} {speedup:>8.2f}×")
print("=" * 80)
print(f"\nAll figures saved to: {OUT_DIR}/")
print("Download them from Colab: Files panel → results/ folder")
