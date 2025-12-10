# quick_lsh_check.py
from lsh_text import MinHashLSH
from typing import List, Set

def jaccard(a:Set[str], b:Set[str]) -> float:
    if not a and not b: return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union

def brute_top1(docs: List[Set[str]], q:Set[str]) -> int:
    scores = [(jaccard(q, docs[i]), i) for i in range(len(docs))]
    scores.sort(reverse=True)
    return scores[0][1]

def test_quick():
    docs = [
        {"action","adventure"},
        {"action","thriller"},
        {"romance"},
        {"action","adventure","fantasy"},
        set(),
    ]

    # --- determinism ---
    lsh1 = MinHashLSH(num_perm=64, num_bands=8, seed=123, fallback_all=False)
    lsh1.fit(docs)
    r1 = lsh1.query({"action","adventure"})

    lsh2 = MinHashLSH(num_perm=64, num_bands=8, seed=123, fallback_all=False)
    lsh2.fit(docs)
    r2 = lsh2.query({"action","adventure"})

    assert r1 == r2, f"Determinism failed: {r1} != {r2}"

    # --- self retrieval (non-empty docs) ---
    for i, s in enumerate(docs):
        if not s:
            continue
        res = lsh1.query(s)
        assert i in res, f"Self-query failed for doc {i}, got {res}"

    # --- tiny accuracy test: top-1 must match brute-force ---
    # create a clear best match
    docs2 = [
        {"a","b"},
        {"a"},
        {"b"},
        {"a","b","c"},
    ]
    lsh3 = MinHashLSH(num_perm=128, num_bands=16, seed=42, fallback_all=False)
    lsh3.fit(docs2)
    q = {"a","b"}
    lsh_top = lsh3.query(q)
    lsh_top1 = lsh_top[0] if lsh_top else None
    brute = brute_top1(docs2, q)
    assert lsh_top1 == brute, f"Top-1 mismatch: LSH gave {lsh_top1}, brute-force {brute}"

    print("All quick LSH checks passed ✅")

if __name__ == "__main__":
    test_quick()
