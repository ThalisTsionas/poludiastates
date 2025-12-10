#KODIKAS GIA TESTING
from lsh_text import MinHashLSH
from typing import List, Set

def jaccard(a:Set[str], b:Set[str]) -> float:
    if not a and not b: return 1.0 #ean ta 2 sets einai adeia tote to similarity=1

    inter = len(a & b) #megethos tou intersection
    union = len(a | b) #megethos tou union
    return 0.0 if union == 0 else inter / union #gia asfaleia ean union = 0

#brute force lisi gia mikro dataset, isos na min einai toso kalo gia to dataset pou exoume tora
def brute_top1(docs: List[Set[str]], q:Set[str]) -> int:
    #ftiaxnoume lista apo score,index gia kathe doc
    scores = [(jaccard(q, docs[i]), i) for i in range(len(docs))]
    #sort analoga me to score kai meta me to index
    scores.sort(reverse=True)
    #return to highest scoring doc
    return scores[0][1]

def test_quick():
    #ta docs exoun 5 pithana idi
    #to proto einai ean to genre einai sxedon idio
    #to deutero ean exei koino ena token me to doc0
    #ean den exei kapoio koino me to action doc
    #superset tou doc0
    #keno set
    docs = [
        {"action","adventure"},
        {"action","thriller"},
        {"romance"},
        {"action","adventure","fantasy"},
        set(),
    ]

    #kanoume 2 idia lsh kai theloume na kanoun match
    lsh1 = MinHashLSH(num_perm=64, num_bands=8, seed=123, fallback_all=False)
    lsh1.fit(docs)
    r1 = lsh1.query({"action","adventure"})

    lsh2 = MinHashLSH(num_perm=64, num_bands=8, seed=123, fallback_all=False)
    lsh2.fit(docs)
    r2 = lsh2.query({"action","adventure"})
    #ean den kanoun match tote den einai deterministic
    assert r1 == r2, f"Determinism failed: {r1} != {r2}"

    for i, s in enumerate(docs):
        if not s:
            continue
        res = lsh1.query(s)
        assert i in res, f"Self-query failed for doc {i}, got {res}"

    #mikro test, to top1 prepei na kanei match (brute force)
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

    print("All test Passed (LSH)")

if __name__ == "__main__":
    test_quick()

