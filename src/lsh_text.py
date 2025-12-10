from __future__ import annotations
from typing import List, Set, Sequence, Dict, Tuple, Iterable, Optional
import hashlib
import math
import random

class MinHashLSH:
    """
    Improved MinHash + LSH for sets of tokens.
    - Deterministic token hashing using SHA-1 (stable across runs).
    - Use tuple(band_slice) as band dictionary key (no Python hash()).
    - Optional fallback behavior controlled by `fallback_all`.
    """
    def __init__(
        self,
        num_perm: int = 64,
        num_bands: int = 8,
        seed: int = 42,
        fallback_all: bool = False,
    ):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.seed = seed
        self.fallback_all = fallback_all

        self._rand = random.Random(seed)
        #xrisimopoioume 64-bit proto arithmo
        self._prime = 18446744073709551557

        self._hash_params = self._generate_hash_params(num_perm)

        self.doc_sets: List[Set[str]] = []
        self.signatures: List[List[int]] = []
        #bucket key: (band_index, tuple_of_ints)
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}

    def _generate_hash_params(self, k: int):
        params = []
        for _ in range(k):
            a = self._rand.randrange(1, self._prime - 1)
            b = self._rand.randrange(0, self._prime - 1)
            params.append((a, b))
        return params

    @staticmethod
    def _token_to_int(token: str) -> int:
        """
        Deterministic integer for a token: take SHA-1 and use first 8 bytes (64-bit).
        This is stable across runs and machines.
        """
        h = hashlib.sha1(token.encode("utf8")).digest()
        return int.from_bytes(h[:8], "big")

    def _minhash_signature(self, s: Iterable[str]) -> List[int]:
        """Compute MinHash signature for a set/iterable of tokens."""
        #an s einai adeio tote epistrefei high sentinel ipografi
        if not s:
            return [self._prime] * self.num_perm

        sig = [self._prime] * self.num_perm
        for token in s:
            x = self._token_to_int(token)
            for i, (a, b) in enumerate(self._hash_params):
                hv = (a * x + b) % self._prime
                if hv < sig[i]:
                    sig[i] = hv
        return sig

    def fit(self, docs: Sequence[Set[str]]) -> None:
        """
        Build LSH index from list of token sets.
        """
        self.doc_sets = [set(d) for d in docs]
        self.signatures = []
        self.buckets = {}

        for doc_id, s in enumerate(self.doc_sets):
            sig = self._minhash_signature(s)
            self.signatures.append(sig)

            #LSH banding pou xrisimopoioume ints gia keys
            for band in range(self.num_bands):
                start = band * self.rows_per_band
                end = start + self.rows_per_band
                band_slice = tuple(sig[start:end])
                key = (band, band_slice)
                self.buckets.setdefault(key, []).append(doc_id)

    def query(self, s: Set[str]) -> List[int]:
        """
        Return candidate doc ids sorted by exact Jaccard similarity (high -> low).
        If fallback_all is False and no candidates found, returns empty list.
        """
        if not self.doc_sets:
            return []

        sig = self._minhash_signature(s)
        candidates: set[int] = set()

        for band in range(self.num_bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(sig[start:end])
            key = (band, band_slice)
            bucket = self.buckets.get(key)
            if bucket:
                candidates.update(bucket)

        if not candidates:
            if self.fallback_all:
                candidates = set(range(len(self.doc_sets)))
            else:
                return []

        #rerank me jaccard similarity
        scored = []
        for doc_id in candidates:
            sim = self.jaccard(s, self.doc_sets[doc_id])
            scored.append((sim, doc_id))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc_id for sim, doc_id in scored]

    @staticmethod
    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

