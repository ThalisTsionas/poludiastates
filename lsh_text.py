from __future__ import annotations
from typing import List, Set, Sequence, Dict, Tuple
import random


class MinHashLSH:
    """
    Απλή υλοποίηση MinHash + LSH για σύνολα tokens (π.χ. genres).
    - Κάθε κείμενο -> σύνολο από tokens (set[str])
    - Υπολογίζουμε MinHash signatures
    - Χρησιμοποιούμε LSH banding για γρήγορο approximate similarity search
    """

    def __init__(self, num_perm: int = 64, num_bands: int = 8, seed: int = 42):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.seed = seed

        self._rand = random.Random(seed)
        self._prime = 4294967311  # μεγάλο πρώτο για hashing

        self._hash_params = self._generate_hash_params(num_perm)

        self.doc_sets: List[Set[str]] = []
        self.signatures: List[List[int]] = []
        self.buckets: Dict[Tuple[int, int], List[int]] = {}

    def _generate_hash_params(self, k: int):
        params = []
        for _ in range(k):
            a = self._rand.randint(1, self._prime - 1)
            b = self._rand.randint(0, self._prime - 1)
            params.append((a, b))
        return params

    def _minhash_signature(self, s: Set[str]) -> List[int]:
        """Υπολογίζει MinHash signature για ένα σύνολο tokens."""
        if not s:
            # για άδειο set, signature με πολύ μεγάλες τιμές
            return [self._prime] * self.num_perm

        sig = [self._prime] * self.num_perm
        for token in s:
            x = hash(token) & 0xFFFFFFFF  # 32-bit
            for i, (a, b) in enumerate(self._hash_params):
                hv = (a * x + b) % self._prime
                if hv < sig[i]:
                    sig[i] = hv
        return sig

    def fit(self, docs: Sequence[Set[str]]) -> None:
        """
        Χτίζει το LSH index πάνω σε λίστα από σύνολα tokens.
        docs: list[set[str]]
        """
        self.doc_sets = [set(d) for d in docs]
        self.signatures = []
        self.buckets = {}

        for doc_id, s in enumerate(self.doc_sets):
            sig = self._minhash_signature(s)
            self.signatures.append(sig)

            # LSH banding
            for band in range(self.num_bands):
                start = band * self.rows_per_band
                end = start + self.rows_per_band
                band_slice = tuple(sig[start:end])
                key = (band, hash(band_slice))
                self.buckets.setdefault(key, []).append(doc_id)

    def query(self, s: Set[str]) -> List[int]:
        """
        Επιστρέφει candidate doc ids ταξινομημένα κατά Jaccard similarity με το s.
        """
        if not self.doc_sets:
            return []

        sig = self._minhash_signature(s)
        candidates: set[int] = set()

        # Συγκέντρωση υποψήφιων από τα buckets
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(sig[start:end])
            key = (band, hash(band_slice))
            bucket = self.buckets.get(key)
            if bucket:
                candidates.update(bucket)

        # Αν δεν βρήκαμε κανέναν υποψήφιο, fallback: όλα τα docs candidates
        if not candidates:
            candidates = set(range(len(self.doc_sets)))

        # Ταξινόμηση των candidate με βάση την exact Jaccard similarity
        scored = []
        for doc_id in candidates:
            sim = self.jaccard(s, self.doc_sets[doc_id])
            scored.append((sim, doc_id))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc_id for sim, doc_id in scored]

    @staticmethod
    def jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard similarity μεταξύ δύο συνόλων."""
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union
