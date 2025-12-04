from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple


@dataclass
class RangeNode:
    """Κόμβος ενός 1D range tree (balanced binary search tree)."""

    value: float               # η τιμή στον 1D άξονα (π.χ. release_year)
    index: int                 # index του σημείου στο αρχικό array
    left: Optional["RangeNode"] = None
    right: Optional["RangeNode"] = None


class RangeTree:
    """
    Απλή υλοποίηση 1D Range Tree (ουσιαστικά balanced BST πάνω σε ένα attribute).
    Τη χρησιμοποιούμε για γρήγορα range queries σε μία διάσταση (π.χ. release_year).
    """

    def __init__(self, values: Sequence[float]):
        """
        values: λίστα ή numpy array με τις τιμές του attribute που κάνουμε index
                (π.χ. όλα τα release_year του base pool).
        """
        self.n_points = len(values)
        self._values = values

        if self.n_points == 0:
            self.root = None
        else:
            # Δημιουργούμε λίστα από (value, index) και τη φτιάχνουμε σε balanced BST.
            pairs: List[Tuple[float, int]] = [(float(v), i) for i, v in enumerate(values)]
            pairs.sort(key=lambda p: p[0])  # ταξινόμηση κατά value
            self.root = self._build(pairs)

    def _build(self, pairs: List[Tuple[float, int]]) -> Optional[RangeNode]:
        """Αναδρομική κατασκευή balanced BST από ταξινομημένη λίστα (value, index)."""
        if not pairs:
            return None

        mid = len(pairs) // 2
        value, index = pairs[mid]

        node = RangeNode(value=value, index=index)
        node.left = self._build(pairs[:mid])
        node.right = self._build(pairs[mid + 1 :])

        return node

    def is_empty(self) -> bool:
        """Επιστρέφει True αν το δέντρο είναι άδειο."""
        return self.root is None

    def __len__(self) -> int:
        """Επιστρέφει πόσα points έχουν γίνει index."""
        return self.n_points

    # ---------- RANGE QUERY ----------

    def range_query(self, low: float, high: float) -> List[int]:
        """
        Επιστρέφει indices σημείων για τα οποία:
        low <= value <= high
        όπου value είναι η τιμή που κάναμε index (π.χ. release_year).
        """
        results: List[int] = []
        self._range_query_node(self.root, low, high, results)
        return results

    def _range_query_node(
        self,
        node: Optional[RangeNode],
        low: float,
        high: float,
        results: List[int],
    ) -> None:
        """Αναδρομική συνάρτηση για range query στον 1D άξονα."""
        if node is None:
            return

        # Αν η τιμή του node είναι μέσα στο [low, high], προσθέτουμε τον index
        if low <= node.value <= high:
            results.append(node.index)

        # Αν low < node.value, μπορεί να υπάρχουν τιμές στο range στο αριστερό υποδέντρο
        if low < node.value:
            self._range_query_node(node.left, low, high, results)

        # Αν node.value < high, μπορεί να υπάρχουν τιμές στο range στο δεξί υποδέντρο
        if node.value < high:
            self._range_query_node(node.right, low, high, results)
