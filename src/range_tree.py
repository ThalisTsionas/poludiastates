from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple


@dataclass
class RangeNode:
    """Κόμβος ενός 1D range tree (balanced binary search tree)."""

    value: float               #to value ston 1d axis
    index: int                 #index sto arxiko array
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
            #ftiaxnoume lista apo value,index kai tin ftiaxnoume se balanced BST
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

        #an i timi einai mesa sto range to prosthetoume to index
        if low <= node.value <= high:
            results.append(node.index)

        #an einai <low tote mporei na iparxoun times sto range sto aristera ipodentro
        if low < node.value:
            self._range_query_node(node.left, low, high, results)

        #allios sto dexia
        if node.value < high:
            self._range_query_node(node.right, low, high, results)

