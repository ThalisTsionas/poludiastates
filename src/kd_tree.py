from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class KDNode:
    """Κόμβος ενός k-d tree."""

    point: Sequence[float]          #to simeio ston k-diastato xoro
    index: int                      #index tou simeiou sto arxiko array
    axis: int                       #poio dimension xrismipoiitai gia to split
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    """Απλή υλοποίηση k-d tree με median split."""

    def __init__(self, points: Sequence[Sequence[float]]):
        """
        points: λίστα ή numpy array από k-διάστατα σημεία
                (π.χ. feature_matrix από το main).
        """
        self.n_points = len(points)
        self.k = len(points[0]) if self.n_points > 0 else 0

        self._points = points

        if self.n_points == 0:
            self.root = None
        else:
            indices = list(range(self.n_points))
            self.root = self._build(indices, depth=0)

    def _build(self, indices: List[int], depth: int) -> Optional[KDNode]:
        """
        Αναδρομική κατασκευή k-d tree.
        indices: λίστα από indices προς τα σημεία στο self._points.
        depth: τρέχον βάθος στο δέντρο, για axis = depth mod k.
        """
        if not indices:
            return None

        axis = depth % self.k

        #taxinomisi ton indexes me vasi tin timi tou axona
        indices.sort(key=lambda i: self._points[i][axis])

        #epilogi median gia isorropimeno dentro
        median_pos = len(indices) // 2
        median_index = indices[median_pos]
        median_point = self._points[median_index]

        #node creation
        node = KDNode(
            point=median_point,
            index=median_index,
            axis=axis,
        )

        #ftiaxnoume anadromika ta ipodentra
        left_indices = indices[:median_pos]
        right_indices = indices[median_pos + 1 :]

        node.left = self._build(left_indices, depth + 1)
        node.right = self._build(right_indices, depth + 1)

        return node

    def is_empty(self) -> bool:
        """Επιστρέφει True αν το δέντρο είναι άδειο."""
        return self.root is None

    def __len__(self) -> int:
        """Επιστρέφει πόσα σημεία περιέχει το δέντρο."""
        return self.n_points
    #gia to range

    def range_query(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> List[int]:
        """
        Επιστρέφει λίστα από indices σημείων που βρίσκονται μέσα στο
        ορθογώνιο/υπερκυβικό range που ορίζεται από lower_bounds και upper_bounds.

        lower_bounds[d] <= point[d] <= upper_bounds[d] για κάθε διάσταση d.
        """
        if self.root is None or self.k == 0:
            return []

        if len(lower_bounds) != self.k or len(upper_bounds) != self.k:
            raise ValueError("Bounds must have length equal to k dimensions")

        results: List[int] = []
        self._range_query(self.root, lower_bounds, upper_bounds, results)
        return results

    def _range_query(
        self,
        node: Optional[KDNode],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        results: List[int],
    ) -> None:
        """Αναδρομική συνάρτηση για range query."""
        if node is None:
            return

        point = node.point
        axis = node.axis

        #checkaroume ean to simeio einai mesa sto range se OLA TA DIMENSIONS
        inside = True
        for dim in range(self.k):
            if point[dim] < lower_bounds[dim] or point[dim] > upper_bounds[dim]:
                inside = False
                break

        if inside:
            results.append(node.index)

        #aristero ipodentro
        if node.left is not None and lower_bounds[axis] <= point[axis]:
            self._range_query(node.left, lower_bounds, upper_bounds, results)

        #dexio ipodentro
        if node.right is not None and upper_bounds[axis] >= point[axis]:
            self._range_query(node.right, lower_bounds, upper_bounds, results)

