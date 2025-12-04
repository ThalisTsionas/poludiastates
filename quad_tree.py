from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple


Bounds = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


@dataclass
class QuadNode:
    """Κόμβος ενός απλού point quad-tree σε 2D."""

    x: float
    y: float
    index: int                    # index του σημείου στον πίνακα points
    bounds: Bounds                # ορθογώνιο κουτί που καλύπτει αυτό το node
    capacity: int = 1             # μέγιστος αριθμός points πριν γίνει subdivision
    points: Optional[List[int]] = None  # λίστα από indices σημείων (αν leaf)
    nw: Optional["QuadNode"] = None
    ne: Optional["QuadNode"] = None
    sw: Optional["QuadNode"] = None
    se: Optional["QuadNode"] = None

    def __post_init__(self):
        # Αρχικοποιούμε τη λίστα points με το index του πρώτου σημείου
        self.points = [self.index]

    def is_leaf(self) -> bool:
        """Επιστρέφει True αν ο κόμβος δεν έχει παιδιά."""
        return self.nw is None and self.ne is None and self.sw is None and self.se is None


class QuadTree:
    """
    Απλή υλοποίηση point quad-tree για 2D σημεία.
    Τη χρησιμοποιούμε σε 2 διαστάσεις (π.χ. popularity, vote_average).
    """

    def __init__(self, points: Sequence[Sequence[float]], bounds: Optional[Bounds] = None, capacity: int = 4):
        """
        points: λίστα ή numpy array από 2D σημεία [x, y]
        bounds: (xmin, ymin, xmax, ymax). Αν είναι None, τα υπολογίζουμε από τα points.
        capacity: πόσα points επιτρέπουμε σε κάθε node πριν subdivide.
        """
        self.points = points
        self.capacity = capacity

        if len(points) == 0:
            self.root = None
            return

        if bounds is None:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            # Προσθέτουμε λίγο "margin" για να μην έχουμε degenerate bounds
            dx = xmax - xmin or 1.0
            dy = ymax - ymin or 1.0
            bounds = (xmin - 0.01 * dx, ymin - 0.01 * dy, xmax + 0.01 * dx, ymax + 0.01 * dy)

        self.root_bounds = bounds
        self.root: Optional[QuadNode] = None

        # Εισάγουμε όλα τα σημεία ένα-ένα
        for idx, p in enumerate(points):
            self.insert(idx, p[0], p[1])

    def insert(self, index: int, x: float, y: float) -> None:
        """Εισάγει ένα point (x, y) με index στο quad-tree."""
        if self.root is None:
            self.root = QuadNode(x=x, y=y, index=index, bounds=self.root_bounds, capacity=self.capacity)
            return

        self._insert_node(self.root, index, x, y)

    def _insert_node(self, node: QuadNode, index: int, x: float, y: float) -> None:
        """Αναδρομική εισαγωγή σε υποδέντρα."""
        if node.is_leaf() and len(node.points) < node.capacity:
            # Υπάρχει χώρος σε αυτό το leaf node
            node.points.append(index)
            return

        if node.is_leaf():
            # Πρέπει να γίνει subdivision πριν εισάγουμε νέο point
            self._subdivide(node)

        # Βρίσκουμε σε ποιο child ανήκει το point
        child = self._choose_child(node, x, y)
        if child is None:
            # Αν για κάποιο λόγο δεν βρίσκεται (αριθμητικά σφάλματα), το βάζουμε εδώ
            node.points.append(index)
        else:
            self._insert_node(child, index, x, y)

    def _subdivide(self, node: QuadNode) -> None:
        """Χωρίζει το node σε 4 παιδιά."""
        xmin, ymin, xmax, ymax = node.bounds
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0

        # Δημιουργία παιδιών με τα κατάλληλα bounds
        node.nw = QuadNode(x=node.x, y=node.y, index=node.points[0],
                           bounds=(xmin, ymid, xmid, ymax), capacity=self.capacity)
        node.ne = QuadNode(x=node.x, y=node.y, index=node.points[0],
                           bounds=(xmid, ymid, xmax, ymax), capacity=self.capacity)
        node.sw = QuadNode(x=node.x, y=node.y, index=node.points[0],
                           bounds=(xmin, ymin, xmid, ymid), capacity=self.capacity)
        node.se = QuadNode(x=node.x, y=node.y, index=node.points[0],
                           bounds=(xmid, ymin, xmax, ymid), capacity=self.capacity)

         # ΝΕΟ: καθαρίζουμε τα points των παιδιών, θα τα ξαναμοιράσουμε
        node.nw.points = []
        node.ne.points = []
        node.sw.points = []
        node.se.points = []

        # Αναδιανομή των ήδη αποθηκευμένων points του node
        old_points = node.points
        node.points = []  # καθαρίζουμε τα points από τον parent

        for idx in old_points:
            px, py = self.points[idx]
            child = self._choose_child(node, px, py)
            if child is not None:
                child.points.append(idx)
            else:
                # Αν δεν βρέθηκε child (σπάνια), το αφήνουμε στον parent (degraded)
                node.points.append(idx)

    def _choose_child(self, node: QuadNode, x: float, y: float) -> Optional[QuadNode]:
        """Επιστρέφει σε ποιο από τα 4 παιδιά πρέπει να πάει το point (x, y)."""
        xmin, ymin, xmax, ymax = node.bounds
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0

        if y >= ymid:
            # πάνω μέρος
            if x < xmid:
                return node.nw
            else:
                return node.ne
        else:
            # κάτω μέρος
            if x < xmid:
                return node.sw
            else:
                return node.se

    # ---------- RANGE QUERY ----------

    def range_query(self, rect: Bounds) -> List[int]:
        """
        Επιστρέφει indices σημείων που βρίσκονται μέσα στο ορθογώνιο rect:
        rect = (xmin, ymin, xmax, ymax)
        """
        results: List[int] = []
        if self.root is None:
            return results

        self._range_query_node(self.root, rect, results)
        return results

    def _rect_intersects(self, a: Bounds, b: Bounds) -> bool:
        """Ελέγχει αν δύο ορθογώνια τέμνονται."""
        axmin, aymin, axmax, aymax = a
        bxmin, bymin, bxmax, bymax = b
        return not (axmax < bxmin or bxmax < axmin or aymax < bymin or bymax < aymin)

    def _rect_contains_point(self, rect: Bounds, x: float, y: float) -> bool:
        """Ελέγχει αν το ορθογώνιο rect περιέχει το σημείο (x, y)."""
        xmin, ymin, xmax, ymax = rect
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def _range_query_node(self, node: QuadNode, rect: Bounds, results: List[int]) -> None:
        """Αναδρομική συνάρτηση για range query."""
        if node is None:
            return

        # Αν το rect δεν τέμνει το node.bounds, αγνοούμε αυτό το υποδέντρο
        if not self._rect_intersects(node.bounds, rect):
            return

        # Ελέγχουμε τα points σε αυτό το node (αν leaf ή αν έχει points)
        if node.points:
            for idx in node.points:
                px, py = self.points[idx]
                if self._rect_contains_point(rect, px, py):
                    results.append(idx)

        # Συνεχίζουμε στα παιδιά
        for child in (node.nw, node.ne, node.sw, node.se):
            if child is not None:
                self._range_query_node(child, rect, results)
