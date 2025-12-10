from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


BBox = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def bbox_area(b: BBox) -> float:
    xmin, ymin, xmax, ymax = b
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def bbox_union(a: BBox, b: BBox) -> BBox:
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    return (
        min(axmin, bxmin),
        min(aymin, bymin),
        max(axmax, bxmax),
        max(aymax, bymax),
    )


def bbox_intersects(a: BBox, b: BBox) -> bool:
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b
    return not (axmax < bxmin or bxmax < axmin or aymax < bymin or bymax < aymin)


@dataclass
class RTreeEntry:
    """Μια εγγραφή σε R-tree: bounding box + είτε child node είτε index point."""
    bbox: BBox
    child: Optional["RTreeNode"] = None  # αν είναι internal node
    index: Optional[int] = None          # αν είναι leaf entry


@dataclass
class RTreeNode:
    """Κόμβος ενός R-tree."""
    is_leaf: bool
    entries: List[RTreeEntry]
    parent: Optional["RTreeNode"] = None


class RTree:
    """
    Απλή υλοποίηση R-tree για 2D points.
    Χρησιμοποιούμε πολύ βασικό linear split για υπερχείλιση.
    """

    def __init__(self, max_entries: int = 16, min_entries: Optional[int] = None):
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max_entries // 2
        self.root = RTreeNode(is_leaf=True, entries=[])

    #insertion
    def insert(self, index: int, bbox: BBox) -> None:
        """Εισάγει ένα point (ως degenerate rectangle) στο R-tree."""
        leaf = self._choose_leaf(self.root, bbox)
        leaf.entries.append(RTreeEntry(bbox=bbox, index=index))

        #overflow -> split
        if len(leaf.entries) > self.max_entries:
            self._handle_overflow(leaf)
        else:
            #update ta bboxes pros ta pano
            self._adjust_bboxes_upwards(leaf)

    def _choose_leaf(self, node: RTreeNode, bbox: BBox) -> RTreeNode:
        """Επιλογή leaf node όπου θα μπει το bbox."""
        if node.is_leaf:
            return node

        #dialegoume to child pou xreiazetai tin ligoteri auxisi emvadou
        best_entry = None
        best_increase = None

        for entry in node.entries:
            old_area = bbox_area(entry.bbox)
            new_area = bbox_area(bbox_union(entry.bbox, bbox))
            increase = new_area - old_area
            if best_increase is None or increase < best_increase:
                best_increase = increase
                best_entry = entry

        return self._choose_leaf(best_entry.child, bbox)

    def _handle_overflow(self, node: RTreeNode) -> None:
        """Διαχειρίζεται υπερχείλιση κόμβου (split)."""
        new_node = self._split_node(node)

        if node.parent is None:
            #nea riza
            new_root = RTreeNode(is_leaf=False, entries=[])
            node.parent = new_root
            new_node.parent = new_root

            #bbox ton 2 children
            bbox1 = self._compute_node_bbox(node)
            bbox2 = self._compute_node_bbox(new_node)

            new_root.entries.append(RTreeEntry(bbox=bbox1, child=node))
            new_root.entries.append(RTreeEntry(bbox=bbox2, child=new_node))

            self.root = new_root
        else:
            #new node -> parent
            parent = node.parent
            bbox_new = self._compute_node_bbox(new_node)
            parent.entries.append(RTreeEntry(bbox=bbox_new, child=new_node))
            new_node.parent = parent

            #update to bbox tou old node ston parent
            for entry in parent.entries:
                if entry.child is node:
                    entry.bbox = self._compute_node_bbox(node)
                    break

            #an kanei overflow o aprent tote kanoume split parapano allios kanoume update ta bbox
            if len(parent.entries) > self.max_entries:
                self._handle_overflow(parent)
            else:
                self._adjust_bboxes_upwards(parent)

    def _adjust_bboxes_upwards(self, node: RTreeNode) -> None:
        """
        Ενημερώνει τα bounding boxes στους προγόνους του node,
        μετά από αλλαγή στις entries του node (insert / split).
        """
        current = node
        while current.parent is not None:
            parent = current.parent
            for entry in parent.entries:
                if entry.child is current:
                    entry.bbox = self._compute_node_bbox(current)
                    break
            current = parent

    def _split_node(self, node: RTreeNode) -> RTreeNode:
        """
        Πολύ απλός linear split:
        - Διαλέγει δύο seeds (τα πιο απομακρυσμένα μεταξύ τους)
        - Μοιράζει τις υπόλοιπες entries με βάση ποιο group αυξάνει λιγότερο το bbox.
        """
        entries = node.entries
        n = len(entries)

        if n <= self.max_entries:
            return node  # no split needed

        #ta 2 seeds
        best_pair = (0, 1)
        best_dist = -1.0

        for i in range(n):
            for j in range(i + 1, n):
                e1 = entries[i].bbox
                e2 = entries[j].bbox
                #apostasi ton kentron ton 2 boxes
                cx1 = (e1[0] + e1[2]) / 2.0
                cy1 = (e1[1] + e1[3]) / 2.0
                cx2 = (e2[0] + e2[2]) / 2.0
                cy2 = (e2[1] + e2[3]) / 2.0
                dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        i1, i2 = best_pair

        #2 nea nodes
        group1_entries = [entries[i1]]
        group2_entries = [entries[i2]]

        bbox1 = entries[i1].bbox
        bbox2 = entries[i2].bbox

        #share ta ipoloipa entries
        for k in range(n):
            if k == i1 or k == i2:
                continue
            e = entries[k]
            b = e.bbox

            #an xreiazetai na exanagkasoume gia min entries
            remaining = n - len(group1_entries) - len(group2_entries)
            if len(group1_entries) + remaining == self.min_entries:
                group1_entries.append(e)
                bbox1 = bbox_union(bbox1, b)
                continue
            if len(group2_entries) + remaining == self.min_entries:
                group2_entries.append(e)
                bbox2 = bbox_union(bbox2, b)
                continue

            #auxisi emvadou gia kathe group
            new_bbox1 = bbox_union(bbox1, b)
            new_bbox2 = bbox_union(bbox2, b)
            inc1 = bbox_area(new_bbox1) - bbox_area(bbox1)
            inc2 = bbox_area(new_bbox2) - bbox_area(bbox2)

            if inc1 < inc2:
                group1_entries.append(e)
                bbox1 = new_bbox1
            else:
                group2_entries.append(e)
                bbox2 = new_bbox2

        #old node -> group1 , group2 -> new node
        node.entries = group1_entries
        new_node = RTreeNode(is_leaf=node.is_leaf, entries=group2_entries, parent=node.parent)

        return new_node

    def _compute_node_bbox(self, node: RTreeNode) -> BBox:
        """Υπολογίζει το bounding box που καλύπτει όλες τις entries ενός node."""
        assert node.entries, "Node has no entries"
        bb = node.entries[0].bbox
        for e in node.entries[1:]:
            bb = bbox_union(bb, e.bbox)
        return bb


    def range_query(self, query_rect: BBox) -> List[int]:
        """
        Επιστρέφει indices σημείων για τα οποία το bbox τους τέμνει το query_rect.
        Στην περίπτωσή μας τα points είναι degenerate rectangles (x, y, x, y).
        """
        results: List[int] = []
        self._range_query_node(self.root, query_rect, results)
        return results

    def _range_query_node(self, node: Optional[RTreeNode], query_rect: BBox, results: List[int]) -> None:
        if node is None:
            return

        for entry in node.entries:
            if not bbox_intersects(entry.bbox, query_rect):
                continue

            if node.is_leaf:
                #an temnei to leaf tote kanoume add to index
                if entry.index is not None:
                    results.append(entry.index)
            else:
                #internal node -> katevainoume sto child
                self._range_query_node(entry.child, query_rect, results)

