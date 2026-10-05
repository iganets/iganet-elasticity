#!/usr/bin/env python3
"""
Prepare the bone multipatch geometry for isogeometric collocation.

Strong-form collocation evaluates the Navier-Lame equation at interior
Greville points, which needs second derivatives and therefore degree >= 2 in
every parametric direction. bone_simplified.xml does not satisfy this: 13 of
its 16 patches carry degree 1 with only two control points in one direction,
so they contain no interior collocation point at all and cannot contribute a
single equation.

This script fixes that by degree elevation and knot insertion. Both operations
leave the geometry itself untouched, they only change how it is described, so
the resulting bone has exactly the same shape as before.

The patches must stay conforming: two patches sharing a face need the same
one-dimensional spline space along that face, otherwise their control points
no longer coincide and the interface coupling breaks. Parametric directions
are therefore grouped across interfaces (union-find over the direction map
stored in the XML) and every direction of a group is refined identically.

Usage:
  python3 -m std_collocation_python.prepare_bone_geometry [in.xml] [out.xml]
                                                          [--min-interior N]

--min-interior sets how many interior collocation points every direction must
end up with (default 2). Higher values give a finer reference solution and a
longer training run.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import splinepy

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "filedata" / "bone_simplified.xml"
DEFAULT_OUT = REPO_ROOT / "filedata" / "bone_collocation_ready.xml"

# Matching tolerance of the C++ multipatch code (set_matching_tolerance).
MATCH_TOL = 1e-6


def read_interface_lines(xml_path: Path):
    """Returns (patch1, side1, patch2, side2, direction_map) per interface.

    The gismo interface line is
        p1 s1 p2 s2 dmap0 dmap1 dmap2 orient0 orient1 orient2
    and direction i of patch1 corresponds to direction dmap[i] of patch2.
    Patch ids are converted to zero-based indices via the id_range offset.
    """
    root = ET.parse(xml_path).getroot()
    multipatch = root.find(".//MultiPatch")
    first_id = int(multipatch.find("patches").text.split()[0])

    result = []
    for block in multipatch.findall("interfaces"):
        for line in block.text.strip().split("\n"):
            values = line.split()
            if len(values) < 7:
                continue
            p1, s1, p2, s2 = (int(values[0]) - first_id, int(values[1]),
                              int(values[2]) - first_id, int(values[3]))
            dmap = [int(v) for v in values[4:7]]
            result.append((p1, s1, p2, s2, dmap))
    return result


def side_direction(side: int) -> int:
    """Parametric direction held fixed by a side (1,2 -> 0; 3,4 -> 1; 5,6 -> 2)."""
    return (side - 1) // 2


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_direction_groups(n_patches: int, interfaces):
    """Groups (patch, direction) pairs that must share the same spline space.

    Only the two in-face directions are linked. The direction perpendicular to
    a shared face may be refined independently without breaking conformity.
    """
    uf = UnionFind()
    for patch in range(n_patches):
        for direction in range(3):
            uf.find((patch, direction))

    for p1, s1, p2, s2, dmap in interfaces:
        fixed1 = side_direction(s1)
        for d in range(3):
            if d == fixed1:
                continue
            uf.union((p1, d), (p2, dmap[d]))

    groups = {}
    for patch in range(n_patches):
        for direction in range(3):
            groups.setdefault(uf.find((patch, direction)), []).append((patch, direction))
    return list(groups.values())


def interior_knots(knot_vector, degree):
    """Interior knot values with their multiplicities."""
    kv = np.asarray(knot_vector, dtype=float)
    inner = kv[degree + 1:len(kv) - degree - 1]
    values, counts = np.unique(np.round(inner, 12), return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def refine_group(patches, group, min_interior: int):
    """Brings every direction of one group onto a common, sufficiently rich
    spline space: first degree elevation, then knot insertion."""
    target_degree = max(2, max(int(patches[p].degrees[d]) for p, d in group))

    for p, d in group:
        missing = target_degree - int(patches[p].degrees[d])
        if missing > 0:
            for _ in range(missing):
                patches[p].elevate_degrees([d])

    # Common interior knots: every value that occurs in any member, with the
    # highest multiplicity seen there.
    common = {}
    for p, d in group:
        for value, count in interior_knots(patches[p].knot_vectors[d], target_degree).items():
            common[value] = max(common.get(value, 0), count)

    # Refine further until the coarsest member has enough interior points.
    # ncp = degree + 1 + (number of interior knots), interior collocation
    # points per direction = ncp - 2.
    while target_degree + 1 + sum(common.values()) - 2 < min_interior:
        edges = sorted([0.0] + list(common.keys()) + [1.0])
        for lo, hi in zip(edges[:-1], edges[1:]):
            mid = round(0.5 * (lo + hi), 12)
            common.setdefault(mid, 1)

    for p, d in group:
        present = interior_knots(patches[p].knot_vectors[d], target_degree)
        to_insert = []
        for value, count in sorted(common.items()):
            to_insert.extend([value] * (count - present.get(value, 0)))
        if to_insert:
            patches[p].insert_knots(d, to_insert)


def report(patches, label):
    total = 0
    worst = None
    for i, s in enumerate(patches):
        ncp = [len(kv) - d - 1 for kv, d in zip(s.knot_vectors, s.degrees)]
        n = int(np.prod([max(c - 2, 0) for c in ncp]))
        total += n
        if worst is None or n < worst[1]:
            worst = (i, n)
    print(f"{label}: {total} innere Kollokationspunkte gesamt, "
          f"Minimum je Patch {worst[1]} (Patch {worst[0]})")
    return total


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    min_interior = 2
    for a in sys.argv[1:]:
        if a.startswith("--min-interior"):
            min_interior = int(a.split("=")[1]) if "=" in a else min_interior
    if "--min-interior" in sys.argv:
        idx = sys.argv.index("--min-interior")
        min_interior = int(sys.argv[idx + 1])

    in_path = Path(args[0]) if args else DEFAULT_IN
    out_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    multipatch, _ = splinepy.io.gismo.load(str(in_path))
    patches = list(multipatch.patches)
    print(f"gelesen: {in_path}  ({len(patches)} Patches)")
    report(patches, "vorher ")

    # Sample points to prove afterwards that the shape did not change.
    rng = np.random.default_rng(0)
    probe = rng.random((200, 3))
    before = [s.evaluate(probe) for s in patches]

    interfaces = read_interface_lines(in_path)
    groups = build_direction_groups(len(patches), interfaces)
    print(f"Richtungsgruppen: {len(groups)} (aus {len(interfaces)} Klebeflaechen)")

    for group in groups:
        refine_group(patches, group, min_interior)

    report(patches, "nachher")

    worst = max(float(np.abs(s.evaluate(probe) - b).max())
                for s, b in zip(patches, before))
    print(f"groesste Geometrieabweichung: {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("Abbruch: die Geometrie hat sich veraendert.")

    refined = splinepy.Multipatch(splines=patches)
    refined.interfaces = multipatch.interfaces
    splinepy.io.gismo.export(str(out_path), refined)
    print(f"geschrieben: {out_path}")


if __name__ == "__main__":
    main()
