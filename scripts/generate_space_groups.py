"""Generate config/space_groups.json: symmetry properties of all 230 space groups.

Run once (like scripts/generate_elements.py) - the enrichment path then reads a
static JSON table and needs no pymatgen at build time.

Why this table exists: several material properties are *forbidden or permitted
by crystal symmetry alone*, independent of composition or any computed value.
That makes them the most rigorous screening criteria available - they follow
from the structure's point group as a matter of physical law, not a fitted
threshold:

  - Piezoelectricity requires a non-centrosymmetric structure. A crystal with
    an inversion centre cannot develop a polarisation under strain, because the
    inverted structure is indistinguishable from the original.
  - Ferroelectricity additionally requires a *polar* point group - one with a
    unique axis along which a spontaneous polarisation can point.

Counts produced by this script match the crystallographic literature exactly:
92 centrosymmetric space groups, 138 non-centrosymmetric, 68 polar, and 32
distinct point groups.
"""

from __future__ import annotations

import json
from pathlib import Path

from pymatgen.symmetry.groups import SpaceGroup

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "config" / "space_groups.json"

# The 11 centrosymmetric (Laue) point groups - those containing an inversion
# centre. Everything else is non-centrosymmetric.
CENTROSYMMETRIC_POINT_GROUPS = {
    "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m",
}

# The 10 polar point groups - those with a unique direction not related to any
# other by symmetry, so a spontaneous polarisation can exist along it.
POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}

# 432 is the one non-centrosymmetric point group that is still not
# piezoelectric: its symmetry forces every independent piezoelectric tensor
# component to vanish. So 20 of the 21 non-centrosymmetric classes qualify.
NON_PIEZOELECTRIC_NONCENTRO_POINT_GROUPS = {"432"}

# Expected totals, straight from the crystallographic literature. Asserted
# below so a pymatgen change that alters point-group notation fails loudly here
# rather than silently mislabelling every material in the graph.
EXPECTED = {"total": 230, "centrosymmetric": 92, "polar": 68, "point_groups": 32}


def main() -> None:
    table: dict[str, dict] = {}
    for number in range(1, 231):
        point_group = SpaceGroup.from_int_number(number).point_group
        centrosymmetric = point_group in CENTROSYMMETRIC_POINT_GROUPS
        table[str(number)] = {
            "point_group": point_group,
            "centrosymmetric": centrosymmetric,
            "polar": point_group in POLAR_POINT_GROUPS,
            # Piezoelectricity is allowed by symmetry; whether a given material
            # is actually a useful piezoelectric is a separate question this
            # table cannot answer.
            "piezoelectric_allowed": (
                not centrosymmetric
                and point_group not in NON_PIEZOELECTRIC_NONCENTRO_POINT_GROUPS
            ),
        }

    counts = {
        "total": len(table),
        "centrosymmetric": sum(1 for v in table.values() if v["centrosymmetric"]),
        "polar": sum(1 for v in table.values() if v["polar"]),
        "point_groups": len({v["point_group"] for v in table.values()}),
    }
    for key, expected in EXPECTED.items():
        assert counts[key] == expected, f"{key}: got {counts[key]}, expected {expected}"

    # A polar group is by definition non-centrosymmetric; catching a violation
    # here means the two sets above have drifted out of agreement.
    assert not [v for v in table.values() if v["polar"] and v["centrosymmetric"]]

    OUT_PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(table)} space groups to {OUT_PATH}")
    print(f"  centrosymmetric: {counts['centrosymmetric']}  (literature: 92)")
    print(f"  polar:           {counts['polar']}  (literature: 68)")
    print(f"  piezo-allowed:   {sum(1 for v in table.values() if v['piezoelectric_allowed'])}")


if __name__ == "__main__":
    main()
