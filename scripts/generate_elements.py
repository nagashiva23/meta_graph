"""Generate config/elements.json from pymatgen for the full periodic table (Z=1-103).

Run once (or whenever the property set needed changes) - the fetch layer does
not depend on pymatgen at request time, only this generation step does.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from pymatgen.core import Element

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "config" / "elements.json"

# Priority order matters: some elements match multiple pymatgen category
# flags (e.g. La is both a transition metal and a lanthanoid) - the first
# matching flag in this list wins.
CATEGORY_FLAGS = [
    ("is_noble_gas", "noble_gas"),
    ("is_halogen", "halogen"),
    ("is_chalcogen", "chalcogen"),
    ("is_lanthanoid", "lanthanide"),
    ("is_actinoid", "actinide"),
    ("is_alkali", "alkali_metal"),
    ("is_alkaline", "alkaline_earth_metal"),
    ("is_metalloid", "metalloid"),
    ("is_transition_metal", "transition_metal"),
    ("is_post_transition_metal", "post_transition_metal"),
]


def categorize(el: Element) -> str:
    for flag, category in CATEGORY_FLAGS:
        if getattr(el, flag, False):
            return category
    return "metal" if el.is_metal else "nonmetal"


def main() -> None:
    table: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected gaps: noble gas X, late-actinide radius
        for z in range(1, 104):
            el = Element.from_Z(z)
            x = el.X
            electronegativity = 0.0 if x is None or x != x else float(x)  # noqa: PLR0124 (NaN check)
            radius_ang = el.atomic_radius or el.atomic_radius_calculated
            atomic_radius_pm = float(radius_ang) * 100 if radius_ang else 200.0

            table[el.symbol] = {
                "name": el.long_name,
                "atomic_number": el.Z,
                "group": el.group,
                "period": el.row,
                "electronegativity": round(electronegativity, 3),
                "covalent_radius_pm": round(atomic_radius_pm, 1),
                "category": categorize(el),
            }

    OUT_PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(table)} elements to {OUT_PATH}")


if __name__ == "__main__":
    main()
