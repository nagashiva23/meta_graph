"""Crystal-symmetry classification and the rules built on it.

The piezoelectric and ferroelectric domains screen on the crystal's point group
rather than on a fitted threshold: symmetry *forbids* these properties outright
in centrosymmetric structures, as a matter of physical law. That makes the
table these rules read the most safety-critical config in the project - a
mislabelled space group silently mis-assigns every material that uses it.
"""

from __future__ import annotations

from matgraph.enrich import load_space_groups

# Reference space groups, chosen because each pins a distinct branch of the
# classification and each has a well-known material sitting in it.
PM3M = 221  # m-3m  - cubic BaTiO3 (paraelectric phase); centrosymmetric
P4MM = 99  # 4mm   - tetragonal BaTiO3 (ferroelectric phase); polar
P63MC = 186  # 6mm   - wurtzite AlN / GaN; polar
F43M = 216  # -43m  - zincblende GaAs; non-centrosymmetric but NOT polar
FD3M = 227  # m-3m  - diamond-structure Si; centrosymmetric
P432 = 207  # 432   - non-centrosymmetric yet NOT piezoelectric (the exception)


class TestSpaceGroupTable:
    def test_covers_all_230_space_groups(self):
        assert len(load_space_groups()) == 230

    def test_counts_match_the_crystallographic_literature(self):
        table = load_space_groups().values()
        assert sum(1 for v in table if v["centrosymmetric"]) == 92
        assert sum(1 for v in table if v["polar"]) == 68
        assert sum(1 for v in table if not v["centrosymmetric"]) == 138

    def test_there_are_exactly_32_point_groups(self):
        assert len({v["point_group"] for v in load_space_groups().values()}) == 32

    def test_polar_implies_non_centrosymmetric(self):
        # A polar axis cannot survive an inversion centre. If this ever fails,
        # the two point-group sets have drifted out of agreement.
        for number, props in load_space_groups().items():
            if props["polar"]:
                assert not props["centrosymmetric"], f"space group {number}"

    def test_piezoelectric_implies_non_centrosymmetric(self):
        for number, props in load_space_groups().items():
            if props["piezoelectric_allowed"]:
                assert not props["centrosymmetric"], f"space group {number}"

    def test_polar_implies_piezoelectric(self):
        # Every polar class is also piezoelectric; ferroelectrics are a subset
        # of piezoelectrics, never the other way round.
        for number, props in load_space_groups().items():
            if props["polar"]:
                assert props["piezoelectric_allowed"], f"space group {number}"


class TestReferenceSpaceGroups:
    def test_centrosymmetric_examples(self):
        table = load_space_groups()
        for number in (PM3M, FD3M):
            props = table[str(number)]
            assert props["centrosymmetric"]
            assert not props["polar"]
            assert not props["piezoelectric_allowed"]

    def test_polar_examples_are_ferroelectric_candidates(self):
        table = load_space_groups()
        for number in (P4MM, P63MC):
            props = table[str(number)]
            assert props["polar"]
            assert props["piezoelectric_allowed"]

    def test_noncentrosymmetric_but_nonpolar(self):
        # Zincblende GaAs can be piezoelectric but never ferroelectric: no
        # unique axis for a spontaneous polarisation to point along.
        props = load_space_groups()[str(F43M)]
        assert not props["centrosymmetric"]
        assert props["piezoelectric_allowed"]
        assert not props["polar"]

    def test_point_group_432_is_the_piezoelectric_exception(self):
        # 432 is the one non-centrosymmetric class whose symmetry forces every
        # piezoelectric tensor component to vanish - 20 of 21, not 21 of 21.
        props = load_space_groups()[str(P432)]
        assert props["point_group"] == "432"
        assert not props["centrosymmetric"]
        assert not props["piezoelectric_allowed"]
