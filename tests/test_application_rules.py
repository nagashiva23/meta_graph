"""Application-domain rules.

These are the project's explainability claim made concrete: each rule that
fires becomes a suitable_for edge carrying the rule text. If a rule silently
changes, the graph asserts something about a material that its stored
justification no longer supports - so each rule is tested at the boundary of
the condition it encodes, in both directions.
"""

from __future__ import annotations

from matgraph.enrich import apply_application_rules

from .conftest import make_raw


def domains(raw, config) -> set[str]:
    return {domain for domain, _rule in apply_application_rules(raw, config)}


class TestBatteryCathode:
    """Rule: contains Li AND a transition metal AND energy_above_hull <= 0.05."""

    def test_fires_for_li_plus_transition_metal_on_hull(self, config):
        assert "battery_cathode" in domains(make_raw(), config)

    def test_requires_lithium(self, config):
        raw = make_raw(elements=["Na", "Fe", "O"], formula_pretty="NaFeO2")
        assert "battery_cathode" not in domains(raw, config)

    def test_requires_a_transition_metal(self, config):
        # Li + Al + O: Al is a post-transition metal, so this must not qualify.
        raw = make_raw(elements=["Li", "Al", "O"], formula_pretty="LiAlO2")
        assert "battery_cathode" not in domains(raw, config)

    def test_stability_cutoff_boundary(self, config):
        assert "battery_cathode" in domains(make_raw(energy_above_hull=0.05), config)
        assert "battery_cathode" not in domains(make_raw(energy_above_hull=0.051), config)


class TestSemiconductorDevice:
    """Rule: 0 < band_gap <= 3.5 AND not metallic."""

    def test_fires_in_range(self, config):
        assert "semiconductor_device" in domains(make_raw(band_gap=1.62), config)

    def test_excludes_zero_gap(self, config):
        # A zero gap is a metal, not a semiconductor - the rule is strictly >0.
        raw = make_raw(band_gap=0.0, is_metal=True)
        assert "semiconductor_device" not in domains(raw, config)

    def test_excludes_metal_flag_even_with_a_gap(self, config):
        # MP can report a small gap on something still flagged metallic; the
        # flag wins, otherwise metals leak into the semiconductor domain.
        raw = make_raw(band_gap=0.4, is_metal=True)
        assert "semiconductor_device" not in domains(raw, config)

    def test_upper_boundary(self, config):
        assert "semiconductor_device" in domains(make_raw(band_gap=3.5), config)
        assert "semiconductor_device" not in domains(make_raw(band_gap=3.51), config)


class TestPhotocatalyst:
    """Rule: contains O AND 1.5 <= band_gap <= 3.5."""

    def test_fires_for_oxide_in_range(self, config):
        assert "photocatalyst" in domains(make_raw(band_gap=2.0), config)

    def test_requires_oxygen(self, config):
        raw = make_raw(elements=["Ga", "N"], formula_pretty="GaN", band_gap=2.0)
        assert "photocatalyst" not in domains(raw, config)

    def test_band_gap_boundaries(self, config):
        assert "photocatalyst" in domains(make_raw(band_gap=1.5), config)
        assert "photocatalyst" not in domains(make_raw(band_gap=1.49), config)
        assert "photocatalyst" in domains(make_raw(band_gap=3.5), config)
        assert "photocatalyst" not in domains(make_raw(band_gap=3.51), config)


class TestDielectric:
    """Rule: band_gap > 2 AND |total_magnetization| < 0.1."""

    def test_fires_for_wide_gap_nonmagnetic(self, config):
        assert "dielectric" in domains(make_raw(band_gap=3.2, total_magnetization=0.0), config)

    def test_excludes_magnetic(self, config):
        raw = make_raw(band_gap=3.2, total_magnetization=2.0)
        assert "dielectric" not in domains(raw, config)

    def test_magnetization_is_compared_by_absolute_value(self, config):
        # MP reports signed magnetization; a negative value is just as magnetic.
        raw = make_raw(band_gap=3.2, total_magnetization=-2.0)
        assert "dielectric" not in domains(raw, config)

    def test_band_gap_boundary_is_strict(self, config):
        assert "dielectric" not in domains(make_raw(band_gap=2.0), config)
        assert "dielectric" in domains(make_raw(band_gap=2.01), config)


class TestMagneticMaterial:
    """Rule: (FM or FiM with |magnetization| >= 0.1) OR AFM ordering."""

    def test_fm_with_magnetization(self, config):
        raw = make_raw(ordering="FM", total_magnetization=4.0)
        assert "magnetic_material" in domains(raw, config)

    def test_fm_without_magnetization_does_not_qualify(self, config):
        raw = make_raw(ordering="FM", total_magnetization=0.0)
        assert "magnetic_material" not in domains(raw, config)

    def test_afm_qualifies_without_net_magnetization(self, config):
        # Antiferromagnets cancel out to ~zero net magnetization, so requiring
        # magnetization would wrongly exclude them. This is why the rule has a
        # separate AFM branch.
        raw = make_raw(ordering="AFM", total_magnetization=0.0)
        assert "magnetic_material" in domains(raw, config)

    def test_nonmagnetic_ordering_does_not_qualify(self, config):
        assert "magnetic_material" not in domains(make_raw(ordering="NM"), config)


class TestRuleText:
    def test_every_fired_rule_carries_its_text(self, config):
        # The rule text is the evidence stored on the edge; an empty or missing
        # string would break the audit trail the whole project rests on.
        matches = apply_application_rules(make_raw(), config)
        assert matches
        for domain, rule in matches:
            assert isinstance(rule, str) and rule.strip(), f"{domain} fired with no rule text"

    def test_rule_text_matches_config(self, config):
        by_domain = {r["domain"]: r["rule"] for r in config["application_rules"]}
        for domain, rule in apply_application_rules(make_raw(), config):
            assert rule == by_domain[domain]


class TestMultipleDomains:
    def test_a_material_can_satisfy_several_rules(self, config):
        # Stable Li + Fe + O with a 1.62 eV gap is simultaneously a cathode
        # candidate, a semiconductor and a photocatalyst. This is why
        # suitable_for edges (1,506) outnumber materials (794) in the graph.
        assert domains(make_raw(), config) == {
            "battery_cathode",
            "semiconductor_device",
            "photocatalyst",
        }

    def test_a_material_can_satisfy_none(self, config):
        # A plain metal matches nothing - and must not be forced into a domain.
        raw = make_raw(
            elements=["Fe"], formula_pretty="Fe", band_gap=0.0, is_metal=True,
            ordering="NM", total_magnetization=0.0, energy_above_hull=0.0,
        )
        assert domains(raw, config) == set()
