"""The application domains added in the Tier 4 coverage pass.

Phase 2 retrieval can only answer questions the graph encodes, so these rules
decide the ceiling on what can be asked. Each is tested either side of its
threshold, and the symmetry-based ones are checked against materials whose real
behaviour is well known (BaTiO3 is ferroelectric; silicon is not).
"""

from __future__ import annotations

import pytest

from matgraph.enrich import (
    APPLICATION_RULE_PREDICATES,
    _assert_rules_match_config,
    apply_application_rules,
    atomic_fractions,
    load_space_groups,
)

from .conftest import make_raw

CENTROSYMMETRIC = 221  # Pm-3m, m-3m
POLAR = 99  # P4mm, 4mm - tetragonal BaTiO3
NONCENTRO_NONPOLAR = 216  # F-43m, -43m - zincblende


@pytest.fixture(scope="session")
def space_groups():
    return load_space_groups()


def domains(raw, config, space_groups) -> set[str]:
    return {domain for domain, _rule in apply_application_rules(raw, config, space_groups)}


def sym(number: int) -> dict:
    return {"symbol": "X", "number": number, "crystal_system": "Cubic"}


class TestConfigCodeConsistency:
    """The two halves of a rule live in different files; neither may drift."""

    def test_config_and_predicates_agree(self, config):
        _assert_rules_match_config(config)

    def test_every_configured_domain_has_a_predicate(self, config):
        configured = {rule["domain"] for rule in config["application_rules"]}
        assert configured == set(APPLICATION_RULE_PREDICATES)

    def test_a_domain_missing_its_predicate_is_rejected(self, config):
        # A suitable_for edge with no rule text would assert something with no
        # stored justification - the one thing this graph must never produce.
        broken = dict(config)
        broken["application_rules"] = [
            *config["application_rules"],
            {"domain": "invented_domain", "rule": "...", "params": {}},
        ]
        with pytest.raises(ValueError, match="invented_domain"):
            _assert_rules_match_config(broken)

    def test_every_rule_has_text_and_params(self, config):
        for rule in config["application_rules"]:
            assert rule["rule"].strip(), f"{rule['domain']} has no rule text"
            assert "params" in rule, f"{rule['domain']} has no params block"


class TestPhotovoltaicAbsorber:
    """direct gap, 1.0 <= gap <= 1.8 eV, not metallic."""

    def test_fires_for_direct_gap_in_window(self, config, space_groups):
        raw = make_raw(band_gap=1.4, is_gap_direct=True)
        assert "photovoltaic_absorber" in domains(raw, config, space_groups)

    def test_requires_a_direct_gap(self, config, space_groups):
        # An indirect-gap absorber needs to be far thicker to absorb the same
        # light, which is the whole reason the rule checks this flag.
        raw = make_raw(band_gap=1.4, is_gap_direct=False)
        assert "photovoltaic_absorber" not in domains(raw, config, space_groups)

    @pytest.mark.parametrize(
        ("band_gap", "fires"), [(0.99, False), (1.0, True), (1.8, True), (1.81, False)]
    )
    def test_band_gap_window(self, band_gap, fires, config, space_groups):
        raw = make_raw(band_gap=band_gap, is_gap_direct=True)
        assert ("photovoltaic_absorber" in domains(raw, config, space_groups)) is fires


class TestSymmetryDomains:
    def test_polar_structure_is_both_ferroelectric_and_piezoelectric(
        self, config, space_groups
    ):
        raw = make_raw(symmetry=sym(POLAR), band_gap=2.0)
        found = domains(raw, config, space_groups)
        assert {"ferroelectric", "piezoelectric"} <= found

    def test_noncentrosymmetric_nonpolar_is_piezoelectric_only(self, config, space_groups):
        # Ferroelectrics are a strict subset of piezoelectrics.
        raw = make_raw(symmetry=sym(NONCENTRO_NONPOLAR), band_gap=2.0)
        found = domains(raw, config, space_groups)
        assert "piezoelectric" in found
        assert "ferroelectric" not in found

    def test_centrosymmetric_is_neither(self, config, space_groups):
        raw = make_raw(symmetry=sym(CENTROSYMMETRIC), band_gap=2.0)
        found = domains(raw, config, space_groups)
        assert "piezoelectric" not in found
        assert "ferroelectric" not in found

    def test_metals_are_excluded_despite_favourable_symmetry(self, config, space_groups):
        # A conductor screens out the internal field, so symmetry alone is not
        # enough - this is why both rules also test is_metal.
        raw = make_raw(symmetry=sym(POLAR), band_gap=0.0, is_metal=True)
        found = domains(raw, config, space_groups)
        assert "piezoelectric" not in found
        assert "ferroelectric" not in found


class TestThermoelectric:
    @pytest.mark.parametrize(
        ("band_gap", "fires"), [(0.09, False), (0.1, True), (0.8, True), (0.81, False)]
    )
    def test_narrow_gap_window(self, band_gap, fires, config, space_groups):
        raw = make_raw(band_gap=band_gap, is_metal=False)
        assert ("thermoelectric" in domains(raw, config, space_groups)) is fires


class TestUvTransparent:
    def test_fires_for_very_wide_gap_nonmagnetic(self, config, space_groups):
        raw = make_raw(band_gap=5.87, total_magnetization=0.0)  # Al2O3-like
        assert "uv_transparent" in domains(raw, config, space_groups)

    def test_boundary(self, config, space_groups):
        assert "uv_transparent" in domains(make_raw(band_gap=4.0), config, space_groups)
        assert "uv_transparent" not in domains(make_raw(band_gap=3.99), config, space_groups)

    def test_magnetic_material_is_excluded(self, config, space_groups):
        raw = make_raw(band_gap=5.0, total_magnetization=3.0)
        assert "uv_transparent" not in domains(raw, config, space_groups)


class TestPartialCoverageDomains:
    """Rules needing elastic (~18%) or dielectric (~20%) data.

    Absence of the edge means "not known", never "not suitable" - and a missing
    modulus must never be read as a modulus of zero.
    """

    def test_hard_structural_fires_when_both_moduli_are_high(self, config, space_groups):
        raw = make_raw(shear_modulus={"vrh": 150.0}, bulk_modulus={"vrh": 200.0})
        assert "hard_structural" in domains(raw, config, space_groups)

    def test_hard_structural_requires_both_moduli(self, config, space_groups):
        raw = make_raw(shear_modulus={"vrh": 150.0})  # no bulk modulus
        assert "hard_structural" not in domains(raw, config, space_groups)

    def test_missing_elastic_data_does_not_match(self, config, space_groups):
        assert "hard_structural" not in domains(make_raw(), config, space_groups)

    @pytest.mark.parametrize(("shear", "fires"), [(99.0, False), (100.0, True)])
    def test_shear_boundary(self, shear, fires, config, space_groups):
        raw = make_raw(shear_modulus={"vrh": shear}, bulk_modulus={"vrh": 200.0})
        assert ("hard_structural" in domains(raw, config, space_groups)) is fires

    def test_high_k_dielectric_fires(self, config, space_groups):
        raw = make_raw(e_total=25.0, band_gap=3.0)
        assert "high_k_dielectric" in domains(raw, config, space_groups)

    def test_high_k_requires_an_insulating_gap(self, config, space_groups):
        raw = make_raw(e_total=25.0, band_gap=1.0)
        assert "high_k_dielectric" not in domains(raw, config, space_groups)

    def test_missing_dielectric_data_does_not_match(self, config, space_groups):
        assert "high_k_dielectric" not in domains(make_raw(band_gap=3.0), config, space_groups)


class TestAtomicFractions:
    def test_fractions_reflect_stoichiometry(self):
        fractions = atomic_fractions("LiFePO4", ["Li", "Fe", "P", "O"])
        assert fractions["O"] == pytest.approx(4 / 7)
        assert fractions["Li"] == pytest.approx(1 / 7)
        # Oxygen is four times as abundant as lithium; unweighted edges cannot
        # express that, which is the point of adding weights.
        assert fractions["O"] > fractions["Li"]

    def test_fractions_sum_to_one(self):
        fractions = atomic_fractions("BaTiO3", ["Ba", "Ti", "O"])
        assert sum(fractions.values()) == pytest.approx(1.0)

    def test_single_element(self):
        assert atomic_fractions("Si", ["Si"]) == {"Si": pytest.approx(1.0)}

    def test_unparseable_formula_returns_none_rather_than_raising(self):
        # The caller then builds unweighted edges instead of inventing a number.
        assert atomic_fractions("not a formula!!", ["X"]) is None

    def test_formula_not_covering_every_element_returns_none(self):
        assert atomic_fractions("LiFePO4", ["Li", "Fe", "P", "O", "Zz"]) is None


class TestEdgeRuleExtraction:
    """Reading the rule text back off a suitable_for edge.

    Cognee stores an Edge's `properties` dict as a *stringified* Python dict
    under a single "properties" key, so the rule sits one level down and
    serialised. A plain edge_props["rule"] returns None for every edge, which
    looks exactly like the evidence was never stored - the failure mode this
    guards against.
    """

    def test_reads_a_rule_from_cognees_stringified_properties(self):
        from matgraph.verify import _edge_rule

        edge = {"properties": "{'rule': '0 < band_gap <= 3.5 eV and not metallic'}"}
        assert _edge_rule(edge) == "0 < band_gap <= 3.5 eV and not metallic"

    def test_reads_a_rule_from_a_real_dict_too(self):
        from matgraph.verify import _edge_rule

        assert _edge_rule({"properties": {"rule": "contains Li"}}) == "contains Li"

    def test_returns_none_when_there_is_no_evidence(self):
        from matgraph.verify import _edge_rule

        assert _edge_rule({}) is None
        assert _edge_rule({"properties": "{}"}) is None
        assert _edge_rule({"properties": "{'rule': '   '}"}) is None

    def test_malformed_properties_do_not_raise(self):
        from matgraph.verify import _edge_rule

        # Parsed with literal_eval, which cannot execute code - this is data
        # read back out of a database.
        assert _edge_rule({"properties": "not a dict at all ["}) is None
