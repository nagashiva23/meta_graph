"""Raw-record validation (the guard added in PR #1).

The point of this layer is that one bad cache record should cost one material,
not the whole build - and that an incomplete record is *skipped*, never
defaulted. Defaulting is the dangerous failure: a missing band gap defaulted to
0.0 classifies the material as a metal, and a missing energy_above_hull
defaulted to 0.0 classifies it as perfectly stable. Both produce a graph that
looks fine and answers wrongly.
"""

from __future__ import annotations

from matgraph.enrich import REQUIRED_RAW_FIELDS, _missing_fields, filter_valid_materials

from .conftest import make_raw


class TestMissingFields:
    def test_a_complete_record_is_valid(self):
        assert _missing_fields(make_raw()) == []

    def test_absent_field_is_reported(self):
        raw = {k: v for k, v in make_raw().items() if k != "density"}
        assert _missing_fields(raw) == ["density"]

    def test_explicit_none_is_reported(self):
        assert _missing_fields(make_raw(band_gap=None)) == ["band_gap"]

    def test_every_required_field_is_actually_checked(self):
        # Guards against a field being added to the schema but forgotten here.
        for field in REQUIRED_RAW_FIELDS:
            raw = {k: v for k, v in make_raw().items() if k != field}
            assert field in _missing_fields(raw), f"{field} is not enforced"


class TestFalsyValuesAreNotMissing:
    """The single most dangerous bug this layer could have.

    Testing missing-ness by falsiness instead of `is None` would treat
    band_gap=0.0 as absent - and band_gap=0.0 is exactly what every metal has.
    That would silently drop an entire class of materials from the graph while
    leaving it looking healthy.
    """

    def test_zero_band_gap_is_valid(self):
        assert _missing_fields(make_raw(band_gap=0.0, is_metal=True)) == []

    def test_false_booleans_are_valid(self):
        raw = make_raw(is_metal=False, is_stable=False, is_gap_direct=False, theoretical=False)
        assert _missing_fields(raw) == []

    def test_zero_magnetization_is_valid(self):
        assert _missing_fields(make_raw(total_magnetization=0.0)) == []

    def test_zero_energy_above_hull_is_valid(self):
        # Every perfectly stable material sits at exactly 0.0.
        assert _missing_fields(make_raw(energy_above_hull=0.0)) == []


class TestStructuralRequirements:
    def test_empty_elements_is_rejected(self):
        # verify.py asserts every Material has at least one contains edge, so a
        # material with no elements would fail verification later anyway -
        # better to catch it here, where the material can be named.
        assert _missing_fields(make_raw(elements=[])) == ["elements (empty)"]

    def test_incomplete_symmetry_is_reported_per_subfield(self):
        raw = make_raw(symmetry={"symbol": "Pmmn"})
        assert _missing_fields(raw) == ["symmetry.number", "symmetry.crystal_system"]

    def test_absent_symmetry_is_reported_once(self):
        # The whole block is missing, so don't also report each subfield.
        assert _missing_fields(make_raw(symmetry=None)) == ["symmetry"]


class TestOptionalFields:
    """Fields that are legitimately absent must not cause a skip."""

    def test_missing_description_is_allowed(self):
        # Not every material has robocrys text; build_graph substitutes a
        # placeholder rather than dropping the material.
        raw = {k: v for k, v in make_raw().items() if k != "description"}
        assert _missing_fields(raw) == []

    def test_empty_oxidation_states_are_allowed(self):
        # Confirmed against the live API: possible_species comes back empty for
        # some materials. They simply get no has_oxidation_state edges. This is
        # why oxidation-state coverage is ~93%, not 100%.
        assert _missing_fields(make_raw(possible_species=[])) == []

    def test_missing_elastic_and_dielectric_are_allowed(self):
        # Coverage is 18.0% and 19.9%; requiring these would discard most of
        # the graph.
        assert _missing_fields(make_raw()) == []


class TestFilterValidMaterials:
    def test_valid_records_pass_through_unchanged(self):
        materials = [make_raw(material_id="mp-1"), make_raw(material_id="mp-2")]
        assert filter_valid_materials(materials) == materials

    def test_invalid_records_are_dropped_and_valid_ones_kept(self):
        good = make_raw(material_id="mp-good")
        bad = {k: v for k, v in make_raw(material_id="mp-bad").items() if k != "band_gap"}
        assert filter_valid_materials([good, bad, good]) == [good, good]

    def test_an_empty_record_is_dropped_without_raising(self):
        # Must not KeyError while trying to report what's wrong with it.
        assert filter_valid_materials([{}]) == []

    def test_filtering_never_raises_on_a_record_missing_material_id(self):
        raw = {k: v for k, v in make_raw().items() if k != "material_id"}
        assert filter_valid_materials([raw]) == []
