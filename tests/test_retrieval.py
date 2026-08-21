"""Phase 2 retrieval: constraint traversal and evidence assembly.

Built on a small hand-made graph rather than the real one, so these run without
cognee, a database, or a built graph — and so each case can pin an exact
expected answer instead of a count that shifts when the dataset is refetched.

The synthetic graph deliberately reproduces cognee's real storage quirks: edge
`properties` arrive as a *stringified* dict, and `weight` arrives as a
*string*. Tests written against tidied-up fixtures would pass while the real
retrieval path returned nothing.
"""

from __future__ import annotations

import pytest

from matgraph.graph_store import Graph, edge_properties, edge_rule, edge_weight
from matgraph.retrieval import (
    Constraints,
    NumericFilter,
    UnknownEntity,
    connections_between,
    find_materials,
    materials_in_class,
    materials_in_crystal_system,
    materials_with,
    similar_materials,
)

CATHODE_RULE = "contains Li and a transition metal, and energy_above_hull <= 0.05 eV/atom"


@pytest.fixture
def graph() -> Graph:
    """Three materials, wired the way enrich.py wires the real ones.

    LiFePO4  stable battery cathode, orthorhombic, contains Li/Fe/P/O
    Fe2O3    stable magnetic oxide, trigonal, contains Fe/O  (shares Fe+O)
    Si       metastable semiconductor, cubic, contains Si    (shares nothing)
    """
    nodes = [
        ("m_lfp", {"type": "Material", "material_id": "mp-1", "formula": "LiFePO4",
                   "band_gap": 3.7, "energy_above_hull": 0.0, "density": 3.6}),
        ("m_fe2o3", {"type": "Material", "material_id": "mp-2", "formula": "Fe2O3",
                     "band_gap": 2.1, "energy_above_hull": 0.0, "density": 5.2}),
        ("m_si", {"type": "Material", "material_id": "mp-3", "formula": "Si",
                  "band_gap": 0.61, "energy_above_hull": 0.02, "density": 2.3}),
        ("e_li", {"type": "Element", "symbol": "Li"}),
        ("e_fe", {"type": "Element", "symbol": "Fe"}),
        ("e_p", {"type": "Element", "symbol": "P"}),
        ("e_o", {"type": "Element", "symbol": "O"}),
        ("e_si", {"type": "Element", "symbol": "Si"}),
        ("pc_stable", {"type": "PropertyClass", "kind": "stability", "name": "stable",
                       "description": "stability classified as 'stable'"}),
        ("pc_meta", {"type": "PropertyClass", "kind": "stability", "name": "metastable",
                     "description": "stability classified as 'metastable'"}),
        # Same name as a stability bucket would collide without `kind`.
        ("pc_wide", {"type": "PropertyClass", "kind": "band_gap", "name": "wide_gap",
                     "description": "band_gap classified as 'wide_gap'"}),
        ("ad_cathode", {"type": "ApplicationDomain", "name": "battery_cathode"}),
        ("ad_magnetic", {"type": "ApplicationDomain", "name": "magnetic_material"}),
        ("sg_pnma", {"type": "SpaceGroup", "symbol": "Pnma", "number": 62}),
        ("sg_r3c", {"type": "SpaceGroup", "symbol": "R-3c", "number": 167}),
        ("sg_fd3m", {"type": "SpaceGroup", "symbol": "Fd-3m", "number": 227}),
        ("cs_ortho", {"type": "CrystalSystem", "name": "Orthorhombic"}),
        ("cs_trig", {"type": "CrystalSystem", "name": "Trigonal"}),
        ("cs_cubic", {"type": "CrystalSystem", "name": "Cubic"}),
        ("ox_o2m", {"type": "OxidationState", "species": "O2-"}),
        ("ox_fe3p", {"type": "OxidationState", "species": "Fe3+"}),
        ("fp_abc4", {"type": "FormulaPattern", "pattern": "ABCD4"}),
        ("chem_lfpo", {"type": "ChemicalSystem", "chemsys": "Fe-Li-O-P"}),
    ]

    def contains(weight: float) -> dict:
        # cognee returns weights as strings
        return {"weight": str(weight)}

    def rule(text: str) -> dict:
        # cognee nests properties as a stringified dict
        return {"properties": str({"rule": text})}

    edges = [
        ("m_lfp", "e_li", "contains", contains(1 / 7)),
        ("m_lfp", "e_fe", "contains", contains(1 / 7)),
        ("m_lfp", "e_p", "contains", contains(1 / 7)),
        ("m_lfp", "e_o", "contains", contains(4 / 7)),
        ("m_fe2o3", "e_fe", "contains", contains(0.4)),
        ("m_fe2o3", "e_o", "contains", contains(0.6)),
        ("m_si", "e_si", "contains", contains(1.0)),
        ("m_lfp", "pc_stable", "classified_as", {}),
        ("m_lfp", "pc_wide", "classified_as", {}),
        ("m_fe2o3", "pc_stable", "classified_as", {}),
        ("m_si", "pc_meta", "classified_as", {}),
        ("m_lfp", "ad_cathode", "suitable_for", rule(CATHODE_RULE)),
        ("m_fe2o3", "ad_magnetic", "suitable_for", rule("magnetic ordering AFM")),
        ("m_lfp", "sg_pnma", "has_space_group", {}),
        ("m_fe2o3", "sg_r3c", "has_space_group", {}),
        ("m_si", "sg_fd3m", "has_space_group", {}),
        ("sg_pnma", "cs_ortho", "crystal_system", {}),
        ("sg_r3c", "cs_trig", "crystal_system", {}),
        ("sg_fd3m", "cs_cubic", "crystal_system", {}),
        ("m_lfp", "ox_o2m", "has_oxidation_state", {}),
        ("m_fe2o3", "ox_o2m", "has_oxidation_state", {}),
        ("m_fe2o3", "ox_fe3p", "has_oxidation_state", {}),
        ("m_lfp", "fp_abc4", "has_formula_pattern", {}),
        ("m_lfp", "chem_lfpo", "member_of", {}),
        ("m_lfp", "m_fe2o3", "similar_to", {"weight": "0.91"}),
        ("m_lfp", "m_si", "similar_to", {"weight": "0.42"}),
    ]
    return Graph(nodes, edges)


def formulas(hits) -> set[str]:
    return {hit.formula for hit in hits}


class TestCogneeStorageQuirks:
    """Guards against the failure mode that looks exactly like missing data."""

    def test_rule_is_read_out_of_the_stringified_properties_blob(self):
        assert edge_rule({"properties": str({"rule": CATHODE_RULE})}) == CATHODE_RULE

    def test_naive_lookup_would_have_returned_nothing(self):
        # Documents *why* edge_rule exists: the obvious access returns None,
        # which is indistinguishable from the rule never being stored.
        raw = {"properties": str({"rule": CATHODE_RULE})}
        assert raw.get("rule") is None
        assert edge_rule(raw) == CATHODE_RULE

    def test_weight_is_coerced_from_string_to_float(self):
        assert edge_weight({"weight": "0.9837107575817106"}) == pytest.approx(0.98371, rel=1e-4)

    def test_string_weights_would_sort_wrongly_uncompared(self):
        # "0.9" > "0.15" as text but not as numbers - silent mis-ranking.
        assert "0.9" > "0.15"
        assert edge_weight({"weight": "0.9"}) > edge_weight({"weight": "0.15"})

    def test_malformed_properties_do_not_raise(self):
        assert edge_properties({"properties": "["}) == {}
        assert edge_rule({}) is None
        assert edge_weight({"weight": "not a number"}) is None


class TestSingleHopTraversal:
    def test_materials_containing_an_element(self, graph):
        assert materials_with(graph, "element", "Fe") == {"m_lfp", "m_fe2o3"}
        assert materials_with(graph, "element", "Si") == {"m_si"}

    def test_shared_nodes_are_hubs(self, graph):
        # Oxygen is one node reached by several materials - the property that
        # makes inbound traversal cheap on the real graph.
        assert materials_with(graph, "element", "O") == {"m_lfp", "m_fe2o3"}

    def test_materials_by_application_domain(self, graph):
        assert materials_with(graph, "application_domain", "battery_cathode") == {"m_lfp"}

    def test_materials_by_oxidation_state(self, graph):
        assert materials_with(graph, "oxidation_state", "O2-") == {"m_lfp", "m_fe2o3"}

    def test_property_class_requires_both_kind_and_name(self, graph):
        assert materials_in_class(graph, "stability", "stable") == {"m_lfp", "m_fe2o3"}
        assert materials_in_class(graph, "band_gap", "wide_gap") == {"m_lfp"}

    def test_unknown_entity_is_an_error_not_an_empty_result(self, graph):
        # "nothing contains Unobtainium" and "you misspelled it" are different
        # answers; conflating them misleads whatever consumes retrieval.
        with pytest.raises(UnknownEntity, match="Element"):
            materials_with(graph, "element", "Xx")
        with pytest.raises(UnknownEntity, match="PropertyClass"):
            materials_in_class(graph, "stability", "nonexistent")


class TestMultiHopTraversal:
    def test_crystal_system_is_reached_through_the_space_group(self, graph):
        # Material -> SpaceGroup -> CrystalSystem. There is no direct edge.
        assert materials_in_crystal_system(graph, "Orthorhombic") == {"m_lfp"}
        assert materials_in_crystal_system(graph, "Cubic") == {"m_si"}

    def test_every_material_lands_in_exactly_one_crystal_system(self, graph):
        systems = ["Orthorhombic", "Trigonal", "Cubic"]
        found = [materials_in_crystal_system(graph, s) for s in systems]
        assert sum(len(f) for f in found) == 3
        assert set().union(*found) == {"m_lfp", "m_fe2o3", "m_si"}

    def test_unknown_crystal_system_raises(self, graph):
        with pytest.raises(UnknownEntity, match="CrystalSystem"):
            materials_in_crystal_system(graph, "Dodecahedral")


class TestConstraintFiltering:
    def test_constraints_are_anded_not_ored(self, graph):
        # Fe alone matches two; Fe AND stable AND cathode narrows to one.
        assert formulas(find_materials(graph, Constraints(elements=("Fe",)))) == {
            "LiFePO4", "Fe2O3"
        }
        narrowed = find_materials(
            graph,
            Constraints(elements=("Fe",), application_domain="battery_cathode"),
        )
        assert formulas(narrowed) == {"LiFePO4"}

    def test_multiple_elements_must_all_be_present(self, graph):
        assert formulas(find_materials(graph, Constraints(elements=("Li", "Fe")))) == {"LiFePO4"}
        # Si and Fe never co-occur here
        assert find_materials(graph, Constraints(elements=("Si", "Fe"))) == []

    def test_numeric_filter_on_a_material_attribute(self, graph):
        hits = find_materials(
            graph, Constraints(numeric=(NumericFilter("band_gap", minimum=2.0),))
        )
        assert formulas(hits) == {"LiFePO4", "Fe2O3"}

    def test_numeric_range_is_inclusive(self, graph):
        hits = find_materials(
            graph, Constraints(numeric=(NumericFilter("band_gap", minimum=2.1, maximum=2.1),))
        )
        assert formulas(hits) == {"Fe2O3"}

    def test_structural_and_numeric_combine(self, graph):
        hits = find_materials(
            graph,
            Constraints(
                elements=("O",),
                property_classes=(("stability", "stable"),),
                numeric=(NumericFilter("band_gap", minimum=3.0),),
            ),
        )
        assert formulas(hits) == {"LiFePO4"}

    def test_missing_attribute_never_satisfies_a_range(self, graph):
        # Absence means "not known", not "passes" - the stance the
        # partial-coverage application rules take too.
        assert find_materials(graph, Constraints(numeric=(NumericFilter("bulk_modulus_vrh", minimum=1),))) == []

    def test_empty_constraints_are_refused(self, graph):
        with pytest.raises(ValueError, match="no constraints"):
            find_materials(graph, Constraints())

    def test_impossible_combination_returns_empty_not_an_error(self, graph):
        # A real "no such material" answer, distinct from UnknownEntity.
        assert find_materials(
            graph, Constraints(elements=("Si",), application_domain="battery_cathode")
        ) == []


class TestOrdering:
    def test_results_are_sorted_most_stable_first(self, graph):
        hits = find_materials(graph, Constraints(numeric=(NumericFilter("band_gap", minimum=0),)))
        hulls = [h.energy_above_hull for h in hits]
        assert hulls == sorted(hulls)
        assert hits[-1].formula == "Si"  # the only metastable one

    def test_order_by_can_be_changed(self, graph):
        hits = find_materials(
            graph, Constraints(numeric=(NumericFilter("band_gap", minimum=0),)), order_by="band_gap"
        )
        assert [h.formula for h in hits] == ["Si", "Fe2O3", "LiFePO4"]

    def test_limit_truncates_after_ordering(self, graph):
        hits = find_materials(
            graph, Constraints(numeric=(NumericFilter("band_gap", minimum=0),)),
            limit=1, order_by="band_gap",
        )
        assert [h.formula for h in hits] == ["Si"]


class TestEvidence:
    def test_every_hit_carries_evidence(self, graph):
        hits = find_materials(graph, Constraints(elements=("Fe",)))
        assert all(hit.evidence for hit in hits)

    def test_application_domain_evidence_includes_the_stored_rule(self, graph):
        # The audit trail: the answer carries the rule that qualified it.
        hit = find_materials(graph, Constraints(application_domain="battery_cathode"))[0]
        details = [e.detail for e in hit.evidence if e.source == "suitable_for"]
        assert details == [CATHODE_RULE]

    def test_element_evidence_includes_the_atomic_fraction(self, graph):
        hit = find_materials(graph, Constraints(elements=("O",), application_domain="battery_cathode"))[0]
        oxygen = next(e for e in hit.evidence if e.claim == "contains O")
        assert "57.1%" in oxygen.detail  # 4/7 of LiFePO4

    def test_numeric_evidence_states_the_value_and_the_bound(self, graph):
        hits = find_materials(graph, Constraints(numeric=(NumericFilter("band_gap", minimum=3.0),)))
        item = next(e for e in hits[0].evidence if e.source == "attribute")
        assert "3.7" in item.claim and ">= 3" in item.claim

    def test_evidence_covers_only_what_was_asked(self, graph):
        # Dumping all ~19 edges would bury the reason the material matched, and
        # this list is the only context the generation step will be given.
        hit = find_materials(graph, Constraints(elements=("Li",)))[0]
        assert [e.claim for e in hit.evidence] == ["contains Li"]

    def test_multi_hop_evidence_names_the_path(self, graph):
        hit = find_materials(graph, Constraints(crystal_system="Orthorhombic"))[0]
        item = next(e for e in hit.evidence if "crystal_system" in e.source)
        assert "Pnma" in item.detail  # the space group it was reached through


class TestSimilarity:
    def test_neighbours_are_returned_strongest_first(self, graph):
        hits = similar_materials(graph, "LiFePO4")
        assert [h.formula for h in hits] == ["Fe2O3", "Si"]

    def test_similarity_evidence_reports_the_weight(self, graph):
        hit = similar_materials(graph, "LiFePO4", limit=1)[0]
        assert "0.910" in hit.evidence[0].detail

    def test_limit_is_respected(self, graph):
        assert len(similar_materials(graph, "LiFePO4", limit=1)) == 1

    def test_unknown_formula_raises(self, graph):
        with pytest.raises(UnknownEntity, match="Material"):
            similar_materials(graph, "NotAMaterial")


class TestConnectionsBetween:
    def test_finds_every_shared_node(self, graph):
        claims = [e.claim for e in connections_between(graph, "LiFePO4", "Fe2O3")]
        assert "both are linked to element Fe" in claims
        assert "both are linked to element O" in claims
        assert "both are linked to property class stable" in claims
        assert "both are linked to oxidation state O2-" in claims

    def test_unrelated_materials_share_nothing(self, graph):
        assert connections_between(graph, "LiFePO4", "Si") == []

    def test_unknown_formula_raises(self, graph):
        with pytest.raises(UnknownEntity, match="NotAMaterial"):
            connections_between(graph, "LiFePO4", "NotAMaterial")
