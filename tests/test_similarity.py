"""Material-material similarity (the similar_to edges).

similar_to is the largest edge type in the graph (3,964 of 14,681), and it is
the slot Phase 3 will replace with CGCNN structural embeddings - so its
contract matters beyond Phase 1: top-K per material, no self-edges, and
missing data handled by mean imputation rather than dropped or zeroed.
"""

from __future__ import annotations

import math

import pytest

from matgraph.enrich import (
    SIMILARITY_FIELDS,
    _cosine,
    _similarity_value,
    _zscore_vectors,
    compute_similarity_edges,
)

from .conftest import make_raw


class TestSimilarityValueExtraction:
    """Raw MP JSON does not use the flattened names the Material node uses.

    Regression guard: a naive raw.get("bulk_modulus_vrh") returns None for
    every material, which would silently drop 3 of the 10 dimensions without
    any error - the vector would still compute, just on less information.
    """

    def test_elastic_moduli_are_unwrapped_from_the_vrh_dict(self):
        raw = make_raw(bulk_modulus={"voigt": 110.0, "reuss": 90.0, "vrh": 100.0})
        assert _similarity_value(raw, "bulk_modulus_vrh") == 100.0

    def test_missing_elastic_dict_is_none_not_an_error(self):
        assert _similarity_value(make_raw(), "bulk_modulus_vrh") is None

    def test_dielectric_uses_its_raw_key(self):
        # dielectric_total is the DataPoint name; e_total is what MP returns.
        assert _similarity_value(make_raw(e_total=12.5), "dielectric_total") == 12.5

    def test_plain_fields_pass_through(self):
        assert _similarity_value(make_raw(band_gap=1.62), "band_gap") == 1.62


class TestZScoring:
    def test_vector_has_one_component_per_similarity_field(self):
        vectors = _zscore_vectors([make_raw(material_id="mp-1")])
        assert len(vectors["mp-1"]) == len(SIMILARITY_FIELDS) == 10

    def test_missing_values_are_mean_imputed_to_zero(self):
        # Elastic data exists for only 18% of materials. A missing value is
        # imputed with the column mean, which z-scores to exactly 0 - i.e. it
        # contributes neutrally instead of pulling the material toward either
        # extreme. Zero-filling instead would make every incomplete material
        # look identically "soft", fabricating similarity that isn't there.
        materials = [
            make_raw(material_id="mp-1", bulk_modulus={"vrh": 100.0}),
            make_raw(material_id="mp-2", bulk_modulus={"vrh": 200.0}),
            make_raw(material_id="mp-3"),  # no elastic data
        ]
        index = SIMILARITY_FIELDS.index("bulk_modulus_vrh")
        vectors = _zscore_vectors(materials)
        assert vectors["mp-3"][index] == 0.0
        # and the materials that do have data sit either side of it
        assert vectors["mp-1"][index] < 0 < vectors["mp-2"][index]

    def test_a_constant_column_does_not_divide_by_zero(self):
        # Every material sharing one value gives std=0. Guarded with `or 1.0`,
        # otherwise the whole build dies on a ZeroDivisionError.
        vectors = _zscore_vectors(
            [make_raw(material_id="mp-1", density=4.3), make_raw(material_id="mp-2", density=4.3)]
        )
        index = SIMILARITY_FIELDS.index("density")
        assert vectors["mp-1"][index] == vectors["mp-2"][index] == 0.0

    def test_all_values_are_finite(self):
        vectors = _zscore_vectors([make_raw(material_id=f"mp-{i}") for i in range(3)])
        for vector in vectors.values():
            assert all(math.isfinite(v) for v in vector)


class TestCosine:
    # Compared with approx: cosine divides by square roots, so exact equality
    # is the wrong assertion even when the maths is right (opposite vectors
    # come out at -0.9999999999999998).

    def test_identical_vectors_score_one(self):
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_opposite_vectors_score_minus_one(self):
        assert _cosine([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        # A material identical to the dataset mean on every field z-scores to
        # all zeros - real, not hypothetical, in a small dataset.
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSimilarityEdges:
    def _spread(self, n: int) -> list[dict]:
        """n materials with genuinely different properties."""
        return [
            make_raw(
                material_id=f"mp-{i}",
                band_gap=float(i),
                density=2.0 + i,
                formation_energy_per_atom=-1.0 * i,
                total_magnetization=float(i % 3),
            )
            for i in range(n)
        ]

    def test_no_material_is_similar_to_itself(self):
        # A self-edge would be a meaningless loop in the graph and would always
        # rank first, crowding out a real neighbour from the top-K.
        edges = compute_similarity_edges(self._spread(5), min_weight=-1.0, top_k=10)
        for material_id, neighbours in edges.items():
            assert material_id not in [other for other, _weight in neighbours]

    def test_top_k_caps_the_neighbour_count(self):
        # The reason top-K exists: a flat cosine threshold produced tens of
        # thousands of "similar enough" pairs at a few hundred materials.
        edges = compute_similarity_edges(self._spread(10), min_weight=-1.0, top_k=3)
        assert all(len(neighbours) <= 3 for neighbours in edges.values())

    def test_neighbours_are_sorted_by_descending_weight(self):
        edges = compute_similarity_edges(self._spread(8), min_weight=-1.0, top_k=5)
        for neighbours in edges.values():
            weights = [weight for _other, weight in neighbours]
            assert weights == sorted(weights, reverse=True)

    def test_min_weight_excludes_weak_pairs(self):
        edges = compute_similarity_edges(self._spread(8), min_weight=0.9, top_k=5)
        for neighbours in edges.values():
            assert all(weight >= 0.9 for _other, weight in neighbours)

    def test_every_material_gets_an_entry_even_with_no_neighbours(self):
        # An unreachable threshold must still yield an entry per material;
        # build_graph looks every material up by id when wiring edges.
        edges = compute_similarity_edges(self._spread(4), min_weight=1.1, top_k=5)
        assert set(edges) == {"mp-0", "mp-1", "mp-2", "mp-3"}
        assert all(neighbours == [] for neighbours in edges.values())

    def test_identical_materials_are_mutually_most_similar(self):
        materials = [
            make_raw(material_id="twin-a", band_gap=2.0, density=5.0),
            make_raw(material_id="twin-b", band_gap=2.0, density=5.0),
            make_raw(material_id="other", band_gap=0.0, density=1.0, total_magnetization=5.0),
        ]
        edges = compute_similarity_edges(materials, min_weight=-1.0, top_k=1)
        assert edges["twin-a"][0][0] == "twin-b"
        assert edges["twin-b"][0][0] == "twin-a"
