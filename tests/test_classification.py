"""Property-bucket classification.

These buckets decide which PropertyClass nodes a material connects to, so a
silent change here quietly re-labels large parts of the graph - e.g. every
material whose gap sits near a boundary. The boundary cases are therefore
pinned explicitly, including the tie-break behaviour where two buckets overlap.
"""

from __future__ import annotations

import pytest

from matgraph.enrich import classify_band_gap, classify_ordering, classify_stability


class TestBandGap:
    @pytest.mark.parametrize(
        ("band_gap", "expected"),
        [
            (0.0, "metal"),
            (0.5, "narrow_gap"),
            (1.62, "semiconductor"),
            (5.87, "wide_gap"),  # Al2O3, the widest gap in the built graph
        ],
    )
    def test_buckets(self, band_gap, expected, config):
        assert classify_band_gap(band_gap, config)["name"] == expected

    @pytest.mark.parametrize(
        ("band_gap", "expected"),
        [
            # Buckets share their endpoints (narrow_gap is 0.0-1.0 and
            # semiconductor is 1.0-3.0), so a value landing exactly on a
            # boundary matches two buckets. _bucket returns the first match, so
            # the lower bucket wins. Pinned because it is a real decision, not
            # an accident - and silently flipping it would move materials
            # between PropertyClass nodes.
            (0.0, "metal"),
            (1.0, "narrow_gap"),
            (3.0, "semiconductor"),
        ],
    )
    def test_boundary_lower_bucket_wins(self, band_gap, expected, config):
        assert classify_band_gap(band_gap, config)["name"] == expected

    def test_description_records_the_value(self, config):
        # The description is what makes a classification auditable in the graph,
        # so it has to carry the number that triggered it.
        assert "1.620" in classify_band_gap(1.62, config)["description"]

    @pytest.mark.parametrize("bad", [float("nan"), None])
    def test_unusable_value_returns_none_rather_than_raising(self, bad, config):
        # Regression: this used to raise ValueError and abort the entire build
        # over one malformed material. NaN is the realistic trigger - it fails
        # every comparison and falls through all buckets.
        assert classify_band_gap(bad, config) is None


class TestStability:
    @pytest.mark.parametrize(
        ("e_hull", "expected"),
        [
            (0.0, "stable"),
            (0.03, "metastable"),
            (0.2, "unstable"),
            # DFT noise can push a ground-state entry slightly below the hull;
            # it must still classify as stable, not fall through to nothing.
            (-1e-9, "stable"),
        ],
    )
    def test_buckets(self, e_hull, expected, config):
        assert classify_stability(e_hull, config)["name"] == expected

    @pytest.mark.parametrize(("e_hull", "expected"), [(1e-6, "stable"), (0.05, "metastable")])
    def test_boundary_lower_bucket_wins(self, e_hull, expected, config):
        assert classify_stability(e_hull, config)["name"] == expected

    @pytest.mark.parametrize("bad", [float("nan"), None])
    def test_unusable_value_returns_none_rather_than_raising(self, bad, config):
        assert classify_stability(bad, config) is None


class TestOrdering:
    @pytest.mark.parametrize("ordering", ["FM", "AFM", "FiM", "NM"])
    def test_reported_ordering_is_passed_through(self, ordering):
        # Ordering is categorical from MP - it is not bucketed, just carried
        # over, so every value MP reports must survive intact.
        result = classify_ordering(ordering)
        assert result["kind"] == "magnetic_ordering"
        assert result["name"] == ordering

    @pytest.mark.parametrize("ordering", ["Unknown", "", None])
    def test_unusable_ordering_is_skipped(self, ordering):
        # "Unknown" is MP's own placeholder: it means no ordering was
        # determined, so creating a PropertyClass node named "Unknown" would
        # invent a category that doesn't exist.
        assert classify_ordering(ordering) is None
