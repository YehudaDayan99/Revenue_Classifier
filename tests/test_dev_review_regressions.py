"""
P2.6: Regression tests for Dev Review fixes.

These tests verify the critical correctness fixes don't regress:
1. P0.1: Validation must fail when table_total mismatches
2. P0.2: META mixed-dimension double counting must be prevented
3. P0.3: GOOGL adjustment rows must be emitted so sums reconcile
"""
import pytest
from revseg.extraction.validation import validate_extraction, ValidationResult


class TestP01ValidationFailOpen:
    """P0.1: Validation must not 'fail-open' when table_total exists and mismatches."""
    
    def test_mismatch_with_known_table_total_fails(self):
        """If table_total is known and sum mismatches, must return ok=False."""
        result = validate_extraction(
            segment_revenues={"Segment A": 100_000_000, "Segment B": 50_000_000},
            adjustment_revenues={},
            table_total=200_000_000,  # Known table total
            tolerance_pct=0.02,
        )
        # Sum is 150M, table_total is 200M -> 25% mismatch -> must fail
        assert result.ok is False
        assert "mismatch" in result.notes.lower() or "fail" in result.notes.lower()
    
    def test_match_within_tolerance_passes(self):
        """If sum matches table_total within tolerance, must pass."""
        result = validate_extraction(
            segment_revenues={"Segment A": 100_000_000, "Segment B": 98_000_000},
            adjustment_revenues={},
            table_total=200_000_000,
            tolerance_pct=0.02,  # 2% tolerance
        )
        # Sum is 198M, table_total is 200M -> 1% mismatch -> should pass
        assert result.ok is True
    
    def test_no_table_total_uses_fallback(self):
        """If table_total is None, fallback logic should apply."""
        result = validate_extraction(
            segment_revenues={"Segment A": 100_000_000, "Segment B": 50_000_000},
            adjustment_revenues={},
            table_total=None,  # No table total known
            min_segments=2,
        )
        # Should accept via fallback (2 segments with positive values)
        assert result.ok is True


class TestP02MetaDoubleCounting:
    """P0.2: META double-counting prevention via is_subtotal_row."""
    
    def test_family_of_apps_is_subtotal(self):
        """Family of Apps should be recognized as subtotal for META."""
        from revseg.mappings import is_subtotal_row
        
        assert is_subtotal_row("Family of Apps", "META") is True
        assert is_subtotal_row("Family of Apps (FoA)", "META") is True
        assert is_subtotal_row("Reality Labs", "META") is True
    
    def test_advertising_is_not_subtotal(self):
        """Advertising (granular item) should NOT be a subtotal."""
        from revseg.mappings import is_subtotal_row
        
        assert is_subtotal_row("Advertising", "META") is False
        assert is_subtotal_row("Other revenue", "META") is False


class TestP03AdjustmentRowsEmitted:
    """P0.3: Adjustment rows must be included for reconciliation."""
    
    def test_googl_hedging_is_adjustment(self):
        """GOOGL hedging gains/losses should be recognized as adjustment."""
        from revseg.mappings import is_adjustment_item
        
        assert is_adjustment_item("GOOGL", "Hedging gains (losses)") is True
        assert is_adjustment_item("GOOGL", "Hedging gains") is True
        assert is_adjustment_item("GOOGL", "Hedging losses") is True
    
    def test_normal_segment_is_not_adjustment(self):
        """Normal segment names should NOT be adjustments."""
        from revseg.mappings import is_adjustment_item
        
        assert is_adjustment_item("GOOGL", "Google Services") is False
        assert is_adjustment_item("GOOGL", "Google Cloud") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
