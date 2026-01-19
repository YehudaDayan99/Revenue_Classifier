import unittest
from pathlib import Path

from revseg.react_agents import extract_segment_revenue_from_segment_results_grid
from revseg.table_candidates import extract_table_grid_normalized


class TestMSFTSegmentResultsExtraction(unittest.TestCase):
    def test_msft_t0071_segment_revenue(self) -> None:
        html_path = Path(
            "data/10k/MSFT/2025-07-30_000095017025100235/primary_document.html"
        )
        grid = extract_table_grid_normalized(html_path, "t0071", max_rows=160)
        out = extract_segment_revenue_from_segment_results_grid(
            grid,
            segments=[
                "Productivity and Business Processes",
                "Intelligent Cloud",
                "More Personal Computing",
            ],
        )
        self.assertEqual(out["year"], 2025)
        seg_totals = out["segment_totals"]
        # Values are in millions (per table units)
        self.assertEqual(seg_totals["Productivity and Business Processes"], 120810)
        self.assertEqual(seg_totals["Intelligent Cloud"], 106265)
        self.assertEqual(seg_totals["More Personal Computing"], 54649)
        self.assertEqual(out["total_value"], 281724)


if __name__ == "__main__":
    unittest.main()

