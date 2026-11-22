import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = Path(__file__).parents[1] / "figures" / "metric_tradeoffs.svg"
CONDITIONS = ["epoch20-B0", "epoch20-B3", "E3-B0", "C0-P0", "C0-B3", "E3-B3"]


class SummaryFigureTests(unittest.TestCase):
    def test_svg_is_well_formed_and_includes_all_conditions(self):
        root = ET.parse(SVG).getroot()
        self.assertTrue(root.tag.endswith("svg"))
        text = " ".join(element.text or "" for element in root.iter() if element.tag.endswith("text"))
        for condition in CONDITIONS:
            self.assertIn(condition, text)
        self.assertIn("全有効14px MAE", text)
        self.assertIn("高差分", text)


if __name__ == "__main__":
    unittest.main()
