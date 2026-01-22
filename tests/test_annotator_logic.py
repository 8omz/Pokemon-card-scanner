
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src', 'card_annotation_tool'))

import annotator

class TestAnnotatorLogic(unittest.TestCase):
    def test_compute_display_scale_fit_width(self):
        # Image 3200x2400, Max 1600x1200
        # Should scale by 0.5
        scale = annotator.compute_display_scale(3200, 2400, 1600, 1200)
        self.assertAlmostEqual(scale, 0.5)

    def test_compute_display_scale_fit_height(self):
        # Image 1000x4000, Max 1600x1200
        # Height ratio: 1200/4000 = 0.3
        # Width ratio: 1600/1000 = 1.6
        # Should pick 0.3
        scale = annotator.compute_display_scale(1000, 4000, 1600, 1200)
        self.assertAlmostEqual(scale, 0.3)

    def test_display_to_original(self):
        # Scale 0.5. Display (100, 100) -> Original (200, 200)
        x, y = annotator.display_to_original(100, 100, 0.5)
        self.assertEqual(x, 200)
        self.assertEqual(y, 200)

    def test_original_to_display(self):
        # Scale 0.5. Original (200, 200) -> Display (100, 100)
        x, y = annotator.original_to_display(200, 200, 0.5)
        self.assertEqual(x, 100)
        self.assertEqual(y, 100)

if __name__ == '__main__':
    unittest.main()
