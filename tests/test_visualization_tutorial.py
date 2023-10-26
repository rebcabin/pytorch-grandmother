from unittest import TestCase
from visualization_tutorial import visualization_tutorial


class Test(TestCase):
    def test_show_images_in_matplotlib(self):
        visualization_tutorial()
        self.assertTrue(True)
