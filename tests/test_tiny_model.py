from unittest import TestCase
from tiny_model import TinyModel


class TestTinyModel(TestCase):
    def test_constructor(self):
        tinymodel = TinyModel()

        print('The model:')
        print(tinymodel)

        print('\n\nJust one layer:')
        print(tinymodel.linear2)

        print('\n\nModel params:')
        for param in tinymodel.parameters():
            print(param)

        print('\n\nLayer params:')
        for param in tinymodel.linear2.parameters():
            print(param)

        self.assertTrue(True)
