import unittest

from service import generation_service


class GenerationServiceTest(unittest.TestCase):
    def test_reproduce_generation(self):
        generation_service.reproduce_generation(1)
