import unittest

from service import generation_service


class GenerationServiceTest(unittest.TestCase):
    def test_create_generation_report(self):
        generation_service.create_generation_report(2)

    def test_reproduce_generation(self):
        generation_service.reproduce_generation(4)
