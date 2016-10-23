import unittest

from dataset import Dataset


class TestDataSetConstruction(unittest.TestCase):
    def test_bigrams_construction(self):
        text = 'aabbccdd'
        data = Dataset.create_from_text(text)

        self.assertTrue(data.get_bigram_id("aa") == 0)
        self.assertTrue(data.get_bigram_id("bb") == 1)
        self.assertTrue(data.get_bigram_id("cc") == 2)
        self.assertTrue(data.get_bigram_id("dd") == 3)
        self.assertTrue(data.text_from_bigrams == [0, 1, 2, 3])

    def test_bigrams_indexing(self):
        text = 'aabbccdd'
        data = Dataset.create_from_text(text)

        self.assertTrue(data.get_bigram_id("aa") == 0)
        self.assertTrue(data.get_bigram_id("bb") == 1)
        self.assertTrue(data.get_bigram_id("cc") == 2)
        self.assertTrue(data.get_bigram_id("dd") == 3)
        self.assertTrue(data.text_from_bigrams == [0, 1, 2, 3])
