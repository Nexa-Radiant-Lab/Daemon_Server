"""
    Unit tests for the chunk_prompt function in the chunk_data module.
"""

import unittest
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunk_data import chunk_prompt


class TestChunkPrompt(unittest.TestCase):

    def test_basic_chunking(self):
        text = "This is a simple test case to check chunking functionality."
        chunk_size = 10
        result = chunk_prompt(text, chunk_size)
        expected = ["This is a", "simple", "test case", "to check", "chunking", "functionality."]
        self.assertEqual(result, expected)

    def test_large_chunk_size(self):
        text = "Chunking is fun!"
        chunk_size = 50
        result = chunk_prompt(text, chunk_size)
        expected = ["Chunking is fun!"]  # One chunk because size is large
        self.assertEqual(result, expected)

    def test_small_chunk_size(self):
        text = "Test small chunk"
        chunk_size = 4
        result = chunk_prompt(text, chunk_size)
        expected = ["Test", "small", "chunk"]  # Each word fits separately
        self.assertEqual(result, expected)

    def test_chunking_empty_string(self):
        text = ""
        chunk_size = 10
        result = chunk_prompt(text, chunk_size)
        expected = []  # No content, so no chunks
        self.assertEqual(result, expected)

    def test_invalid_chunk_size(self):
        text = "This should raise an error."
        chunk_size = 0
        with self.assertRaises(ValueError):
            chunk_prompt(text, chunk_size)

    def test_invalid_text_input(self):
        text = 1234  # Not a string
        chunk_size = 10
        with self.assertRaises(ValueError):
            chunk_prompt(text, chunk_size)

if __name__ == "__main__":
    unittest.main()
 