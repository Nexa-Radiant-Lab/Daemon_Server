"""
    Unit tests for the TagGenerator class in the tag_generator module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the directory containing tag_generator to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tag_generator import TagGenerator, ollama


class TestTagGenerator(unittest.TestCase):

    def setUp(self):
        self.task = """
        You are AI Career Tag generator, an AI system designed to analyze content
        and compare it to a list of career titles provided to you.
        Your role is to evaluate the content, identify the key skills, topics,
        or themes, and determine which careers from the provided list are
        relevant to the content.

        When analyzing content:

        Compare the key skills, topics, or themes in the content to
        the careers in the list.
        Identify which career titles from the list are relevant to the content.
        Respond with the names of the relevant career titles from the list and only
        the career-list, or if none are relevant, respond with 'No relevant
        careers found.'
        """
        self.content = ("This content talks about backend development,"
                        "APIs, databases, and server-side programming.")
        self.career_list = ["Backend Developer", "Frontend Developer",
                            "Database Administrator", "Data Scientist"]
        self.tag_generator = TagGenerator(self.task, self.content, self.career_list)

    def test_validate_input_valid(self):
        # Test valid inputs
        try:
            self.tag_generator.validate_input(self.task, self.content, self.career_list)
        except ValueError:
            self.fail("validate_input raised ValueError unexpectedly!")

    def test_validate_input_invalid_task(self):
        # Test invalid task
        with self.assertRaises(ValueError):
            self.tag_generator.validate_input("", self.content, self.career_list)

    def test_validate_input_invalid_content(self):
        # Test invalid content
        with self.assertRaises(ValueError):
            self.tag_generator.validate_input(self.task, "", self.career_list)

    def test_validate_input_invalid_career_list(self):
        # Test invalid career list
        with self.assertRaises(ValueError):
            self.tag_generator.validate_input(self.task, self.content, [])

    @patch('tag_generator.ollama.chat')
    def test_agent(self, mock_chat):
        # Mock the response from ollama.chat
        mock_response = MagicMock()
        mock_response['message']['content'] = "Backend Developer, Database Administrator"
        mock_chat.return_value = mock_response

        results = self.tag_generator.agent()
        self.assertIn("Backend Developer", results[0])
        self.assertIn("Database Administrator", results[0])

    @patch('tag_generator.ollama.chat')
    def test_agent_no_relevant_careers(self, mock_chat):
        # Mock the response from ollama.chat
        mock_response = MagicMock()
        mock_response['message']['content'] = "No relevant careers found."
        mock_chat.return_value = mock_response

        results = self.tag_generator.agent()
        self.assertIn("No relevant careers found.", results[0])

if __name__ == '__main__':
    unittest.main()
