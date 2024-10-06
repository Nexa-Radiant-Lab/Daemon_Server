"""
    Unit tests for the ContentGuard class in the content_guard module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust the sys.path to include the path where content_guard.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from content_guard import ContentGuard


class TestContentGuard(unittest.TestCase):

    def setUp(self):
        """
        Sets up the necessary inputs for the tests.
        """
        self.task = """
        You are AI Content Guard, an AI system designed to detect harmful
        or inappropriate content. Your role is to analyze data or content and
        identify forbidden content types.
        """
        self.content = "This is a test content."

    def test_initialization(self):
        """
        Test the initialization of the ContentGuard object.
        """
        guard = ContentGuard(self.task, self.content)
        self.assertEqual(guard.task, self.task)
        self.assertEqual(guard.content, self.content)

    def test_invalid_task(self):
        """
        Test that initializing with an empty task raises ValueError.
        """
        with self.assertRaises(ValueError):
            ContentGuard("", self.content)

    def test_invalid_content(self):
        """
        Test that initializing with empty content raises ValueError.
        """
        with self.assertRaises(ValueError):
            ContentGuard(self.task, "")

    @patch('utils.chunk_data.chunk_prompt')
    @patch('ollama.chat')
    def test_agent_method(self, mock_ollama_chat, mock_chunk_prompt):
        """
        Test the agent method's functionality.
        Mocks the Ollama API call and chunk_prompt function.
        """
        # Mock chunk_prompt to return a predefined chunk
        mock_chunk_prompt.return_value = ["This is a chunk of content."]
        
        # Mock ollama.chat to return a fake response
        mock_ollama_chat.return_value = {
            'message': {
                'content': "No, no forbidden content found."
            }
        }
        
        guard = ContentGuard(self.task, self.content)
        result = guard.agent()
        
        # Ensure chunk_prompt was called correctly
        mock_chunk_prompt.assert_called_once_with(self.content, chunk_size=1000)
        
        # Ensure ollama.chat was called with the expected prompt
        mock_ollama_chat.assert_called_once_with(
            model='phi3',
            messages=[
                {'role': 'user', 'content': f"{self.task}\nThis is a chunk of content."}
            ]
        )
        
        # Check that the result is as expected
        self.assertEqual(result, ["No, no forbidden content found."])

    @patch('ollama.chat', side_effect=Exception("API Error"))
    def test_agent_method_ollama_error(self, mock_ollama_chat):
        """
        Test that the agent method raises an error when the Ollama API fails.
        """
        guard = ContentGuard(self.task, self.content)
        
        with self.assertRaises(Exception) as context:
            guard.agent()
        
        self.assertIn("API Error", str(context.exception))
