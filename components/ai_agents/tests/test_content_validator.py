import unittest
from unittest.mock import patch, MagicMock
import logging
import ollama
import os
import sys

# Adjust the sys.path to include the parent directory where
# content_guard.py or ai_agent_module.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import ContentValidator from the module,
# assuming it's in the parent directory
from ai_agent_module import ContentValidator


class TestContentValidator(unittest.TestCase):
    """
    Unit tests for the ContentValidator class.

    The ContentValidator class validates AI-generated content using the
    Ollama API, and these tests ensure that it behaves correctly under
    various conditions.
    """

    @patch('ollama.chat')  # Mock the Ollama API call
    @patch('logging.info')  # Mock the logging.info function
    @patch('os.makedirs')  # Mock os.makedirs
    def test_content_validator_agent(self, mock_makedirs, mock_logging_info,
                                     mock_ollama_chat):
        """
        Test that the ContentValidator can process content correctly.

        This test verifies that the ContentValidator interacts with the
        Ollama API and logs the correct information during content
        validation.
        """
        # Arrange
        task = "AI Content Validator task"
        content = "Sample content for testing"

        # Simulate a successful response from Ollama API
        mock_ollama_chat.return_value = {
            'message': {'content': 'No, the content is complete.'}
        }

        # Act
        validator = ContentValidator(task, content)
        result = validator.agent()

        # Assert
        self.assertEqual(result, 'No, the content is complete.')
        mock_logging_info.assert_called_with("Sending prompt to Ollama API...")
        mock_ollama_chat.assert_called_once()

    @patch('ollama.chat')  # Mock the Ollama API call
    def test_content_validator_empty_task(self, mock_ollama_chat):
        """
        Test that ContentValidator raises a ValueError for an empty task.

        This test verifies that the ContentValidator raises an appropriate
        exception when the task is empty.
        """
        # Arrange
        empty_task = ""
        content = "Sample content"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            ContentValidator(empty_task, content)
        self.assertEqual(str(context.exception), "Task cannot be empty.")

    @patch('ollama.chat')  # Mock the Ollama API call
    def test_content_validator_empty_content(self, mock_ollama_chat):
        """
        Test that ContentValidator raises a ValueError for empty content.

        This test verifies that the ContentValidator raises an appropriate
        exception when the content is empty.
        """
        # Arrange
        task = "AI Content Validator task"
        empty_content = ""

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            ContentValidator(task, empty_content)
        self.assertEqual(str(context.exception), "Content cannot be empty.")

    @patch('ollama.chat', side_effect=ollama.ResponseError("API error"))
    def test_content_validator_api_error(self, mock_ollama_chat):
        """
        Test that the ContentValidator handles API errors.

        This test ensures that the ContentValidator raises an exception
        when the Ollama API returns an error.
        """
        # Arrange
        task = "AI Content Validator task"
        content = "Sample content for testing"

        # Act & Assert
        validator = ContentValidator(task, content)
        with self.assertRaises(ollama.ResponseError):
            validator.agent()


if __name__ == '__main__':
    unittest.main()
