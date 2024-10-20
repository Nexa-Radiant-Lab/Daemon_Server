#!/usr/bin/env python3
"""
Unit tests for the URL_Filter class in the ai_agent_module.

This module contains test cases for the functionality of the URL_Filter class,
which analyzes a list of URLs and detects if any forbidden content exists.
"""

import unittest
from unittest.mock import patch
import ollama
import logging
import os
import sys

# Adjust the sys.path to include the path where ai_agent_module.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming the script is saved in a module called 'ai_agent_module'
from ai_agent_module import URL_Filter


class TestURLFilter(unittest.TestCase):
    """
    Test cases for the URL_Filter class.

    This class tests the functionality of the URL_Filter class, which is
    designed to scan a list of URLs for forbidden content and return
    the appropriate response.
    """

    @patch('ollama.chat')
    @patch('logging.info')
    def test_url_filter_forbidden_content_detected(self, mock_logging_info, mock_ollama_chat):
        """
        Test the URL_Filter agent method when forbidden content is detected.

        The method should return a list of URLs that contain forbidden content.
        """
        # Arrange
        task = "Test forbidden content detection task"
        URLlist = ["http://example.com/torrents/download/movies",
                   "http://facebook.com/user/profile",
                   "http://safewebsite.com/home"]

        mock_ollama_chat.return_value = {
            'message': {
                'content': '["http://example.com/torrents/download/movies", '
                           '"http://facebook.com/user/profile"]'
            }
        }

        # Act
        url_filter = URL_Filter(task, URLlist)
        result = url_filter.agent()

        # Assert
        self.assertEqual(result, '["http://example.com/torrents/download/movies", '
                                 '"http://facebook.com/user/profile"]')
        mock_logging_info.assert_called_with("Sending prompt to Ollama API...")
        mock_ollama_chat.assert_called_once()

    @patch('ollama.chat')
    def test_url_filter_no_forbidden_content(self, mock_ollama_chat):
        """
        Test the URL_Filter agent method when no forbidden content is detected.

        The method should return 'No forbidden content detected.'.
        """
        # Arrange
        task = "Test no forbidden content"
        URLlist = ["http://safewebsite.com/home"]

        mock_ollama_chat.return_value = {
            'message': {'content': 'No forbidden content detected.'}
        }

        # Act
        url_filter = URL_Filter(task, URLlist)
        result = url_filter.agent()

        # Assert
        self.assertEqual(result, 'No forbidden content detected.')
        mock_ollama_chat.assert_called_once()

    def test_url_filter_empty_task(self):
        """
        Test the URL_Filter class when the task input is empty.

        The class should raise a ValueError if the task is empty.
        """
        # Arrange
        task = ""
        URLlist = ["http://safewebsite.com/home"]

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            URL_Filter(task, URLlist)
        self.assertEqual(str(context.exception), "Task cannot be empty.")

    def test_url_filter_invalid_url_list(self):
        """
        Test the URL_Filter class when the URL list input is invalid.

        The class should raise a ValueError if the URL list contains invalid data.
        """
        # Arrange
        task = "Test task"
        invalid_URLlist = [123, None, "http://safewebsite.com/home"]

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            URL_Filter(task, invalid_URLlist)
        self.assertEqual(str(context.exception), "URL list must contain valid URL strings.")

    @patch('ollama.chat', side_effect=ollama.ResponseError("API error"))
    def test_url_filter_api_error(self, mock_ollama_chat):
        """
        Test the URL_Filter agent method when an API error occurs.

        The method should raise an Ollama.ResponseError.
        """
        # Arrange
        task = "Test API error handling"
        URLlist = ["http://safewebsite.com/home"]

        # Act & Assert
        url_filter = URL_Filter(task, URLlist)
        with self.assertRaises(ollama.ResponseError):
            url_filter.agent()


if __name__ == '__main__':
    unittest.main()
