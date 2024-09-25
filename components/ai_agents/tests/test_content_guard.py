import unittest
from unittest.mock import patch
import sys
import os
import ollama

# Ensure the tag_generator module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tag_generator import ContentGuard  # Adjust the import according to your project structure

class TestContentGuard(unittest.TestCase):
    
    def setUp(self):
        self.task = "You are an AI Career Tag Validator."
        self.career_list = [
            "Backend Developer", 
            "Frontend Developer", 
            "Database Administrator", 
            "Data Scientist"
        ]
        self.dummy_content = "This is a sample content for testing career tags." 
        self.guard = ContentGuard(self.task, self.dummy_content, self.career_list)
        self.harmful_content = "This message promotes hate speech and discrimination against certain groups."
        self.empty_content = ""  # For testing empty content scenarios

    @patch('ollama.chat')
    def test_valid_content(self, mock_chat):
        mock_chat.return_value = {'message': {'content': 'Backend Developer, Database Administrator'}}
        self.guard.content = "This content discusses backend development and databases."
        result = self.guard.agent()  
        self.assertEqual(result, 'Backend Developer, Database Administrator')

    @patch('ollama.chat')
    def test_harmful_content(self, mock_chat):
        mock_chat.return_value = {'message': {'content': 'No relevant careers found.'}}
        self.guard.content = self.harmful_content
        result = self.guard.agent()  
        self.assertEqual(result, 'No relevant careers found.')

    @patch('ollama.chat')
    def test_empty_content(self, mock_chat):
        """Test the handling of empty content input."""
        
        # Mock chat return value for empty content
        mock_chat.return_value = {'message': {'content': 'Failed to generate response.'}}  # Return as dictionary

        result = self.guard.agent("")  # Ensure this method handles empty content
        self.assertEqual(result, "Failed to generate response.")

    @patch('ollama.chat')
    def test_no_careers_found(self, mock_chat):
        mock_chat.return_value = {'message': {'content': 'No relevant careers found.'}}
        self.guard.content = "This content does not mention any valid career."
        result = self.guard.agent()  
        self.assertEqual(result, 'No relevant careers found.')

    

if __name__ == '__main__':
    unittest.main()
