"""
An AI agent module that assesses the completeness of written content.
"""

import ollama
import logging
import os

# Define the log directory and file path for logging AI agent activities
log_dir = "/var/log/NRL-product-1/Daemon_Server"
log_file = "ai_agents.log"
log_path = os.path.join(log_dir, log_file)

# Ensure the log directory exists; if not, create it
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Configure logging to output to the specified log file
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Logging ContentValidator...")  # Log the initialization of the ContentValidator

# Define the primary task for the AI Content Validator
task = """
You are AI Content Validator, an AI system designed to analyze content and determine whether it is complete or requires more information. Your role is to assess data, articles, or any provided content to check if it is fully developed or if there are missing elements, missing information, missing essay body or missing conclusion.

When analyzing content:

If the content is incomplete or needs more information, respond with 'Yes, more information is needed.'
If the content is complete and no further information is required, respond with 'No, the content is complete.'
Do not provide any additional explanation or context.
"""

# Sample content to be analyzed
content = "Dummy data for testing"

class ContentValidator:
    """
    AI Content Validator class designed to analyze content
    and determine its completeness.

    Attributes:
        task (str): The primary instruction for the AI model.
        content (str): The text content to be analyzed for completeness.

    Methods:
        agent(task_prompt=None, content_prompt=None): Analyzes
        the content based on the given prompts.
    """

    def __init__(self, task, content):
        """
        Initializes the ContentValidator object with external values.

        Args:
            task (str): The primary instruction for the AI model.
            content (str): The text content to be analyzed for completeness.
        """
        self.validate_input(task, content)  # Validate inputs before setting them
        self.task = task
        self.content = content

    def validate_input(self, task, content):
        """
        Validates the task and content inputs.

        Raises:
            ValueError: If any of the inputs are invalid or empty.
        """
        if not task.strip():
            raise ValueError("Task cannot be empty.")  # Raise an error if the task is empty
        if not content.strip():
            raise ValueError("Content cannot be empty.")  # Raise an error if the content is empty

    def agent(self, task_prompt=None, content_prompt=None):
        """
        Analyzes the content based on the given prompts.

        Args:
            task_prompt (str, optional): Additional instructions
            to append to the main task. Defaults to None.
            content_prompt (str, optional): Additional content
            to append for analysis. Defaults to None.

        Returns:
            str: A response indicating whether more information is needed or if the content is complete.

        Raises:
            ollama.OllamaError: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt  # Append any additional task instructions
        if content_prompt:
            self.content += "\n" + content_prompt  # Append any additional content for analysis

        # Combine the task and content into a single prompt for the AI model
        full_prompt = f"{self.task}\n{self.content}"

        try:
            logging.info("Sending prompt to Ollama API...")  # Log the API request
            # Send the combined prompt to the Ollama API and receive a response
            response = ollama.chat(
                model='phi3',
                messages=[{
                    'role': 'user',
                    'content': full_prompt
                }],
            )
            return response['message']['content']  # Return the content of the response
        except ollama.ResponseError as e:
            logging.error(f"Ollama API error: {e}")  # Log any API-specific errors
            raise e
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")  # Log any unexpected errors
            raise e


# Create an instance of ContentValidator with the predefined task and content
validator = ContentValidator(task, content)
result = validator.agent()  # Analyze the content and get the result
print(result)  # Output the result of the content analysis
