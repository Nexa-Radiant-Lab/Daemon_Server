"""
An AI agent module that creates career tags based on provided content or feeds.
"""

import ollama
import logging
from utils.chunk_data import chunk_prompt  # Import the chunking utility

# Define the log directory and file
log_dir = "/var/log/NRL-product-1/Daemon_Server"
log_file = "ai_agents.log"
log_path = os.path.join(log_dir, log_file)

# Ensure the log directory exists, create it if necessary
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Configure logging to output to the specified log file
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Logging TagGenerator...")

task = """
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

Below is the content to be analyzed:

"""

content = ("This content talks about backend development,"
            "APIs, databases, and server-side programming.")

career_list = ["Backend Developer", "Frontend Developer",
                "Database Administrator", "Data Scientist"]


class TagGenerator:
    """
    AI Career Tag Validator class designed to analyze content
    and match it with relevant career titles.

    Attributes:
        task (str): The primary instruction for the AI model.
        content (str): The text content to be analyzed for
        career relevance.
        career_list (list): A list of career titles to compare
        against the content.

    Methods:
        agent(task_prompt=None, content_prompt=None): Generates career
        tags based on the given prompts.
    """

    def __init__(self, task, content, career_list):
        """
        Initializes the ContentGuard object with external values.

        Args:
            task (str): The primary instruction for the AI model.
            content (str): The text content to be analyzed for
            career relevance.
            career_list (list): A list of career titles to compare
            against the content.
        """
        self.validate_input(task, content, career_list)
        self.task = task
        self.content = content
        self.career_list = career_list

    def validate_input(self, task, content, career_list):
        """
        Validates the task, content, and career_list inputs.

        Raises:
            ValueError: If any of the inputs are invalid or empty.
        """
        if not task.strip():
            raise ValueError("Task cannot be empty.")
        if not content.strip():
            raise ValueError("Content cannot be empty.")
        if not career_list or not all(
                isinstance(career, str) for career in career_list):
            raise ValueError("Career list must contain valid career titles.")

    def agent(self, task_prompt=None, content_prompt=None):
        """
        Generates career tags based on the given prompts.

        Args:
            task_prompt (str, optional): Additional instructions
            to append to the main task. Defaults to None.
            content_prompt (str, optional): Additional content
            to analyze. Defaults to None.

        Returns:
            list: A list of strings indicating relevant career titles
            for each chunk or 'No relevant careers found.'

        Raises:
            ollama.ResponseError: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt

        if content_prompt:
            self.content += "\n" + content_prompt

        # Chunk the content
        content_chunks = chunk_prompt(self.content, chunk_size=1000)  # Example chunk size of 1000 characters
        responses = []

        for chunk in content_chunks:
            full_prompt = f"{self.task}\n{chunk}\n{', '.join(self.career_list)}"

            try:
                logging.info("Sending prompt to Ollama API...")
                response = ollama.chat(
                    model='phi3',
                    messages=[
                        {
                            'role': 'user',
                            'content': full_prompt
                        }
                    ],
                )
                responses.append(response['message']['content'])
            except ollama.ResponseError as e:
                logging.error(f"Ollama API error: {e}")
                raise e
            except Exception as e:
                logging.error(f"Unexpected error occurred: {e}")
                raise  e

        return responses  # Return responses for each chunk


guard = TagGenerator(task, content, career_list)
results = guard.agent()
for result in results:
    print(result)