#!/usr/bin/env python3

import ollama
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

task = """
You are AI Content Guard, an AI system designed to detect harmful or
inappropriate content. Your role is to analyze data or content and
identify whether any of the following types of forbidden content are present:

- Abusive content
- Pornographic material
- Discriminatory or exclusive content
- Violent content
- Hate speech
- Promotion of self-harm or suicide
- Drug-related content
- Cyberbullying or harassment
- Terrorism or extremist propaganda
- Graphic violence or gore
- Promotion of illegal activities
- Child exploitation

When analyzing content, if any of these forbidden types are detected,
respond with 'Yes' followed by the specific types of forbidden content
found (e.g., 'Yes: Pornographic material, Hate speech'). If no forbidden
content is found, respond with 'No, no forbidden content found.' Do not
provide any additional explanation or context.
"""

content = "dummy data"


class ContentGuard:
    """
    AI Content Guard class designed to analyze content
    and identify any harmful or inappropriate types.

    Attributes:
        task (str): The primary instruction for the AI model.
        content (str): The text content to be analyzed for
        forbidden content types.

    Methods:
        agent(task_prompt=None, content_prompt=None): Analyzes
        the content based on the given prompts.
    """

    def __init__(self, task, content):
        """
        Initializes the ContentGuard object with external values.

        Args:
            task (str): The primary instruction for the AI model.
            content (str): The text content to be analyzed for
            forbidden content.
        """
        self.validate_input(task, content)
        self.task = task
        self.content = content

    def validate_input(self, task, content):
        """
        Validates the task and content inputs.

        Raises:
            ValueError: If any of the inputs are invalid or empty.
        """
        if not task.strip():
            raise ValueError("Task cannot be empty.")
        if not content.strip():
            raise ValueError("Content cannot be empty.")

    def agent(self, task_prompt=None, content_prompt=None):
        """
        Analyzes the content based on the given prompts.

        Args:
            task_prompt (str, optional): Additional instructions
            to append to the main task. Defaults to None.
            content_prompt (str, optional): Additional content
            to analyze. Defaults to None.

        Returns:
            str: A string indicating whether forbidden content
            was found or not.

        Raises:
            ollama.OllamaError: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt

        if content_prompt:
            self.content += "\n" + content_prompt

        full_prompt = f"{self.task}\n{self.content}"

        try:
            logging.info("Sending prompt to Ollama API...")
            response = ollama.chat(
                model='Phi3',
                messages=[
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
            )
            return response['message']['content']
        except ollama.OllamaError as e:
            logging.error(f"Ollama API error: {e}")
            return "Failed to generate response."
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            return "Failed to generate response."


guard = ContentGuard(task, content)
print(guard.agent())
