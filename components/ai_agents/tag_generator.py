#!/usr/bin/env python3

import ollama

class ContentGuard:
    """
    AI Career Tag Validator class designed to analyze content and match it with relevant career titles.

    Attributes:
        task (str): The primary instruction for the AI model.
        content (str): The text content to be analyzed for career relevance.
        career_list (list): A list of career titles to compare against the content.

    Methods:
        agent(task_prompt=None, content_prompt=None): Generates career tags based on the given prompts.
    """

    def __init__(self):
        """
        Initializes the ContentGuard object with default values.

        Sets up the initial task description, sample content, and a list of careers.
        These can be modified later using the agent method.
        """
        self.task = """
You are AI Career Tag Validator, an AI system designed to analyze content and compare it to a list of career titles provided to you. Your role is to evaluate the content, identify the key skills, topics, or themes, and determine which careers from the provided list are relevant to the content.

When analyzing content:

Compare the key skills, topics, or themes in the content to the careers in the list.
Identify which career titles from the list are relevant to the content.
Respond with the names of the relevant career titles from the list and only the career-list, or if none are relevant, respond with 'No relevant careers found.'
Do not provide any other explanations or context.
"""
        self.content = "This content talks about backend development, APIs, databases, and server-side programming."
        self.career_list = ["Backend Developer", "Frontend Developer", "Database Administrator", "Data Scientist"]

    def agent(self, task_prompt=None, content_prompt=None):
        """
        Generates career tags based on the given prompts.

        Args:
            task_prompt (str, optional): Additional instructions to append to the main task. Defaults to None.
            content_prompt (str, optional): Additional content to analyze. Defaults to None.

        Returns:
            str: A string containing the names of relevant career titles or 'No relevant careers found.'

        Raises:
            Exception: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt

        if content_prompt:
            self.content += "\n" + content_prompt

        full_prompt = self.task + "\n" + self.content + "\n" + ', '.join(self.career_list)

        try:
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
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Failed to generate response."

guard = ContentGuard()
print(guard.agent())