#!/usr/bin/env python3
import ollama


class ContentGuard:
    def __init__(self, task=None, content=None):
        """Initialize the ContentGuard with task and content."""
        self.task = task
        self.content = content

    def agent(self):
        """Concatenate task and content prompts, pass
        to the model, and return response.

        Returns:
            str: The model's response based on the concatenated prompt.
        """
        if not self.task:
            return "Task is missing."

        if not self.content:
            return "Content is not available for analysis."

        prompt = self.task.strip() + "\n" + self.content.strip()
        return self.analyze_content(prompt)

    def analyze_content(self, prompt):
        """Call the model with the prompt and return the response.

        Args:
            prompt (str): The combined prompt for analysis.

        Returns:
            str: Response from the model based on the prompt.
        """
        try:
            response = ollama.generate(prompt)
            return response['response']
        except Exception as e:
            return f"Error analyzing content: {str(e)}"


if __name__ == "__main__":
    task = """
    You are AI Content Guard, an AI system designed to detect harmful or
    inappropriate content.
    Your role is to analyze data or content and identify whether any of the
    following types of forbidden content are present:

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
    found(e.g., 'Yes: Pornographic material, Hate speech'). If no forbidden
    content is found, respond with 'No, no forbidden content found.'
    Do not provide any additional explanation or context.
    """

    content = "dummy data"

    content_guard = ContentGuard(task=task, content=content)

    response = content_guard.agent()
    print(response)
