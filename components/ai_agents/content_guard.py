# components/ai_agents/content_guard.py
import ollama

task = """
You are AI Content Guard, an AI system designed to detect harmful or inappropriate content. 
Your role is to analyze data or content and identify whether any of the following types of forbidden content are present:

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

When analyzing content, if any of these forbidden types are detected, respond with 'Yes' followed by the specific types of forbidden content found 
(e.g., 'Yes: Pornographic material, Hate speech'). If no forbidden content is found, respond with 'No, no forbidden content found.' 
Do not provide any additional explanation or context.
"""

content = "dummy data"


class ContentGuard:
    def agent(self, task, content):
        """Concatenate task and content prompts, pass to the model, and return response.

        Args:
            task (str): The prompt detailing the AI's task.
            content (str): The content to analyze.

        Returns:
            str: The model's response based on the concatenated prompt.
        """
        prompt = task.strip() + "\n" + content.strip()

        return self.analyze_content(prompt)

    def analyze_content(self, prompt):
        """Call the model with the prompt and return the response.

        Args:
            prompt (str): The combined prompt for analysis.

        Returns:
            str: Response from the model based on the prompt.
        """
        response = ollama.generate(prompt)

        return response['response']
if __name__ == "__main__":
    content_guard = ContentGuard()
    
    response = content_guard.agent(task, content)
    

    print(response)
