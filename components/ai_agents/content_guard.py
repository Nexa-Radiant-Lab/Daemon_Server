"""
An AI agent module that analyses the content of given feeds or data.
"""

import ollama
import logging
from utils.chunk_data import chunk_prompt

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

content = """
In the world of artificial intelligence, the importance of ethical considerations has grown significantly as the technology continues to evolve and impact various aspects of society. As AI systems become increasingly capable of performing complex tasks, understanding the implications of their deployment becomes crucial. The need for responsible AI development stems from the potential risks associated with biased algorithms, privacy violations, and unintended consequences that can arise from autonomous decision-making processes.

One of the key ethical concerns is algorithmic bias. AI systems learn from historical data, and if that data reflects existing prejudices or inequalities, the AI can perpetuate and even amplify those biases. This can lead to unfair treatment of individuals based on race, gender, or socioeconomic status in areas such as hiring, lending, and law enforcement. To combat this issue, it is essential to implement fairness in AI practices, including thorough data auditing, bias detection techniques, and diverse training datasets.

Privacy is another major consideration. The widespread use of AI often involves the collection and analysis of vast amounts of personal data. This raises questions about consent and data ownership, as well as the potential for misuse. Organizations must prioritize transparent data practices and ensure that individuals are aware of how their information is being used. Robust data protection measures should also be in place to prevent unauthorized access and breaches.

Additionally, the rise of autonomous systems brings forth the challenge of accountability. When an AI makes a decision that results in harm or loss, it can be difficult to determine who is responsible. This ambiguity can complicate legal frameworks and hinder the pursuit of justice for those affected. As such, establishing clear guidelines for accountability and liability in AI systems is critical.

Moreover, there is a pressing need to consider the social implications of AI technology. Automation and AI-driven solutions can displace jobs and disrupt traditional industries. While these technologies have the potential to enhance productivity and create new opportunities, it is vital to address the socioeconomic impacts on workers and communities. Policies and initiatives that support workforce transition and skill development can help mitigate these effects.

To address these ethical concerns, collaboration between technologists, ethicists, policymakers, and the public is necessary. Engaging diverse stakeholders in discussions about AI governance and ethics can lead to more inclusive and effective solutions. Establishing interdisciplinary committees and fostering an open dialogue about the ethical use of AI will contribute to the development of guidelines and standards that prioritize the well-being of individuals and society.

Furthermore, education plays a pivotal role in fostering ethical AI practices. Incorporating ethics into computer science curricula and training programs for AI practitioners can help instill a sense of responsibility and awareness about the potential consequences of their work. By equipping future generations of technologists with the knowledge and skills to navigate ethical dilemmas, we can build a more responsible AI landscape.

In conclusion, as artificial intelligence continues to advance, addressing the ethical implications of its use is paramount. By prioritizing fairness, transparency, accountability, and collaboration, we can harness the benefits of AI while minimizing the risks. The journey toward responsible AI development is ongoing, and it requires the collective efforts of all stakeholders involved. Together, we can shape a future where AI serves humanity positively and equitably.

"""

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

    def agent(self, task_prompt=None):
        """
            Analyzes the content based on the given prompts.

            Args:
                task_prompt (str, optional): Additional instructions
                to append to the main task. Defaults to None.

            Returns:
                list: A list of strings indicating whether forbidden content
                was found or not for each chunk.

            Raises:
                ollama.OllamaError: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt

        # Chunk the content
        content_chunks = chunk_prompt(self.content, chunk_size=1000)  # Example chunk size of 1000 characters
        responses = []

        for chunk in content_chunks:
            full_prompt = f"{self.task}\n{chunk}"

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
            except Exception as e:
                logging.error(f"Unexpected error occurred: {e}")
                responses.append("Failed to generate response for this chunk.")

        return responses  # Return responses for each chunk


guard = ContentGuard(task, content)
results = guard.agent()
for result in results:
    print(result)
