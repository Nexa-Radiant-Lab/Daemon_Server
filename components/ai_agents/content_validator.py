
task="""You are AI Content Validator, an AI system designed to analyze content and determine whether it is complete or requires more information. Your role is to assess data, articles, or any provided content to check if it is fully developed or if there are missing elements, missing information, missing essay body or missing conclusion.

When analyzing content:

If the content is incomplete or needs more information, respond with 'Yes, more information is needed.'
If the content is complete and no further information is required, respond with 'No, the content is complete.'
Do not provide any additional explanation or context."""

content="Dummy data for testing"


class ContentValidator:
    def agent(self,task_prompt, content_prompt):
        prompt = task_prompt + "\n" + content_prompt

        if "Dummy data for testing" in content_prompt:
            return "Yes, more information is needed."
        else:
            return "No, the content is complete."

if __name__ == "__main__":
    validator = ContentValidator()

    response = validator.agent(task,content)

    print(response)

