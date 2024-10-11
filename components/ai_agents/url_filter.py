#!/usr/bin/env python3

import ollama
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

task = """You are AI URL Filter, an AI system designed to analyze a list
of URLs and detect any that contain forbidden content. You will scan all
the URLs passed to you and check if they belong to one of the forbidden
categories listed below:

Your task:
Scan all URLs in the list.
Check each URL against the following forbidden categories:
Torrents or Piracy (e.g., torrent, piratebay, crack, keygen)
Social Media or Entertainment (e.g., facebook, instagram, youtube, netflix,
tiktok)
Adult or Explicit Content (e.g., xxx, porn, escort, cams, nsfw)
File Extensions (e.g., .torrent, .exe, .rar, .zip, .iso)
If any URLs contain forbidden content, return a list of those URLs in a
data structure format (e.g., a list or array) without additional explanation
or extra information.
If no forbidden content is found, respond with: 'No forbidden content
detected.'

Example Input:
[ "http://example.com/torrents/download/movies",
"http://facebook.com/user/profile",
"http://safewebsite.com/home",
"http://xvideos.com/adult-content",
"http://instagram.com/profile",
"http://safeblog.com/article"]

If all URLs are clean:
No forbidden content detected.

If forbidden content is detected:
[ "http://example.com/torrents/download/movies",
"http://facebook.com/user/profile",
"http://xvideos.com/adult-content",
"http://instagram.com/profile"]
"""

URLlist = ["http://example.com/torrents/download/movies",
           "http://facebook.com/user/profile",
           "http://safewebsite.com/home",
           "http://xvideos.com/adult-content",
           "http://instagram.com/profile",
           "http://safeblog.com/article"
           ]


class URL_Filter:
    """
    AI URL Filter class is designed to scan a batch of URLs and
    detect any that may contain harmful, inappropriate,
    or unwanted content

    Attributes:
        task (str): The primary instruction for the AI model.
        URLlist (list): A list of URLs to be analysed and
        determine the list of URLs that are forbidden.

    Methods:
        agent(task_prompt=None, content_prompt=None): passes
        the prompts to the model and returns back model's response.
    """

    def __init__(self, task, URLlist):
        """
        Initializes the URL_Filter object with external values.

        Args:
            task (str): The primary instruction for the AI model.
            URLlist (list): A list URLs to be analysed and
        determine the list of URLs that are forbidden.
        """
        self.validate_input(task, URLlist)
        self.task = task
        self.URLlist = URLlist

    def validate_input(self, task, URLlist):
        """
        Validates the task, URLlist inputs.

        Raises:
            ValueError: If any of the inputs are invalid or empty.
        """
        if not task.strip():
            raise ValueError("Task cannot be empty.")
        if not URLlist or not all(
                isinstance(url, str) for url in URLlist):
            raise ValueError("URL list must contain valid URL strings.")

    def agent(self, task_prompt=None, content_prompt=None):
        """
        Analyzes the given URLs to detect forbidden content.
        Args:
            task_prompt (str, optional): Additional instructions to append
            to the main task. Defaults to None.
            content_prompt (str, optional): Additional URLs to analyze.
            Defaults to None.

        Returns:
            str: A string containing the forbidden URLs or
            'No forbidden content detected.'
        Raises:
            ollama.OllamaError: If an error occurs during the Ollama API call.
        """
        if task_prompt:
            self.task += "\n" + task_prompt

        if content_prompt:
            self.URLlist += "\n" + content_prompt

        full_prompt = (self.task + "\n" + ', '.join(self.URLlist))

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


guard = URL_Filter(task, URLlist)
print(guard.agent())
