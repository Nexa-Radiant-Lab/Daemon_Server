"""
A module that chunks data into a provided max figure.
"""

from typing import List

def chunk_prompt(text: str, chunk_size: int) -> List[str]:
    """
    Splits the provided text into chunks of specified size.

    Parameters:
    text (str): The text to be chunked.
    chunk_size (int): The maximum size of each chunk.

    Returns:
    List[str]: A list of text chunks.

    Raises:
    ValueError: If the text is not a string or if the chunk size is not a positive integer.
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    # Split the text into words
    words = text.split()
    
    # Initialize an empty list to hold the chunks
    chunks: List[str] = []  # Specify that chunks is a list of strings
    
    # Create a variable to hold the current chunk
    if not words:
        return chunks

    current_chunk: str = words[0]  # Start with the first word

    for word in words[1:]:
        # If adding the next word exceeds the chunk size, save the current chunk
        if len(current_chunk) + len(word) + 1 > chunk_size:
            chunks.append(current_chunk)
            current_chunk = word  # Start a new chunk with the current word
        else:
            current_chunk += " " + word  # Add the word to the current chunk

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks