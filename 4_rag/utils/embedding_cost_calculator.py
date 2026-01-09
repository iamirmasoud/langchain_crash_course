import os

import tiktoken

# Define the file path for the document
file_path = os.path.join(os.path.dirname(__file__), "..", "books", "odyssey.txt")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # Use the appropriate encoding for the model

# Tokenize the text and count the tokens
tokens = tokenizer.encode(text)
total_tokens = len(tokens)

# Print the results
print(f"Total number of tokens: {total_tokens}")
print(
    "If you're using local Hugging Face / SentenceTransformers embeddings (as in this repo), "
    "the API cost is $0 (you only pay compute on your machine)."
)
