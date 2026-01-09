# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# Hugging Face (Inference API) Documents:
# - https://python.langchain.com/v0.2/docs/integrations/llms/huggingface_endpoint/
# - https://python.langchain.com/v0.2/docs/integrations/chat/huggingface/

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env
load_dotenv()

# Requirements:
# - set `HUGGINGFACEHUB_API_TOKEN` in your `.env`
#
# Notes:
# - `HuggingFaceEndpoint` uses Hugging Face's Inference API (hosted model).
# - `ChatHuggingFace` wraps the endpoint so you get a chat-model style response
#   (e.g. `.content`) similar to other LangChain chat models.
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=128,
    temperature=0.0,
)
model = ChatHuggingFace(llm=llm)

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
