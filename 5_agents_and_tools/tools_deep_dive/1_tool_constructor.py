# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent

from langchain_core.tools import StructuredTool, Tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


# Functions for the tools
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"


def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


import re

def concatenate_from_text(text: str) -> str:
    # extract two quoted chunks (single or double quotes)
    parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
    vals = [a or b for a, b in parts]
    if len(vals) >= 2:
        return vals[0] + vals[1]
    # fallback: split by comma
    items = [x.strip() for x in text.split(",")]
    if len(items) >= 2:
        return items[0].strip("'\"") + items[1].strip("'\"")
    raise ValueError("Need two strings to concatenate.")


# Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


# Create tools using the Tool and StructuredTool constructor approach
tools = [
    # Use Tool for simpler functions with a single input parameter.
    # This is straightforward and doesn't require an input schema.
    Tool(
        name="GreetUser",  # Name of the tool
        func=greet_user,  # Function to execute
        description="Greets the user by name.",  # Description of the tool
    ),
    # Use Tool for another simple function with a single input parameter.
    Tool(
        name="ReverseString",  # Name of the tool
        func=reverse_string,  # Function to execute
        description="Reverses the given string.",  # Description of the tool
    ),
    # Use StructuredTool for more complex functions that require multiple input parameters.
    # StructuredTool allows us to define an input schema using Pydantic, ensuring proper validation and description.
    StructuredTool.from_function(
        func=concatenate_from_text,  # Function to execute
        name="ConcatenateStrings",  # Name of the tool
        description="Concatenates two strings.",  # Description of the tool
        args_schema=ConcatenateStringsArgs,  # Schema defining the tool's input arguments
    ),
]

# Initialize a Hugging Face model (using Mistral)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.01,  # HuggingFace requires temperature > 0
)
model = ChatHuggingFace(llm=llm)

# Pull the prompt template from the hub (using react prompt for HuggingFace compatibility)
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=model,  # Language model to use
    tools=tools,  # List of tools available to the agent
    prompt=prompt,  # Prompt template to guide the agent's responses
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # The agent to execute
    tools=tools,  # List of tools available to the agent
    verbose=True,  # Enable verbose logging
    handle_parsing_errors=True,  # Handle parsing errors gracefully
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)
