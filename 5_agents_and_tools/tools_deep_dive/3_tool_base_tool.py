# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
import os
from typing import Type

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Pydantic models for tool arguments


class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")


# Custom tool with only custom input


class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for: {query}\n\n\n{results}\n"


# Custom tool with custom input and output
class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(
        self,
        x: float,
        y: float,
    ) -> str:
        """Use the tool."""
        result = x * y
        return f"The product of {x} and {y} is {result}"


# Create tools using the Pydantic subclass approach
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
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
    llm=model,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response)
