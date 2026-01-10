from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from langchain_core.tools import Tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


# List of tools available to the agent
tools = [
    Tool(
        name="current_time",  # Name of the tool
        func=get_current_time,  # Function that the tool will execute
        # Description of the tool
        description='Returns the current local time. Input must be an empty string "".',
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
# prompt = hub.pull("hwchase17/react")

react_template = """You are a helpful assistant that can use tools.

Tools:
{tools}

Tool names:
{tool_names}

CRITICAL FORMAT RULES:
- You must output EITHER:
  (A) a tool call, OR
  (B) a final answer
- NEVER output both a tool call and a final answer in the same message.

When you need a tool, output EXACTLY two lines and then STOP:
Action: <one of [{tool_names}]>
Action Input: <string>

Do NOT include Thought, Observation, or Final Answer when calling a tool.
Do NOT use parentheses in Action (never write current_time()).
For current_time, Action Input must be "".

After you receive an Observation (provided by the system), then you may output:
Final Answer: <answer>

User question: {input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(react_template)

# Initialize a Hugging Face model (using Mistral)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.001,  # HuggingFace requires temperature > 0
)
model = ChatHuggingFace(llm=llm)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
    stop_sequence=False
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it?"})

# Print the response from the agent
print("response:", response)
