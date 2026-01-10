from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import create_structured_chat_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

def get_current_time(_input: str = "") -> str:
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query: str) -> str:
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except Exception:
        return "I couldn't find any information on that."

tools = [
    Tool(
        name="current_time",
        func=get_current_time,
        description='Returns the current local time. Input must be an empty string "".',
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Get a short Wikipedia summary for a topic. Input is the search query string.",
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")

# ✅ Make HF generation params consistent
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    return_full_text=False,
)
model = ChatHuggingFace(llm=llm)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_message(
    SystemMessage(
        content=(
            "You are a helpful assistant. You can use tools when needed.\n"
            "Available tools: current_time, Wikipedia."
        )
    )
)

agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="force",
)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # ✅ Do NOT manually add Human/AI messages; executor + memory handles it
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])