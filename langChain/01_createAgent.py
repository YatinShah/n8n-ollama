# ref - https://docs.langchain.com/oss/python/langchain/quickstart

# export LANGSMITH_TRACING=true
# export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# export LANGSMITH_PROJECT=langchain_tuts

# tracing output can be seen at https://smith.langchain.com/o/573d44b7-52a1-58b8-a2cf-9dc3eb05137c/projects/p/e2a0acea-20ed-495e-8710-285a0007c555?timeModel=%7B%22duration%22%3A%227d%22%7D&peek=f4961c7c-a484-4e6e-899e-6112445872ac&peeked_trace=f4961c7c-a484-4e6e-899e-6112445872ac

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Define the weather function
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Initialize the language model
llm = ChatOllama(
    model="gemma:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)