
from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# The API keys should be set in your Replit Secrets
# Make sure you have the following secrets set up:
# - SERPER_API_KEY
# - OPENROUTER_API_KEY
# - JINA_API_KEY

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
    reranker="jina"
)

model = LiteLLMModel(
    "openrouter/google/gemini-2.0-flash-001",
    temperature=0.2
)

# Make sure the search agent is set up
if not search_agent.is_initialized:
    search_agent.setup()

code_agent = CodeAgent(tools=[search_agent], model=model)
query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
result = code_agent.run(query)

print(result)
