from smolagents import CodeAgent, GradioUI, LiteLLMModel
from opendeepsearch import OpenDeepSearchTool
import os
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Run the Gradio demo with custom models')
parser.add_argument('--model-name',
                   default=os.getenv("LITELLM_SEARCH_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")),
                   help='Model name for search')
parser.add_argument('--orchestrator-model',
                   default=os.getenv("LITELLM_ORCHESTRATOR_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001")),
                   help='Model name for orchestration')
parser.add_argument('--reranker',
                   choices=['jina', 'infinity'],
                   default='jina',
                   help='Reranker to use (jina or infinity)')
parser.add_argument('--search-provider',
                   choices=['serper', 'searxng'],
                   default='serper',
                   help='Search provider to use (serper or searxng)')
parser.add_argument('--searxng-instance',
                   help='SearXNG instance URL (required if search-provider is searxng)')
parser.add_argument('--searxng-api-key',
                   help='SearXNG API key (optional)')
parser.add_argument('--serper-api-key',
                   help='Serper API key (optional, will use SERPER_API_KEY env var if not provided)')
parser.add_argument('--openai-base-url',
                   help='OpenAI API base URL (optional, will use OPENAI_BASE_URL env var if not provided)')
parser.add_argument('--server-port',
                   type=int,
                   default=7860,
                   help='Port to run the Gradio server on')

args = parser.parse_args()

# Validate arguments
if args.search_provider == 'searxng' and not (args.searxng_instance or os.getenv('SEARXNG_INSTANCE_URL')):
    parser.error("--searxng-instance is required when using --search-provider=searxng")

# Set OpenAI base URL if provided via command line
if args.openai_base_url:
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url

# Use the command line arguments
search_tool = OpenDeepSearchTool(
    model_name=args.model_name,
    reranker=args.reranker,
    search_provider=args.search_provider,
    serper_api_key=args.serper_api_key,
    searxng_instance_url=args.searxng_instance,
    searxng_api_key=args.searxng_api_key
)
model = LiteLLMModel(
    model_id=args.orchestrator_model,
    temperature=0.2,
)

# Initialize the agent with the search tool
agent = CodeAgent(tools=[search_tool], model=model)

# Add a name when initializing GradioUI
# Using 0.0.0.0 to make it accessible in Replit environment
try:
    # First try with standard parameters for Gradio 4.x
    GradioUI(agent).launch(server_name="0.0.0.0", server_port=args.server_port, share=True)
except Exception as e:
    print(f"Error launching with standard parameters: {e}")
    # Fall back to parameters that might work with other Gradio versions
    import gradio as gr
    print(f"Using Gradio version: {gr.__version__}")
    GradioUI(agent).launch(server_name="0.0.0.0", server_port=args.server_port, 
             share=True, show_error=True)
