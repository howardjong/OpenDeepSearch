
from smolagents import CodeAgent, LiteLLMModel
from opendeepsearch import OpenDeepSearchTool
import os
from dotenv import load_dotenv
import argparse
import gradio as gr
import sys

# Load environment variables
load_dotenv()

# Print Gradio version for debugging
print(f"Using Gradio version: {gr.__version__}")

# Ensure API keys are available from Replit Secrets
api_keys = {
    "JINA_API_KEY": os.environ.get("JINA_API_KEY"),
    "SERPER_API_KEY": os.environ.get("SERPER_API_KEY"),
    "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "FIREWORKS_API_KEY": os.environ.get("FIREWORKS_API_KEY")
}

# Log which API keys are available (without exposing the values)
print("API keys detected in environment:")
for key, value in api_keys.items():
    print(f"  - {key}: {'✓ Available' if value else '✗ Missing'}")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the Gradio demo with custom models')
parser.add_argument('--model-name',
                   default=os.getenv("LITELLM_SEARCH_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "anthropic/claude-3-7-sonnet-20250219")),
                   help='Model name for search')
parser.add_argument('--orchestrator-model',
                   default=os.getenv("LITELLM_ORCHESTRATOR_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "anthropic/claude-3-7-sonnet-20250219")),
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

try:
    # Create the search tool
    print(f"Initializing search tool with reranker: {args.reranker}")
    search_tool = OpenDeepSearchTool(
        model_name=args.model_name,
        reranker=args.reranker,
        search_provider=args.search_provider,
        serper_api_key=args.serper_api_key,
        searxng_instance_url=args.searxng_instance,
        searxng_api_key=args.searxng_api_key
    )
    print(f"Search tool initialized with model {args.model_name} and reranker {args.reranker}")
    
    # Create the model
    model = LiteLLMModel(
        model_id=args.orchestrator_model,
        temperature=0.2,
    )
    
    # Initialize the agent with the search tool
    agent = CodeAgent(tools=[search_tool], model=model)
    
    # Define a simple Gradio interface directly without using GradioUI from smolagents
    def process_query(query):
        try:
            return agent.run(query)
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    # Create a basic Gradio interface
    with gr.Blocks(title="OpenDeepSearch Demo") as demo:
        gr.Markdown("# 🔍 OpenDeepSearch Demo")
        gr.Markdown("Ask any question and get answers powered by AI search")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What is the fastest land animal?",
                lines=2
            )
        
        submit_btn = gr.Button("Search")
        
        output = gr.Textbox(
            label="Answer",
            lines=10
        )
        
        submit_btn.click(fn=process_query, inputs=query_input, outputs=output)
    
    # Launch the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=True)

except Exception as e:
    print(f"Fatal error initializing application: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
