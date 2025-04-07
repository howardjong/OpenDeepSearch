from opendeepsearch import OpenDeepSearchTool
from smolagents import LiteLLMModel, CodeAgent
import os
from dotenv import load_dotenv
import logging
import sys
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Argument parser for model parameters and query
parser = argparse.ArgumentParser(description="Test OpenDeepSearch with configurable parameters.")
parser.add_argument("--model", default="openrouter/google/gemini-2.0-flash-001", help="Name of the LLM model to use.")
parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for LLM generation.")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for LLM generation.")
parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens for LLM generation.")
parser.add_argument("--query", help="Query to run. If not provided, a default query will be used.")
args = parser.parse_args()

logger.info(f"Testing with model: {args.model}")

try:
    # Initialize the search agent with the specified model
    logger.info(f"Initializing OpenDeepSearchTool with model {args.model}")
    search_agent = OpenDeepSearchTool(
        model_name=args.model,
        reranker="jina"
    )

    # Initialize the model with configurable parameters
    logger.info(f"Initializing LiteLLMModel with model {args.model}")
    model = LiteLLMModel(
        args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Set up the search agent
    logger.info("Setting up search agent")
    if not hasattr(search_agent, 'is_initialized') or not search_agent.is_initialized:
        search_agent.setup()

    # Create the CodeAgent
    logger.info("Creating CodeAgent")
    code_agent = CodeAgent(tools=[search_agent], model=model)

    # Test query (use command line argument or default)
    query = args.query if args.query else "What is the likelihood that the VIX index will continue to rise the next trading day after closing at a value of 45 the previous day?"
    logger.info(f"Running query: {query}")

    # Execute the query with configured parameters
    result = code_agent.run(query)

    logger.info("Query completed successfully")
    print("\n======= RESULT =======")
    print(result)
    print("======================\n")

except Exception as e:
    logger.error(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)