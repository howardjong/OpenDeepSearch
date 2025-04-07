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
    # First attempt with the specified model
    try:
        # Check if the model is Perplexity Sonar and use direct API
        if "perplexity/sonar" in args.model:
            logger.info("Using Perplexity AI directly instead of OpenRouter")
            direct_model_name = "perplexity/sonar"
            # Make sure PERPLEXITY_API_KEY is set in your environment variables
            if not os.getenv("PERPLEXITY_API_KEY"):
                logger.warning("PERPLEXITY_API_KEY not found in environment variables")
        else:
            direct_model_name = args.model
            
        # Initialize the search agent with the specified model
        logger.info(f"Initializing OpenDeepSearchTool with model {direct_model_name}")
        search_agent = OpenDeepSearchTool(
            model_name=direct_model_name,
            reranker="jina"
        )

        # Initialize the model with configurable parameters
        logger.info(f"Initializing LiteLLMModel with model {direct_model_name}")
        model = LiteLLMModel(
            direct_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        # Test if model works with a minimal query
        logger.info("Testing model connection...")
        test_response = model.client.completion(
            model=direct_model_name,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=args.temperature,
            max_tokens=10
        )
        logger.info(f"Successfully connected to model: {direct_model_name}")
        
    except Exception as model_error:
        # Fallback to a different model if the first one fails
        fallback_model = "fireworks_ai/accounts/fireworks/models/deepseek-r1-basic"
        logger.warning(f"Error with primary model {args.model}: {str(model_error)}")
        logger.info(f"Falling back to model: {fallback_model}")
        
        search_agent = OpenDeepSearchTool(
            model_name=fallback_model,
            reranker="jina"
        )
        
        model = LiteLLMModel(
            fallback_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
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
    
    # Extract more details from litellm errors
    if hasattr(e, '__cause__') and e.__cause__:
        logger.error(f"Cause: {str(e.__cause__)}")
        
        # Check for API-specific errors
        if 'OpenrouterException' in str(e.__cause__):
            logger.error("OpenRouter API Error detected. This could be due to:")
            logger.error("- Rate limits or quota exceeded")
            logger.error("- Invalid model name")
            logger.error("- Authentication issues")
        elif 'Perplexity' in str(e.__cause__):
            logger.error("Perplexity API Error detected. Check your API key and model name.")
    
    import traceback
    traceback.print_exc()
    sys.exit(1)