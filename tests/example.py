
from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os
import sys
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Check for required API keys
required_keys = ["SERPER_API_KEY", "OPENROUTER_API_KEY", "JINA_API_KEY"]
print("Checking API keys:")
for key in required_keys:
    value = os.getenv(key)
    if not value:
        logger.warning(f"Missing required API key: {key}")
        print(f"- {key}: MISSING")
    else:
        # Only print first 3 and last 3 characters for security
        masked_value = value[:3] + "***" + value[-3:] if len(value) > 6 else "***"
        print(f"- {key}: {masked_value}")

try:
    # Using Serper (default)
    logger.info("Initializing OpenDeepSearchTool with Gemini Flash model")
    search_agent = OpenDeepSearchTool(
        model_name="openrouter/google/gemini-2.0-flash-001",
        reranker="jina"
    )
    
    logger.info("Initializing LiteLLMModel")
    model = LiteLLMModel(
        "openrouter/google/gemini-2.0-flash-001",
        temperature=0.2
    )
    
    # Make sure the search agent is set up
    logger.info("Setting up search agent")
    if not hasattr(search_agent, 'is_initialized') or not search_agent.is_initialized:
        search_agent.setup()
    
    logger.info("Creating CodeAgent")
    code_agent = CodeAgent(tools=[search_agent], model=model)
    
    # Example query for the cheetah speed test
    query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
    logger.info(f"Running query: {query}")
    
    # Monitor the number of search results
    original_get_sources = search_agent.search_tool.serp_search.get_sources
    
    def get_sources_with_logging(*args, **kwargs):
        result = original_get_sources(*args, **kwargs)
        if result and hasattr(result, 'data') and 'organic' in result.data:
            num_results = len(result.data['organic'])
            print(f"\n>>> Number of search results received: {num_results} <<<\n")
        return result
    
    # Replace the method temporarily
    search_agent.search_tool.serp_search.get_sources = get_sources_with_logging
    
    result = code_agent.run(query)
    
    logger.info("Query completed successfully")
    print("\n======= RESULT =======")
    print(result)
    print("======================\n")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}")
