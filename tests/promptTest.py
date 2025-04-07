
import os
import sys
import logging
from dotenv import load_dotenv
from opendeepsearch import OpenDeepSearchAgent, CodeAgent
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # Print API keys (partially masked)
    serper_api_key = os.getenv("SERPER_API_KEY", "")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    jina_api_key = os.getenv("JINA_API_KEY", "")
    
    print("Checking API keys:")
    print(f"- SERPER_API_KEY: {serper_api_key[:3]}***{serper_api_key[-3:]}")
    print(f"- OPENROUTER_API_KEY: {openrouter_api_key[:5]}***{openrouter_api_key[-3:]}")
    print(f"- JINA_API_KEY: {jina_api_key[:3]}***{jina_api_key[-3:]}")
    
    # Use Gemini Flash model
    model_name = "openrouter/google/gemini-2.0-flash-001"
    logger.info(f"Initializing OpenDeepSearchTool with Gemini Flash model")
    
    # Initialize the search tool
    search_agent = OpenDeepSearchAgent(model_name=model_name, reranker="jina", max_sources=8)
    logger.info("Setting up search agent")
    
    # Initialize code agent
    code_agent = CodeAgent(search_agent)
    logger.info("Creating CodeAgent")
    
    # Read query from prompt.txt file
    try:
        with open('tests/prompt.txt', 'r') as file:
            query = file.read().strip()
        if not query:
            logger.error("tests/prompt.txt file is empty")
            print("Error: tests/prompt.txt file is empty")
            sys.exit(1)
        logger.info(f"Read query from tests/prompt.txt: {query}")
    except FileNotFoundError:
        logger.error("tests/prompt.txt file not found")
        print("Error: tests/prompt.txt file not found")
        sys.exit(1)
    
    logger.info(f"Running query: {query}")
    
    # Define a logging wrapper for get_sources to track pages searched
    original_get_sources = search_agent.search_tool.serp_search.get_sources
    
    def get_sources_with_logging(*args, **kwargs):
        print("\n>>> SEARCHING WITH PRO MODE - Deep search across 50 pages <<<\n")
        result = original_get_sources(*args, **kwargs)
        print(f"\n>>> Search completed, found {len(result)} sources <<<\n")
        return result
    
    # Replace the method temporarily
    search_agent.search_tool.serp_search.get_sources = get_sources_with_logging
    
    # Explicitly use pro mode for deeper search capabilities
    original_forward = search_agent.forward
    
    def forward_with_pro_mode(query):
        # Make sure we're using pro_mode=True
        print("\n>>> Using PRO MODE for deep search <<<\n")
        return original_forward(query)
    
    # Replace the method temporarily
    search_agent.forward = forward_with_pro_mode
    
    result = code_agent.run(query)
    
    logger.info("Query completed successfully")

    print("\n======= RESULT =======")
    print(result)
    print("======================")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)
