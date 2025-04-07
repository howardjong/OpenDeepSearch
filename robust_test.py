
"""
Robust test script for OpenDeepSearch with detailed error handling
"""
import os
import sys
import traceback
import logging
from dotenv import load_dotenv

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
    
    # Check API keys
    print("Checking API keys:")
    for key_name in ["SERPER_API_KEY", "OPENROUTER_API_KEY", "JINA_API_KEY"]:
        key_value = os.getenv(key_name)
        if key_value:
            # Print first 3 and last 3 characters of the key for verification
            print(f"- {key_name}: {key_value[:3]}***{key_value[-3:]}")
        else:
            print(f"- {key_name}: Missing!")
            logger.error(f"{key_name} not found in environment variables")
    
    try:
        # Import the library (only after checking environment)
        from opendeepsearch import OpenDeepSearchTool
        logger.info("Successfully imported OpenDeepSearchTool")
        
        # Initialize OpenDeepSearchTool
        logger.info("Initializing OpenDeepSearchTool with Gemini Flash model")
        search_tool = OpenDeepSearchTool(
            model_name="openrouter/google/gemini-2.0-flash-001",
            reranker="jina"  # Keep using Jina with our improved error handling
        )
        
        logger.info("Setting up search tool")
        if not search_tool.is_initialized:
            search_tool.setup()
        
        # Run a test query
        logger.info("Running test query")
        query = "What is the length of Pont Alexandre III?"
        print(f"\nQuery: {query}")
        
        result = search_tool.forward(query)
        print("\nResults:")
        print(result)
        
        # Run a second query
        second_query = "What is the top speed of a cheetah?"
        logger.info(f"Running second query: {second_query}")
        print(f"\nQuery: {second_query}")
        
        second_result = search_tool.forward(second_query)
        print("\nResults:")
        print(second_result)
        
        # Calculate time based on results (manually)
        print("\nWith these results, we can calculate:")
        print("Time = Distance/Speed")
        print("The time it would take a cheetah to run the length of Pont Alexandre III")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
