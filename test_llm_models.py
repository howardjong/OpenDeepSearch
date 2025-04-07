
from opendeepsearch import OpenDeepSearchTool
from smolagents import LiteLLMModel, CodeAgent
import os
from dotenv import load_dotenv
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for models passed as command line arguments
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "openrouter/google/gemini-2.0-flash-001"  # Default model

logger.info(f"Testing with model: {model_name}")

try:
    # Initialize the search agent with the specified model
    logger.info(f"Initializing OpenDeepSearchTool with model {model_name}")
    search_agent = OpenDeepSearchTool(
        model_name=model_name,
        reranker="jina"
    )
    
    # Initialize the model
    logger.info(f"Initializing LiteLLMModel with model {model_name}")
    model = LiteLLMModel(
        model_name,
        temperature=0.2
    )
    
    # Set up the search agent
    logger.info("Setting up search agent")
    if not hasattr(search_agent, 'is_initialized') or not search_agent.is_initialized:
        search_agent.setup()
    
    # Create the CodeAgent
    logger.info("Creating CodeAgent")
    code_agent = CodeAgent(tools=[search_agent], model=model)
    
    # Test query
    query = "What is the likelihood of the VIX index continuing to rise the next trading day after closing at a value of 45 the day before?"
    logger.info(f"Running query: {query}")
    
    # Execute the query
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
