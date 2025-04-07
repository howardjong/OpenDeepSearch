
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
    # Read query from prompt.txt file
    try:
        with open('prompt.txt', 'r') as file:
            query = file.read().strip()
        if not query:
            logger.error("prompt.txt file is empty")
            print("Error: prompt.txt file is empty")
            sys.exit(1)
        logger.info(f"Read query from prompt.txt: {query}")
    except FileNotFoundError:
        logger.error("prompt.txt file not found")
        print("Error: prompt.txt file not found")
        sys.exit(1)
    
    # Using Serper with DeepSeek model on Fireworks AI
    logger.info("Initializing OpenDeepSearchTool with DeepSeek model")
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/accounts/fireworks/models/deepseek-r1-basic",
        reranker="jina"
    )
    
    logger.info("Initializing LiteLLMModel")
    model = LiteLLMModel(
        "fireworks_ai/accounts/fireworks/models/deepseek-r1-basic",
        temperature=0.2
    )
    
    # Make sure the search agent is set up
    logger.info("Setting up search agent")
    if not hasattr(search_agent, 'is_initialized') or not search_agent.is_initialized:
        search_agent.setup()
    
    logger.info("Creating CodeAgent")
    code_agent = CodeAgent(tools=[search_agent], model=model)
    
    # Monitor the number of search results
    logger.info(f"Running query: {query}")
    original_get_sources = search_agent.search_tool.serp_search.get_sources
    
    def get_sources_with_logging(*args, **kwargs):
        result = original_get_sources(*args, **kwargs)
        if result and hasattr(result, 'data') and 'organic' in result.data:
            num_results = len(result.data['organic'])
            print(f"\n>>> Number of search results received: {num_results} <<<\n")
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
    print("======================\n")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}")
