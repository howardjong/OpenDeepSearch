
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent
import os
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
required_keys = ["SERPER_API_KEY", "JINA_API_KEY", "WOLFRAM_ALPHA_APP_ID"]
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
    # Initialize the LLM model
    logger.info("Initializing LiteLLMModel")
    model = LiteLLMModel(
        "openrouter/google/gemini-2.0-flash-001",  # Using Gemini model which we know works
        temperature=0.7
    )
    
    # Initialize the search agent
    logger.info("Initializing OpenDeepSearchTool")
    search_agent = OpenDeepSearchTool(
        model_name="openrouter/google/gemini-2.0-flash-001", 
        reranker="jina"
    )
    
    # Initialize the Wolfram Alpha tool
    logger.info("Initializing WolframAlphaTool")
    wolfram_tool = WolframAlphaTool(app_id=os.getenv("WOLFRAM_ALPHA_APP_ID"))
    
    # Set up the tools
    logger.info("Setting up tools")
    if not search_agent.is_initialized:
        search_agent.setup()
    
    # Initialize the React Agent with search and wolfram tools
    logger.info("Creating ToolCallingAgent with REACT_PROMPT")
    react_agent = ToolCallingAgent(
        tools=[search_agent, wolfram_tool],
        model=model,
        prompt_templates=REACT_PROMPT  # Using REACT_PROMPT as system prompt
    )
    
    # Example query for the React Agent
    query = "What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice"
    logger.info(f"Running query: {query}")
    
    result = react_agent.run(query)
    
    logger.info("Query completed successfully")
    print("\n======= RESULT =======")
    print(result)
    print("======================\n")
    
except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}")
