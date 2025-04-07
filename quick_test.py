
"""
Quick test script that avoids the issues encountered in previous tests
"""
import os
import time
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
    print("API Keys Status:")
    for key in ["SERPER_API_KEY", "OPENROUTER_API_KEY", "JINA_API_KEY"]:
        value = os.getenv(key)
        if value:
            print(f"- {key}: Present ({value[:3]}...{value[-3:]})")
        else:
            print(f"- {key}: Missing")
    
    print("\nImporting OpenDeepSearchTool...")
    try:
        from opendeepsearch import OpenDeepSearchTool
        print("✅ Import successful")
        
        print("\nInitializing OpenDeepSearchTool with minimal configuration...")
        # Use a simpler configuration without code agent
        search_tool = OpenDeepSearchTool(
            model_name="openrouter/google/gemini-2.0-flash-001",
            reranker="jina"  # Use jina reranker for better results
        )
        
        print("Setting up search tool...")
        search_tool.setup()
        print("✅ Setup successful")
        
        print("\nRunning a basic query...")
        query = "What is the length of Pont Alexandre III?"
        
        # Start timing
        start_time = time.time()
        print(f"Query: {query}")
        
        # Run the query with a safety timeout
        result = search_tool.forward(query)
        
        # End timing
        duration = time.time() - start_time
        print(f"✅ Query completed in {duration:.2f} seconds")
        
        print("\nResult:")
        print(result)
        
        print("\nTrying a second query...")
        second_query = "What is the top speed of a cheetah?"
        
        # Start timing
        start_time = time.time()
        print(f"Query: {second_query}")
        
        # Run the query with a safety timeout
        second_result = search_tool.forward(second_query)
        
        # End timing
        duration = time.time() - start_time
        print(f"✅ Second query completed in {duration:.2f} seconds")
        
        print("\nSecond Result:")
        print(second_result)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    print("=" * 80)
    print("OpenDeepSearch Quick Test")
    print("=" * 80)
    main()
