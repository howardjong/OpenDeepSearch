
from opendeepsearch import OpenDeepSearchAgent
import os
from dotenv import load_dotenv
import time
import asyncio

# Load environment variables
load_dotenv()

def main():
    print("=" * 80)
    print("OpenDeepSearch Pro Mode Example")
    print("=" * 80)
    
    # Check API keys
    print("Checking API keys:")
    for key_name in ["SERPER_API_KEY", "OPENROUTER_API_KEY", "JINA_API_KEY"]:
        key_value = os.getenv(key_name)
        if key_value:
            # Print first 3 and last 3 characters of the key for verification
            print(f"- {key_name}: {key_value[:3]}***{key_value[-3:]}")
        else:
            print(f"- {key_name}: Missing!")
    
    # Initialize the OpenDeepSearchAgent
    print("\nInitializing OpenDeepSearchAgent with Jina reranker...")
    search_agent = OpenDeepSearchAgent(
        model="openrouter/google/gemini-2.0-flash-001",
        reranker="jina",  # Using Jina reranker for better results
        source_processor_config={
            "top_results": 5,  # Process more results for better context
            "filter_content": True
        }
    )
    
    # Run a query in default mode
    print("\nRunning a query in DEFAULT mode...")
    query = "What are the main challenges in quantum computing?"
    
    start_time = time.time()
    result_default = search_agent.ask_sync(query, max_sources=2, pro_mode=False)
    default_duration = time.time() - start_time
    
    print(f"\nDEFAULT MODE RESULT (took {default_duration:.2f} seconds):")
    print("-" * 40)
    print(result_default)
    
    # Run the same query in pro mode
    print("\nRunning the same query in PRO mode...")
    
    start_time = time.time()
    result_pro = search_agent.ask_sync(query, max_sources=3, pro_mode=True)
    pro_duration = time.time() - start_time
    
    print(f"\nPRO MODE RESULT (took {pro_duration:.2f} seconds):")
    print("-" * 40)
    print(result_pro)
    
    # Run a complex multi-hop query that benefits from Pro Mode
    print("\nRunning a complex multi-hop query in PRO mode...")
    complex_query = "What is the relationship between quantum entanglement and quantum teleportation, and how does it relate to quantum computing?"
    
    start_time = time.time()
    result_complex = search_agent.ask_sync(complex_query, max_sources=4, pro_mode=True)
    complex_duration = time.time() - start_time
    
    print(f"\nCOMPLEX QUERY RESULT (took {complex_duration:.2f} seconds):")
    print("-" * 40)
    print(result_complex)

if __name__ == "__main__":
    main()
