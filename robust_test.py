
"""
Robust test script for OpenDeepSearch with detailed error handling and timeouts
"""
import os
import sys
import traceback
import logging
import signal
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout=60, *args, **kwargs):
    """Run a function with a timeout"""
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        print(f"❌ Operation timed out after {timeout} seconds")
        raise
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled

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
        print("\n1. Importing OpenDeepSearchTool...")
        from opendeepsearch import OpenDeepSearchTool
        logger.info("Successfully imported OpenDeepSearchTool")
        
        # Initialize OpenDeepSearchTool
        print("\n2. Initializing OpenDeepSearchTool with Gemini Flash model...")
        logger.info("Initializing OpenDeepSearchTool with Gemini Flash model")
        search_tool = OpenDeepSearchTool(
            model_name="openrouter/google/gemini-2.0-flash-001",
            reranker="jina"  # Using Jina with our improved error handling
        )
        
        print("\n3. Setting up search tool...")
        logger.info("Setting up search tool")
        if not hasattr(search_tool, 'is_initialized') or not search_tool.is_initialized:
            search_tool.setup()
        print("✅ Setup complete")
        
        # Run a test query with timeout
        print("\n4. Running first test query with 60-second timeout...")
        query = "What is the length of Pont Alexandre III?"
        print(f"\nQuery: {query}")
        
        def run_query(q):
            logger.info(f"Executing query: {q}")
            start_time = time.time()
            result = search_tool.forward(q)
            duration = time.time() - start_time
            logger.info(f"Query completed in {duration:.2f} seconds")
            return result
        
        try:
            result = run_with_timeout(run_query, 60, query)
            print("\nResults:")
            print(result)
            print("\n✅ First query successful")
        except Exception as e:
            logger.error(f"First query failed: {str(e)}")
            print(f"\n❌ First query failed: {str(e)}")
            traceback.print_exc()
            return 1
        
        # Run a second query
        print("\n5. Running second test query...")
        second_query = "What is the top speed of a cheetah?"
        logger.info(f"Running second query: {second_query}")
        print(f"\nQuery: {second_query}")
        
        try:
            second_result = run_with_timeout(run_query, 60, second_query)
            print("\nResults:")
            print(second_result)
            print("\n✅ Second query successful")
        except Exception as e:
            logger.error(f"Second query failed: {str(e)}")
            print(f"\n❌ Second query failed: {str(e)}")
            traceback.print_exc()
            return 1
        
        # Calculate time based on results
        print("\nWith these results, we can calculate:")
        print("Time = Distance/Speed")
        
        # Extract numeric values using simple parsing (this is just for demonstration)
        try:
            # Extract length (in meters) from first result
            length_text = result.lower()
            length = None
            for line in length_text.split("."):
                if "meter" in line or "metres" in line or " m " in line:
                    words = line.split()
                    for i, word in enumerate(words):
                        if word.replace(".", "").isdigit():
                            length = float(word)
                            break
            
            # Extract speed (in m/s) from second result
            speed_text = second_result.lower()
            speed_mph = None
            for line in speed_text.split("."):
                if "mph" in line or "miles per hour" in line:
                    words = line.split()
                    for i, word in enumerate(words):
                        if word.replace(".", "").isdigit():
                            speed_mph = float(word)
                            break
            
            # Convert mph to m/s if needed
            if length and speed_mph:
                speed_ms = speed_mph * 0.44704  # Convert mph to m/s
                time_seconds = length / speed_ms
                print(f"Bridge length: approximately {length} meters")
                print(f"Cheetah speed: approximately {speed_mph} mph ({speed_ms:.2f} m/s)")
                print(f"Time to cross bridge: {time_seconds:.2f} seconds")
            else:
                print("Couldn't extract precise numbers for calculation")
                
        except Exception as e:
            print(f"Error during calculation: {str(e)}")
        
        logger.info("Test completed successfully")
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    print("=" * 80)
    print("OpenDeepSearch Robust Test Script")
    print("=" * 80)
    sys.exit(main())
