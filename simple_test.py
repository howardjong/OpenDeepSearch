
from opendeepsearch import OpenDeepSearchTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print API keys status (safely)
print("API Keys Status:")
for key in ["SERPER_API_KEY", "OPENROUTER_API_KEY", "JINA_API_KEY"]:
    value = os.getenv(key)
    if value:
        print(f"- {key}: Present ({value[:3]}...{value[-3:]})")
    else:
        print(f"- {key}: Missing")

# Create a simple search tool
try:
    print("\nInitializing OpenDeepSearchTool...")
    search_tool = OpenDeepSearchTool(
        model_name="openrouter/google/gemini-2.0-flash-001",
        reranker="jina"
    )
    
    print("Setting up search tool...")
    if not search_tool.is_initialized:
        search_tool.setup()
    
    # Try a basic search
    print("\nTrying a simple search query...")
    query = "What is the length of Pont Alexandre III?"
    result = search_tool.forward(query)
    
    print("\nSearch Result:")
    print(result)
    
except Exception as e:
    print(f"\nError encountered: {str(e)}")
    import traceback
    traceback.print_exc()
