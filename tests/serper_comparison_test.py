
import os
import json
import requests
from dotenv import load_dotenv
from src.opendeepsearch.serp_search.serp_search import SerperAPI

# Load environment variables
load_dotenv()

def test_serper_comparison():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("Error: SERPER_API_KEY environment variable not set")
        return
    
    # Test query
    query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
    num_results = 50  # Request 50 results
    
    print(f"\n=== Comparing Direct API vs Library Implementation (num={num_results}) ===\n")
    
    # === Method 1: Direct API call (like your example) ===
    print("Method 1: Direct API call")
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": num_results
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        direct_data = response.json()
        
        # Count organic results
        direct_organic_results = direct_data.get("organic", [])
        print(f"Direct API: Requested {num_results} results, got {len(direct_organic_results)} organic results")
        
        # Save the full response to a file for inspection
        with open("direct_api_response.json", "w") as f:
            json.dump(direct_data, f, indent=2)
        print("Full direct API response saved to direct_api_response.json")
    except Exception as e:
        print(f"Direct API Error: {str(e)}")
    
    # === Method 2: Using our library's SerperAPI implementation ===
    print("\nMethod 2: Library Implementation")
    try:
        serper_api = SerperAPI(api_key=api_key)
        result = serper_api.get_sources(query=query, num_results=num_results)
        
        if result.success:
            lib_data = result.data
            lib_organic_results = lib_data.get("organic", [])
            print(f"Library API: Requested {num_results} results, got {len(lib_organic_results)} organic results")
            
            # Save the full response to a file for inspection
            with open("library_api_response.json", "w") as f:
                json.dump(lib_data, f, indent=2)
            print("Full library API response saved to library_api_response.json")
            
            # Print comparison
            print(f"\nComparison: Direct API got {len(direct_organic_results)} results, Library got {len(lib_organic_results)} results")
            
            # Check if the results count matches
            if len(direct_organic_results) != len(lib_organic_results):
                print("WARNING: Result count mismatch! Library implementation is not getting the same number of results.")
                print("This suggests there may be filtering happening in the library implementation.")
        else:
            print(f"Library API Error: {result.error}")
    except Exception as e:
        print(f"Library API Error: {str(e)}")

if __name__ == "__main__":
    test_serper_comparison()
