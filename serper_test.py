
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_serper_limits():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("Error: SERPER_API_KEY environment variable not set")
        return
    
    # Test different limits
    for num_results in [8, 24, 50, 100]:
        print(f"\n--- Testing with num={num_results} ---")
        
        # Make direct API call to Serper
        url = "https://google.serper.dev/search"
        payload = {
            "q": "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
            "num": num_results
        }
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Count organic results
            organic_results = data.get("organic", [])
            print(f"Requested {num_results} results, got {len(organic_results)} organic results")
            
            # Save the full response to a file for inspection
            filename = f"serper_response_{num_results}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Full response saved to {filename}")
            
            # Print the first few and last few results to verify
            if organic_results:
                print(f"\nFirst 2 results:")
                for i, result in enumerate(organic_results[:2]):
                    print(f"{i+1}. {result.get('title')} - {result.get('link')}")
                
                if len(organic_results) > 3:
                    print(f"\nLast 2 results:")
                    for i, result in enumerate(organic_results[-2:]):
                        print(f"{len(organic_results)-1+i}. {result.get('title')} - {result.get('link')}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_serper_limits()
