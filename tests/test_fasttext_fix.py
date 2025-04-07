
"""
Test script to verify FastText model loading with automatic download
"""
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("FastText Model Fix Test")
    print("=" * 80)
    
    # Check if model already exists
    model_path = "lid.176.bin"
    if os.path.exists(model_path):
        print(f"Model file {model_path} exists ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
    else:
        print(f"Model file {model_path} does not exist")
    
    # Import our module with the fix
    print("\nLoading FastText model using our improved module...")
    from opendeepsearch.context_scraping import utils
    
    # Test prediction function
    print("\nTesting prediction function...")
    sample_text = ["This is a test sentence in English.", 
                  "Это предложение на русском языке.", 
                  "Dies ist ein Testsatz auf Deutsch."]
    
    educational_values = utils.predict_educational_value(sample_text)
    print("\nPrediction results:")
    for i, (text, value) in enumerate(zip(sample_text, educational_values)):
        print(f"Text {i+1}: {value:.2f} - {text[:30]}...")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    main()
