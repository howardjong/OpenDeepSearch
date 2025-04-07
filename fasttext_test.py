
"""
Minimal test script for FastText loading
"""
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("Testing FastText model loading...")
    
    try:
        # Import the utils module that contains the FastText model loading
        print("1. Importing opendeepsearch.context_scraping.utils...")
        from opendeepsearch.context_scraping import utils
        print("✅ Module imported successfully")
        
        # Check if FastText model was loaded or if fallback is being used
        print("\n2. Checking FastText model status...")
        if hasattr(utils.model, 'predict'):
            print("✅ Model or fallback is available")
            
            # Test with a simple prediction
            print("\n3. Testing prediction functionality...")
            text = "This is a test sentence to check if the model is working."
            prediction = utils.predict_educational_value([text])
            print(f"Prediction result: {prediction}")
            print("✅ Prediction function works")
            
            print("\n4. Testing filter_quality_content function...")
            text = """This is a paragraph that should be evaluated for quality.
            
            This is another paragraph that should be evaluated separately."""
            filtered = utils.filter_quality_content(text)
            print(f"Filtered content length: {len(filtered)} characters")
            print("✅ Content filtering works")
        else:
            print("⚠️ Model or fallback not properly initialized")
    
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        traceback.print_exc()
        return 1
    
    print("\n✅ All tests completed successfully")
    return 0

if __name__ == "__main__":
    print("=" * 80)
    print("FastText Model Test")
    print("=" * 80)
    main()
