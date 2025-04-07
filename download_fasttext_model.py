
"""
Script to download FastText language identification model
"""
import os
import sys
import logging
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastText model URL
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_FILENAME = "lid.176.bin"

def download_file(url, filename):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download
        filename: Local filename to save to
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading {filename} ({total_size/1024/1024:.1f} MB)")
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        logger.info(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def main():
    """Main function to download FastText model"""
    try:
        # Check if model already exists
        if os.path.exists(MODEL_FILENAME):
            logger.info(f"Model file {MODEL_FILENAME} already exists. Skipping download.")
            return 0
        
        logger.info("Starting FastText model download...")
        success = download_file(FASTTEXT_MODEL_URL, MODEL_FILENAME)
        
        if success:
            logger.info("Model downloaded successfully")
            return 0
        else:
            logger.error("Failed to download model")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
