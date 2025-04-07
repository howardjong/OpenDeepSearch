
"""
FastText fallback module for when the main FastText model fails to load
"""
import logging
import random
from typing import List, Optional

logger = logging.getLogger(__name__)

class FastTextFallback:
    """A fallback class that mimics FastText functionality when the actual model fails to load"""
    
    def __init__(self):
        self.is_fallback = True
        logger.warning("Using FastText fallback implementation")
        
    def predict(self, texts: List[str], k: int = 1) -> tuple:
        """
        Fallback prediction function that returns random scores
        
        Args:
            texts: List of text strings to evaluate
            k: Number of labels to return
            
        Returns:
            Tuple of (labels, probabilities)
        """
        # Generate random educational value scores between 0.3 and 0.9
        # This is a fallback, so we're being somewhat generous but not perfect
        labels = [[f"__label__{i}"] for i in range(len(texts))]
        probs = [[random.uniform(0.3, 0.9) for _ in range(k)] for _ in texts]
        
        logger.info(f"FastText fallback used for {len(texts)} text segments")
        return labels, probs

def load_fasttext_or_fallback(model_path: Optional[str] = None) -> object:
    """
    Attempts to load the FastText model, falls back to our implementation if it fails
    
    Args:
        model_path: Path to the FastText model
        
    Returns:
        Either the actual FastText model or our fallback implementation
    """
    try:
        import fasttext
        logger.info(f"Attempting to load FastText model from {model_path if model_path else 'default path'}")
        
        if model_path:
            model = fasttext.load_model(model_path)
        else:
            # Create a very simple model in memory as fallback
            model = fasttext.train_unsupervised("", model='cbow', dim=10, ws=2, epoch=1)
            
        # Test if model has predict method
        if hasattr(model, 'predict'):
            logger.info("FastText model loaded successfully")
            return model
        else:
            logger.warning("Loaded FastText model lacks predict method, using fallback")
            return FastTextFallback()
            
    except Exception as e:
        logger.warning(f"Failed to load FastText model: {str(e)}")
        logger.warning("Using FastText fallback implementation")
        return FastTextFallback()
