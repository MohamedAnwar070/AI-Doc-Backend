from abc import ABC, abstractmethod
from PIL import Image
from typing import Optional
import numpy as np


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self):
        """Initialize OCR engine"""
        self.engine_name = self.__class__.__name__
    
    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the OCR engine with necessary configurations"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Optional preprocessing step
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def get_engine_info(self) -> dict:
        """Get engine information"""
        return {
            "name": self.engine_name,
            "initialized": hasattr(self, '_initialized') and self._initialized
        }
