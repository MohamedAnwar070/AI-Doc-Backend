from .base import OCREngine
from PIL import Image
import pytesseract
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation"""
    
    def __init__(self, lang: str = 'eng', config: str = ''):
        super().__init__()
        self.lang = lang
        self.config = config or '--oem 3 --psm 6'
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize Tesseract engine"""
        if self._initialized:
            return
        
        try:
            # Test if tesseract is available
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            self._initialized = True
            logger.info("Tesseract initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise RuntimeError(
                "Tesseract not found. Please install: sudo apt-get install tesseract-ocr"
            )
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return f"[ERROR: {str(e)}]"
    
    def cleanup(self) -> None:
        """Cleanup Tesseract resources"""
        self._initialized = False
        logger.info("Tesseract cleaned up")
