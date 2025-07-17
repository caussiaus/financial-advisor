"""
LayoutLM extractor for document layout understanding.

Uses Microsoft's LayoutLM model for document understanding with layout information.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import pdf2image

from . import IExtractor

logger = logging.getLogger(__name__)


class LayoutLMExtractor(IExtractor):
    """LayoutLM-based document understanding extractor."""
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.logger.info(f"LayoutLM model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load LayoutLM model: {e}")
            raise
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using LayoutLM model."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            results = []
            for i, image in enumerate(images):
                # Process image with LayoutLM
                encoding = self.processor(
                    image, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                for key, value in encoding.items():
                    encoding[key] = value.to(self.device)
                
                # Get model output
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    confidence = torch.max(probabilities).item()
                
                # Extract text from image (simplified)
                text_content = self._extract_text_from_image(image)
                
                results.append({
                    'page': i + 1,
                    'text': text_content,
                    'confidence': confidence,
                    'layout_analysis': self._analyze_layout(image)
                })
            
            return {
                'pages': results,
                'full_text': '\n'.join([r['text'] for r in results]),
                'layout_analysis': [r['layout_analysis'] for r in results],
                'metadata': {
                    'model': self.model_name,
                    'device': self.device,
                    'num_pages': len(images)
                }
            }
            
        except Exception as e:
            self.logger.error(f"LayoutLM extraction failed: {str(e)}")
            raise
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score for LayoutLM extraction."""
        pages = extraction_result.get('pages', [])
        if not pages:
            return 0.0
        
        # Average confidence across pages
        total_confidence = sum(page.get('confidence', 0.0) for page in pages)
        return total_confidence / len(pages)
    
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        return f"LayoutLMExtractor({self.model_name})"
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return ""
    
    def _analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout structure."""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL image to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect lines and rectangles
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            
            # Detect rectangles (tables, forms)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = []
            for contour in contours:
                if len(contour) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 50 and h > 20:  # Filter small rectangles
                        rectangles.append({'x': x, 'y': y, 'width': w, 'height': h})
            
            return {
                'lines_detected': len(lines) if lines is not None else 0,
                'rectangles_detected': len(rectangles),
                'image_size': image.size,
                'layout_confidence': min(len(rectangles) / 10.0, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Layout analysis failed: {e}")
            return {
                'lines_detected': 0,
                'rectangles_detected': 0,
                'image_size': image.size,
                'layout_confidence': 0.0
            } 