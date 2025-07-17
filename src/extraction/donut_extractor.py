"""
Donut extractor for document understanding.

Uses the Donut model for document understanding and information extraction.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from . import IExtractor

logger = logging.getLogger(__name__)


class DonutExtractor(IExtractor):
    """Donut-based document understanding extractor."""
    
    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.logger.info(f"Donut model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load Donut model: {e}")
            raise
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using Donut model."""
        try:
            from PIL import Image
            import pdf2image
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            results = []
            for i, image in enumerate(images):
                # Prepare image for model
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=512,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )
                
                # Decode generated text
                generated_text = self.processor.batch_decode(
                    generated_ids.sequences, skip_special_tokens=True
                )[0]
                
                results.append({
                    'page': i + 1,
                    'text': generated_text,
                    'confidence': self._calculate_confidence(generated_text)
                })
            
            return {
                'pages': results,
                'full_text': '\n'.join([r['text'] for r in results]),
                'metadata': {
                    'model': self.model_name,
                    'device': self.device,
                    'num_pages': len(images)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Donut extraction failed: {str(e)}")
            raise
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score for Donut extraction."""
        pages = extraction_result.get('pages', [])
        if not pages:
            return 0.0
        
        # Average confidence across pages
        total_confidence = sum(page.get('confidence', 0.0) for page in pages)
        return total_confidence / len(pages)
    
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        return f"DonutExtractor({self.model_name})"
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence based on text quality and length."""
        if not text.strip():
            return 0.0
        
        # Basic confidence based on text length and content
        confidence = min(len(text) / 1000.0, 0.8)  # Cap at 0.8
        
        # Check for financial keywords
        financial_keywords = [
            'income', 'salary', 'investment', 'asset', 'debt',
            'expense', 'budget', 'savings', 'retirement'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in text_lower)
        confidence += min(keyword_matches * 0.1, 0.2)
        
        return min(confidence, 1.0) 