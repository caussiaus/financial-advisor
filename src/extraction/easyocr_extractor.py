"""
EasyOCR extractor for text recognition.

Uses EasyOCR for text recognition with support for multiple languages.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import easyocr
import pdf2image
from PIL import Image

from . import IExtractor

logger = logging.getLogger(__name__)


class EasyOCRExtractor(IExtractor):
    """EasyOCR-based text recognition extractor."""
    
    def __init__(self, languages: List[str] = ['en']):
        self.logger = logging.getLogger(__name__)
        self.languages = languages
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
            self.logger.info(f"EasyOCR model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load EasyOCR model: {e}")
            raise
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using EasyOCR."""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            results = []
            for i, image in enumerate(images):
                # Run OCR
                ocr_result = self.reader.readtext(image)
                
                # Process results
                text_blocks = []
                total_confidence = 0.0
                block_count = 0
                
                for bbox, text, confidence in ocr_result:
                    text_blocks.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
                    total_confidence += confidence
                    block_count += 1
                
                # Calculate average confidence
                avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
                
                # Extract full text
                full_text = ' '.join([block['text'] for block in text_blocks])
                
                results.append({
                    'page': i + 1,
                    'text': full_text,
                    'confidence': avg_confidence,
                    'text_blocks': text_blocks,
                    'financial_data': self._extract_financial_data(full_text)
                })
            
            return {
                'pages': results,
                'full_text': '\n'.join([r['text'] for r in results]),
                'metadata': {
                    'model': 'EasyOCR',
                    'device': self.device,
                    'languages': self.languages,
                    'num_pages': len(images)
                }
            }
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {str(e)}")
            raise
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score for EasyOCR extraction."""
        pages = extraction_result.get('pages', [])
        if not pages:
            return 0.0
        
        # Average confidence across pages
        total_confidence = sum(page.get('confidence', 0.0) for page in pages)
        return total_confidence / len(pages)
    
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        return f"EasyOCRExtractor({','.join(self.languages)})"
    
    def _extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial data from text."""
        import re
        
        financial_data = {
            'amounts': [],
            'percentages': [],
            'dates': [],
            'account_numbers': [],
            'financial_keywords': []
        }
        
        # Extract amounts
        amount_pattern = r'\$[\d,]+\.?\d*'
        amounts = re.findall(amount_pattern, text)
        financial_data['amounts'] = [amount.replace('$', '').replace(',', '') for amount in amounts]
        
        # Extract percentages
        percentage_pattern = r'\d+\.?\d*%'
        percentages = re.findall(percentage_pattern, text)
        financial_data['percentages'] = [pct.replace('%', '') for pct in percentages]
        
        # Extract dates
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        financial_data['dates'] = re.findall(date_pattern, text)
        
        # Extract account numbers
        account_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        financial_data['account_numbers'] = re.findall(account_pattern, text)
        
        # Find financial keywords
        financial_keywords = [
            'income', 'salary', 'wage', 'earnings', 'revenue',
            'expense', 'cost', 'payment', 'bill', 'debt',
            'asset', 'investment', 'portfolio', 'savings',
            'retirement', 'pension', 'insurance', 'tax',
            'budget', 'cash flow', 'net worth', 'liability'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        financial_data['financial_keywords'] = found_keywords
        
        return financial_data 