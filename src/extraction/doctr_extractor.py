"""
docTR extractor for document understanding.

Uses the docTR library for document understanding and information extraction.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from . import IExtractor

logger = logging.getLogger(__name__)


class DocTRExtractor(IExtractor):
    """docTR-based document understanding extractor."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.predictor = ocr_predictor(pretrained=True)
            self.predictor.to(self.device)
            self.logger.info(f"docTR model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load docTR model: {e}")
            raise
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using docTR model."""
        try:
            # Load document
            doc = DocumentFile.from_pdf(pdf_path)
            
            # Run OCR
            result = self.predictor(doc)
            
            # Extract text and structure
            pages = []
            for i, page in enumerate(result.pages):
                page_data = {
                    'page': i + 1,
                    'blocks': [],
                    'text': "",
                    'confidence': 0.0
                }
                
                # Extract blocks and text
                total_confidence = 0.0
                block_count = 0
                
                for block in page.blocks:
                    block_data = {
                        'geometry': block.geometry,
                        'lines': []
                    }
                    
                    for line in block.lines:
                        line_data = {
                            'geometry': line.geometry,
                            'words': []
                        }
                        
                        for word in line.words:
                            word_data = {
                                'text': word.value,
                                'confidence': word.confidence,
                                'geometry': word.geometry
                            }
                            line_data['words'].append(word_data)
                            total_confidence += word.confidence
                            block_count += 1
                        
                        block_data['lines'].append(line_data)
                    
                    page_data['blocks'].append(block_data)
                
                # Calculate average confidence
                if block_count > 0:
                    page_data['confidence'] = total_confidence / block_count
                
                # Extract full text
                page_data['text'] = result.pages[i].render()
                pages.append(page_data)
            
            return {
                'pages': pages,
                'full_text': '\n'.join([page['text'] for page in pages]),
                'metadata': {
                    'model': 'docTR',
                    'device': self.device,
                    'num_pages': len(pages)
                }
            }
            
        except Exception as e:
            self.logger.error(f"docTR extraction failed: {str(e)}")
            raise
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score for docTR extraction."""
        pages = extraction_result.get('pages', [])
        if not pages:
            return 0.0
        
        # Average confidence across pages
        total_confidence = sum(page.get('confidence', 0.0) for page in pages)
        return total_confidence / len(pages)
    
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        return "DocTRExtractor"
    
    def _extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial data from text."""
        import re
        
        financial_data = {
            'amounts': [],
            'percentages': [],
            'dates': [],
            'account_numbers': []
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
        
        return financial_data 