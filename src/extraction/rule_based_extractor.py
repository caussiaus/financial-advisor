"""
Rule-based PDF extractor as fallback for ML models.

This extractor uses traditional PDF processing libraries and pattern matching
to extract financial data when ML models have low confidence.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import pdfplumber
import PyPDF2
from pdfminer.high_level import extract_text

from . import IExtractor

logger = logging.getLogger(__name__)


class RuleBasedExtractor(IExtractor):
    """Rule-based PDF extractor using traditional libraries."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.financial_patterns = {
            'amount': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'account_number': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        self.financial_keywords = [
            'income', 'salary', 'wage', 'earnings', 'revenue',
            'expense', 'cost', 'payment', 'bill', 'debt',
            'asset', 'investment', 'portfolio', 'savings',
            'retirement', 'pension', 'insurance', 'tax',
            'budget', 'cash flow', 'net worth', 'liability'
        ]
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract financial data using rule-based approach."""
        try:
            # Extract text using multiple methods
            text_content = self._extract_text_content(pdf_path)
            
            # Parse structured data
            structured_data = self._parse_structured_data(text_content)
            
            # Extract financial information
            financial_data = self._extract_financial_data(text_content)
            
            # Combine results
            result = {
                'text_content': text_content,
                'structured_data': structured_data,
                'financial_data': financial_data,
                'metadata': {
                    'extraction_method': 'rule_based',
                    'file_path': str(pdf_path),
                    'text_length': len(text_content)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rule-based extraction failed: {str(e)}")
            raise
    
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction quality."""
        confidence = 0.5  # Base confidence for rule-based
        
        # Check if we found financial data
        financial_data = extraction_result.get('financial_data', {})
        if financial_data:
            confidence += 0.2
        
        # Check text quality
        text_content = extraction_result.get('text_content', '')
        if len(text_content) > 100:
            confidence += 0.1
        
        # Check for structured data
        structured_data = extraction_result.get('structured_data', {})
        if structured_data:
            confidence += 0.1
        
        # Check for financial patterns
        pattern_matches = 0
        for pattern in self.financial_patterns.values():
            if re.search(pattern, text_content):
                pattern_matches += 1
        
        if pattern_matches > 0:
            confidence += min(0.1 * pattern_matches, 0.2)
        
        return min(confidence, 1.0)
    
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        return "RuleBasedExtractor"
    
    def _extract_text_content(self, pdf_path: str) -> str:
        """Extract text content using multiple PDF libraries."""
        text_content = ""
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
        except Exception as e:
            self.logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        if not text_content.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() or ""
            except Exception as e:
                self.logger.warning(f"PyPDF2 failed: {e}")
        
        # Final fallback to pdfminer
        if not text_content.strip():
            try:
                text_content = extract_text(pdf_path)
            except Exception as e:
                self.logger.warning(f"pdfminer failed: {e}")
        
        return text_content
    
    def _parse_structured_data(self, text_content: str) -> Dict[str, Any]:
        """Parse structured data from text content."""
        structured_data = {
            'tables': [],
            'forms': [],
            'lists': []
        }
        
        # Simple table detection (lines with consistent separators)
        lines = text_content.split('\n')
        for line in lines:
            if line.count('\t') > 2 or line.count('  ') > 3:
                structured_data['tables'].append(line.strip())
        
        # Form field detection
        form_patterns = [
            r'([A-Za-z\s]+):\s*([^\n]+)',
            r'([A-Za-z\s]+)\s*=\s*([^\n]+)'
        ]
        
        for pattern in form_patterns:
            matches = re.findall(pattern, text_content)
            for field, value in matches:
                structured_data['forms'].append({
                    'field': field.strip(),
                    'value': value.strip()
                })
        
        return structured_data
    
    def _extract_financial_data(self, text_content: str) -> Dict[str, Any]:
        """Extract financial data using pattern matching."""
        financial_data = {
            'amounts': [],
            'percentages': [],
            'dates': [],
            'account_numbers': [],
            'financial_keywords': []
        }
        
        # Extract amounts
        amounts = re.findall(self.financial_patterns['amount'], text_content)
        financial_data['amounts'] = [amount.replace('$', '').replace(',', '') for amount in amounts]
        
        # Extract percentages
        percentages = re.findall(self.financial_patterns['percentage'], text_content)
        financial_data['percentages'] = [pct.replace('%', '') for pct in percentages]
        
        # Extract dates
        dates = re.findall(self.financial_patterns['date'], text_content)
        financial_data['dates'] = dates
        
        # Extract account numbers
        account_numbers = re.findall(self.financial_patterns['account_number'], text_content)
        financial_data['account_numbers'] = account_numbers
        
        # Find financial keywords
        text_lower = text_content.lower()
        found_keywords = []
        for keyword in self.financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        financial_data['financial_keywords'] = found_keywords
        
        return financial_data 