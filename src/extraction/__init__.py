"""
PDF Extraction Module

This module provides ML-powered PDF extraction capabilities with multiple models
and confidence-based fallback to rule-based extraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IExtractor(ABC):
    """Interface for PDF extraction models."""
    
    @abstractmethod
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data from PDF and return structured output."""
        pass
    
    @abstractmethod
    def get_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction result."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the extraction model."""
        pass


class ExtractionResult:
    """Container for extraction results with metadata."""
    
    def __init__(self, 
                 data: Dict[str, Any],
                 confidence: float,
                 model_name: str,
                 extraction_time: float,
                 fallback_used: bool = False):
        self.data = data
        self.confidence = confidence
        self.model_name = model_name
        self.extraction_time = extraction_time
        self.fallback_used = fallback_used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'extraction_time': self.extraction_time,
            'fallback_used': self.fallback_used
        }


class HybridExtractionPipeline:
    """Hybrid pipeline combining ML and rule-based extraction."""
    
    def __init__(self, 
                 ml_extractors: List[IExtractor],
                 rule_based_extractor: 'RuleBasedExtractor',
                 confidence_threshold: float = 0.7):
        self.ml_extractors = ml_extractors
        self.rule_based_extractor = rule_based_extractor
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def extract(self, pdf_path: str) -> ExtractionResult:
        """Extract data using hybrid approach with confidence-based fallback."""
        import time
        start_time = time.time()
        
        # Try ML extractors first
        for extractor in self.ml_extractors:
            try:
                self.logger.info(f"Trying {extractor.get_model_name()} extraction")
                result = extractor.extract(pdf_path)
                confidence = extractor.get_confidence_score(result)
                
                if confidence >= self.confidence_threshold:
                    extraction_time = time.time() - start_time
                    return ExtractionResult(
                        data=result,
                        confidence=confidence,
                        model_name=extractor.get_model_name(),
                        extraction_time=extraction_time,
                        fallback_used=False
                    )
                else:
                    self.logger.warning(f"Low confidence ({confidence:.3f}) for {extractor.get_model_name()}")
                    
            except Exception as e:
                self.logger.error(f"Error with {extractor.get_model_name()}: {str(e)}")
                continue
        
        # Fallback to rule-based extraction
        self.logger.info("Falling back to rule-based extraction")
        try:
            result = self.rule_based_extractor.extract(pdf_path)
            confidence = self.rule_based_extractor.get_confidence_score(result)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                data=result,
                confidence=confidence,
                model_name=self.rule_based_extractor.get_model_name(),
                extraction_time=extraction_time,
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based extraction failed: {str(e)}")
            raise


# Import specific extractors
try:
    from .donut_extractor import DonutExtractor
    from .layoutlm_extractor import LayoutLMExtractor
    from .doctr_extractor import DocTRExtractor
    from .easyocr_extractor import EasyOCRExtractor
    from .rule_based_extractor import RuleBasedExtractor
except ImportError as e:
    logger.warning(f"Some extractors not available: {e}")


__all__ = [
    'IExtractor',
    'ExtractionResult', 
    'HybridExtractionPipeline',
    'DonutExtractor',
    'LayoutLMExtractor',
    'DocTRExtractor',
    'EasyOCRExtractor',
    'RuleBasedExtractor'
] 