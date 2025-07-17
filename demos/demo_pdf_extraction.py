"""
Demo script for PDF extraction capabilities.

This script demonstrates the hybrid PDF extraction pipeline with ML models
and confidence-based fallback to rule-based extraction.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from extraction import (
    HybridExtractionPipeline,
    DonutExtractor,
    LayoutLMExtractor,
    DocTRExtractor,
    EasyOCRExtractor,
    RuleBasedExtractor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_extraction_pipeline():
    """Set up the hybrid extraction pipeline."""
    try:
        # Initialize ML extractors
        ml_extractors = []
        
        # Try to initialize each ML extractor
        try:
            donut_extractor = DonutExtractor()
            ml_extractors.append(donut_extractor)
            logger.info("Donut extractor initialized successfully")
        except Exception as e:
            logger.warning(f"Donut extractor failed to initialize: {e}")
        
        try:
            layoutlm_extractor = LayoutLMExtractor()
            ml_extractors.append(layoutlm_extractor)
            logger.info("LayoutLM extractor initialized successfully")
        except Exception as e:
            logger.warning(f"LayoutLM extractor failed to initialize: {e}")
        
        try:
            doctr_extractor = DocTRExtractor()
            ml_extractors.append(doctr_extractor)
            logger.info("DocTR extractor initialized successfully")
        except Exception as e:
            logger.warning(f"DocTR extractor failed to initialize: {e}")
        
        try:
            easyocr_extractor = EasyOCRExtractor()
            ml_extractors.append(easyocr_extractor)
            logger.info("EasyOCR extractor initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR extractor failed to initialize: {e}")
        
        # Initialize rule-based extractor
        rule_based_extractor = RuleBasedExtractor()
        
        # Create hybrid pipeline
        pipeline = HybridExtractionPipeline(
            ml_extractors=ml_extractors,
            rule_based_extractor=rule_based_extractor,
            confidence_threshold=0.7
        )
        
        logger.info(f"Hybrid pipeline initialized with {len(ml_extractors)} ML extractors")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to set up extraction pipeline: {e}")
        raise


def extract_pdf(pipeline, pdf_path: str):
    """Extract data from PDF using the hybrid pipeline."""
    try:
        logger.info(f"Extracting data from: {pdf_path}")
        
        # Run extraction
        result = pipeline.extract(pdf_path)
        
        # Log results
        logger.info(f"Extraction completed:")
        logger.info(f"  Model used: {result.model_name}")
        logger.info(f"  Confidence: {result.confidence:.3f}")
        logger.info(f"  Extraction time: {result.extraction_time:.2f}s")
        logger.info(f"  Fallback used: {result.fallback_used}")
        
        # Log financial data summary
        if 'financial_data' in result.data:
            financial_data = result.data['financial_data']
            logger.info(f"  Financial data found:")
            logger.info(f"    Amounts: {len(financial_data.get('amounts', []))}")
            logger.info(f"    Percentages: {len(financial_data.get('percentages', []))}")
            logger.info(f"    Dates: {len(financial_data.get('dates', []))}")
            logger.info(f"    Keywords: {len(financial_data.get('financial_keywords', []))}")
        
        return result
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


def main():
    """Main demo function."""
    logger.info("Starting PDF extraction demo")
    
    # Set up pipeline
    pipeline = setup_extraction_pipeline()
    
    # Test with sample PDF
    sample_pdf = Path("data/inputs/uploads/Case_1_IPS_Individual.pdf")
    
    if not sample_pdf.exists():
        logger.warning(f"Sample PDF not found: {sample_pdf}")
        logger.info("Please place a PDF file in data/inputs/uploads/ to test extraction")
        return
    
    # Extract data
    result = extract_pdf(pipeline, str(sample_pdf))
    
    # Save results
    output_dir = Path("data/outputs/extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"extraction_result_{sample_pdf.stem}.json"
    with open(output_file, 'w') as f:
        import json
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_file}")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 