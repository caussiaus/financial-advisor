# PDF Extraction System Documentation

## Overview

The PDF extraction system provides a hybrid approach combining multiple ML models with rule-based fallback for robust document understanding and financial data extraction.

## Architecture

### Hybrid Pipeline
```
PDF Input → ML Models (Donut, LayoutLM, DocTR, EasyOCR) → Confidence Scoring → Rule-based Fallback → JSON Output
```

### Components

#### 1. ML Extractors
- **DonutExtractor**: Document understanding with visual-text alignment
- **LayoutLMExtractor**: Layout-aware document understanding
- **DocTRExtractor**: Document text recognition with structure
- **EasyOCRExtractor**: Multi-language OCR with financial keyword detection

#### 2. Rule-based Extractor
- **RuleBasedExtractor**: Traditional PDF processing with pattern matching
- Fallback for low-confidence ML results
- Financial data extraction using regex patterns

#### 3. Hybrid Pipeline
- **HybridExtractionPipeline**: Orchestrates ML and rule-based extraction
- Confidence-based model selection
- Automatic fallback to rule-based extraction

## Installation

### System Dependencies

**Linux (GPU Support):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Mac (CPU Only):**
```bash
# Install system dependencies
brew install tesseract tesseract-lang

# Install PyTorch CPU version
pip install torch torchvision torchaudio
```

### Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Download pre-trained models
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base'); AutoModel.from_pretrained('microsoft/layoutlmv3-base')"
```

## Usage

### Basic Usage

```python
from src.extraction import HybridExtractionPipeline, RuleBasedExtractor

# Initialize pipeline
pipeline = HybridExtractionPipeline(
    ml_extractors=[],  # Will be auto-initialized
    rule_based_extractor=RuleBasedExtractor(),
    confidence_threshold=0.7
)

# Extract data
result = pipeline.extract("path/to/document.pdf")

# Access results
print(f"Model used: {result.model_name}")
print(f"Confidence: {result.confidence}")
print(f"Data: {result.data}")
```

### Advanced Usage

```python
from src.extraction import (
    DonutExtractor,
    LayoutLMExtractor,
    DocTRExtractor,
    EasyOCRExtractor,
    RuleBasedExtractor,
    HybridExtractionPipeline
)

# Initialize specific extractors
ml_extractors = [
    DonutExtractor(),
    LayoutLMExtractor(),
    DocTRExtractor(),
    EasyOCRExtractor()
]

rule_extractor = RuleBasedExtractor()

# Create custom pipeline
pipeline = HybridExtractionPipeline(
    ml_extractors=ml_extractors,
    rule_based_extractor=rule_extractor,
    confidence_threshold=0.8  # Higher threshold
)

# Extract with custom settings
result = pipeline.extract("document.pdf")
```

## Model Details

### DonutExtractor
- **Model**: `naver-clova-ix/donut-base-finetuned-docvqa`
- **Capabilities**: Document understanding with visual-text alignment
- **Best for**: Complex documents with mixed content
- **GPU**: Recommended for optimal performance

### LayoutLMExtractor
- **Model**: `microsoft/layoutlmv3-base`
- **Capabilities**: Layout-aware document understanding
- **Best for**: Forms, tables, structured documents
- **GPU**: Recommended for optimal performance

### DocTRExtractor
- **Model**: Pre-trained docTR OCR model
- **Capabilities**: Text recognition with document structure
- **Best for**: High-quality text extraction
- **GPU**: Recommended for optimal performance

### EasyOCRExtractor
- **Model**: EasyOCR with multiple language support
- **Capabilities**: Multi-language OCR with financial keyword detection
- **Best for**: Multi-language documents, financial forms
- **GPU**: Optional, works on CPU

### RuleBasedExtractor
- **Libraries**: pdfplumber, PyPDF2, pdfminer
- **Capabilities**: Pattern-based financial data extraction
- **Best for**: Fallback when ML models fail
- **GPU**: Not required

## Financial Data Extraction

### Supported Patterns
- **Amounts**: `$1,234.56`, `$500`, `$1,000,000`
- **Percentages**: `5.5%`, `10%`, `25.75%`
- **Dates**: `01/15/2024`, `12-31-2023`, `2024-01-15`
- **Account Numbers**: `1234-5678-9012-3456`
- **SSN**: `123-45-6789`
- **Phone Numbers**: `555-123-4567`

### Financial Keywords
- Income: salary, wage, earnings, revenue
- Expenses: cost, payment, bill, debt
- Assets: investment, portfolio, savings
- Planning: retirement, pension, insurance, tax

## Confidence Scoring

### ML Model Confidence
- **Donut**: Based on text quality and financial keyword presence
- **LayoutLM**: Model output probabilities
- **DocTR**: Word-level confidence scores
- **EasyOCR**: Character-level confidence scores

### Rule-based Confidence
- **Base**: 0.5 for rule-based extraction
- **Financial Data**: +0.2 if financial patterns found
- **Text Quality**: +0.1 if text length > 100 characters
- **Structured Data**: +0.1 if tables/forms detected
- **Pattern Matches**: +0.1 per financial pattern found

## Performance Optimization

### GPU Acceleration
```python
# Check GPU availability
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Batch Processing
```python
# Process multiple documents
results = []
for pdf_path in pdf_files:
    result = pipeline.extract(pdf_path)
    results.append(result)
```

### Memory Management
```python
# Clear GPU memory after processing
import torch
torch.cuda.empty_cache()
```

## Error Handling

### Model Loading Errors
```python
try:
    extractor = DonutExtractor()
except Exception as e:
    logger.warning(f"Donut extractor failed: {e}")
    # Continue with other extractors
```

### Extraction Errors
```python
try:
    result = pipeline.extract(pdf_path)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    # Handle error appropriately
```

## Output Format

### ExtractionResult
```python
{
    "data": {
        "text_content": "Extracted text...",
        "financial_data": {
            "amounts": ["1234.56", "500.00"],
            "percentages": ["5.5", "10.0"],
            "dates": ["01/15/2024"],
            "financial_keywords": ["income", "investment"]
        },
        "structured_data": {
            "tables": [...],
            "forms": [...]
        }
    },
    "confidence": 0.85,
    "model_name": "DonutExtractor",
    "extraction_time": 2.34,
    "fallback_used": false
}
```

## Testing

### Unit Tests
```bash
python -m pytest tests/test_extraction.py
```

### Integration Tests
```bash
python demos/demo_pdf_extraction.py
```

### Performance Tests
```bash
python tests/test_extraction_performance.py
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU
   torch.cuda.empty_cache()
   ```

2. **Model Download Failures**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   ```

3. **OCR Quality Issues**
   ```python
   # Adjust EasyOCR parameters
   reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')
   ```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Custom Model Training**: Fine-tune models on financial documents
- **Multi-page Analysis**: Cross-page information extraction
- **Table Detection**: Advanced table structure recognition
- **Form Field Mapping**: Automatic form field identification
- **Confidence Calibration**: Improved confidence scoring

### Performance Improvements
- **Model Quantization**: Reduced memory usage
- **Parallel Processing**: Multi-GPU support
- **Caching**: Model and result caching
- **Streaming**: Large document processing

## License

MIT License - see LICENSE file for details. 