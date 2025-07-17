# Financial Mesh Training System with Advanced PDF Extraction

## Overview

This system combines **advanced PDF extraction** with **financial mesh training** to create a comprehensive pipeline for:
- **ML-powered document understanding** with confidence scoring
- **Synthetic people generation** with realistic financial profiles
- **Financial shock simulation** and recovery analysis
- **Commutator route learning** for optimal financial strategies
- **Mesh-based portfolio comparison** and optimization

## Key Components

### PDF Extraction & Document Understanding
- `src/extraction/` - ML-based PDF extraction with multiple models
- `src/enhanced_pdf_processor.py` - Advanced PDF processing with NLP
- `src/layers/pdf_processor.py` - Document layout analysis and text extraction

### Financial Mesh Training System
- `src/training/mesh_training_engine.py` - Core training engine for synthetic scenarios
- `src/training/training_controller.py` - Controller for training system management
- `src/commutator_decision_engine.py` - Optimal financial strategy learning
- `src/core/stochastic_mesh_engine.py` - Stochastic mesh generation and evolution

### Core Mesh Engines
- `src/core/state_space_mesh_engine.py` - State-space mesh with full cash flow tracking
- `src/core/time_uncertainty_mesh.py` - Time uncertainty modeling
- `src/integration/mesh_engine_layer.py` - Unified mesh engine interface

### Analysis & Visualization
- `src/analysis/mesh_congruence_engine.py` - Congruence triangle matching
- `src/analysis/mesh_vector_database.py` - Vector database for similarity matching
- `src/visualization/mesh_3d_visualizer.py` - 3D mesh visualization
- `src/visualization/flexibility_comfort_mesh.py` - Financial comfort analysis

### Synthetic Data Generation
- `src/synthetic_lifestyle_engine.py` - Synthetic lifestyle generation
- `src/synthetic_data_generator.py` - Synthetic data generation
- `src/json_to_vector_converter.py` - JSON to vector conversion

## Quick Start

### 1. Environment Setup

**For Linux (GPU Support):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Create conda environment
conda create -n financial-mesh python=3.11
conda activate financial-mesh

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

**For Mac (CPU Only):**
```bash
# Install system dependencies
brew install tesseract tesseract-lang

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. PDF Extraction Setup

```bash
# Download pre-trained models
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base'); AutoModel.from_pretrained('microsoft/layoutlmv3-base')"

# Test PDF extraction
python demos/demo_pdf_extraction.py
```

### 3. Training System Setup

```bash
# Run training demo
python demos/demo_training_engine.py

# Run basic training
python run_training.py 50

# Check system status
python run_training.py --status
```

### 4. Portfolio Comparison

```bash
# Run portfolio comparison demo
python demos/demo_portfolio_comparison.py

# Start 3D visualizer
python start_3d_visualizer.py
```

## Architecture

### PDF Extraction Pipeline
```
PDF Input → Layout Analysis → Text Extraction → ML Classification → Confidence Scoring → JSON Output
```

### Training Pipeline
```
Synthetic People → Financial Shocks → Mesh Simulation → Commutator Learning → Route Optimization
```

### Portfolio Comparison Pipeline
```
Two Portfolios → Mesh Evolution → Path Analysis → Commutator Comparison → Strategy Optimization
```

## Key Features

### 1. Advanced PDF Extraction
- **Multiple ML Models**: Donut, LayoutLMv3, docTR, EasyOCR
- **Hybrid Approach**: ML + Rule-based extraction with confidence thresholds
- **Layout Analysis**: Detect tables, forms, and complex document structures
- **Confidence Scoring**: Automatic fallback to rule-based extraction for low-confidence cases
- **GPU Acceleration**: Optimized for NVIDIA RTX 4090 with mixed precision

### 2. Financial Mesh Training
- **Synthetic People Generation**: Realistic financial profiles with age-based distributions
- **Financial Shock Simulation**: Market crashes, job loss, medical emergencies, etc.
- **Commutator Route Learning**: Optimal sequences of financial moves for recovery
- **Edge Path Tracking**: Complete tracking of financial state transitions
- **Success Metrics**: Recovery rate, time, and strategy effectiveness

### 3. Portfolio Comparison
- **Side-by-Side Analysis**: Compare two portfolios under identical conditions
- **Mesh Evolution Visualization**: 3D visualization of financial state evolution
- **Commutator Strategy Comparison**: Compare recovery strategies between portfolios
- **Performance Metrics**: Risk-adjusted returns, drawdown analysis, path dependency

### 4. Mesh Congruence Engine
- **Delaunay Triangulation**: Optimal mesh structure for financial state space
- **Centroidal Voronoi Tessellations**: Density optimization for mesh nodes
- **Edge Collapsing**: Mesh decimation for performance optimization
- **Congruence Scoring**: Validation of mesh structure quality

## Data Structure

### PDF Extraction Output
```
data/outputs/extraction/
├── raw_extractions/          # Raw ML model outputs
├── processed_extractions/    # Post-processed JSON data
├── confidence_scores/        # Confidence metrics
└── fallback_logs/           # Rule-based fallback logs
```

### Training Data
```
data/outputs/training/
├── successful_routes.json    # Successful commutator routes
├── training_history.json     # Training session history
├── training_analysis.png     # Visualization of results
└── model_checkpoints/        # Saved model states
```

### Portfolio Comparison
```
data/outputs/comparison/
├── portfolio_pairs/          # Paired portfolio analysis
├── mesh_evolution/           # Mesh state evolution data
├── commutator_comparison/    # Strategy comparison results
└── visualizations/           # Comparison plots and charts
```

## Development

### Adding New PDF Extraction Models
1. Implement `IExtractor` interface in `src/extraction/`
2. Add confidence scoring and fallback logic
3. Update the hybrid pipeline in `src/enhanced_pdf_processor.py`

### Extending Training System
1. Add new shock types in `src/training/mesh_training_engine.py`
2. Implement new commutator operations in `src/commutator_decision_engine.py`
3. Create new visualization components in `src/visualization/`

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_extraction.py
python -m pytest tests/test_training.py
python -m pytest tests/test_mesh.py
```

### Documentation
- PDF Extraction: `docs/PDF_EXTRACTION_README.md`
- Training System: `docs/TRAINING_ENGINE_README.md`
- Mesh Architecture: `docs/MESH_SYSTEM_SUMMARY.md`

## Performance Optimization

### GPU Acceleration
- **CUDA Support**: Automatic detection and utilization of NVIDIA GPUs
- **Mixed Precision**: FP16 training for faster convergence
- **Batch Processing**: Optimized batch sizes for memory efficiency

### Memory Management
- **Mesh Memory Manager**: Efficient storage and retrieval of mesh nodes
- **Compressed Representations**: Lossless compression of financial state data
- **Lazy Loading**: On-demand loading of mesh components

### Cross-Platform Compatibility
- **Linux (GPU)**: Full CUDA acceleration with all ML models
- **Mac (CPU)**: CPU-only fallback with optimized performance
- **Docker Support**: Containerized deployment for consistent environments

## License

MIT License - see LICENSE file for details.
