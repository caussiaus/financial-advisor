# MINTT v1 - Multiple INterpolation Trial Triangle

## Overview

MINTT v1 is a refactored financial analysis system that focuses on:
- **Multiple profile ingestion** from PDF documents
- **Feature selection** with dynamic unit detection
- **Congruence triangle matching** for mesh interpolation
- **Context-aware summarization** with number detection
- **Real-time service** for PDF processing and analysis

## Key Components

### Core MINTT System
- `src/mintt_core.py` - Core MINTT system with feature selection
- `src/mintt_interpolation.py` - Multiple profile interpolation
- `src/mintt_service.py` - Service for number detection and context analysis

### PDF Processing
- `src/enhanced_pdf_processor.py` - Advanced PDF processing with NLP
- `src/trial_people_manager.py` - Trial people management and interpolation

### Mesh Engine
- `src/mesh_congruence_engine.py` - Congruence triangle matching
- `src/mesh_vector_database.py` - Vector database for similarity matching
- `src/stochastic_mesh_engine.py` - Stochastic mesh generation

### Synthetic Data
- `src/synthetic_lifestyle_engine.py` - Synthetic lifestyle generation
- `src/json_to_vector_converter.py` - JSON to vector conversion
- `src/synthetic_data_generator.py` - Synthetic data generation

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the MINTT service:**
   ```bash
   python -c "from src.mintt_service import MINTTService; service = MINTTService()"
   ```

3. **Process PDFs with feature selection:**
   ```python
   from src.mintt_core import MINTTCore
   
   mintt = MINTTCore()
   result = mintt.process_pdf_with_feature_selection("path/to/document.pdf")
   ```

4. **Interpolate multiple profiles:**
   ```python
   from src.mintt_interpolation import MINTTInterpolation
   
   interpolation = MINTTInterpolation(mintt_core, trial_manager)
   result = interpolation.interpolate_profiles(target_id, source_ids)
   ```

## Architecture

### Feature Selection Pipeline
```
PDF Input → Feature Detection → Unit Normalization → Context Analysis → Feature Summary
```

### Interpolation Pipeline
```
Multiple Profiles → Congruence Triangle Matching → Feature Interpolation → Quality Assessment
```

### Service Pipeline
```
Service Request → Number Detection → Context Analysis → Summarization → Response
```

## Key Features

### 1. Advanced Feature Selection
- Automatic detection of financial amounts, dates, and categorical data
- Dynamic unit detection and conversion
- Confidence scoring for extracted features
- Context-aware feature analysis

### 2. Multiple Profile Interpolation
- Congruence triangle matching for similarity
- Multiple interpolation methods (linear, polynomial, spline, RBF)
- Quality assessment and confidence scoring
- Real-time interpolation network

### 3. Context-Aware Service
- Number detection with context analysis
- Unit conversion and normalization
- Summarization with backing analysis
- Real-time processing capabilities

### 4. Mesh Congruence Engine
- Delaunay triangulation for optimal mesh structure
- Centroidal Voronoi tessellations for density optimization
- Edge collapsing for mesh decimation
- Congruence scoring and validation

## Data Structure

### Trial People
```
data/inputs/trial_people/
├── person_1/
│   ├── PERSONAL_INFO.json
│   ├── LIFESTYLE_EVENTS.json
│   ├── FINANCIAL_PROFILE.json
│   └── GOALS.json
└── person_2/
    └── ...
```

### Outputs
```
data/outputs/
├── trial_analysis/          # Trial people analysis
├── mesh_congruence/         # Congruence analysis
├── vector_db/              # Vector database
├── visual_timelines/       # Visualizations
└── reports/                # Analysis reports
```

## Development

### Adding New Features
1. Extend `MINTTCore` for new feature types
2. Add interpolation methods to `MINTTInterpolation`
3. Implement service endpoints in `MINTTService`

### Testing
```bash
python -m pytest tests/
```

### Documentation
- Core modules: `CORE_MODULES_SUMMARY.md`
- Mesh database: `MESH_VECTOR_DATABASE_SUMMARY.md`
- Synthetic engine: `SYNTHETIC_LIFESTYLE_ENGINE_SUMMARY.md`

## License

MIT License - see LICENSE file for details.
