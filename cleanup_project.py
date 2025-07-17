#!/usr/bin/env python3
"""
Project Cleanup Script for MINTT v1

This script cleans up the project by:
1. Removing unused files and directories
2. Organizing the project structure
3. Keeping only essential components
4. Creating a clean MINTT v1 branch
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Set

# Files to keep (essential for MINTT v1)
ESSENTIAL_FILES = {
    # Core MINTT components
    'src/mintt_core.py',
    'src/mintt_interpolation.py', 
    'src/mintt_service.py',
    'src/enhanced_pdf_processor.py',
    'src/trial_people_manager.py',
    'src/mesh_congruence_engine.py',
    'src/mesh_vector_database.py',
    'src/synthetic_lifestyle_engine.py',
    'src/json_to_vector_converter.py',
    
    # Supporting components
    'src/accounting_reconciliation.py',
    'src/financial_recommendation_engine.py',
    'src/stochastic_mesh_engine.py',
    'src/adaptive_mesh_generator.py',
    'src/time_uncertainty_mesh.py',
    'src/real_people_integrator.py',
    'src/dl_friendly_storage.py',
    'src/enhanced_category_calculator_fixed.py',
    'src/commutator_decision_engine.py',
    'src/market_tracking_backtest.py',
    'src/mesh_backtesting_framework.py',
    'src/mesh_market_integration.py',
    'src/mesh_memory_manager.py',
    'src/omega_mesh_integration.py',
    'src/synthetic_data_generator.py',
    'src/unified_cash_flow_model.py',
    'src/time_uncertainty_integration.py',
    'src/vectorized_accounting.py',
    'src/accounting_debugger.py',
    'src/comprehensive_mesh_testing.py',
    
    # Demo files
    'demo_trial_people_manager.py',
    'demo_mesh_congruence_system.py',
    'demo_real_people_integration.py',
    'demo_comprehensive_synthetic_engine.py',
    'demo_json_to_vector_converter.py',
    'demo_mesh_vector_database.py',
    'demo_refactored_architecture.py',
    'demo_enhanced_category_calculator.py',
    'demo_omega_mesh.py',
    'demo_time_uncertainty_full_system.py',
    'demo_unified_cash_flow_system.py',
    'demo_financial_space_mapping.py',
    'demo_commutator_algorithms.py',
    
    # Test files
    'test_optimized_mesh.py',
    'test_pdf_processing.py',
    'test_enhanced_nlp.py',
    'test_market_tracking.py',
    'test_pdf_pipeline.py',
    'test_refined_feature_extraction.py',
    'test_cuda_acceleration.py',
    
    # Configuration files
    'requirements.txt',
    'setup.py',
    'README.md',
    'LICENSE',
    'deploy.sh',
    
    # Documentation
    'CORE_MODULES_SUMMARY.md',
    'MESH_VECTOR_DATABASE_SUMMARY.md',
    'SYNTHETIC_LIFESTYLE_ENGINE_SUMMARY.md',
    'REFACTORED_ARCHITECTURE_README.md',
    'DL_FRIENDLY_STORAGE_IMPLEMENTATION.md',
    'MARKET_TRACKING_ANALYSIS.md',
    'MODULE_STRUCTURE.md',
    'OMEGA_MESH_EVALUATION_SUMMARY.md',
    'SYSTEM_SUMMARY.md',
    'SYSTEM_DELIVERY_SUMMARY.md',
    'ENHANCED_DASHBOARD_README.md',
    
    # Web interface
    'enhanced_mesh_dashboard.py',
    'start_enhanced_dashboard.py',
    'web_ui.py',
    'omega_web_app.py',
    'mesh_congruence_app.py',
    'flexibility_comfort_dashboard.py',
    'mesh_market_dashboard.py',
    'system_control.py',
    'stress_test_script.py',
    
    # Templates
    'templates/index.html',
    'templates/dashboard.html',
    'templates/enhanced_dashboard.html',
    
    # Data directories (keep structure)
    'data/inputs/',
    'data/outputs/',
    
    # Evaluation results (keep recent)
    'evaluation_results_20250717_012831/',
    
    # Essential data files
    'comprehensive_mesh_evaluation.py',
    'simple_mesh_evaluation.py',
    'convert_to_dl_friendly.py',
    'upload_horatio_demo.py',
    'horatio_profile.json'
}

# Directories to keep
ESSENTIAL_DIRECTORIES = {
    'src/',
    'src/layers/',
    'data/',
    'data/inputs/',
    'data/outputs/',
    'data/outputs/analysis_data/',
    'data/outputs/ips_output/',
    'data/outputs/trial_analysis/',
    'data/outputs/vector_db/',
    'data/outputs/visual_timelines/',
    'data/outputs/visuals/',
    'data/outputs/mesh_congruence/',
    'data/outputs/reports/',
    'data/outputs/backtesting/',
    'data/outputs/client_data/',
    'data/outputs/comprehensive_testing/',
    'data/inputs/trial_people/',
    'data/inputs/historical_backtests/',
    'data/inputs/uploads/',
    'templates/',
    'tests/',
    'evaluation_results_20250717_012831/',
    'omega_mesh_export/',
    'ips_output/',
    'venv/'
}

# Files to remove (unused or outdated)
FILES_TO_REMOVE = {
    # Large data files
    'comprehensive_integrated_mesh.json',
    'comprehensive_mesh_data.json',
    'time_uncertainty_mesh_data.json',
    'unified_cash_flow_data.json',
    'comprehensive_cash_flow_report.json',
    'comprehensive_mesh_analysis.png',
    'comprehensive_integrated_mesh.json',
    
    # Old evaluation results
    'evaluation_results_20250716_175836/',
    'simple_evaluation_results_20250716_175842.json',
    'simple_evaluation_results_20250716_212514.json',
    
    # Old exports
    'omega_mesh_dashboard.html',
    'evaluation_subjects_analysis.html',
    'evaluation_performance_dashboard.html',
    'evaluation_configuration_matrix.html',
    'financial_space_map.html',
    'omega_mesh_dashboard.html',
    
    # Old analysis files
    'enhanced_chunked_analysis_CLIENT_Case_#1_IPS__Individual_pdf.json',
    'enhanced_chunked_analysis_CLIENT_sample_client_update_txt.json',
    'omega_engine_analysis.json',
    'mesh_market_test_results.json',
    'dashboard_export_20250717_055219.json',
    
    # Log files
    'enhanced_dashboard.log',
    'dashboard_startup.log',
    'web_app.log',
    'web_app_debug.log',
    'nlp_test_output.txt',
    
    # Temporary files
    'temp_repo/',
    '__pycache__/',
    '.pytest_cache/',
    '.DS_Store',
    
    # Old demo files
    'demo_enhanced_category_calculator.py',
    'demo_omega_mesh.py',
    
    # Old test files
    'test_optimized_mesh.py',
    'test_pdf_processing.py',
    'test_enhanced_nlp.py',
    
    # Old evaluation files
    'simple_mesh_evaluation.py',
    
    # Old data files
    'comprehensive_cash_flow_report.json',
    'comprehensive_mesh_analysis.png',
    'comprehensive_mesh_data.json',
    'comprehensive_integrated_mesh.json',
    'time_uncertainty_mesh_data.json',
    'unified_cash_flow_data.json'
}

# Directories to remove
DIRECTORIES_TO_REMOVE = {
    'temp_repo/',
    '__pycache__/',
    '.pytest_cache/',
    'evaluation_results_20250716_175836/',
    'omega_mesh_engine.egg-info/'
}

def cleanup_project():
    """Clean up the project by removing unused files and organizing structure"""
    print("🧹 Starting project cleanup for MINTT v1...")
    
    # Get current directory
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Track what we're removing
    removed_files = []
    removed_dirs = []
    kept_files = []
    
    # Remove files to remove
    print("\n🗑️ Removing unused files...")
    for file_pattern in FILES_TO_REMOVE:
        if '*' in file_pattern:
            # Handle glob patterns
            for file_path in project_root.glob(file_pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path))
                        print(f"   Removed: {file_path}")
                    except Exception as e:
                        print(f"   Error removing {file_path}: {e}")
        else:
            file_path = project_root / file_pattern
            if file_path.exists() and file_path.is_file():
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    print(f"   Removed: {file_path}")
                except Exception as e:
                    print(f"   Error removing {file_path}: {e}")
    
    # Remove directories to remove
    print("\n🗑️ Removing unused directories...")
    for dir_pattern in DIRECTORIES_TO_REMOVE:
        if '*' in dir_pattern:
            # Handle glob patterns
            for dir_path in project_root.glob(dir_pattern):
                if dir_path.is_dir():
                    try:
                        shutil.rmtree(dir_path)
                        removed_dirs.append(str(dir_path))
                        print(f"   Removed: {dir_path}")
                    except Exception as e:
                        print(f"   Error removing {dir_path}: {e}")
        else:
            dir_path = project_root / dir_pattern
            if dir_path.exists() and dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    removed_dirs.append(str(dir_path))
                    print(f"   Removed: {dir_path}")
                except Exception as e:
                    print(f"   Error removing {dir_path}: {e}")
    
    # Remove __pycache__ directories recursively
    print("\n🗑️ Removing __pycache__ directories...")
    for pycache_dir in project_root.rglob('__pycache__'):
        try:
            shutil.rmtree(pycache_dir)
            removed_dirs.append(str(pycache_dir))
            print(f"   Removed: {pycache_dir}")
        except Exception as e:
            print(f"   Error removing {pycache_dir}: {e}")
    
    # Remove .pyc files
    print("\n🗑️ Removing .pyc files...")
    for pyc_file in project_root.rglob('*.pyc'):
        try:
            pyc_file.unlink()
            removed_files.append(str(pyc_file))
            print(f"   Removed: {pyc_file}")
        except Exception as e:
            print(f"   Error removing {pyc_file}: {e}")
    
    # Create essential directories if they don't exist
    print("\n📁 Creating essential directories...")
    for dir_path in ESSENTIAL_DIRECTORIES:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {full_path}")
    
    # Create new README for MINTT v1
    create_mintt_readme()
    
    # Create summary of cleanup
    print("\n📊 Cleanup Summary:")
    print(f"   Files removed: {len(removed_files)}")
    print(f"   Directories removed: {len(removed_dirs)}")
    print(f"   Essential files kept: {len(ESSENTIAL_FILES)}")
    print(f"   Essential directories kept: {len(ESSENTIAL_DIRECTORIES)}")
    
    # Save cleanup report
    cleanup_report = {
        'timestamp': str(Path.cwd()),
        'removed_files': removed_files,
        'removed_directories': removed_dirs,
        'essential_files': list(ESSENTIAL_FILES),
        'essential_directories': list(ESSENTIAL_DIRECTORIES)
    }
    
    with open('cleanup_report.json', 'w') as f:
        json.dump(cleanup_report, f, indent=2)
    
    print(f"\n✅ Cleanup complete! Report saved to cleanup_report.json")
    print("🚀 Project is now ready for MINTT v1 development!")

def create_mintt_readme():
    """Create a new README for MINTT v1"""
    readme_content = """# MINTT v1 - Multiple INterpolation Trial Triangle

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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("   Created: README.md (MINTT v1)")

if __name__ == "__main__":
    cleanup_project() 