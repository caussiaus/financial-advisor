# 🚀 Quick Start Guide

## Fresh Clone Setup

This guide will help you set up the Financial Mesh System from a fresh clone.

### Prerequisites

1. **Conda/Miniconda**: Install from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. **Git**: For cloning the repository
3. **Linux (Recommended)**: For GPU acceleration
4. **NVIDIA GPU (Optional)**: For optimal performance

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/caussiaus/financial-advisor.git
cd financial-advisor

# Switch to the PDF extraction branch
git checkout pdf-extraction-enhancement
```

### Step 2: Run the Setup Script

```bash
# Make the setup script executable (if not already)
chmod +x setup.sh

# Run the automated setup
./setup.sh
```

The setup script will:
- ✅ Check system requirements
- ✅ Install system dependencies
- ✅ Create conda environment with all dependencies
- ✅ Download pre-trained ML models
- ✅ Create necessary directories
- ✅ Test the installation
- ✅ Create activation script

### Step 3: Activate the Environment

```bash
# Activate the environment
source activate_env.sh
```

### Step 4: Test the System

```bash
# Test PDF extraction
python demos/demo_pdf_extraction.py

# Test training system
python demos/demo_training_engine.py

# Start 3D visualizer
python start_3d_visualizer.py
```

## Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install System Dependencies

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 2. Create Conda Environment

```bash
# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate financial-mesh
```

### 3. Download Models

```bash
# Download pre-trained models
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base')
AutoModel.from_pretrained('microsoft/layoutlmv3-base')
AutoTokenizer.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
AutoModel.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Create Directories

```bash
mkdir -p data/outputs/{extraction,training,comparison,visualizations}
mkdir -p logs models
```

## Environment Information

- **Environment Name**: `financial-mesh`
- **Python Version**: 3.11
- **GPU Support**: NVIDIA CUDA 12.1
- **Key Features**: PDF extraction, ML training, 3D visualization

## Troubleshooting

### Common Issues

1. **Conda not found**
   ```bash
   # Install Miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **CUDA out of memory**
   ```bash
   # Clear GPU memory
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Model download failures**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   ```

4. **Permission denied**
   ```bash
   # Fix permissions
   chmod +x setup.sh
   chmod +x activate_env.sh
   ```

### GPU Verification

```bash
# Check GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## Quick Commands

```bash
# Activate environment
source activate_env.sh

# Test PDF extraction
python demos/demo_pdf_extraction.py

# Run training
python run_training.py 50

# Start 3D visualizer
python start_3d_visualizer.py

# Check system status
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

## Documentation

- **Main README**: `README.md`
- **PDF Extraction**: `docs/PDF_EXTRACTION_README.md`
- **Training System**: `docs/TRAINING_ENGINE_README.md`
- **Architecture**: `docs/ARCHITECTURE_README.md`

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the detailed documentation
3. Check system requirements
4. Verify GPU drivers (if using GPU)

---

🎉 **You're ready to use the Financial Mesh System!** 