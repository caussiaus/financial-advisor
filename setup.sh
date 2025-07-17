#!/bin/bash

# Financial Mesh System Setup Script
# This script sets up the complete environment for the financial mesh system

set -e  # Exit on any error

echo "🚀 Starting Financial Mesh System Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Miniconda or Anaconda first."
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda is installed"
}

# Check system requirements
check_system() {
    print_status "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux system detected"
        SYSTEM="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS system detected"
        SYSTEM="macos"
    else
        print_warning "Unknown OS type: $OSTYPE"
        SYSTEM="unknown"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        GPU="nvidia"
    else
        print_warning "No NVIDIA GPU detected, will use CPU"
        GPU="cpu"
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$SYSTEM" == "linux" ]]; then
        # Linux dependencies
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                libgcc-s1
        elif command -v yum &> /dev/null; then
            sudo yum update
            sudo yum install -y \
                gcc \
                gcc-c++ \
                mesa-libGL \
                mesa-libGL-devel
        fi
    elif [[ "$SYSTEM" == "macos" ]]; then
        # macOS dependencies
        if command -v brew &> /dev/null; then
            brew update
            brew install tesseract tesseract-lang
        else
            print_warning "Homebrew not found. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
    fi
}

# Create conda environment
create_environment() {
    print_status "Creating conda environment..."
    
    # Remove existing environment if it exists
    if conda env list | grep -q "financial-mesh"; then
        print_warning "Environment 'financial-mesh' already exists. Removing..."
        conda env remove -n financial-mesh -y
    fi
    
    # Create environment from YAML
    conda env create -f environment.yml
    
    print_success "Conda environment created successfully"
}

# Activate environment and install additional dependencies
setup_environment() {
    print_status "Setting up environment..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate financial-mesh
    
    # Download pre-trained models
    print_status "Downloading pre-trained models..."
    python -c "
import torch
from transformers import AutoTokenizer, AutoModel
print('Downloading LayoutLM model...')
AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base')
AutoModel.from_pretrained('microsoft/layoutlmv3-base')
print('Downloading Donut model...')
AutoTokenizer.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
AutoModel.from_pretrained('naver-clova-ix/donut-base-finetuned-docvqa')
print('Models downloaded successfully!')
"
    
    # Download spaCy model
    print_status "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    
    print_success "Environment setup completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/outputs/extraction
    mkdir -p data/outputs/training
    mkdir -p data/outputs/comparison
    mkdir -p data/outputs/visualizations
    mkdir -p logs
    mkdir -p models
    
    print_success "Directories created"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate financial-mesh
    
    # Test basic imports
    python -c "
import sys
print('Testing imports...')

# Test core libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✓ Core libraries imported')

# Test PDF processing
import pdfplumber
import PyPDF2
print('✓ PDF processing libraries imported')

# Test ML libraries
import transformers
import doctr
import easyocr
print('✓ ML libraries imported')

# Test visualization
import plotly
import dash
print('✓ Visualization libraries imported')

# Test our modules
sys.path.append('src')
from extraction import HybridExtractionPipeline, RuleBasedExtractor
print('✓ Custom modules imported')

print('🎉 All tests passed!')
"
    
    print_success "Installation test completed successfully"
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Financial Mesh Environment Activation Script

echo "🔧 Activating Financial Mesh Environment"
echo "======================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate financial-mesh

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Check GPU availability
python -c "
import torch
if torch.cuda.is_available():
    print(f'🚀 GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('💻 Using CPU')
"

echo "✅ Environment activated successfully!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Environment: financial-mesh"
echo ""
echo "🚀 Ready to run:"
echo "   python demos/demo_pdf_extraction.py"
echo "   python demos/demo_training_engine.py"
echo "   python start_3d_visualizer.py"
EOF

    chmod +x activate_env.sh
    print_success "Activation script created: ./activate_env.sh"
}

# Main setup function
main() {
    echo ""
    print_status "Starting Financial Mesh System Setup"
    echo ""
    
    # Check prerequisites
    check_conda
    check_system
    
    # Install system dependencies
    install_system_deps
    
    # Create conda environment
    create_environment
    
    # Setup environment
    setup_environment
    
    # Create directories
    create_directories
    
    # Test installation
    test_installation
    
    # Create activation script
    create_activation_script
    
    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Activate the environment: source activate_env.sh"
    echo "   2. Test PDF extraction: python demos/demo_pdf_extraction.py"
    echo "   3. Test training system: python demos/demo_training_engine.py"
    echo "   4. Start 3D visualizer: python start_3d_visualizer.py"
    echo ""
    echo "📚 Documentation:"
    echo "   - README.md: Main documentation"
    echo "   - docs/PDF_EXTRACTION_README.md: PDF extraction guide"
    echo "   - docs/TRAINING_ENGINE_README.md: Training system guide"
    echo ""
    echo "🔧 Environment info:"
    echo "   - Name: financial-mesh"
    echo "   - Python: 3.11"
    echo "   - GPU: $GPU"
    echo "   - System: $SYSTEM"
    echo ""
}

# Run main function
main "$@" 