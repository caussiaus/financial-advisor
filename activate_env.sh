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
