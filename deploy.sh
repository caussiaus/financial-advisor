#!/bin/bash

# Exit on error
set -e

echo "🚀 Deploying Omega Mesh Engine..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python version $python_version is not compatible. Please install Python >= 3.8"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "📊 CUDA Driver Version: $cuda_version"
else
    echo "⚠️ No NVIDIA GPU detected - will run in CPU mode"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install base requirements
echo "📚 Installing base requirements..."
pip install -e .

# Install CUDA requirements if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 Installing CUDA acceleration packages..."
    pip install -e .[cuda]
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/inputs/uploads
mkdir -p data/outputs/analysis_data
mkdir -p data/outputs/client_data
mkdir -p data/outputs/reports
mkdir -p data/outputs/visual_timelines

# Set up configuration
echo "⚙️ Setting up configuration..."
if [ ! -f "config.json" ]; then
    cat > config.json << EOF
{
    "server": {
        "host": "0.0.0.0",
        "port": 8081,
        "debug": false
    },
    "mesh": {
        "default_time_horizon": 10,
        "max_nodes": 10000,
        "use_cuda": true
    },
    "accounting": {
        "initial_wealth": 2000000,
        "currency": "USD"
    }
}
EOF
fi

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/omega-mesh.service > /dev/null << EOF
[Unit]
Description=Omega Mesh Financial Engine
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python omega_web_app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

echo "✨ Installation complete!"
echo "To start the service:"
echo "  sudo systemctl start omega-mesh"
echo "To enable on boot:"
echo "  sudo systemctl enable omega-mesh"
echo "To check status:"
echo "  sudo systemctl status omega-mesh"
echo "To view logs:"
echo "  journalctl -u omega-mesh -f" 