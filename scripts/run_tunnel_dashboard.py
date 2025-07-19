#!/usr/bin/env python3
"""
Run Tunnel Dashboard on Port 5020

Simple script to start the high-dimensional tunnel dashboard
on a dedicated port to avoid conflicts.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import and modify the tunnel dashboard
from src.visualization.tunnel.tunnel_dashboard import app, main

if __name__ == "__main__":
    print("🚀 High-Dimensional Tunnel Dashboard")
    print("=" * 50)
    print("📍 Starting on port 5020...")
    print("🌐 Access at: http://localhost:5020")
    print("💡 Press Ctrl+C to stop")
    
    # Run on port 5020
    app.run(host='0.0.0.0', port=5020, debug=True) 