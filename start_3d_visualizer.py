#!/usr/bin/env python3
"""
3D Mesh Visualizer Startup Script

Starts the 3D mesh visualizer web application with all features enabled.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('3d_visualizer_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    print("🎨 Starting 3D Mesh Visualizer...")
    print("=" * 60)
    
    try:
        # Import and create 3D visualizer app
        from dashboards.mesh_3d_visualizer_app import create_3d_visualizer_app
        
        # Create the application
        app = create_3d_visualizer_app()
        
        print("\n🚀 Launching 3D Mesh Visualizer...")
        print("📍 Access the visualizer at: http://localhost:5002")
        print("📊 API endpoints available at: http://localhost:5002/api/")
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the application
        app.run(host='localhost', port=5002, debug=True)
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        print(f"❌ Failed to import required modules: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install flask flask-cors plotly numpy pandas")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        print(f"❌ Failed to start 3D Mesh Visualizer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 