#!/usr/bin/env python3
"""
Enhanced Mesh Dashboard Startup Script
Simple startup script for the TradingView-style dashboard
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the enhanced dashboard"""
    try:
        # Import the dashboard
        from dashboards.enhanced_mesh_dashboard import EnhancedMeshDashboard
        
        # Create and run dashboard
        dashboard = EnhancedMeshDashboard()
        
        logger.info("Starting TradingView-style Enhanced Mesh Dashboard...")
        logger.info("🌐 Access at: http://localhost:5001")
        logger.info("📊 Health check: http://localhost:5001/api/health")
        logger.info("🔧 System controls available in the dashboard")
        
        dashboard.run(host='0.0.0.0', port=5001, debug=True)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 