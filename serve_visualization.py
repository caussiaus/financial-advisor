#!/usr/bin/env python3
"""
Visualization Service Launcher
Launches the life events visualization server
"""

import sys
import os
import subprocess

def main():
    """Launch the visualization service"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the actual visualization script
    viz_script = os.path.join(script_dir, 'visualizations', 'serve_visualization.py')
    
    if not os.path.exists(viz_script):
        print(f"Error: Visualization script not found at {viz_script}")
        sys.exit(1)
    
    # Launch the visualization script with the same arguments
    cmd = [sys.executable, viz_script] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization service: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nVisualization service stopped by user")

if __name__ == "__main__":
    main() 