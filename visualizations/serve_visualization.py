#!/usr/bin/env python3
"""
Life Events Portfolio Visualization Server

This script serves the D3.js visualization for life events and portfolio impact.
"""

from flask import Flask, send_from_directory, render_template_string
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main visualization page"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_file = os.path.join(script_dir, 'strategy_comparison_visualization.html')
        
        with open(html_file, 'r') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Visualization file not found. Please ensure strategy_comparison_visualization.html exists.", 404

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "life_events_visualization"}

@app.route('/api/data/life_events')
def get_life_events_data():
    """API endpoint to get life events data"""
    import json
    try:
        # Get the project root directory (two levels up from visualizations/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_file = os.path.join(project_root, 'data/outputs/analysis_data/realistic_life_events_analysis.json')
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "Life events data not found"}, 404

@app.route('/api/data/horatio_profile')
def get_horatio_profile():
    """API endpoint to get Horatio's profile data"""
    import json
    try:
        # Get the project root directory (two levels up from visualizations/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        profile_file = os.path.join(project_root, 'horatio_profile.json')
        
        with open(profile_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "Horatio profile not found"}, 404

if __name__ == '__main__':
    port = 5003
    logger.info(f"Starting Life Events Visualization Server on port {port}")
    logger.info(f"🌐 Access at: http://localhost:{port}")
    logger.info(f"📊 Health check: http://localhost:{port}/api/health")
    app.run(host='0.0.0.0', port=port, debug=True) 