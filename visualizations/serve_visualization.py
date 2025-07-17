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
        with open('life_events_portfolio_visualization.html', 'r') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "Visualization file not found. Please ensure life_events_portfolio_visualization.html exists.", 404

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "life_events_visualization"}

@app.route('/api/data/life_events')
def get_life_events_data():
    """API endpoint to get life events data"""
    import json
    try:
        with open('data/outputs/analysis_data/realistic_life_events_analysis.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": "Life events data not found"}, 404

@app.route('/api/data/horatio_profile')
def get_horatio_profile():
    """API endpoint to get Horatio's profile data"""
    import json
    try:
        with open('horatio_profile.json', 'r') as f:
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