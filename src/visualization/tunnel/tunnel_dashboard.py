#!/usr/bin/env python3
"""
High-Dimensional Tunnel Dashboard

Interactive Flask dashboard for high-dimensional tunnel visualization
with real-time data exploration and filtering capabilities.
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from flask import Flask, render_template, jsonify, request
import traceback

# Import the tunnel visualizer
from .guaranteed_tunnel_visualizer import GuaranteedTunnelVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global visualizer instance
visualizer = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('tunnel_dashboard.html')

@app.route('/api/tunnel-data')
def get_tunnel_data():
    """Get tunnel visualization data"""
    try:
        global visualizer
        
        if not visualizer:
            visualizer = GuaranteedTunnelVisualizer()
        
        if not visualizer.data:
            return jsonify({'error': 'No data available'}), 400
        
        # Get parameters from request
        time_window = request.args.get('time_window')
        node_filter = request.args.get('node_filter')
        
        # Parse time window
        parsed_time_window = None
        if time_window:
            try:
                start, end = map(int, time_window.split(','))
                parsed_time_window = (start, end)
            except:
                pass
        
        # Parse node filter
        parsed_node_filter = None
        if node_filter:
            parsed_node_filter = node_filter.split(',')
        
        # Create visualization
        fig = visualizer.create_guaranteed_visualization(
            time_window=parsed_time_window,
            node_filter=parsed_node_filter
        )
        
        if fig:
            # Convert to JSON
            fig_json = fig.to_json()
            return jsonify({
                'success': True,
                'figure': fig_json,
                'data_summary': {
                    'total_snapshots': len(visualizer.data['snapshots']),
                    'total_data_points': len(visualizer.time_series_data['timestamps']) if visualizer.time_series_data else 0,
                    'time_range': f"{visualizer.data['snapshots'][0]['snapshot_time']} to {visualizer.data['snapshots'][-1]['snapshot_time']}"
                }
            })
        else:
            return jsonify({'error': 'Failed to create visualization'}), 500
            
    except Exception as e:
        logger.error(f"Error creating tunnel visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-summary')
def get_data_summary():
    """Get summary of available data"""
    try:
        global visualizer
        
        if not visualizer:
            from .high_dimensional_tunnel_visualizer import HighDimensionalTunnelVisualizer
            visualizer = HighDimensionalTunnelVisualizer()
        
        if not visualizer.data:
            return jsonify({'error': 'No data available'}), 400
        
        # Calculate data statistics
        if visualizer.time_series_data:
            total_wealth_stats = {
                'min': min(visualizer.time_series_data['total_wealth']),
                'max': max(visualizer.time_series_data['total_wealth']),
                'mean': np.mean(visualizer.time_series_data['total_wealth']),
                'std': np.std(visualizer.time_series_data['total_wealth'])
            }
            
            cash_stats = {
                'min': min(visualizer.time_series_data['cash']),
                'max': max(visualizer.time_series_data['cash']),
                'mean': np.mean(visualizer.time_series_data['cash']),
                'std': np.std(visualizer.time_series_data['cash'])
            }
            
            investments_stats = {
                'min': min(visualizer.time_series_data['investments']),
                'max': max(visualizer.time_series_data['investments']),
                'mean': np.mean(visualizer.time_series_data['investments']),
                'std': np.std(visualizer.time_series_data['investments'])
            }
            
            prob_stats = {
                'min': min(visualizer.time_series_data['probabilities']),
                'max': max(visualizer.time_series_data['probabilities']),
                'mean': np.mean(visualizer.time_series_data['probabilities']),
                'std': np.std(visualizer.time_series_data['probabilities'])
            }
        else:
            total_wealth_stats = cash_stats = investments_stats = prob_stats = {}
        
        return jsonify({
            'success': True,
            'data_summary': {
                'total_snapshots': len(visualizer.data['snapshots']),
                'total_data_points': len(visualizer.time_series_data['timestamps']) if visualizer.time_series_data else 0,
                'time_range': {
                    'start': visualizer.data['snapshots'][0]['snapshot_time'],
                    'end': visualizer.data['snapshots'][-1]['snapshot_time']
                },
                'statistics': {
                    'total_wealth': total_wealth_stats,
                    'cash': cash_stats,
                    'investments': investments_stats,
                    'probabilities': prob_stats
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def export_visualization():
    """Export visualization as HTML"""
    try:
        global visualizer
        
        if not visualizer:
            from .high_dimensional_tunnel_visualizer import HighDimensionalTunnelVisualizer
            visualizer = HighDimensionalTunnelVisualizer()
        
        if not visualizer.data:
            return jsonify({'error': 'No data available'}), 400
        
        # Get parameters
        time_window = request.args.get('time_window')
        node_filter = request.args.get('node_filter')
        
        # Parse parameters
        parsed_time_window = None
        if time_window:
            try:
                start, end = map(int, time_window.split(','))
                parsed_time_window = (start, end)
            except:
                pass
        
        parsed_node_filter = None
        if node_filter:
            parsed_node_filter = node_filter.split(',')
        
        # Create visualization
        fig = visualizer.create_guaranteed_visualization(
            time_window=parsed_time_window,
            node_filter=parsed_node_filter
        )
        
        if fig:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tunnel_visualization_{timestamp}.html"
            
            # Save to file
            fig.write_html(filename)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': f'Visualization exported to {filename}'
            })
        else:
            return jsonify({'error': 'Failed to create visualization'}), 500
            
    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_templates_directory():
    """Create templates directory and HTML template"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>High-Dimensional Tunnel Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .control-group {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #495057;
        }
        .control-group input, .control-group select {
            padding: 8px 12px;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .control-group input:focus, .control-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .visualization {
            padding: 20px;
            min-height: 800px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            color: #6c757d;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            margin: 20px;
        }
        .info-panel {
            background: #e3f2fd;
            padding: 15px;
            margin: 20px;
            border-radius: 6px;
            border-left: 4px solid #2196f3;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h4 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 14px;
        }
        .stat-value {
            font-size: 18px;
            font-weight: 600;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 High-Dimensional Tunnel Dashboard</h1>
            <p>Explore financial state evolution through 4D tunnel visualizations</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="timeWindow">Time Window (start,end):</label>
                <input type="text" id="timeWindow" placeholder="0,50" value="0,50">
            </div>
            <div class="control-group">
                <label for="resolution">Tunnel Resolution:</label>
                <select id="resolution">
                    <option value="30">Low (30)</option>
                    <option value="50" selected>Medium (50)</option>
                    <option value="80">High (80)</option>
                    <option value="100">Very High (100)</option>
                </select>
            </div>
            <div class="control-group">
                <label for="nodeFilter">Node Filter (comma-separated):</label>
                <input type="text" id="nodeFilter" placeholder="omega_0_0,omega_0_1">
            </div>
            <div class="control-group">
                <button class="btn" onclick="updateVisualization()">🔄 Update Visualization</button>
                <button class="btn" onclick="exportVisualization()">📊 Export HTML</button>
            </div>
        </div>
        
        <div id="infoPanel" class="info-panel" style="display: none;">
            <h3>📊 Data Summary</h3>
            <div id="dataStats" class="stats-grid"></div>
        </div>
        
        <div id="visualization" class="visualization">
            <div class="loading">
                <h3>🔄 Loading tunnel visualization...</h3>
                <p>Please wait while we prepare the high-dimensional tunnel surfaces.</p>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentFigure = null;
        
        // Initialize dashboard
        $(document).ready(function() {
            loadDataSummary();
            updateVisualization();
        });
        
        function loadDataSummary() {
            $.get('/api/data-summary')
                .done(function(data) {
                    if (data.success) {
                        displayDataSummary(data.data_summary);
                    }
                })
                .fail(function(xhr) {
                    console.error('Failed to load data summary:', xhr.responseJSON);
                });
        }
        
        function displayDataSummary(summary) {
            $('#infoPanel').show();
            
            const stats = summary.statistics;
            let statsHtml = '';
            
            // Total Wealth Stats
            if (stats.total_wealth) {
                statsHtml += `
                    <div class="stat-card">
                        <h4>💰 Total Wealth</h4>
                        <div class="stat-value">$${stats.total_wealth.mean.toLocaleString()}</div>
                        <small>Range: $${stats.total_wealth.min.toLocaleString()} - $${stats.total_wealth.max.toLocaleString()}</small>
                    </div>
                `;
            }
            
            // Cash Stats
            if (stats.cash) {
                statsHtml += `
                    <div class="stat-card">
                        <h4>💵 Cash</h4>
                        <div class="stat-value">$${stats.cash.mean.toLocaleString()}</div>
                        <small>Range: $${stats.cash.min.toLocaleString()} - $${stats.cash.max.toLocaleString()}</small>
                    </div>
                `;
            }
            
            // Investments Stats
            if (stats.investments) {
                statsHtml += `
                    <div class="stat-card">
                        <h4>📈 Investments</h4>
                        <div class="stat-value">$${stats.investments.mean.toLocaleString()}</div>
                        <small>Range: $${stats.investments.min.toLocaleString()} - $${stats.investments.max.toLocaleString()}</small>
                    </div>
                `;
            }
            
            // General Stats
            statsHtml += `
                <div class="stat-card">
                    <h4>📊 Data Points</h4>
                    <div class="stat-value">${summary.total_data_points.toLocaleString()}</div>
                    <small>Across ${summary.total_snapshots} snapshots</small>
                </div>
            `;
            
            $('#dataStats').html(statsHtml);
        }
        
        function updateVisualization() {
            const timeWindow = $('#timeWindow').val();
            const resolution = $('#resolution').val();
            const nodeFilter = $('#nodeFilter').val();
            
            $('#visualization').html(`
                <div class="loading">
                    <h3>🔄 Creating tunnel visualization...</h3>
                    <p>Generating high-dimensional surfaces with resolution ${resolution}</p>
                </div>
            `);
            
            $.get('/api/tunnel-data', {
                time_window: timeWindow,
                resolution: resolution,
                node_filter: nodeFilter
            })
            .done(function(data) {
                if (data.success) {
                    displayVisualization(data.figure);
                } else {
                    showError('Failed to create visualization: ' + data.error);
                }
            })
            .fail(function(xhr) {
                showError('Failed to load visualization: ' + (xhr.responseJSON?.error || 'Unknown error'));
            });
        }
        
        function displayVisualization(figureJson) {
            try {
                const figure = JSON.parse(figureJson);
                currentFigure = figure;
                
                Plotly.newPlot('visualization', figure.data, figure.layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
                });
                
                // Add event listeners for 3D surface interactions
                document.getElementById('visualization').on('plotly_click', function(data) {
                    console.log('Surface clicked:', data);
                });
                
            } catch (error) {
                showError('Failed to display visualization: ' + error.message);
            }
        }
        
        function exportVisualization() {
            const timeWindow = $('#timeWindow').val();
            const resolution = $('#resolution').val();
            const nodeFilter = $('#nodeFilter').val();
            
            $.get('/api/export', {
                time_window: timeWindow,
                resolution: resolution,
                node_filter: nodeFilter
            })
            .done(function(data) {
                if (data.success) {
                    alert('✅ ' + data.message);
                } else {
                    alert('❌ Export failed: ' + data.error);
                }
            })
            .fail(function(xhr) {
                alert('❌ Export failed: ' + (xhr.responseJSON?.error || 'Unknown error'));
            });
        }
        
        function showError(message) {
            $('#visualization').html(`
                <div class="error">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                </div>
            `);
        }
    </script>
</body>
</html>'''
    
    with open(templates_dir / 'tunnel_dashboard.html', 'w') as f:
        f.write(html_template)

def main():
    """Main function to run the tunnel dashboard"""
    print("🚀 High-Dimensional Tunnel Dashboard")
    print("=" * 50)
    
    # Create templates directory and HTML template
    create_templates_directory()
    
    # Initialize visualizer
    global visualizer
    visualizer = GuaranteedTunnelVisualizer()
    
    if not visualizer.data:
        print("❌ No mesh data available")
        return
    
    print("✅ Data loaded successfully")
    print(f"📊 {len(visualizer.data['snapshots'])} snapshots available")
    
    # Start Flask app
    print("\n🌐 Starting tunnel dashboard...")
    print("📍 Access at: http://localhost:5010")
    print("💡 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5010, debug=True)

if __name__ == "__main__":
    main() 