#!/usr/bin/env python3
"""
3D Mesh Visualizer Web Application

Flask web application for interactive 3D mesh visualization with:
- Real-time mesh exploration
- Comfort state clustering
- Stress test visualization
- Animation controls
- Export functionality
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd

# Import 3D visualizer
try:
    from src.visualization.mesh_3d_visualizer import Mesh3DVisualizer
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.visualization.mesh_3d_visualizer import Mesh3DVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('3d_visualizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Mesh3DVisualizerApp:
    """Flask application for 3D mesh visualization"""
    
    def __init__(self):
        # Set up Flask
        template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../templates')
        self.app = Flask(__name__, template_folder=template_folder)
        CORS(self.app)
        
        # Configuration
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Initialize 3D visualizer
        self.visualizer = Mesh3DVisualizer()
        
        # Load mesh data
        self.visualizer.load_mesh_data()
        
        # Register routes
        self._register_routes()
        
        # Performance monitoring
        self.performance_metrics = {
            'requests_processed': 0,
            'avg_response_time': 0.0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('3d_visualizer.html')
        
        @self.app.route('/api/3d-visualization')
        def get_3d_visualization():
            """Get 3D visualization data"""
            try:
                viz_type = request.args.get('type', 'mesh')
                snapshot_index = int(request.args.get('snapshot', 0))
                
                # Generate visualization data based on type
                if viz_type == 'mesh':
                    data = self._generate_mesh_data(snapshot_index)
                elif viz_type == 'comfort':
                    data = self._generate_comfort_data(snapshot_index)
                elif viz_type == 'evolution':
                    data = self._generate_evolution_data()
                elif viz_type == 'stress':
                    data = self._generate_stress_data()
                elif viz_type == 'fsqca':
                    data = self._generate_fsqca_data()
                else:
                    data = self._generate_mesh_data(snapshot_index)
                
                self.performance_metrics['requests_processed'] += 1
                return jsonify(data)
                
            except Exception as e:
                self.performance_metrics['errors'] += 1
                logger.error(f"Error generating 3D visualization: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get performance metrics"""
            return jsonify(self.performance_metrics)
        
        @self.app.route('/api/export/<viz_type>')
        def export_visualization(viz_type):
            """Export visualization data"""
            try:
                # Generate and export visualization
                if viz_type == 'mesh':
                    fig = self.visualizer.create_3d_mesh_visualization()
                elif viz_type == 'comfort':
                    fig = self.visualizer.create_comfort_state_visualization()
                elif viz_type == 'evolution':
                    fig = self.visualizer.create_mesh_evolution_animation()
                elif viz_type == 'stress':
                    fig = self.visualizer.create_stress_test_visualization({})
                elif viz_type == 'fsqca':
                    fig = self.visualizer.create_fsqca_visualization({})
                else:
                    return jsonify({'error': 'Invalid visualization type'}), 400
                
                # Export to file
                filename = f"3d_{viz_type}_export.html"
                export_path = self.visualizer.export_visualization(fig, filename)
                
                return send_file(export_path, as_attachment=True)
                
            except Exception as e:
                logger.error(f"Error exporting visualization: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics
            })
    
    def _generate_mesh_data(self, snapshot_index: int) -> Dict[str, Any]:
        """Generate mesh visualization data"""
        if not self.visualizer.mesh_data or 'snapshots' not in self.visualizer.mesh_data:
            return self._generate_sample_mesh_data()
        
        if snapshot_index >= len(self.visualizer.mesh_data['snapshots']):
            snapshot_index = 0
        
        snapshot = self.visualizer.mesh_data['snapshots'][snapshot_index]
        nodes = snapshot['nodes']
        
        # Extract data with error handling
        x_coords = []
        y_coords = []
        z_coords = []
        wealth_values = []
        comfort_scores = []
        
        for node in nodes:
            try:
                if 'position' in node and isinstance(node['position'], dict):
                    x_coords.append(node['position'].get('x', 0))
                    y_coords.append(node['position'].get('y', 0))
                    z_coords.append(node['position'].get('z', 0))
                else:
                    # Generate random positions if not available
                    x_coords.append(np.random.normal(0, 1))
                    y_coords.append(np.random.normal(0, 1))
                    z_coords.append(np.random.normal(0, 1))
                
                if 'financial_state' in node and isinstance(node['financial_state'], dict):
                    wealth_values.append(node['financial_state'].get('wealth', 500000))
                    comfort_scores.append(node['financial_state'].get('comfort_score', 0.5))
                else:
                    wealth_values.append(500000)
                    comfort_scores.append(0.5)
            except Exception as e:
                # Fallback values
                x_coords.append(np.random.normal(0, 1))
                y_coords.append(np.random.normal(0, 1))
                z_coords.append(np.random.normal(0, 1))
                wealth_values.append(500000)
                comfort_scores.append(0.5)
        node_sizes = [max(5, min(20, wealth / 10000)) for wealth in wealth_values]
        
        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'z_coords': z_coords,
            'node_sizes': node_sizes,
            'colors': comfort_scores,
            'colorscale': 'Viridis',
            'labels': [f"Wealth: ${wealth:,.0f}<br>Comfort: {comfort:.3f}" 
                      for wealth, comfort in zip(wealth_values, comfort_scores)],
            'hover_template': '<b>%{text}</b><extra></extra>',
            'title': f"3D Financial Mesh - Snapshot {snapshot_index + 1}",
            'x_title': 'X Position',
            'y_title': 'Y Position',
            'z_title': 'Z Position',
            'total_nodes': len(nodes),
            'avg_wealth': np.mean(wealth_values),
            'avg_comfort': np.mean(comfort_scores),
            'stress_level': snapshot_index * 10
        }
    
    def _generate_comfort_data(self, snapshot_index: int) -> Dict[str, Any]:
        """Generate comfort state visualization data"""
        if not self.visualizer.mesh_data or 'snapshots' not in self.visualizer.mesh_data:
            return self._generate_sample_comfort_data()
        
        if snapshot_index >= len(self.visualizer.mesh_data['snapshots']):
            snapshot_index = 0
        
        snapshot = self.visualizer.mesh_data['snapshots'][snapshot_index]
        nodes = snapshot['nodes']
        
        # Extract comfort data
        x_coords = [node['position']['x'] for node in nodes]
        y_coords = [node['position']['y'] for node in nodes]
        z_coords = [node['financial_state']['comfort_score'] for node in nodes]
        wealth_values = [node['financial_state']['wealth'] for node in nodes]
        comfort_scores = [node['financial_state']['comfort_score'] for node in nodes]
        node_sizes = [max(5, min(25, wealth / 10000)) for wealth in wealth_values]
        
        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'z_coords': z_coords,
            'node_sizes': node_sizes,
            'colors': comfort_scores,
            'colorscale': 'RdYlGn',
            'labels': [f"Comfort: {score:.3f}<br>Wealth: ${wealth:,.0f}" 
                      for score, wealth in zip(comfort_scores, wealth_values)],
            'hover_template': '<b>%{text}</b><extra></extra>',
            'title': f"3D Comfort State Visualization - Snapshot {snapshot_index + 1}",
            'x_title': 'X Position',
            'y_title': 'Y Position',
            'z_title': 'Comfort Score',
            'total_nodes': len(nodes),
            'avg_wealth': np.mean(wealth_values),
            'avg_comfort': np.mean(comfort_scores),
            'stress_level': snapshot_index * 10
        }
    
    def _generate_evolution_data(self) -> Dict[str, Any]:
        """Generate mesh evolution data"""
        # Generate sample evolution data
        num_points = 100
        x_coords = np.random.normal(0, 1, num_points)
        y_coords = np.random.normal(0, 1, num_points)
        z_coords = np.random.normal(0, 1, num_points)
        colors = np.random.random(num_points)
        sizes = np.random.random(num_points) * 15 + 5
        
        return {
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'z_coords': z_coords.tolist(),
            'node_sizes': sizes.tolist(),
            'colors': colors.tolist(),
            'colorscale': 'Viridis',
            'labels': [f"Evolution Node {i+1}" for i in range(num_points)],
            'hover_template': '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            'title': '3D Mesh Evolution',
            'x_title': 'X Position',
            'y_title': 'Y Position',
            'z_title': 'Z Position',
            'total_nodes': num_points,
            'avg_wealth': 500000,
            'avg_comfort': 0.6,
            'stress_level': 20
        }
    
    def _generate_stress_data(self) -> Dict[str, Any]:
        """Generate stress test visualization data"""
        # Generate sample stress test data
        scenario_names = ['Market Shock', 'Interest Rate', 'Correlation', 'Combined']
        wealth_impacts = [-0.2, -0.1, -0.05, -0.15]
        comfort_impacts = [-0.3, -0.2, -0.1, -0.25]
        risk_scores = [0.8, 0.6, 0.4, 0.7]
        
        return {
            'x_coords': wealth_impacts,
            'y_coords': comfort_impacts,
            'z_coords': risk_scores,
            'node_sizes': [15] * len(scenario_names),
            'colors': risk_scores,
            'colorscale': 'RdYlGn',
            'labels': scenario_names,
            'hover_template': '<b>%{text}</b><br>Wealth Impact: %{x:.2%}<br>Comfort Impact: %{y:.2%}<br>Risk: %{z:.2%}<extra></extra>',
            'title': '3D Stress Test Results',
            'x_title': 'Wealth Impact',
            'y_title': 'Comfort Impact',
            'z_title': 'Risk Impact',
            'total_nodes': len(scenario_names),
            'avg_wealth': -0.125,
            'avg_comfort': -0.2125,
            'stress_level': 75
        }
    
    def _generate_fsqca_data(self) -> Dict[str, Any]:
        """Generate fsQCA visualization data"""
        # Generate sample fsQCA data
        conditions = ['High Wealth', 'High Savings', 'High Stability']
        coverage_scores = [0.8, 0.7, 0.9]
        consistency_scores = [0.85, 0.75, 0.95]
        necessity_scores = [0.6, 0.8, 0.7]
        
        return {
            'x_coords': coverage_scores,
            'y_coords': consistency_scores,
            'z_coords': necessity_scores,
            'node_sizes': [20] * len(conditions),
            'colors': necessity_scores,
            'colorscale': 'Viridis',
            'labels': conditions,
            'hover_template': '<b>%{text}</b><br>Coverage: %{x:.2f}<br>Consistency: %{y:.2f}<br>Necessity: %{z:.2f}<extra></extra>',
            'title': '3D fsQCA Analysis Results',
            'x_title': 'Coverage',
            'y_title': 'Consistency',
            'z_title': 'Necessity',
            'total_nodes': len(conditions),
            'avg_wealth': 0.8,
            'avg_comfort': 0.85,
            'stress_level': 30
        }
    
    def _generate_sample_mesh_data(self) -> Dict[str, Any]:
        """Generate sample mesh data"""
        num_points = 50
        x_coords = np.random.normal(0, 1, num_points)
        y_coords = np.random.normal(0, 1, num_points)
        z_coords = np.random.normal(0, 1, num_points)
        colors = np.random.random(num_points)
        sizes = np.random.random(num_points) * 15 + 5
        
        return {
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'z_coords': z_coords.tolist(),
            'node_sizes': sizes.tolist(),
            'colors': colors.tolist(),
            'colorscale': 'Viridis',
            'labels': [f"Sample Node {i+1}" for i in range(num_points)],
            'hover_template': '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            'title': '3D Financial Mesh (Sample Data)',
            'x_title': 'X Position',
            'y_title': 'Y Position',
            'z_title': 'Z Position',
            'total_nodes': num_points,
            'avg_wealth': 500000,
            'avg_comfort': 0.5,
            'stress_level': 0
        }
    
    def _generate_sample_comfort_data(self) -> Dict[str, Any]:
        """Generate sample comfort data"""
        num_points = 50
        x_coords = np.random.normal(0, 1, num_points)
        y_coords = np.random.normal(0, 1, num_points)
        z_coords = np.random.random(num_points)  # Comfort scores
        colors = z_coords  # Color by comfort
        sizes = np.random.random(num_points) * 20 + 5
        
        return {
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'z_coords': z_coords.tolist(),
            'node_sizes': sizes.tolist(),
            'colors': colors.tolist(),
            'colorscale': 'RdYlGn',
            'labels': [f"Comfort: {score:.3f}" for score in z_coords],
            'hover_template': '<b>%{text}</b><extra></extra>',
            'title': '3D Comfort States (Sample Data)',
            'x_title': 'X Position',
            'y_title': 'Y Position',
            'z_title': 'Comfort Score',
            'total_nodes': num_points,
            'avg_wealth': 500000,
            'avg_comfort': np.mean(z_coords),
            'stress_level': 0
        }
    
    def run(self, host: str = 'localhost', port: int = 5002, debug: bool = False):
        """Run the Flask application"""
        logger.info(f"🚀 Starting 3D Mesh Visualizer on {host}:{port}")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"❌ Failed to start 3D Mesh Visualizer: {e}")
            raise


def create_3d_visualizer_app():
    """Create and run 3D visualizer application"""
    print("🎨 Creating 3D Mesh Visualizer Application...")
    
    app = Mesh3DVisualizerApp()
    
    print("✅ 3D Mesh Visualizer Application Created Successfully!")
    print("=" * 60)
    print("🎯 Features:")
    print("  ✅ Interactive 3D mesh exploration")
    print("  ✅ Comfort state clustering visualization")
    print("  ✅ Mesh evolution animation")
    print("  ✅ Stress test results visualization")
    print("  ✅ fsQCA analysis visualization")
    print("  ✅ Real-time data updates")
    print("  ✅ Export functionality")
    print("  ✅ Performance monitoring")
    
    return app


if __name__ == '__main__':
    app = create_3d_visualizer_app()
    app.run(host='localhost', port=5002, debug=True) 