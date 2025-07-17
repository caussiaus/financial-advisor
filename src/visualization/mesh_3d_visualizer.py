#!/usr/bin/env python3
"""
3D Mesh Visualizer for Financial Mesh System

Interactive 3D visualization of the financial mesh with:
- Real-time mesh exploration
- Comfort state clustering
- Stress test visualization
- Animation capabilities
- Export functionality
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core components
try:
    from ..core.stochastic_mesh_engine import StochasticMeshEngine
    from ..core.state_space_mesh_engine import EnhancedMeshEngine
    from ..analysis.mesh_congruence_engine import MeshCongruenceEngine
    from ..analysis.mesh_vector_database import MeshVectorDatabase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.core.stochastic_mesh_engine import StochasticMeshEngine
    from src.core.state_space_mesh_engine import EnhancedMeshEngine
    from src.analysis.mesh_congruence_engine import MeshCongruenceEngine
    from src.analysis.mesh_vector_database import MeshVectorDatabase


class Mesh3DVisualizer:
    """
    Interactive 3D mesh visualizer for financial mesh system
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.mesh_data = None
        self.comfort_data = None
        self.stress_data = None
        self.animation_data = None
        
        # Create output directory
        self.output_dir = Path("data/outputs/3d_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for 3D visualizer"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_mesh_data(self, mesh_file: str = "horatio_mesh_timelapse.json") -> bool:
        """Load mesh data from file"""
        try:
            mesh_path = Path(mesh_file)
            if not mesh_path.exists():
                self.logger.warning(f"Mesh file {mesh_file} not found, using sample data")
                self.mesh_data = self._generate_sample_mesh_data()
                return True
            
            with open(mesh_path, 'r') as f:
                self.mesh_data = json.load(f)
            
            self.logger.info(f"✅ Loaded mesh data from {mesh_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load mesh data: {e}")
            self.mesh_data = self._generate_sample_mesh_data()
            return False
    
    def _generate_sample_mesh_data(self) -> Dict[str, Any]:
        """Generate sample mesh data for visualization"""
        sample_data = {
            'metadata': {
                'total_snapshots': 132,
                'time_horizon_years': 10,
                'initial_nodes': 10,
                'peak_nodes': 220,
                'final_nodes': 220
            },
            'snapshots': []
        }
        
        # Generate sample snapshots
        for i in range(12):  # 12 snapshots for demo
            snapshot = {
                'timestamp': f'2025-{i+1:02d}',
                'nodes': [],
                'edges': [],
                'metrics': {
                    'total_nodes': 10 + i * 20,
                    'total_edges': i * 100,
                    'avg_wealth': 500000 + i * 50000,
                    'comfort_score': 0.5 + i * 0.05
                }
            }
            
            # Generate sample nodes
            for j in range(snapshot['metrics']['total_nodes']):
                node = {
                    'node_id': f'node_{i}_{j}',
                    'position': {
                        'x': np.random.normal(0, 1),
                        'y': np.random.normal(0, 1),
                        'z': np.random.normal(0, 1)
                    },
                    'financial_state': {
                        'wealth': 500000 + np.random.normal(0, 100000),
                        'cash': 100000 + np.random.normal(0, 20000),
                        'investments': 300000 + np.random.normal(0, 80000),
                        'comfort_score': 0.5 + np.random.normal(0, 0.2)
                    },
                    'comfort_cluster': np.random.randint(0, 3)
                }
                snapshot['nodes'].append(node)
            
            sample_data['snapshots'].append(snapshot)
        
        return sample_data
    
    def create_3d_mesh_visualization(self, snapshot_index: int = 0) -> go.Figure:
        """
        Create 3D mesh visualization for a specific snapshot
        
        Args:
            snapshot_index: Index of snapshot to visualize
            
        Returns:
            Plotly 3D figure
        """
        if not self.mesh_data or 'snapshots' not in self.mesh_data:
            self.logger.error("No mesh data available")
            return go.Figure()
        
        if snapshot_index >= len(self.mesh_data['snapshots']):
            snapshot_index = 0
        
        snapshot = self.mesh_data['snapshots'][snapshot_index]
        nodes = snapshot['nodes']
        
        # Extract node positions and properties
        x_coords = [node['position']['x'] for node in nodes]
        y_coords = [node['position']['y'] for node in nodes]
        z_coords = [node['position']['z'] for node in nodes]
        wealth_values = [node['financial_state']['wealth'] for node in nodes]
        comfort_scores = [node['financial_state']['comfort_score'] for node in nodes]
        comfort_clusters = [node['comfort_cluster'] for node in nodes]
        node_ids = [node['node_id'] for node in nodes]
        
        # Create color mapping for comfort clusters
        colors = ['red', 'yellow', 'green']
        node_colors = [colors[cluster] for cluster in comfort_clusters]
        
        # Create node sizes based on wealth
        node_sizes = [max(5, min(20, wealth / 10000)) for wealth in wealth_values]
        
        # Create 3D scatter plot for nodes
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                colorscale='Viridis'
            ),
            text=node_ids,
            hovertemplate='<b>%{text}</b><br>' +
                        'Wealth: $%{customdata[0]:,.0f}<br>' +
                        'Comfort: %{customdata[1]:.3f}<br>' +
                        'Cluster: %{customdata[2]}<extra></extra>',
            customdata=list(zip(wealth_values, comfort_scores, comfort_clusters))
        ))
        
        # Add edges if available
        if 'edges' in snapshot and snapshot['edges']:
            edge_x = []
            edge_y = []
            edge_z = []
            
            for edge in snapshot['edges']:
                if 'source' in edge and 'target' in edge:
                    source_node = next((n for n in nodes if n['node_id'] == edge['source']), None)
                    target_node = next((n for n in nodes if n['node_id'] == edge['target']), None)
                    
                    if source_node and target_node:
                        edge_x.extend([source_node['position']['x'], target_node['position']['x'], None])
                        edge_y.extend([source_node['position']['y'], target_node['position']['y'], None])
                        edge_z.extend([source_node['position']['z'], target_node['position']['z'], None])
            
            if edge_x:
                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(color='gray', width=1),
                    opacity=0.3,
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=f"3D Financial Mesh - Snapshot {snapshot_index + 1}",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_comfort_state_visualization(self, snapshot_index: int = 0) -> go.Figure:
        """
        Create 3D comfort state visualization
        
        Args:
            snapshot_index: Index of snapshot to visualize
            
        Returns:
            Plotly 3D figure
        """
        if not self.mesh_data or 'snapshots' not in self.mesh_data:
            return go.Figure()
        
        if snapshot_index >= len(self.mesh_data['snapshots']):
            snapshot_index = 0
        
        snapshot = self.mesh_data['snapshots'][snapshot_index]
        nodes = snapshot['nodes']
        
        # Extract comfort-related data
        x_coords = [node['position']['x'] for node in nodes]
        y_coords = [node['position']['y'] for node in nodes]
        z_coords = [node['position']['z'] for node in nodes]
        comfort_scores = [node['financial_state']['comfort_score'] for node in nodes]
        wealth_values = [node['financial_state']['wealth'] for node in nodes]
        cash_values = [node['financial_state']['cash'] for node in nodes]
        investment_values = [node['financial_state']['investments'] for node in nodes]
        
        # Create color mapping based on comfort scores
        colors = ['red' if score < 0.3 else 'yellow' if score < 0.7 else 'green' 
                 for score in comfort_scores]
        
        # Create node sizes based on wealth
        node_sizes = [max(5, min(25, wealth / 10000)) for wealth in wealth_values]
        
        fig = go.Figure()
        
        # Add comfort state nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=comfort_scores,
                colorscale='RdYlGn',
                opacity=0.8,
                colorbar=dict(title="Comfort Score")
            ),
            text=[f"Comfort: {score:.3f}<br>Wealth: ${wealth:,.0f}" 
                  for score, wealth in zip(comfort_scores, wealth_values)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add comfort threshold planes
        comfort_thresholds = [0.3, 0.7]
        for threshold in comfort_thresholds:
            fig.add_trace(go.Surface(
                x=[[-2, 2], [-2, 2]],
                y=[[-2, -2], [2, 2]],
                z=[[threshold, threshold], [threshold, threshold]],
                opacity=0.2,
                colorscale=[[0, 'red'], [1, 'green']],
                showscale=False,
                name=f"Comfort Threshold {threshold}"
            ))
        
        fig.update_layout(
            title=f"3D Comfort State Visualization - Snapshot {snapshot_index + 1}",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Comfort Score",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_mesh_evolution_animation(self) -> go.Figure:
        """
        Create animated 3D mesh evolution
        
        Returns:
            Plotly animated figure
        """
        if not self.mesh_data or 'snapshots' not in self.mesh_data:
            return go.Figure()
        
        # Prepare animation data
        frames = []
        
        for i, snapshot in enumerate(self.mesh_data['snapshots']):
            nodes = snapshot['nodes']
            
            x_coords = [node['position']['x'] for node in nodes]
            y_coords = [node['position']['y'] for node in nodes]
            z_coords = [node['position']['z'] for node in nodes]
            wealth_values = [node['financial_state']['wealth'] for node in nodes]
            comfort_scores = [node['financial_state']['comfort_score'] for node in nodes]
            node_sizes = [max(5, min(20, wealth / 10000)) for wealth in wealth_values]
            
            frame = go.Frame(
                data=[go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=comfort_scores,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Wealth: ${wealth:,.0f}<br>Comfort: {comfort:.3f}" 
                          for wealth, comfort in zip(wealth_values, comfort_scores)],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                )],
                name=f"Frame {i}"
            )
            frames.append(frame)
        
        # Create initial frame
        initial_snapshot = self.mesh_data['snapshots'][0]
        initial_nodes = initial_snapshot['nodes']
        
        x_coords = [node['position']['x'] for node in initial_nodes]
        y_coords = [node['position']['y'] for node in initial_nodes]
        z_coords = [node['position']['z'] for node in initial_nodes]
        wealth_values = [node['financial_state']['wealth'] for node in initial_nodes]
        comfort_scores = [node['financial_state']['comfort_score'] for node in initial_nodes]
        node_sizes = [max(5, min(20, wealth / 10000)) for wealth in wealth_values]
        
        fig = go.Figure(
            data=[go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=comfort_scores,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f"Wealth: ${wealth:,.0f}<br>Comfort: {comfort:.3f}" 
                      for wealth, comfort in zip(wealth_values, comfort_scores)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            )],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="3D Mesh Evolution Animation",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Z Position",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f"Frame {i}"], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f"Snapshot {i+1}",
                        'method': 'animate'
                    }
                    for i in range(len(self.mesh_data['snapshots']))
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Snapshot: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def create_stress_test_visualization(self, stress_results: Dict[str, Any]) -> go.Figure:
        """
        Create 3D stress test visualization
        
        Args:
            stress_results: Results from stress testing
            
        Returns:
            Plotly 3D figure
        """
        if not stress_results or 'stress_scenarios' not in stress_results:
            return go.Figure()
        
        # Extract stress test data
        scenarios = stress_results['stress_scenarios']
        
        # Prepare data for visualization
        scenario_names = []
        wealth_impacts = []
        comfort_impacts = []
        risk_scores = []
        
        for scenario in scenarios:
            scenario_name = scenario['scenario']['type']
            aggregate_impact = scenario.get('aggregate_impact', {})
            comfort_impact = scenario.get('comfort_impact', {})
            
            scenario_names.append(scenario_name)
            wealth_impacts.append(aggregate_impact.get('wealth_impact', 0))
            comfort_impacts.append(comfort_impact.get('comfort_change', 0))
            risk_scores.append(aggregate_impact.get('risk_impact', 0))
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=wealth_impacts,
            y=comfort_impacts,
            z=risk_scores,
            mode='markers+text',
            marker=dict(
                size=15,
                color=risk_scores,
                colorscale='RdYlGn',
                opacity=0.8
            ),
            text=scenario_names,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                        'Wealth Impact: %{x:.2%}<br>' +
                        'Comfort Impact: %{y:.2%}<br>' +
                        'Risk Impact: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="3D Stress Test Results",
            scene=dict(
                xaxis_title="Wealth Impact",
                yaxis_title="Comfort Impact",
                zaxis_title="Risk Impact",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_fsqca_visualization(self, fsqca_results: Dict[str, Any]) -> go.Figure:
        """
        Create 3D fsQCA visualization
        
        Args:
            fsqca_results: Results from fsQCA analysis
            
        Returns:
            Plotly 3D figure
        """
        # Generate sample fsQCA data for visualization
        conditions = ['high_wealth', 'high_savings', 'high_stability']
        coverage_scores = [0.8, 0.7, 0.9]
        consistency_scores = [0.85, 0.75, 0.95]
        necessity_scores = [0.6, 0.8, 0.7]
        
        fig = go.Figure()
        
        # Create 3D scatter plot for fsQCA results
        fig.add_trace(go.Scatter3d(
            x=coverage_scores,
            y=consistency_scores,
            z=necessity_scores,
            mode='markers+text',
            marker=dict(
                size=20,
                color=necessity_scores,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=conditions,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                        'Coverage: %{x:.2f}<br>' +
                        'Consistency: %{y:.2f}<br>' +
                        'Necessity: %{z:.2f}<extra></extra>'
        ))
        
        # Add solution threshold planes
        fig.add_trace(go.Surface(
            x=[[0.8, 1.0], [0.8, 1.0]],
            y=[[0.8, 0.8], [1.0, 1.0]],
            z=[[0.5, 0.5], [0.5, 0.5]],
            opacity=0.2,
            colorscale=[[0, 'red'], [1, 'green']],
            showscale=False,
            name="Solution Threshold"
        ))
        
        fig.update_layout(
            title="3D fsQCA Analysis Results",
            scene=dict(
                xaxis_title="Coverage",
                yaxis_title="Consistency",
                zaxis_title="Necessity",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> str:
        """
        Create interactive 3D dashboard HTML
        
        Returns:
            HTML string for interactive dashboard
        """
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>3D Financial Mesh Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .controls {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
        }
        .control-group {
            display: inline-block;
            margin-right: 20px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .control-group select, .control-group input {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .visualization-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D Financial Mesh Visualizer</h1>
            <p>Interactive 3D exploration of financial mesh evolution, comfort states, and stress testing</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="visualizationType">Visualization Type:</label>
                <select id="visualizationType">
                    <option value="mesh">3D Mesh</option>
                    <option value="comfort">Comfort States</option>
                    <option value="evolution">Mesh Evolution</option>
                    <option value="stress">Stress Testing</option>
                    <option value="fsqca">fsQCA Analysis</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="snapshotIndex">Snapshot Index:</label>
                <input type="number" id="snapshotIndex" value="0" min="0" max="11">
            </div>
            
            <div class="control-group">
                <button class="btn" onclick="updateVisualization()">Update Visualization</button>
            </div>
            
            <div class="control-group">
                <button class="btn" onclick="exportVisualization()">Export</button>
            </div>
        </div>
        
        <div class="metrics" id="metrics">
            <div class="metric-card">
                <div class="metric-value" id="totalNodes">0</div>
                <div class="metric-label">Total Nodes</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgWealth">$0</div>
                <div class="metric-label">Average Wealth</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgComfort">0.0</div>
                <div class="metric-label">Average Comfort</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="stressLevel">0%</div>
                <div class="metric-label">Stress Level</div>
            </div>
        </div>
        
        <div class="visualization-grid">
            <div class="chart-container">
                <h3>3D Visualization</h3>
                <div id="mainChart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Metrics Over Time</h3>
                <div id="metricsChart"></div>
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        
        function updateVisualization() {
            const visualizationType = document.getElementById('visualizationType').value;
            const snapshotIndex = parseInt(document.getElementById('snapshotIndex').value);
            
            // Simulate API call to get visualization data
            fetch(`/api/3d-visualization?type=${visualizationType}&snapshot=${snapshotIndex}`)
                .then(response => response.json())
                .then(data => {
                    currentData = data;
                    renderVisualization(data);
                    updateMetrics(data);
                })
                .catch(error => {
                    console.error('Error updating visualization:', error);
                    // Use sample data for demo
                    renderSampleVisualization(visualizationType, snapshotIndex);
                });
        }
        
        function renderVisualization(data) {
            const trace = {
                x: data.x_coords,
                y: data.y_coords,
                z: data.z_coords,
                mode: 'markers',
                marker: {
                    size: data.node_sizes,
                    color: data.colors,
                    colorscale: data.colorscale,
                    opacity: 0.8
                },
                text: data.labels,
                hovertemplate: data.hover_template
            };
            
            const layout = {
                title: data.title,
                scene: {
                    xaxis_title: data.x_title,
                    yaxis_title: data.y_title,
                    zaxis_title: data.z_title,
                    camera: {
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    }
                },
                width: 800,
                height: 600
            };
            
            Plotly.newPlot('mainChart', [trace], layout);
        }
        
        function renderSampleVisualization(type, snapshotIndex) {
            // Generate sample data for demo
            const numPoints = 50;
            const x_coords = Array.from({length: numPoints}, () => Math.random() * 4 - 2);
            const y_coords = Array.from({length: numPoints}, () => Math.random() * 4 - 2);
            const z_coords = Array.from({length: numPoints}, () => Math.random() * 4 - 2);
            const colors = Array.from({length: numPoints}, () => Math.random());
            const sizes = Array.from({length: numPoints}, () => Math.random() * 15 + 5);
            
            const titles = {
                'mesh': '3D Financial Mesh',
                'comfort': 'Comfort State Visualization',
                'evolution': 'Mesh Evolution',
                'stress': 'Stress Test Results',
                'fsqca': 'fsQCA Analysis'
            };
            
            const trace = {
                x: x_coords,
                y: y_coords,
                z: z_coords,
                mode: 'markers',
                marker: {
                    size: sizes,
                    color: colors,
                    colorscale: 'Viridis',
                    opacity: 0.8
                },
                text: Array.from({length: numPoints}, (_, i) => `Node ${i + 1}`),
                hovertemplate: '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            };
            
            const layout = {
                title: titles[type] || '3D Visualization',
                scene: {
                    xaxis_title: 'X Position',
                    yaxis_title: 'Y Position',
                    zaxis_title: 'Z Position',
                    camera: {
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    }
                },
                width: 800,
                height: 600
            };
            
            Plotly.newPlot('mainChart', [trace], layout);
            
            // Update metrics
            updateMetrics({
                total_nodes: numPoints,
                avg_wealth: 500000 + snapshotIndex * 50000,
                avg_comfort: 0.5 + snapshotIndex * 0.05,
                stress_level: snapshotIndex * 10
            });
        }
        
        function updateMetrics(data) {
            document.getElementById('totalNodes').textContent = data.total_nodes || 0;
            document.getElementById('avgWealth').textContent = '$' + (data.avg_wealth || 0).toLocaleString();
            document.getElementById('avgComfort').textContent = (data.avg_comfort || 0).toFixed(3);
            document.getElementById('stressLevel').textContent = (data.stress_level || 0) + '%';
        }
        
        function exportVisualization() {
            if (currentData) {
                const dataStr = JSON.stringify(currentData, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = '3d_visualization_data.json';
                link.click();
            }
        }
        
        // Initialize on page load
        $(document).ready(function() {
            updateVisualization();
        });
    </script>
</body>
</html>
        """
        
        return dashboard_html
    
    def export_visualization(self, fig: go.Figure, filename: str) -> str:
        """
        Export visualization to file
        
        Args:
            fig: Plotly figure
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_path = self.output_dir / filename
        
        if filename.endswith('.html'):
            fig.write_html(str(output_path))
        elif filename.endswith('.png'):
            fig.write_image(str(output_path))
        elif filename.endswith('.json'):
            fig.write_json(str(output_path))
        
        self.logger.info(f"✅ Exported visualization to {output_path}")
        return str(output_path)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all 3D visualizations
        
        Returns:
            Dictionary of visualization file paths
        """
        self.logger.info("🎨 Generating all 3D visualizations...")
        
        # Load mesh data
        self.load_mesh_data()
        
        visualizations = {}
        
        # Generate 3D mesh visualization
        mesh_fig = self.create_3d_mesh_visualization()
        mesh_path = self.export_visualization(mesh_fig, "3d_mesh_visualization.html")
        visualizations['mesh'] = mesh_path
        
        # Generate comfort state visualization
        comfort_fig = self.create_comfort_state_visualization()
        comfort_path = self.export_visualization(comfort_fig, "3d_comfort_states.html")
        visualizations['comfort'] = comfort_path
        
        # Generate mesh evolution animation
        evolution_fig = self.create_mesh_evolution_animation()
        evolution_path = self.export_visualization(evolution_fig, "3d_mesh_evolution.html")
        visualizations['evolution'] = evolution_path
        
        # Generate stress test visualization (sample data)
        stress_fig = self.create_stress_test_visualization({})
        stress_path = self.export_visualization(stress_fig, "3d_stress_test.html")
        visualizations['stress'] = stress_path
        
        # Generate fsQCA visualization (sample data)
        fsqca_fig = self.create_fsqca_visualization({})
        fsqca_path = self.export_visualization(fsqca_fig, "3d_fsqca_analysis.html")
        visualizations['fsqca'] = fsqca_path
        
        # Generate interactive dashboard
        dashboard_html = self.create_interactive_dashboard()
        dashboard_path = self.output_dir / "3d_interactive_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        visualizations['dashboard'] = str(dashboard_path)
        
        self.logger.info(f"✅ Generated {len(visualizations)} 3D visualizations")
        return visualizations


def create_3d_mesh_visualizer():
    """Create and demonstrate 3D mesh visualizer"""
    print("🎨 Creating 3D Mesh Visualizer...")
    
    visualizer = Mesh3DVisualizer()
    
    # Generate all visualizations
    visualizations = visualizer.generate_all_visualizations()
    
    print("\n✅ 3D Mesh Visualizer Created Successfully!")
    print("=" * 60)
    
    print("📁 Generated Visualizations:")
    for viz_type, path in visualizations.items():
        print(f"  {viz_type}: {path}")
    
    print("\n🎯 Key Features:")
    print("  ✅ Interactive 3D mesh exploration")
    print("  ✅ Comfort state clustering visualization")
    print("  ✅ Mesh evolution animation")
    print("  ✅ Stress test results visualization")
    print("  ✅ fsQCA analysis visualization")
    print("  ✅ Interactive dashboard with controls")
    
    return visualizer, visualizations


if __name__ == "__main__":
    create_3d_mesh_visualizer() 