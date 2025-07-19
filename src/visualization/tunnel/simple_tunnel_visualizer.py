#!/usr/bin/env python3
"""
Simple Tunnel Visualizer - Guaranteed to Work

This creates a simpler but more reliable tunnel visualization
that will definitely produce visible 3D surfaces.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTunnelVisualizer:
    """Simple but reliable tunnel visualization"""
    
    def __init__(self, data_path="outputs/data/horatio_mesh_timelapse.json"):
        self.data_path = Path(data_path)
        self.data = None
        self.time_series_data = None
        self.load_data()
        
    def load_data(self):
        """Load mesh data and prepare time series"""
        if not self.data_path.exists():
            print(f"❌ Mesh data not found at {self.data_path}")
            return
        
        print(f"📊 Loading neural mesh data from {self.data_path}")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"✅ Loaded {len(self.data['snapshots'])} snapshots")
        print(f"📈 Timeline: {self.data['snapshots'][0]['snapshot_time']} to {self.data['snapshots'][-1]['snapshot_time']}")
        
        # Prepare time series data
        self._prepare_time_series_data()
    
    def _prepare_time_series_data(self):
        """Prepare time series data for tunnel visualization"""
        if not self.data or 'snapshots' not in self.data:
            return
        
        print("🔄 Preparing time series data for tunnel visualization...")
        
        # Extract time series data
        time_series = {
            'timestamps': [],
            'total_wealth': [],
            'cash': [],
            'investments': [],
            'probabilities': [],
            'node_ids': []
        }
        
        for snapshot_idx, snapshot in enumerate(self.data['snapshots']):
            timestamp = snapshot['snapshot_time']
            
            for node in snapshot['nodes']:
                time_series['timestamps'].append(snapshot_idx)
                time_series['total_wealth'].append(float(node['financial_state']['total_wealth']))
                time_series['cash'].append(float(node['financial_state']['cash']))
                time_series['investments'].append(float(node['financial_state']['investments']))
                time_series['probabilities'].append(node['probability'])
                time_series['node_ids'].append(node['node_id'])
        
        self.time_series_data = time_series
        
        print(f"✅ Prepared {len(time_series['timestamps'])} data points across {len(self.data['snapshots'])} snapshots")
    
    def create_simple_tunnel_visualization(self, 
                                         time_window: Optional[Tuple[int, int]] = None,
                                         node_filter: Optional[List[str]] = None) -> go.Figure:
        """
        Create simple but reliable tunnel visualization
        
        Args:
            time_window: Optional time window filter
            node_filter: Optional node filter
            
        Returns:
            Plotly figure with 4 tunnel surfaces
        """
        if not self.time_series_data:
            print("❌ No time series data available")
            return None
        
        print("🎨 Creating simple tunnel visualization...")
        
        # Filter data if needed
        filtered_data = self._filter_time_series_data(time_window, node_filter)
        
        # Create 4 subplots for the tunnel surfaces
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Wealth Evolution', 'Cash Flow Evolution',
                           'Investment Portfolio Evolution', 'State Confidence Evolution'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Create simple surfaces
        surfaces = self._create_simple_surfaces(filtered_data)
        
        # Add surfaces to subplots
        for i, surface in enumerate(surfaces):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.add_trace(surface, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Simple Financial State Evolution Tunnels",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1600,
            height=1200,
            showlegend=False,
            scene=dict(
                xaxis_title="Time (Snapshot Index)",
                yaxis_title="Value Range",
                zaxis_title="Total Wealth ($)",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            scene2=dict(
                xaxis_title="Time (Snapshot Index)",
                yaxis_title="Value Range",
                zaxis_title="Cash ($)",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            scene3=dict(
                xaxis_title="Time (Snapshot Index)",
                yaxis_title="Value Range",
                zaxis_title="Investments ($)",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            scene4=dict(
                xaxis_title="Time (Snapshot Index)",
                yaxis_title="Value Range",
                zaxis_title="Probability",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            )
        )
        
        return fig
    
    def _filter_time_series_data(self, time_window: Optional[Tuple[int, int]], 
                                node_filter: Optional[List[str]]) -> Dict[str, List]:
        """Filter time series data based on time window and node filter"""
        data = self.time_series_data.copy()
        
        if time_window:
            start_idx, end_idx = time_window
            mask = [(start_idx <= t <= end_idx) for t in data['timestamps']]
            for key in data:
                data[key] = [val for i, val in enumerate(data[key]) if mask[i]]
        
        if node_filter:
            mask = [node_id in node_filter for node_id in data['node_ids']]
            for key in data:
                data[key] = [val for i, val in enumerate(data[key]) if mask[i]]
        
        return data
    
    def _create_simple_surfaces(self, data: Dict[str, List]) -> List[go.Surface]:
        """Create simple but reliable surfaces"""
        surfaces = []
        
        # Prepare data for each metric
        metrics = [
            ('total_wealth', 'Total Wealth ($)', 'viridis', 'Total Wealth Evolution'),
            ('cash', 'Cash ($)', 'plasma', 'Cash Flow Evolution'),
            ('investments', 'Investments ($)', 'inferno', 'Investment Portfolio Evolution'),
            ('probabilities', 'Probability', 'magma', 'State Confidence Evolution')
        ]
        
        for metric_name, axis_label, color_scale, title in metrics:
            surface = self._create_simple_single_surface(
                data['timestamps'], data[metric_name], 
                axis_label, color_scale, title
            )
            surfaces.append(surface)
        
        return surfaces
    
    def _create_simple_single_surface(self, timestamps: List[int],
                                    values: List[float],
                                    axis_label: str,
                                    color_scale: str,
                                    title: str) -> go.Surface:
        """Create a simple but reliable surface"""
        
        if not timestamps or not values:
            # Return empty surface
            return go.Surface(
                x=np.array([[0]]), y=np.array([[0]]), z=np.array([[0]]),
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label)
            )
        
        # Group by time
        time_groups = {}
        for t, v in zip(timestamps, values):
            if t not in time_groups:
                time_groups[t] = []
            time_groups[t].append(v)
        
        # Create simple surface data
        time_points = sorted(time_groups.keys())
        
        if len(time_points) < 2:
            # Return empty surface if not enough data
            return go.Surface(
                x=np.array([[0]]), y=np.array([[0]]), z=np.array([[0]]),
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label)
            )
        
        # Create simple grid
        X = []
        Y = []
        Z = []
        
        for t in time_points:
            values_at_time = time_groups[t]
            if values_at_time:
                min_val = min(values_at_time)
                max_val = max(values_at_time)
                mean_val = np.mean(values_at_time)
                
                # Create simple range
                value_range = np.linspace(min_val, max_val, 20)
                
                for val in value_range:
                    # Simple linear interpolation
                    tunnel_height = mean_val + (val - mean_val) * 0.1
                    
                    X.append(t)
                    Y.append(val)
                    Z.append(tunnel_height)
        
        # Convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        if len(X) == 0:
            # Return empty surface
            return go.Surface(
                x=np.array([[0]]), y=np.array([[0]]), z=np.array([[0]]),
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label)
            )
        
        # Reshape for surface plotting
        unique_times = np.unique(X)
        unique_values = np.unique(Y)
        
        if len(unique_times) > 1 and len(unique_values) > 1:
            # Create grid
            X_grid, Y_grid = np.meshgrid(unique_times, unique_values)
            Z_grid = np.zeros_like(X_grid)
            
            # Simple interpolation
            for i, t in enumerate(unique_times):
                for j, v in enumerate(unique_values):
                    mask = (X == t) & (Y == v)
                    if np.any(mask):
                        Z_grid[j, i] = np.mean(Z[mask])
                    else:
                        # Simple fallback
                        Z_grid[j, i] = np.mean(Z) if len(Z) > 0 else 0
            
            return go.Surface(
                x=X_grid, y=Y_grid, z=Z_grid,
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label),
                opacity=0.8
            )
        
        # Fallback: scatter surface
        return go.Surface(
            x=np.array([X]), y=np.array([Y]), z=np.array([Z]),
            colorscale=color_scale,
            name=title,
            showscale=True,
            colorbar=dict(title=axis_label)
        )

def main():
    """Main function to demonstrate simple tunnel visualization"""
    print("🚀 Simple Tunnel Visualizer")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = SimpleTunnelVisualizer()
    
    if not visualizer.data:
        print("❌ No mesh data available")
        return
    
    # Create simple tunnel visualization
    print("\n🔄 Creating simple tunnel visualization...")
    fig = visualizer.create_simple_tunnel_visualization()
    
    if fig:
        fig.write_html("simple_tunnel_visualization.html")
        print("✅ Simple tunnel visualization saved")
    
    # Create filtered visualization
    print("\n🔄 Creating filtered tunnel visualization...")
    fig2 = visualizer.create_simple_tunnel_visualization(
        time_window=(0, 50),
        node_filter=['omega_0_0', 'omega_0_1', 'omega_1_0', 'omega_1_1']
    )
    
    if fig2:
        fig2.write_html("filtered_tunnel_visualization.html")
        print("✅ Filtered tunnel visualization saved")
    
    print("\n✅ Simple tunnel visualization complete!")

if __name__ == "__main__":
    main() 