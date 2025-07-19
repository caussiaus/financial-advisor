#!/usr/bin/env python3
"""
High Dimensional Tunnel Visualizer

This module creates a high-dimensional visualization with 4 surface graphs where:
- X-axis is time (snapshot progression)
- Y-axis is the range for each financial metric
- Z-axis is the value of the financial metric
- Each surface represents a "tunnel" in the volume
- Color coding shows the state/evolution of financial data

The 4 surfaces represent:
1. Total Wealth Tunnel
2. Cash Tunnel  
3. Investments Tunnel
4. Probability Tunnel (state confidence)
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

class HighDimensionalTunnelVisualizer:
    """High-dimensional tunnel visualization for neural mesh evolution"""
    
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
    
    def create_tunnel_surfaces(self, 
                              time_window: Optional[Tuple[int, int]] = None,
                              node_filter: Optional[List[str]] = None) -> go.Figure:
        """
        Create 4 tunnel surfaces showing financial evolution over time
        
        Args:
            time_window: Optional tuple (start_snapshot, end_snapshot) to filter time range
            node_filter: Optional list of node IDs to include
            
        Returns:
            Plotly figure with 4 tunnel surfaces
        """
        if not self.time_series_data:
            print("❌ No time series data available")
            return None
        
        print("🎨 Creating high-dimensional tunnel visualization...")
        
        # Filter data if needed
        filtered_data = self._filter_time_series_data(time_window, node_filter)
        
        # Create 4 subplots for the tunnel surfaces
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Wealth Tunnel', 'Cash Tunnel', 
                           'Investments Tunnel', 'Probability Tunnel'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Create tunnel surfaces
        surfaces = self._create_tunnel_surfaces(filtered_data)
        
        # Add surfaces to subplots
        for i, surface in enumerate(surfaces):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.add_trace(surface, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="High-Dimensional Financial Tunnel Visualization",
            width=1400,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis_title="Time (Snapshot)",
                yaxis_title="Value Range",
                zaxis_title="Financial Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene2=dict(
                xaxis_title="Time (Snapshot)",
                yaxis_title="Value Range", 
                zaxis_title="Financial Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene3=dict(
                xaxis_title="Time (Snapshot)",
                yaxis_title="Value Range",
                zaxis_title="Financial Value", 
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene4=dict(
                xaxis_title="Time (Snapshot)",
                yaxis_title="Value Range",
                zaxis_title="Probability Value",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
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
    
    def _create_tunnel_surfaces(self, data: Dict[str, List]) -> List[go.Surface]:
        """Create 4 tunnel surfaces from the data"""
        surfaces = []
        
        # 1. Total Wealth Tunnel
        wealth_surface = self._create_single_tunnel_surface(
            data['timestamps'], data['total_wealth'], 
            'Total Wealth ($)', 'viridis', 'Total Wealth Evolution'
        )
        surfaces.append(wealth_surface)
        
        # 2. Cash Tunnel
        cash_surface = self._create_single_tunnel_surface(
            data['timestamps'], data['cash'],
            'Cash ($)', 'plasma', 'Cash Evolution'
        )
        surfaces.append(cash_surface)
        
        # 3. Investments Tunnel
        investments_surface = self._create_single_tunnel_surface(
            data['timestamps'], data['investments'],
            'Investments ($)', 'inferno', 'Investments Evolution'
        )
        surfaces.append(investments_surface)
        
        # 4. Probability Tunnel
        prob_surface = self._create_single_tunnel_surface(
            data['timestamps'], data['probabilities'],
            'Probability', 'magma', 'State Confidence Evolution'
        )
        surfaces.append(prob_surface)
        
        return surfaces
    
    def _create_single_tunnel_surface(self, timestamps: List[int], 
                                     values: List[float], 
                                     value_name: str,
                                     color_scale: str,
                                     title: str) -> go.Surface:
        """Create a single tunnel surface"""
        
        # Create time-value pairs and sort by time
        time_value_pairs = list(zip(timestamps, values))
        time_value_pairs.sort(key=lambda x: x[0])
        
        # Group by time to create tunnel structure
        time_groups = {}
        for t, v in time_value_pairs:
            if t not in time_groups:
                time_groups[t] = []
            time_groups[t].append(v)
        
        # Create tunnel surface data
        time_points = sorted(time_groups.keys())
        tunnel_data = []
        
        for t in time_points:
            values_at_time = time_groups[t]
            if values_at_time:
                # Create a range of values to form the tunnel
                min_val = min(values_at_time)
                max_val = max(values_at_time)
                mean_val = np.mean(values_at_time)
                
                # Create tunnel cross-section
                num_points = 20
                value_range = np.linspace(min_val, max_val, num_points)
                
                # Add some variation to create tunnel effect
                for i, val in enumerate(value_range):
                    # Add noise to create tunnel shape
                    noise = np.random.normal(0, (max_val - min_val) * 0.1)
                    tunnel_data.append([t, val, mean_val + noise])
        
        # Convert to numpy arrays for surface plotting
        if tunnel_data:
            tunnel_array = np.array(tunnel_data)
            x = tunnel_array[:, 0]
            y = tunnel_array[:, 1] 
            z = tunnel_array[:, 2]
            
            # Reshape for surface plotting
            unique_times = np.unique(x)
            unique_values = np.unique(y)
            
            if len(unique_times) > 1 and len(unique_values) > 1:
                # Create grid for surface
                X, Y = np.meshgrid(unique_times, unique_values)
                Z = np.zeros_like(X)
                
                # Interpolate Z values
                for i, t in enumerate(unique_times):
                    for j, v in enumerate(unique_values):
                        mask = (x == t) & (y == v)
                        if np.any(mask):
                            Z[j, i] = np.mean(z[mask])
                        else:
                            # Interpolate from nearby points
                            nearby_mask = (np.abs(x - t) <= 1) & (np.abs(y - v) <= (max_val - min_val) * 0.1)
                            if np.any(nearby_mask):
                                Z[j, i] = np.mean(z[nearby_mask])
                
                return go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale=color_scale,
                    name=title,
                    showscale=True,
                    colorbar=dict(title=value_name)
                )
        
        # Fallback: simple scatter surface
        return go.Surface(
            x=np.array([x]), y=np.array([y]), z=np.array([z]),
            colorscale=color_scale,
            name=title,
            showscale=True,
            colorbar=dict(title=value_name)
        )
    
    def create_enhanced_tunnel_visualization(self, 
                                           time_window: Optional[Tuple[int, int]] = None,
                                           node_filter: Optional[List[str]] = None,
                                           tunnel_resolution: int = 50) -> go.Figure:
        """
        Create enhanced tunnel visualization with better surface generation
        
        Args:
            time_window: Optional time window filter
            node_filter: Optional node filter
            tunnel_resolution: Resolution for tunnel surface generation
            
        Returns:
            Enhanced tunnel visualization figure
        """
        if not self.time_series_data:
            print("❌ No time series data available")
            return None
        
        print("🎨 Creating enhanced tunnel visualization...")
        
        # Filter data
        filtered_data = self._filter_time_series_data(time_window, node_filter)
        
        # Create enhanced surfaces
        surfaces = self._create_enhanced_tunnel_surfaces(filtered_data, tunnel_resolution)
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Wealth Evolution Tunnel', 'Cash Flow Tunnel',
                           'Investment Portfolio Tunnel', 'State Confidence Tunnel'),
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add surfaces
        for i, surface in enumerate(surfaces):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.add_trace(surface, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "High-Dimensional Financial State Evolution Tunnels",
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
    
    def _create_enhanced_tunnel_surfaces(self, data: Dict[str, List], 
                                       resolution: int) -> List[go.Surface]:
        """Create enhanced tunnel surfaces with better interpolation"""
        surfaces = []
        
        # Prepare data for each metric
        metrics = [
            ('total_wealth', 'Total Wealth ($)', 'viridis', 'Total Wealth Evolution'),
            ('cash', 'Cash ($)', 'plasma', 'Cash Flow Evolution'),
            ('investments', 'Investments ($)', 'inferno', 'Investment Portfolio Evolution'),
            ('probabilities', 'Probability', 'magma', 'State Confidence Evolution')
        ]
        
        for metric_name, axis_label, color_scale, title in metrics:
            surface = self._create_enhanced_single_tunnel(
                data['timestamps'], data[metric_name], 
                axis_label, color_scale, title, resolution
            )
            surfaces.append(surface)
        
        return surfaces
    
    def _create_enhanced_single_tunnel(self, timestamps: List[int],
                                     values: List[float],
                                     axis_label: str,
                                     color_scale: str,
                                     title: str,
                                     resolution: int) -> go.Surface:
        """Create enhanced single tunnel surface with better interpolation"""
        
        # Create time-value pairs
        time_value_pairs = list(zip(timestamps, values))
        
        if not time_value_pairs:
            return go.Surface()
        
        # Group by time
        time_groups = {}
        for t, v in time_value_pairs:
            if t not in time_groups:
                time_groups[t] = []
            time_groups[t].append(v)
        
        # Create enhanced tunnel surface
        time_points = sorted(time_groups.keys())
        
        if len(time_points) < 2:
            return go.Surface()
        
        # Create tunnel cross-sections
        X, Y, Z = [], [], []
        
        for t in time_points:
            values_at_time = time_groups[t]
            if values_at_time:
                min_val = min(values_at_time)
                max_val = max(values_at_time)
                mean_val = np.mean(values_at_time)
                std_val = max(np.std(values_at_time), (max_val - min_val) * 0.1)  # Ensure non-zero std
                
                # Create tunnel cross-section with Gaussian distribution
                value_range = np.linspace(min_val - std_val, max_val + std_val, resolution)
                
                for val in value_range:
                    # Create tunnel shape using Gaussian distribution
                    distance_from_mean = abs(val - mean_val)
                    tunnel_height = mean_val * np.exp(-(distance_from_mean ** 2) / (2 * std_val ** 2))
                    
                    # Add some variation for realistic tunnel effect
                    noise = np.random.normal(0, std_val * 0.05)
                    tunnel_height += noise
                    
                    # Ensure we have valid values
                    if not np.isnan(tunnel_height) and not np.isinf(tunnel_height):
                        X.append(t)
                        Y.append(val)
                        Z.append(tunnel_height)
        
        # Convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        # Reshape for surface plotting
        unique_times = np.unique(X)
        unique_values = np.unique(Y)
        
        if len(unique_times) > 1 and len(unique_values) > 1:
            # Create grid
            X_grid, Y_grid = np.meshgrid(unique_times, unique_values)
            Z_grid = np.zeros_like(X_grid)
            
            # Interpolate Z values
            for i, t in enumerate(unique_times):
                for j, v in enumerate(unique_values):
                    mask = (X == t) & (Y == v)
                    if np.any(mask):
                        z_mean = np.mean(Z[mask])
                        Z_grid[j, i] = z_mean if not np.isnan(z_mean) else 0
                    else:
                        # Interpolate from nearby points
                        nearby_mask = (np.abs(X - t) <= 1) & (np.abs(Y - v) <= np.std(Y) * 0.2)
                        if np.any(nearby_mask):
                            z_mean = np.mean(Z[nearby_mask])
                            Z_grid[j, i] = z_mean if not np.isnan(z_mean) else 0
                        else:
                            Z_grid[j, i] = 0
            
            return go.Surface(
                x=X_grid, y=Y_grid, z=Z_grid,
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label),
                opacity=0.8
            )
        
        # Fallback: scatter surface
        if len(X) > 0:
            return go.Surface(
                x=np.array([X]), y=np.array([Y]), z=np.array([Z]),
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label)
            )
        else:
            # Empty surface if no data
            return go.Surface(
                x=np.array([[0]]), y=np.array([[0]]), z=np.array([[0]]),
                colorscale=color_scale,
                name=title,
                showscale=True,
                colorbar=dict(title=axis_label)
            )

def main():
    """Main function to demonstrate high-dimensional tunnel visualization"""
    print("🚀 High-Dimensional Tunnel Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = HighDimensionalTunnelVisualizer()
    
    if not visualizer.data:
        print("❌ No mesh data available")
        return
    
    # Create basic tunnel visualization
    print("\n🔄 Creating basic tunnel visualization...")
    fig1 = visualizer.create_tunnel_surfaces()
    
    if fig1:
        fig1.write_html("basic_tunnel_visualization.html")
        print("✅ Basic tunnel visualization saved")
    
    # Create enhanced tunnel visualization
    print("\n🔄 Creating enhanced tunnel visualization...")
    fig2 = visualizer.create_enhanced_tunnel_visualization()
    
    if fig2:
        fig2.write_html("enhanced_tunnel_visualization.html")
        print("✅ Enhanced tunnel visualization saved")
    
    # Create time-windowed visualization
    print("\n🔄 Creating time-windowed tunnel visualization...")
    fig3 = visualizer.create_enhanced_tunnel_visualization(
        time_window=(0, 50)  # First 50 snapshots
    )
    
    if fig3:
        fig3.write_html("time_windowed_tunnel_visualization.html")
        print("✅ Time-windowed tunnel visualization saved")
    
    print("\n✅ High-dimensional tunnel visualization complete!")

if __name__ == "__main__":
    main() 