"""
Tunnel Visualization Module

This module provides high-dimensional tunnel visualization capabilities
for financial data analysis and neural mesh exploration.

Components:
- GuaranteedTunnelVisualizer: Simple, reliable 3D scatter plots
- SimpleTunnelVisualizer: Basic tunnel surface generation
- HighDimensionalTunnelVisualizer: Advanced tunnel with statistical interpolation
- TunnelDashboard: Interactive Flask web interface
"""

from .guaranteed_tunnel_visualizer import GuaranteedTunnelVisualizer
from .simple_tunnel_visualizer import SimpleTunnelVisualizer
from .high_dimensional_tunnel_visualizer import HighDimensionalTunnelVisualizer
from .tunnel_dashboard import app as tunnel_dashboard_app

__all__ = [
    'GuaranteedTunnelVisualizer',
    'SimpleTunnelVisualizer', 
    'HighDimensionalTunnelVisualizer',
    'tunnel_dashboard_app'
] 