#!/usr/bin/env python3
"""
Visualization Tools Module
Comprehensive collection of visualization tools for the financial advisor system
"""

from .charts.financial_charts import FinancialChartGenerator
from .charts.fingerprint_charts import FingerprintVisualizer
from .charts.portfolio_charts import PortfolioChartGenerator
from .charts.risk_charts import RiskChartGenerator
from .charts.behavioral_charts import BehavioralChartGenerator

from .dashboards.ui_layer import UILayer
from .dashboards.dashboard_generator import DashboardGenerator
from .dashboards.interactive_dashboard import InteractiveDashboard

from .d3d_visualizations.mesh_3d_visualizer import Mesh3DVisualizer
from .d3d_visualizations.topology_visualizer import TopologyVisualizer
from .d3d_visualizations.stress_test_visualizer import StressTestVisualizer

from .utils.chart_utils import ChartUtils
from .utils.color_utils import ColorUtils
from .utils.export_utils import ExportUtils

# Main visualization engine
from .visualization_engine import FinancialVisualizationEngine

__all__ = [
    # Chart generators
    'FinancialChartGenerator',
    'FingerprintVisualizer', 
    'PortfolioChartGenerator',
    'RiskChartGenerator',
    'BehavioralChartGenerator',
    
    # Dashboard components
    'UILayer',
    'DashboardGenerator',
    'InteractiveDashboard',
    
    # 3D visualizations
    'Mesh3DVisualizer',
    'TopologyVisualizer',
    'StressTestVisualizer',
    
    # Utilities
    'ChartUtils',
    'ColorUtils',
    'ExportUtils',
    
    # Main engine
    'FinancialVisualizationEngine'
]

# Version info
__version__ = "1.0.0"
__author__ = "Financial Advisor System"
__description__ = "Comprehensive visualization tools for financial analysis" 