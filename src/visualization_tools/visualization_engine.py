#!/usr/bin/env python3
"""
Financial Visualization Engine
Comprehensive visualization system for financial advisor
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify, session
import threading
import time
from dataclasses import dataclass

# Import visualization components
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization engine"""
    theme: str = "plotly_white"
    default_width: int = 800
    default_height: int = 600
    export_format: str = "html"  # html, png, pdf, svg
    auto_save: bool = True
    output_dir: str = "data/outputs/visualizations"
    enable_3d: bool = True
    enable_interactive: bool = True

class FinancialVisualizationEngine:
    """Main visualization engine for financial advisor system"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualization engine"""
        self.config = config or VisualizationConfig()
        
        # Initialize components
        self.chart_generator = FinancialChartGenerator()
        self.fingerprint_visualizer = FingerprintVisualizer()
        self.portfolio_charts = PortfolioChartGenerator()
        self.risk_charts = RiskChartGenerator()
        self.behavioral_charts = BehavioralChartGenerator()
        
        self.ui_layer = UILayer()
        self.dashboard_generator = DashboardGenerator()
        self.interactive_dashboard = InteractiveDashboard()
        
        self.mesh_3d_visualizer = Mesh3DVisualizer()
        self.topology_visualizer = TopologyVisualizer()
        self.stress_test_visualizer = StressTestVisualizer()
        
        self.chart_utils = ChartUtils()
        self.color_utils = ColorUtils()
        self.export_utils = ExportUtils()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Financial Visualization Engine initialized")
    
    def create_financial_health_chart(self, financial_data: Dict) -> Union[str, go.Figure]:
        """Create financial health visualization"""
        try:
            return self.chart_generator.create_financial_health_chart(financial_data)
        except Exception as e:
            logger.error(f"Error creating financial health chart: {e}")
            return self._create_error_chart("Financial Health Chart", str(e))
    
    def create_portfolio_allocation_chart(self, portfolio_data: Dict) -> Union[str, go.Figure]:
        """Create portfolio allocation visualization"""
        try:
            return self.portfolio_charts.create_allocation_chart(portfolio_data)
        except Exception as e:
            logger.error(f"Error creating portfolio allocation chart: {e}")
            return self._create_error_chart("Portfolio Allocation Chart", str(e))
    
    def create_risk_assessment_chart(self, risk_data: Dict) -> Union[str, go.Figure]:
        """Create risk assessment visualization"""
        try:
            return self.risk_charts.create_risk_chart(risk_data)
        except Exception as e:
            logger.error(f"Error creating risk assessment chart: {e}")
            return self._create_error_chart("Risk Assessment Chart", str(e))
    
    def create_behavioral_analysis_chart(self, behavioral_data: Dict) -> Union[str, go.Figure]:
        """Create behavioral analysis visualization"""
        try:
            return self.behavioral_charts.create_behavioral_chart(behavioral_data)
        except Exception as e:
            logger.error(f"Error creating behavioral analysis chart: {e}")
            return self._create_error_chart("Behavioral Analysis Chart", str(e))
    
    def create_fingerprint_visualization(self, fingerprint_data: List[Dict]) -> Union[str, go.Figure]:
        """Create financial fingerprint visualization"""
        try:
            return self.fingerprint_visualizer.create_fingerprint_chart(fingerprint_data)
        except Exception as e:
            logger.error(f"Error creating fingerprint visualization: {e}")
            return self._create_error_chart("Fingerprint Visualization", str(e))
    
    def create_3d_mesh_visualization(self, mesh_data: Dict) -> Union[str, go.Figure]:
        """Create 3D mesh visualization"""
        try:
            if not self.config.enable_3d:
                logger.warning("3D visualizations are disabled")
                return self._create_error_chart("3D Mesh Visualization", "3D visualizations disabled")
            
            return self.mesh_3d_visualizer.create_mesh_visualization(mesh_data)
        except Exception as e:
            logger.error(f"Error creating 3D mesh visualization: {e}")
            return self._create_error_chart("3D Mesh Visualization", str(e))
    
    def create_topology_visualization(self, topology_data: Dict) -> Union[str, go.Figure]:
        """Create topology visualization"""
        try:
            if not self.config.enable_3d:
                logger.warning("3D visualizations are disabled")
                return self._create_error_chart("Topology Visualization", "3D visualizations disabled")
            
            return self.topology_visualizer.create_topology_chart(topology_data)
        except Exception as e:
            logger.error(f"Error creating topology visualization: {e}")
            return self._create_error_chart("Topology Visualization", str(e))
    
    def create_stress_test_visualization(self, stress_data: Dict) -> Union[str, go.Figure]:
        """Create stress test visualization"""
        try:
            return self.stress_test_visualizer.create_stress_test_chart(stress_data)
        except Exception as e:
            logger.error(f"Error creating stress test visualization: {e}")
            return self._create_error_chart("Stress Test Visualization", str(e))
    
    def create_comprehensive_dashboard(self, analysis_data: Dict) -> str:
        """Create comprehensive dashboard with all visualizations"""
        try:
            dashboard_html = self.dashboard_generator.create_dashboard(analysis_data)
            
            if self.config.auto_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comprehensive_dashboard_{timestamp}.html"
                filepath = self.output_dir / filename
                self.export_utils.save_html(dashboard_html, filepath)
                logger.info(f"✅ Dashboard saved to {filepath}")
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return self._create_error_dashboard(str(e))
    
    def create_interactive_dashboard(self, analysis_data: Dict, port: int = 5000) -> str:
        """Create and start interactive dashboard"""
        try:
            if not self.config.enable_interactive:
                logger.warning("Interactive dashboards are disabled")
                return self.create_comprehensive_dashboard(analysis_data)
            
            dashboard_url = self.interactive_dashboard.start_dashboard(analysis_data, port)
            logger.info(f"✅ Interactive dashboard started at {dashboard_url}")
            return dashboard_url
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return self.create_comprehensive_dashboard(analysis_data)
    
    def export_visualization(self, visualization: Union[str, go.Figure], 
                           filename: str, format: str = None) -> str:
        """Export visualization to file"""
        try:
            export_format = format or self.config.export_format
            filepath = self.output_dir / f"{filename}.{export_format}"
            
            if isinstance(visualization, go.Figure):
                if export_format == "html":
                    visualization.write_html(str(filepath))
                elif export_format == "png":
                    visualization.write_image(str(filepath))
                elif export_format == "pdf":
                    visualization.write_image(str(filepath))
                else:
                    visualization.write_html(str(filepath))
            else:
                # Assume it's HTML string
                self.export_utils.save_html(visualization, filepath)
            
            logger.info(f"✅ Visualization exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return ""
    
    def batch_create_visualizations(self, data_dict: Dict[str, Any]) -> Dict[str, str]:
        """Create multiple visualizations in batch"""
        results = {}
        
        try:
            # Financial health chart
            if "financial_health" in data_dict:
                results["financial_health"] = self.create_financial_health_chart(
                    data_dict["financial_health"]
                )
            
            # Portfolio allocation chart
            if "portfolio_allocation" in data_dict:
                results["portfolio_allocation"] = self.create_portfolio_allocation_chart(
                    data_dict["portfolio_allocation"]
                )
            
            # Risk assessment chart
            if "risk_assessment" in data_dict:
                results["risk_assessment"] = self.create_risk_assessment_chart(
                    data_dict["risk_assessment"]
                )
            
            # Behavioral analysis chart
            if "behavioral_analysis" in data_dict:
                results["behavioral_analysis"] = self.create_behavioral_analysis_chart(
                    data_dict["behavioral_analysis"]
                )
            
            # Fingerprint visualization
            if "fingerprint_data" in data_dict:
                results["fingerprint"] = self.create_fingerprint_visualization(
                    data_dict["fingerprint_data"]
                )
            
            # 3D mesh visualization
            if "mesh_data" in data_dict:
                results["mesh_3d"] = self.create_3d_mesh_visualization(
                    data_dict["mesh_data"]
                )
            
            # Topology visualization
            if "topology_data" in data_dict:
                results["topology"] = self.create_topology_visualization(
                    data_dict["topology_data"]
                )
            
            # Stress test visualization
            if "stress_test_data" in data_dict:
                results["stress_test"] = self.create_stress_test_visualization(
                    data_dict["stress_test_data"]
                )
            
            logger.info(f"✅ Created {len(results)} visualizations")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch visualization creation: {e}")
            return {"error": str(e)}
    
    def get_available_chart_types(self) -> List[str]:
        """Get list of available chart types"""
        return [
            "financial_health",
            "portfolio_allocation", 
            "risk_assessment",
            "behavioral_analysis",
            "fingerprint",
            "mesh_3d",
            "topology",
            "stress_test"
        ]
    
    def get_chart_config(self, chart_type: str) -> Dict:
        """Get configuration for specific chart type"""
        configs = {
            "financial_health": {
                "width": 800,
                "height": 600,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "portfolio_allocation": {
                "width": 600,
                "height": 400,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "risk_assessment": {
                "width": 800,
                "height": 600,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "behavioral_analysis": {
                "width": 800,
                "height": 600,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "fingerprint": {
                "width": 1000,
                "height": 800,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "mesh_3d": {
                "width": 1000,
                "height": 800,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "topology": {
                "width": 1000,
                "height": 800,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            },
            "stress_test": {
                "width": 800,
                "height": 600,
                "theme": "plotly_white",
                "export_formats": ["html", "png", "pdf"]
            }
        }
        
        return configs.get(chart_type, {})
    
    def _create_error_chart(self, title: str, error_message: str) -> str:
        """Create error chart when visualization fails"""
        error_html = f"""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>{title}</h3>
            <p>❌ Error creating visualization</p>
            <p><small>{error_message}</small></p>
        </div>
        """
        return error_html
    
    def _create_error_dashboard(self, error_message: str) -> str:
        """Create error dashboard when dashboard creation fails"""
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Dashboard - Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 50px; }}
                .error-container {{ text-align: center; padding: 50px; }}
                .error-message {{ color: #d32f2f; }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Financial Dashboard</h1>
                <div class="error-message">
                    <h2>❌ Error Creating Dashboard</h2>
                    <p>{error_message}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return error_html

def main():
    """Test the visualization engine"""
    # Initialize visualization engine
    config = VisualizationConfig(
        theme="plotly_white",
        default_width=800,
        default_height=600,
        export_format="html",
        auto_save=True,
        output_dir="data/outputs/visualizations",
        enable_3d=True,
        enable_interactive=True
    )
    
    viz_engine = FinancialVisualizationEngine(config)
    
    # Test data
    test_data = {
        "financial_health": {
            "score": 75.5,
            "net_worth": 150000,
            "savings_rate": 0.25,
            "debt_ratio": 0.15
        },
        "portfolio_allocation": {
            "cash": 20000,
            "investments": 50000,
            "debt_reduction": 15000,
            "emergency_fund": 25000,
            "discretionary_spending": 10000
        },
        "risk_assessment": {
            "expected_return": 0.08,
            "volatility": 0.15,
            "scenarios": {
                "recession": {"probability": 0.15, "impact": -0.25},
                "market_correction": {"probability": 0.25, "impact": -0.10},
                "normal_growth": {"probability": 0.45, "impact": 0.08},
                "bull_market": {"probability": 0.15, "impact": 0.20}
            }
        },
        "behavioral_analysis": {
            "save_motivation": 0.7,
            "invest_motivation": 0.6,
            "spend_motivation": 0.3,
            "behavioral_biases": {
                "loss_aversion": 0.6,
                "overconfidence": 0.4,
                "herding": 0.3,
                "anchoring": 0.4
            }
        }
    }
    
    # Create batch visualizations
    results = viz_engine.batch_create_visualizations(test_data)
    
    print("✅ Visualization engine test completed")
    print(f"📊 Created {len(results)} visualizations")
    
    # Export one visualization as example
    if "financial_health" in results:
        filepath = viz_engine.export_visualization(
            results["financial_health"], 
            "test_financial_health"
        )
        print(f"📁 Exported to: {filepath}")

if __name__ == "__main__":
    main() 