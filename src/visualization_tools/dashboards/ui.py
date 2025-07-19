"""
UI Layer

Responsible for:
- Web interface and visualization
- Dashboard generation
- Interactive charts and graphs
- User interaction handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Protocol
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import os


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    chart_type: str  # 'line', 'bar', 'scatter', 'pie', 'heatmap'
    title: str
    x_axis: str
    y_axis: str
    width: int = 800
    height: int = 600
    template: str = "plotly_white"
    metadata: Dict = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation"""
    dashboard_id: str
    title: str
    layout: str  # 'grid', 'tabs', 'accordion'
    charts: List[ChartConfig]
    refresh_interval: int = 300  # seconds
    metadata: Dict = field(default_factory=dict)


class ChartGenerator(Protocol):
    """Protocol for chart generation capabilities"""
    
    def generate_chart(self, data: Dict, config: ChartConfig) -> go.Figure:
        """Generate a chart from data and config"""
        ...


class DashboardBuilder(Protocol):
    """Protocol for dashboard building capabilities"""
    
    def build_dashboard(self, data: Dict, config: DashboardConfig) -> str:
        """Build dashboard HTML"""
        ...


class UILayer:
    """
    UI Layer - Clean API for web interface and visualization
    
    Responsibilities:
    - Web interface and visualization
    - Dashboard generation
    - Interactive charts and graphs
    - User interaction handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.app = Flask(__name__)
        self.chart_templates = self._initialize_chart_templates()
        self.dashboard_templates = self._initialize_dashboard_templates()
        
        # Register routes
        self._register_routes()
    
    def _initialize_chart_templates(self) -> Dict[str, ChartConfig]:
        """Initialize chart templates"""
        templates = {}
        
        # Portfolio allocation pie chart
        templates['portfolio_allocation'] = ChartConfig(
            chart_type='pie',
            title='Portfolio Allocation',
            x_axis='',
            y_axis='',
            width=600,
            height=400
        )
        
        # Net worth timeline
        templates['net_worth_timeline'] = ChartConfig(
            chart_type='line',
            title='Net Worth Over Time',
            x_axis='Date',
            y_axis='Net Worth ($)',
            width=800,
            height=400
        )
        
        # Risk analysis heatmap
        templates['risk_analysis'] = ChartConfig(
            chart_type='heatmap',
            title='Risk Analysis Matrix',
            x_axis='Asset Class',
            y_axis='Risk Metric',
            width=600,
            height=400
        )
        
        # Commutator sequence visualization
        templates['commutator_sequence'] = ChartConfig(
            chart_type='bar',
            title='Recommended Commutator Sequence',
            x_axis='Step',
            y_axis='Impact',
            width=800,
            height=400
        )
        
        return templates
    
    def _initialize_dashboard_templates(self) -> Dict[str, DashboardConfig]:
        """Initialize dashboard templates"""
        templates = {}
        
        # Main financial dashboard
        templates['main_dashboard'] = DashboardConfig(
            dashboard_id="main_dashboard",
            title="Financial Planning Dashboard",
            layout="grid",
            charts=[
                self.chart_templates['portfolio_allocation'],
                self.chart_templates['net_worth_timeline'],
                self.chart_templates['risk_analysis'],
                self.chart_templates['commutator_sequence']
            ]
        )
        
        # Mesh analysis dashboard
        templates['mesh_dashboard'] = DashboardConfig(
            dashboard_id="mesh_dashboard",
            title="Stochastic Mesh Analysis",
            layout="tabs",
            charts=[
                self.chart_templates['net_worth_timeline'],
                self.chart_templates['risk_analysis']
            ]
        )
        
        return templates
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('index.html')
        
        @self.app.route('/api/dashboard/<dashboard_id>')
        def get_dashboard(dashboard_id):
            """Get dashboard data"""
            if dashboard_id not in self.dashboard_templates:
                return jsonify({'error': 'Dashboard not found'}), 404
            
            config = self.dashboard_templates[dashboard_id]
            return jsonify({
                'dashboard_id': dashboard_id,
                'title': config.title,
                'layout': config.layout,
                'charts': [chart.__dict__ for chart in config.charts]
            })
        
        @self.app.route('/api/chart/<chart_type>')
        def get_chart_data(chart_type):
            """Get chart data"""
            if chart_type not in self.chart_templates:
                return jsonify({'error': 'Chart type not found'}), 404
            
            # Generate sample data based on chart type
            data = self._generate_sample_data(chart_type)
            return jsonify(data)
    
    def _generate_sample_data(self, chart_type: str) -> Dict:
        """Generate sample data for charts"""
        if chart_type == 'portfolio_allocation':
            return {
                'labels': ['Cash', 'Bonds', 'Stocks', 'Real Estate', 'Commodities'],
                'values': [20, 30, 35, 10, 5],
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            }
        
        elif chart_type == 'net_worth_timeline':
            dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
            values = [100000 + i * 1000 + np.random.normal(0, 5000) for i in range(len(dates))]
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'values': values
            }
        
        elif chart_type == 'risk_analysis':
            return {
                'x_labels': ['Cash', 'Bonds', 'Stocks', 'Real Estate'],
                'y_labels': ['Risk', 'Liquidity', 'Return'],
                'values': [
                    [0.0, 0.2, 0.6, 0.4],  # Risk
                    [1.0, 0.8, 0.9, 0.3],  # Liquidity
                    [0.02, 0.04, 0.08, 0.06]  # Return
                ]
            }
        
        elif chart_type == 'commutator_sequence':
            return {
                'steps': ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'],
                'impacts': [0.1, 0.15, 0.08, 0.12, 0.05],
                'descriptions': [
                    'Increase cash position',
                    'Reduce stock allocation',
                    'Add bond exposure',
                    'Rebalance real estate',
                    'Final optimization'
                ]
            }
        
        return {}
    
    def generate_portfolio_chart(self, allocation_data: Dict[str, float]) -> go.Figure:
        """
        Generate portfolio allocation pie chart
        
        Args:
            allocation_data: Asset allocation data
            
        Returns:
            Plotly figure
        """
        labels = list(allocation_data.keys())
        values = list(allocation_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            width=600,
            height=400,
            showlegend=True
        )
        
        return fig
    
    def generate_net_worth_chart(self, timeline_data: List[Dict]) -> go.Figure:
        """
        Generate net worth timeline chart
        
        Args:
            timeline_data: Timeline data with dates and net worth values
            
        Returns:
            Plotly figure
        """
        dates = [item['timestamp'] for item in timeline_data]
        net_worth = [item['net_worth'] for item in timeline_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=net_worth,
            mode='lines+markers',
            name='Net Worth',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Net Worth Over Time",
            xaxis_title="Date",
            yaxis_title="Net Worth ($)",
            width=800,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def generate_risk_analysis_chart(self, risk_data: Dict) -> go.Figure:
        """
        Generate risk analysis heatmap
        
        Args:
            risk_data: Risk analysis data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=risk_data['values'],
            x=risk_data['x_labels'],
            y=risk_data['y_labels'],
            colorscale='RdYlGn_r',
            text=risk_data['values'],
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Risk Analysis Matrix",
            xaxis_title="Asset Class",
            yaxis_title="Risk Metric",
            width=600,
            height=400
        )
        
        return fig
    
    def generate_commutator_chart(self, commutator_data: Dict) -> go.Figure:
        """
        Generate commutator sequence chart
        
        Args:
            commutator_data: Commutator sequence data
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=commutator_data['steps'],
            y=commutator_data['impacts'],
            text=commutator_data['descriptions'],
            textposition='auto',
            name='Impact',
            marker_color='#2ca02c'
        ))
        
        fig.update_layout(
            title="Recommended Commutator Sequence",
            xaxis_title="Step",
            yaxis_title="Impact",
            width=800,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def build_dashboard(self, dashboard_data: Dict, 
                       dashboard_type: str = 'main_dashboard') -> str:
        """
        Build complete dashboard HTML
        
        Args:
            dashboard_data: Data for dashboard
            dashboard_type: Type of dashboard to build
            
        Returns:
            Dashboard HTML string
        """
        if dashboard_type not in self.dashboard_templates:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        
        config = self.dashboard_templates[dashboard_type]
        
        # Generate charts
        charts_html = []
        for chart_config in config.charts:
            if chart_config.chart_type == 'pie':
                fig = self.generate_portfolio_chart(dashboard_data.get('allocation', {}))
            elif chart_config.chart_type == 'line':
                fig = self.generate_net_worth_chart(dashboard_data.get('timeline', []))
            elif chart_config.chart_type == 'heatmap':
                fig = self.generate_risk_analysis_chart(dashboard_data.get('risk_analysis', {}))
            elif chart_config.chart_type == 'bar':
                fig = self.generate_commutator_chart(dashboard_data.get('commutators', {}))
            else:
                continue
            
            charts_html.append(fig.to_html(full_html=False))
        
        # Build dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .chart-container {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .dashboard-title {{ text-align: center; color: #333; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1 class="dashboard-title">{config.title}</h1>
                <div class="chart-grid">
                    {''.join([f'<div class="chart-container">{chart}</div>' for chart in charts_html])}
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def create_interactive_dashboard(self, data_sources: Dict) -> str:
        """
        Create interactive dashboard with real-time data
        
        Args:
            data_sources: Dictionary of data sources
            
        Returns:
            Interactive dashboard HTML
        """
        # Create interactive dashboard with JavaScript
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Financial Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .dashboard { max-width: 1400px; margin: 0 auto; }
                .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .control-group { display: inline-block; margin-right: 20px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                select, input { padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
                button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>Interactive Financial Dashboard</h1>
                    <p>Real-time portfolio analysis and recommendations</p>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <label for="risk-preference">Risk Preference:</label>
                        <select id="risk-preference">
                            <option value="conservative">Conservative</option>
                            <option value="moderate" selected>Moderate</option>
                            <option value="aggressive">Aggressive</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label for="time-horizon">Time Horizon (years):</label>
                        <input type="number" id="time-horizon" value="10" min="1" max="30">
                    </div>
                    
                    <div class="control-group">
                        <button onclick="updateDashboard()">Update Analysis</button>
                    </div>
                </div>
                
                <div class="chart-grid">
                    <div class="chart-container">
                        <h3>Portfolio Allocation</h3>
                        <div id="allocation-chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Net Worth Timeline</h3>
                        <div id="timeline-chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Risk Analysis</h3>
                        <div id="risk-chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Recommended Actions</h3>
                        <div id="recommendations-chart"></div>
                    </div>
                </div>
            </div>
            
            <script>
                function updateDashboard() {
                    const riskPreference = document.getElementById('risk-preference').value;
                    const timeHorizon = document.getElementById('time-horizon').value;
                    
                    // Simulate API call to update data
                    fetch(`/api/dashboard/update?risk=${riskPreference}&horizon=${timeHorizon}`)
                        .then(response => response.json())
                        .then(data => {
                            updateCharts(data);
                        })
                        .catch(error => {
                            console.error('Error updating dashboard:', error);
                        });
                }
                
                function updateCharts(data) {
                    // Update allocation chart
                    if (data.allocation) {
                        const allocationData = [{
                            values: Object.values(data.allocation),
                            labels: Object.keys(data.allocation),
                            type: 'pie',
                            hole: 0.3
                        }];
                        
                        Plotly.newPlot('allocation-chart', allocationData, {
                            title: 'Portfolio Allocation',
                            height: 400
                        });
                    }
                    
                    // Update timeline chart
                    if (data.timeline) {
                        const timelineData = [{
                            x: data.timeline.dates,
                            y: data.timeline.values,
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Net Worth'
                        }];
                        
                        Plotly.newPlot('timeline-chart', timelineData, {
                            title: 'Net Worth Over Time',
                            xaxis: { title: 'Date' },
                            yaxis: { title: 'Net Worth ($)' },
                            height: 400
                        });
                    }
                }
                
                // Initialize dashboard on load
                document.addEventListener('DOMContentLoaded', function() {
                    updateDashboard();
                });
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def export_dashboard(self, dashboard_html: str, filepath: str):
        """Export dashboard to HTML file"""
        with open(filepath, 'w') as f:
            f.write(dashboard_html)
    
    def run_server(self, host: str = 'localhost', port: int = 5000, debug: bool = False):
        """Run the Flask server"""
        self.app.run(host=host, port=port, debug=debug) 