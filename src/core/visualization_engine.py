#!/usr/bin/env python3
"""
Core Visualization Engine Module
Consolidated visualization and dashboard system for the financial advisor
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, session
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import threading
import time
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for dashboard and visualization"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True
    auto_reload: bool = True
    chart_theme: str = 'plotly_white'
    update_interval: int = 300  # 5 minutes

@dataclass
class ChartData:
    """Data structure for chart generation"""
    labels: List[str]
    values: List[float]
    colors: Optional[List[str]] = None
    title: str = ""
    chart_type: str = "bar"

class FinancialVisualizationEngine:
    """Main visualization engine for financial advisor"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize the visualization engine"""
        self.config = config or DashboardConfig()
        self.app = None
        self.is_running = False
        self.dashboard_thread = None
        self.current_data = {}
        
        # Initialize Flask app
        self._initialize_flask_app()
        
        logger.info("✅ Financial Visualization Engine initialized")
    
    def _initialize_flask_app(self):
        """Initialize Flask application"""
        self.app = Flask(__name__)
        self.app.secret_key = 'financial_advisor_secret_key'
        
        # Register routes
        self._register_routes()
        
        logger.info("✅ Flask app initialized")
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/start_analysis', methods=['POST'])
        def start_analysis():
            """Start continuous financial analysis"""
            try:
                client_data = request.json
                session['client_data'] = client_data
                
                return jsonify({
                    "status": "success",
                    "message": "Analysis started successfully"
                })
            except Exception as e:
                logger.error(f"Error starting analysis: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/stop_analysis', methods=['POST'])
        def stop_analysis():
            """Stop continuous financial analysis"""
            try:
                return jsonify({
                    "status": "success",
                    "message": "Analysis stopped successfully"
                })
            except Exception as e:
                logger.error(f"Error stopping analysis: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/get_analysis')
        def get_analysis():
            """Get current analysis results"""
            try:
                if self.current_data:
                    return jsonify({
                        "status": "success",
                        "data": self.current_data
                    })
                else:
                    return jsonify({
                        "status": "no_data",
                        "message": "No analysis data available"
                    })
            except Exception as e:
                logger.error(f"Error getting analysis: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/api/sample_client')
        def get_sample_client():
            """Get sample client data for demonstration"""
            sample_client = {
                "name": "Alex Johnson",
                "age": 32,
                "income": 85000,
                "expenses": 65000,
                "assets": {
                    "cash": 15000,
                    "investments": 45000,
                    "retirement": 25000,
                    "real_estate": 200000
                },
                "liabilities": {
                    "student_loans": 35000,
                    "credit_cards": 8000,
                    "mortgage": 180000
                },
                "personality": {
                    "fear_of_loss": 0.6,
                    "greed_factor": 0.4,
                    "social_pressure": 0.3,
                    "patience": 0.7,
                    "financial_literacy": 0.6
                },
                "goals": ["emergency_fund", "debt_free", "retirement"],
                "risk_tolerance": 0.6,
                "life_stage": "early_career"
            }
            
            return jsonify({
                "status": "success",
                "client_data": sample_client
            })
    
    def start_dashboard(self):
        """Start the dashboard server"""
        self.is_running = True
        self.dashboard_thread = threading.Thread(
            target=self._run_dashboard_server
        )
        self.dashboard_thread.start()
        logger.info(f"🔄 Started dashboard server on {self.config.host}:{self.config.port}")
    
    def stop_dashboard(self):
        """Stop the dashboard server"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join()
        logger.info("⏹️ Stopped dashboard server")
    
    def _run_dashboard_server(self):
        """Run the dashboard server"""
        try:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
        except Exception as e:
            logger.error(f"❌ Dashboard server error: {e}")
    
    def update_dashboard_data(self, analysis_data: Dict):
        """Update dashboard with new analysis data"""
        self.current_data = analysis_data
        logger.info("📊 Updated dashboard data")
    
    def generate_financial_health_chart(self, financial_health: Dict) -> str:
        """Generate financial health chart"""
        try:
            # Create subplot for financial health metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Financial Health Score', 'Net Worth', 'Savings Rate', 'Debt Ratio'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Financial health score (gauge chart)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=financial_health["score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 60], 'color': "yellow"},
                            {'range': [60, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Net worth bar chart
            fig.add_trace(
                go.Bar(
                    x=['Net Worth'],
                    y=[financial_health["net_worth"]],
                    name='Net Worth',
                    marker_color='green'
                ),
                row=1, col=2
            )
            
            # Savings rate bar chart
            fig.add_trace(
                go.Bar(
                    x=['Savings Rate'],
                    y=[financial_health["savings_rate"] * 100],
                    name='Savings Rate (%)',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            # Debt ratio bar chart
            fig.add_trace(
                go.Bar(
                    x=['Debt Ratio'],
                    y=[financial_health["debt_ratio"] * 100],
                    name='Debt Ratio (%)',
                    marker_color='red'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Financial Health Overview",
                height=600,
                showlegend=False
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error generating financial health chart: {e}")
            return f"<p>Error generating chart: {e}</p>"
    
    def generate_capital_allocation_chart(self, capital_allocation: Dict) -> str:
        """Generate capital allocation chart"""
        try:
            # Extract allocation data
            labels = ['Cash', 'Investments', 'Debt Reduction', 'Emergency Fund', 'Discretionary']
            values = [
                capital_allocation.get("cash_allocation", 0),
                capital_allocation.get("investment_allocation", 0),
                capital_allocation.get("debt_reduction", 0),
                capital_allocation.get("emergency_fund", 0),
                capital_allocation.get("discretionary_spending", 0)
            ]
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )])
            
            fig.update_layout(
                title="Capital Allocation",
                height=500
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error generating capital allocation chart: {e}")
            return f"<p>Error generating chart: {e}</p>"
    
    def generate_risk_assessment_chart(self, risk_assessment: Dict) -> str:
        """Generate risk assessment chart"""
        try:
            # Create subplot for risk metrics
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Expected Return vs Risk', 'Risk Scenarios'),
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Expected return vs risk scatter
            scenarios = risk_assessment.get("scenarios", {})
            scenario_names = list(scenarios.keys())
            probabilities = [scenarios[s]["probability"] for s in scenario_names]
            impacts = [scenarios[s]["impact"] for s in scenario_names]
            
            fig.add_trace(
                go.Scatter(
                    x=impacts,
                    y=probabilities,
                    mode='markers+text',
                    text=scenario_names,
                    textposition="top center",
                    marker=dict(size=15, color=probabilities, colorscale='Viridis'),
                    name='Risk Scenarios'
                ),
                row=1, col=1
            )
            
            # Risk scenarios bar chart
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=probabilities,
                    name='Probability',
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Risk Assessment",
                height=500,
                showlegend=False
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error generating risk assessment chart: {e}")
            return f"<p>Error generating chart: {e}</p>"
    
    def generate_behavioral_analysis_chart(self, behavioral_analysis: Dict) -> str:
        """Generate behavioral analysis chart"""
        try:
            # Extract behavioral factors
            behavioral_biases = behavioral_analysis.get("behavioral_biases", {})
            motivation_scores = {
                "Save": behavioral_analysis.get("save_motivation", 0),
                "Invest": behavioral_analysis.get("invest_motivation", 0),
                "Spend": behavioral_analysis.get("spend_motivation", 0)
            }
            
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Motivation Scores', 'Behavioral Biases'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Motivation scores
            fig.add_trace(
                go.Bar(
                    x=list(motivation_scores.keys()),
                    y=list(motivation_scores.values()),
                    name='Motivation',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Behavioral biases
            bias_names = list(behavioral_biases.keys())
            bias_values = list(behavioral_biases.values())
            
            fig.add_trace(
                go.Bar(
                    x=bias_names,
                    y=bias_values,
                    name='Bias Strength',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Behavioral Analysis",
                height=500,
                showlegend=False
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error generating behavioral analysis chart: {e}")
            return f"<p>Error generating chart: {e}</p>"
    
    def generate_comprehensive_dashboard(self, analysis_data: Dict) -> str:
        """Generate comprehensive dashboard with all charts"""
        try:
            # Generate individual charts
            financial_health_chart = self.generate_financial_health_chart(
                analysis_data.get("financial_health", {})
            )
            
            capital_allocation_chart = self.generate_capital_allocation_chart(
                analysis_data.get("capital_allocation", {})
            )
            
            risk_assessment_chart = self.generate_risk_assessment_chart(
                analysis_data.get("risk_assessment", {})
            )
            
            behavioral_analysis_chart = self.generate_behavioral_analysis_chart(
                analysis_data.get("behavioral_analysis", {})
            )
            
            # Combine into comprehensive dashboard
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Financial Advisor Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .chart-container {{ margin: 20px 0; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Financial Advisor Dashboard</h1>
                    <p>Last updated: {analysis_data.get("timestamp", "Unknown")}</p>
                </div>
                
                <div class="grid">
                    <div class="chart-container">
                        <h3>Financial Health</h3>
                        {financial_health_chart}
                    </div>
                    
                    <div class="chart-container">
                        <h3>Capital Allocation</h3>
                        {capital_allocation_chart}
                    </div>
                    
                    <div class="chart-container">
                        <h3>Risk Assessment</h3>
                        {risk_assessment_chart}
                    </div>
                    
                    <div class="chart-container">
                        <h3>Behavioral Analysis</h3>
                        {behavioral_analysis_chart}
                    </div>
                </div>
            </body>
            </html>
            """
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"Error generating comprehensive dashboard: {e}")
            return f"<p>Error generating dashboard: {e}</p>"
    
    def export_chart_to_file(self, chart_html: str, filepath: str):
        """Export chart to HTML file"""
        try:
            with open(filepath, 'w') as f:
                f.write(chart_html)
            logger.info(f"✅ Chart exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
    
    def get_dashboard_config(self) -> Dict:
        """Get current dashboard configuration"""
        return {
            "host": self.config.host,
            "port": self.config.port,
            "debug": self.config.debug,
            "auto_reload": self.config.auto_reload,
            "chart_theme": self.config.chart_theme,
            "update_interval": self.config.update_interval
        }

def main():
    """Test the visualization engine"""
    # Initialize visualization engine
    config = DashboardConfig(port=5001)  # Use different port for testing
    viz_engine = FinancialVisualizationEngine(config)
    
    # Test chart generation
    sample_analysis = {
        "timestamp": datetime.now().isoformat(),
        "financial_health": {
            "score": 75.5,
            "net_worth": 150000,
            "savings_rate": 0.25,
            "debt_ratio": 0.15,
            "income_coverage": 1.4,
            "category": "good"
        },
        "capital_allocation": {
            "cash_allocation": 20000,
            "investment_allocation": 50000,
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
    
    # Generate comprehensive dashboard
    dashboard_html = viz_engine.generate_comprehensive_dashboard(sample_analysis)
    
    # Export to file
    viz_engine.export_chart_to_file(dashboard_html, "test_dashboard.html")
    
    print("✅ Visualization engine test completed")
    print("📊 Dashboard exported to test_dashboard.html")

if __name__ == "__main__":
    main() 