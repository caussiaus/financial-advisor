#!/usr/bin/env python3
"""
Mesh-Market Integration Dashboard

Interactive dashboard showing the integration between:
1. Omega mesh personal finance decisions
2. Market tracking and backtesting
3. Investment action associations
4. Performance attribution
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from mesh_market_integration import MeshMarketIntegration, run_comprehensive_mesh_market_analysis

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

class MeshMarketDashboard:
    """
    Comprehensive dashboard for mesh-market integration visualization
    """
    
    def __init__(self):
        self.integration = MeshMarketIntegration()
        self.results = None
        self.analysis = None
        self.report = None
        self.dashboard_data = None
        
    def load_results(self):
        """Load or generate mesh-market integration results"""
        try:
            # Try to load existing results
            with open('mesh_market_integration_results.json', 'r') as f:
                data = json.load(f)
                self.results = data['backtest_result']
                self.analysis = data['analysis']
                self.report = data['report']
                self.dashboard_data = data['dashboard']
                print("✅ Loaded existing results")
        except FileNotFoundError:
            # Generate new results
            print("🔄 Generating new mesh-market integration results...")
            result, analysis, report, dashboard = run_comprehensive_mesh_market_analysis()
            self.results = result
            self.analysis = analysis
            self.report = report
            self.dashboard_data = dashboard
    
    def create_portfolio_timeline_chart(self) -> go.Figure:
        """Create portfolio timeline chart"""
        if not self.dashboard_data or 'portfolio_timeline' not in self.dashboard_data:
            return go.Figure()
        
        timeline_data = self.dashboard_data['portfolio_timeline']
        dates = [item['date'] for item in timeline_data]
        values = [item['portfolio_value'] for item in timeline_data]
        returns = [item['cumulative_return'] for item in timeline_data]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value Over Time', 'Cumulative Return'),
            vertical_spacing=0.1
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#007bff', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Cumulative return
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=returns,
                mode='lines+markers',
                name='Cumulative Return',
                line=dict(color='#28a745', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Portfolio Performance Timeline',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_action_analysis_chart(self) -> go.Figure:
        """Create action analysis chart"""
        if not self.dashboard_data or 'action_analysis' not in self.dashboard_data:
            return go.Figure()
        
        action_data = self.dashboard_data['action_analysis']
        action_distribution = action_data.get('action_distribution', {})
        decision_effectiveness = action_data.get('decision_effectiveness', {})
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Action Distribution', 'Decision Effectiveness'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Action distribution pie chart
        if action_distribution:
            fig.add_trace(
                go.Pie(
                    labels=list(action_distribution.keys()),
                    values=list(action_distribution.values()),
                    name="Action Distribution"
                ),
                row=1, col=1
            )
        
        # Decision effectiveness bar chart
        if decision_effectiveness:
            fig.add_trace(
                go.Bar(
                    x=list(decision_effectiveness.keys()),
                    y=list(decision_effectiveness.values()),
                    name="Decision Effectiveness",
                    marker_color='#ffc107'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Mesh Action Analysis',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_performance_metrics_chart(self) -> go.Figure:
        """Create performance metrics chart"""
        if not self.dashboard_data or 'performance_metrics' not in self.dashboard_data:
            return go.Figure()
        
        metrics = self.dashboard_data['performance_metrics']
        
        # Create radar chart for key metrics
        categories = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown']
        values = [
            metrics.get('total_return', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('win_rate', 0),
            1 - abs(metrics.get('max_drawdown', 0))  # Invert drawdown for radar
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance Metrics',
            line_color='#007bff'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title='Performance Metrics Overview'
        )
        
        return fig
    
    def create_mesh_insights_chart(self) -> go.Figure:
        """Create mesh insights chart"""
        if not self.dashboard_data or 'mesh_insights' not in self.dashboard_data:
            return go.Figure()
        
        insights = self.dashboard_data['mesh_insights']
        
        # Create metrics cards
        fig = go.Figure()
        
        # Add text annotations for key metrics
        fig.add_annotation(
            x=0.5, y=0.8,
            text=f"Mesh Actions Processed: {insights.get('mesh_actions_processed', 0)}",
            showarrow=False,
            font=dict(size=16, color='#007bff')
        )
        
        fig.add_annotation(
            x=0.5, y=0.6,
            text=f"Decision Success Rate: {insights.get('mesh_decision_success_rate', 0):.1%}",
            showarrow=False,
            font=dict(size=16, color='#28a745')
        )
        
        fig.add_annotation(
            x=0.5, y=0.4,
            text=f"Average Action Confidence: {insights.get('average_action_confidence', 0):.1%}",
            showarrow=False,
            font=dict(size=16, color='#ffc107')
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title='Mesh Integration Insights',
            height=300
        )
        
        return fig
    
    def create_market_stress_chart(self) -> go.Figure:
        """Create market stress analysis chart"""
        # Simulate market stress data over time
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
        market_stress = np.random.normal(0.3, 0.15, len(dates))
        market_stress = np.clip(market_stress, 0, 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=market_stress,
            mode='lines',
            name='Market Stress Index',
            line=dict(color='#dc3545', width=2),
            fill='tonexty'
        ))
        
        # Add stress level zones
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="High Stress")
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                     annotation_text="Low Stress")
        
        fig.update_layout(
            title='Market Stress Analysis',
            xaxis_title='Date',
            yaxis_title='Stress Index',
            height=400
        )
        
        return fig
    
    def create_decision_flow_chart(self) -> go.Figure:
        """Create decision flow diagram"""
        # Create a Sankey diagram showing mesh actions to investment decisions
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = ["Income Increase", "Major Expense", "Debt Payment", 
                        "Milestone Achievement", "Buy Equity", "Buy Bonds", 
                        "Increase Cash", "Rebalance Portfolio"],
                color = "blue"
            ),
            link = dict(
                source = [0, 0, 1, 1, 2, 2, 3, 3],  # Source nodes
                target = [4, 6, 4, 6, 5, 7, 7, 4],  # Target nodes
                value = [0.4, 0.1, 0.3, 0.2, 0.6, 0.4, 0.5, 0.3]  # Flow values
            )
        )])
        
        fig.update_layout(
            title_text="Mesh Action to Investment Decision Flow",
            font_size=10,
            height=500
        )
        
        return fig

# Initialize dashboard
dashboard = MeshMarketDashboard()
dashboard.load_results()

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🌐 Mesh-Market Integration Dashboard", 
                   className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Summary metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Portfolio Performance", className="card-title"),
                    html.P(f"Total Return: {dashboard.results.get('total_return', 0):.2%}", className="card-text"),
                    html.P(f"Sharpe Ratio: {dashboard.results.get('sharpe_ratio', 0):.2f}", className="card-text"),
                    html.P(f"Max Drawdown: {dashboard.results.get('max_drawdown', 0):.2%}", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎯 Mesh Actions", className="card-title"),
                    html.P(f"Actions Processed: {dashboard.dashboard_data.get('action_analysis', {}).get('total_actions', 0)}", className="card-text"),
                    html.P(f"Decisions Made: {dashboard.dashboard_data.get('action_analysis', {}).get('total_decisions', 0)}", className="card-text"),
                    html.P(f"Win Rate: {dashboard.results.get('win_rate', 0):.1%}", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Market Analysis", className="card-title"),
                    html.P(f"Annualized Return: {dashboard.results.get('annualized_return', 0):.2%}", className="card-text"),
                    html.P(f"Volatility: {dashboard.results.get('volatility', 0):.2%}", className="card-text"),
                    html.P(f"Profitable Trades: {dashboard.results.get('profitable_trades', 0)}", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("💡 Recommendations", className="card-title"),
                    html.P("View detailed recommendations in the analysis section", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Main charts
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='portfolio-timeline',
                figure=dashboard.create_portfolio_timeline_chart()
            )
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='action-analysis',
                figure=dashboard.create_action_analysis_chart()
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='performance-metrics',
                figure=dashboard.create_performance_metrics_chart()
            )
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='mesh-insights',
                figure=dashboard.create_mesh_insights_chart()
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='market-stress',
                figure=dashboard.create_market_stress_chart()
            )
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='decision-flow',
                figure=dashboard.create_decision_flow_chart()
            )
        ], width=12)
    ], className="mb-4"),
    
    # Detailed analysis section
    dbc.Row([
        dbc.Col([
            html.H3("📋 Detailed Analysis"),
            html.Div(id='detailed-analysis')
        ], width=12)
    ])
    
], fluid=True)

@app.callback(
    Output('detailed-analysis', 'children'),
    Input('portfolio-timeline', 'clickData')
)
def update_detailed_analysis(click_data):
    """Update detailed analysis based on chart interactions"""
    if not dashboard.report:
        return html.P("No detailed analysis available")
    
    analysis_content = []
    
    # Backtest summary
    if 'backtest_summary' in dashboard.report:
        summary = dashboard.report['backtest_summary']
        analysis_content.append(html.H4("Backtest Summary"))
        for key, value in summary.items():
            analysis_content.append(html.P(f"{key.replace('_', ' ').title()}: {value}"))
    
    # Mesh analysis
    if 'mesh_analysis' in dashboard.report:
        mesh_analysis = dashboard.report['mesh_analysis']
        analysis_content.append(html.H4("Mesh Analysis"))
        for key, value in mesh_analysis.items():
            if isinstance(value, dict):
                analysis_content.append(html.P(f"{key.replace('_', ' ').title()}:"))
                for sub_key, sub_value in value.items():
                    analysis_content.append(html.P(f"  {sub_key}: {sub_value}"))
            else:
                analysis_content.append(html.P(f"{key.replace('_', ' ').title()}: {value}"))
    
    # Recommendations
    if 'recommendations' in dashboard.report:
        recommendations = dashboard.report['recommendations']
        analysis_content.append(html.H4("Recommendations"))
        for i, rec in enumerate(recommendations, 1):
            analysis_content.append(html.P(f"{i}. {rec}"))
    
    return analysis_content

if __name__ == '__main__':
    print("🚀 Starting Mesh-Market Integration Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("💡 Use Ctrl+C to stop the server")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050) 