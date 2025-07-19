#!/usr/bin/env python3
"""
Financial Chart Generator
Creates various financial visualization charts
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialChartGenerator:
    """Generator for financial visualization charts"""
    
    def __init__(self):
        """Initialize the financial chart generator"""
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
        
        logger.info("✅ Financial Chart Generator initialized")
    
    def create_financial_health_chart(self, financial_data: Dict) -> Union[str, go.Figure]:
        """Create comprehensive financial health chart"""
        try:
            # Create subplot for financial health metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Financial Health Score', 'Net Worth', 'Savings Rate', 'Debt Ratio'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Financial health score (gauge chart)
            health_score = financial_data.get("score", 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.default_colors['primary']},
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
            net_worth = financial_data.get("net_worth", 0)
            fig.add_trace(
                go.Bar(
                    x=['Net Worth'],
                    y=[net_worth],
                    name='Net Worth',
                    marker_color=self.default_colors['success']
                ),
                row=1, col=2
            )
            
            # Savings rate bar chart
            savings_rate = financial_data.get("savings_rate", 0) * 100
            fig.add_trace(
                go.Bar(
                    x=['Savings Rate'],
                    y=[savings_rate],
                    name='Savings Rate (%)',
                    marker_color=self.default_colors['info']
                ),
                row=2, col=1
            )
            
            # Debt ratio bar chart
            debt_ratio = financial_data.get("debt_ratio", 0) * 100
            fig.add_trace(
                go.Bar(
                    x=['Debt Ratio'],
                    y=[debt_ratio],
                    name='Debt Ratio (%)',
                    marker_color=self.default_colors['warning']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Financial Health Overview",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating financial health chart: {e}")
            return self._create_error_chart("Financial Health Chart", str(e))
    
    def create_net_worth_timeline(self, timeline_data: List[Dict]) -> Union[str, go.Figure]:
        """Create net worth timeline chart"""
        try:
            dates = [item.get('timestamp', item.get('date', '')) for item in timeline_data]
            net_worth = [item.get('net_worth', 0) for item in timeline_data]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=net_worth,
                mode='lines+markers',
                name='Net Worth',
                line=dict(color=self.default_colors['primary'], width=2),
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
            
        except Exception as e:
            logger.error(f"Error creating net worth timeline: {e}")
            return self._create_error_chart("Net Worth Timeline", str(e))
    
    def create_income_expense_chart(self, income_expense_data: Dict) -> Union[str, go.Figure]:
        """Create income vs expense chart"""
        try:
            categories = list(income_expense_data.keys())
            values = list(income_expense_data.values())
            
            # Separate income and expenses
            income_values = [v if v > 0 else 0 for v in values]
            expense_values = [abs(v) if v < 0 else 0 for v in values]
            
            fig = go.Figure()
            
            # Add income bars
            fig.add_trace(go.Bar(
                x=categories,
                y=income_values,
                name='Income',
                marker_color=self.default_colors['success']
            ))
            
            # Add expense bars
            fig.add_trace(go.Bar(
                x=categories,
                y=expense_values,
                name='Expenses',
                marker_color=self.default_colors['warning']
            ))
            
            fig.update_layout(
                title="Income vs Expenses",
                xaxis_title="Category",
                yaxis_title="Amount ($)",
                barmode='group',
                width=800,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating income expense chart: {e}")
            return self._create_error_chart("Income vs Expenses Chart", str(e))
    
    def create_cash_flow_chart(self, cash_flow_data: List[Dict]) -> Union[str, go.Figure]:
        """Create cash flow chart"""
        try:
            dates = [item.get('date', '') for item in cash_flow_data]
            inflows = [item.get('inflow', 0) for item in cash_flow_data]
            outflows = [item.get('outflow', 0) for item in cash_flow_data]
            net_flow = [item.get('net_flow', 0) for item in cash_flow_data]
            
            fig = go.Figure()
            
            # Add inflow line
            fig.add_trace(go.Scatter(
                x=dates,
                y=inflows,
                mode='lines+markers',
                name='Inflows',
                line=dict(color=self.default_colors['success'], width=2),
                marker=dict(size=6)
            ))
            
            # Add outflow line
            fig.add_trace(go.Scatter(
                x=dates,
                y=outflows,
                mode='lines+markers',
                name='Outflows',
                line=dict(color=self.default_colors['warning'], width=2),
                marker=dict(size=6)
            ))
            
            # Add net flow line
            fig.add_trace(go.Scatter(
                x=dates,
                y=net_flow,
                mode='lines+markers',
                name='Net Flow',
                line=dict(color=self.default_colors['primary'], width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Cash Flow Analysis",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                width=800,
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cash flow chart: {e}")
            return self._create_error_chart("Cash Flow Chart", str(e))
    
    def create_financial_goals_chart(self, goals_data: List[Dict]) -> Union[str, go.Figure]:
        """Create financial goals progress chart"""
        try:
            goal_names = [goal.get('name', '') for goal in goals_data]
            target_amounts = [goal.get('target_amount', 0) for goal in goals_data]
            current_amounts = [goal.get('current_amount', 0) for goal in goals_data]
            progress_percentages = [goal.get('progress_percentage', 0) for goal in goals_data]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Goal Progress', 'Progress Percentages'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Progress bars
            fig.add_trace(
                go.Bar(
                    x=goal_names,
                    y=current_amounts,
                    name='Current Amount',
                    marker_color=self.default_colors['info']
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=goal_names,
                    y=target_amounts,
                    name='Target Amount',
                    marker_color=self.default_colors['secondary']
                ),
                row=1, col=1
            )
            
            # Progress percentages
            fig.add_trace(
                go.Bar(
                    x=goal_names,
                    y=progress_percentages,
                    name='Progress %',
                    marker_color=self.default_colors['success']
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Financial Goals Progress",
                height=500,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating financial goals chart: {e}")
            return self._create_error_chart("Financial Goals Chart", str(e))
    
    def create_asset_allocation_chart(self, asset_data: Dict) -> Union[str, go.Figure]:
        """Create asset allocation pie chart"""
        try:
            labels = list(asset_data.keys())
            values = list(asset_data.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                marker_colors=list(self.default_colors.values())[:len(labels)]
            )])
            
            fig.update_layout(
                title="Asset Allocation",
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating asset allocation chart: {e}")
            return self._create_error_chart("Asset Allocation Chart", str(e))
    
    def create_liability_breakdown_chart(self, liability_data: Dict) -> Union[str, go.Figure]:
        """Create liability breakdown chart"""
        try:
            labels = list(liability_data.keys())
            values = list(liability_data.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                marker_colors=[self.default_colors['warning'], self.default_colors['light'], 
                             self.default_colors['dark'], self.default_colors['secondary']]
            )])
            
            fig.update_layout(
                title="Liability Breakdown",
                width=600,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating liability breakdown chart: {e}")
            return self._create_error_chart("Liability Breakdown Chart", str(e))
    
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

def main():
    """Test the financial chart generator"""
    # Initialize chart generator
    chart_gen = FinancialChartGenerator()
    
    # Test data
    test_financial_data = {
        "score": 75.5,
        "net_worth": 150000,
        "savings_rate": 0.25,
        "debt_ratio": 0.15,
        "income_coverage": 1.4,
        "category": "good"
    }
    
    # Create financial health chart
    chart = chart_gen.create_financial_health_chart(test_financial_data)
    
    if isinstance(chart, go.Figure):
        chart.write_html("test_financial_health_chart.html")
        print("✅ Financial health chart created and saved")
    else:
        print("❌ Error creating financial health chart")
    
    # Test timeline data
    timeline_data = [
        {"timestamp": "2023-01", "net_worth": 100000},
        {"timestamp": "2023-02", "net_worth": 105000},
        {"timestamp": "2023-03", "net_worth": 110000},
        {"timestamp": "2023-04", "net_worth": 115000},
        {"timestamp": "2023-05", "net_worth": 120000}
    ]
    
    timeline_chart = chart_gen.create_net_worth_timeline(timeline_data)
    
    if isinstance(timeline_chart, go.Figure):
        timeline_chart.write_html("test_net_worth_timeline.html")
        print("✅ Net worth timeline chart created and saved")
    else:
        print("❌ Error creating net worth timeline chart")

if __name__ == "__main__":
    main() 