#!/usr/bin/env python
# Enhanced Interactive Dashboard for Dynamic Portfolio Engine
# Features dropdown life event selection and detailed portfolio analysis
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

class InteractiveDashboard:
    """Enhanced interactive dashboard with life event dropdowns and detailed analysis"""
    
    def __init__(self, portfolio_engine):
        """Initialize dashboard with portfolio engine data"""
        self.engine = portfolio_engine
        self.data = portfolio_engine.export_data()
        
    def create_enhanced_dashboard(self):
        """Create an enhanced dashboard with interactive features"""
        # Create main dashboard
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Portfolio Allocation Over Time', 'Comfort Score & Market Stress',
                'Life Events Impact', 'Rebalancing History',
                'Risk Metrics', 'Contribution Analysis',
                'Life Event Timeline', 'Portfolio Performance'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Get time series data
        dates = [s['date'] for s in self.data['portfolio_snapshots']]
        equity_alloc = [s['portfolio']['equity'] for s in self.data['portfolio_snapshots']]
        bonds_alloc = [s['portfolio']['bonds'] for s in self.data['portfolio_snapshots']]
        cash_alloc = [s['portfolio']['cash'] for s in self.data['portfolio_snapshots']]
        comfort_scores = [s['comfort_metrics']['comfort_score'] for s in self.data['portfolio_snapshots']]
        market_stress = [s['market_data']['market_stress'] for s in self.data['portfolio_snapshots']]
        
        # 1. Portfolio Allocation (top left)
        fig.add_trace(
            go.Scatter(x=dates, y=equity_alloc, name='Equity', fill='tonexty', 
                      line=dict(color='#1f77b4'), showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=bonds_alloc, name='Bonds', fill='tonexty', 
                      line=dict(color='#ff7f0e'), showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=cash_alloc, name='Cash', fill='tonexty', 
                      line=dict(color='#2ca02c'), showlegend=True),
            row=1, col=1
        )
        
        # 2. Comfort Score & Market Stress (top right)
        fig.add_trace(
            go.Scatter(x=dates, y=comfort_scores, name='Comfort Score', 
                      line=dict(color='#d62728', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=market_stress, name='Market Stress', 
                      line=dict(color='#9467bd', width=3)),
            row=1, col=2
        )
        
        # 3. Life Events Impact (second row left)
        if self.data['life_events_log']:
            event_dates = [e['date'] for e in self.data['life_events_log']]
            event_impacts = [e['impact_score'] for e in self.data['life_events_log']]
            event_types = [e['type'] for e in self.data['life_events_log']]
            
            fig.add_trace(
                go.Bar(x=event_dates, y=event_impacts, name='Life Event Impact',
                      marker_color=['red' if x < 0 else 'green' for x in event_impacts],
                      text=event_types, textposition='outside'),
                row=2, col=1
            )
        
        # 4. Rebalancing History (second row right)
        if self.data['rebalancing_history']:
            rebal_dates = [r['date'] for r in self.data['rebalancing_history']]
            rebal_scores = [r['comfort_score'] for r in self.data['rebalancing_history']]
            
            fig.add_trace(
                go.Bar(x=rebal_dates, y=rebal_scores, name='Rebalancing Comfort Score',
                      marker_color='orange'),
                row=2, col=2
            )
        
        # 5. Risk Metrics (third row left)
        volatilities = [s['comfort_metrics']['volatility'] for s in self.data['portfolio_snapshots']]
        drawdowns = [s['comfort_metrics']['potential_drawdown'] for s in self.data['portfolio_snapshots']]
        
        fig.add_trace(
            go.Scatter(x=dates, y=volatilities, name='Portfolio Volatility',
                      line=dict(color='#e377c2')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=drawdowns, name='Potential Drawdown',
                      line=dict(color='#8c564b')),
            row=3, col=1
        )
        
        # 6. Contribution Analysis (third row right)
        if len(dates) > 1:
            equity_changes = [equity_alloc[i] - equity_alloc[i-1] for i in range(1, len(equity_alloc))]
            bonds_changes = [bonds_alloc[i] - bonds_alloc[i-1] for i in range(1, len(bonds_alloc))]
            change_dates = dates[1:]
            
            fig.add_trace(
                go.Scatter(x=change_dates, y=equity_changes, name='Equity Changes',
                          line=dict(color='#1f77b4')),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(x=change_dates, y=bonds_changes, name='Bonds Changes',
                          line=dict(color='#ff7f0e')),
                row=3, col=2
            )
        
        # 7. Life Event Timeline (fourth row left)
        if self.data['life_events_log']:
            event_y_positions = [0.8 - i * 0.1 for i in range(len(self.data['life_events_log']))]
            
            for i, event in enumerate(self.data['life_events_log']):
                fig.add_trace(
                    go.Scatter(x=[event['date']], y=[event_y_positions[i]], 
                              mode='markers+text',
                              marker=dict(size=15, color='red' if event['impact_score'] < 0 else 'green'),
                              text=[event['type']], textposition='top center',
                              name=event['type'], showlegend=False),
                    row=4, col=1
                )
        
        # 8. Portfolio Performance (fourth row right)
        # Simulate portfolio value changes
        portfolio_values = []
        base_value = self.data['client_config']['portfolio_value']
        
        for i, snapshot in enumerate(self.data['portfolio_snapshots']):
            # Simple performance simulation
            if i == 0:
                portfolio_values.append(base_value)
            else:
                # Simulate returns based on allocation and market conditions
                equity_return = snapshot['market_data']['equity_returns'] * snapshot['portfolio']['equity']
                bond_return = snapshot['market_data']['bond_returns'] * snapshot['portfolio']['bonds']
                total_return = equity_return + bond_return
                portfolio_values.append(portfolio_values[-1] * (1 + total_return))
        
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_values, name='Portfolio Value',
                      line=dict(color='#17a2b8', width=3)),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Dynamic Portfolio Engine - Enhanced Dashboard",
            height=1200,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Allocation %", row=1, col=1)
        fig.update_yaxes(title_text="Comfort Score", row=1, col=2)
        fig.update_yaxes(title_text="Impact Score", row=2, col=1)
        fig.update_yaxes(title_text="Comfort Score", row=2, col=2)
        fig.update_yaxes(title_text="Risk Metrics", row=3, col=1)
        fig.update_yaxes(title_text="Allocation Changes", row=3, col=2)
        fig.update_yaxes(title_text="Life Events", row=4, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=4, col=2)
        
        # Update y-axis for comfort and stress
        fig.update_yaxes(title_text="Score", row=1, col=2)
        
        return fig
    
    def create_life_event_analysis(self):
        """Create detailed life event analysis"""
        if not self.data['life_events_log']:
            return None
        
        # Prepare life event data
        events = self.data['life_events_log']
        
        # Create analysis figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Life Event Impact Scores', 'Portfolio Changes by Event',
                'Event Timeline', 'Comfort Score Changes'
            ),
            vertical_spacing=0.15
        )
        
        # 1. Impact scores
        event_types = [e['type'] for e in events]
        impact_scores = [e['impact_score'] for e in events]
        
        fig.add_trace(
            go.Bar(x=event_types, y=impact_scores,
                  marker_color=['red' if x < 0 else 'green' for x in impact_scores],
                  text=[f"{x:+.2f}" for x in impact_scores],
                  textposition='outside'),
            row=1, col=1
        )
        
        # 2. Portfolio changes
        equity_changes = [e['allocation_change']['equity'] for e in events]
        bonds_changes = [e['allocation_change']['bonds'] for e in events]
        cash_changes = [e['allocation_change']['cash'] for e in events]
        
        fig.add_trace(
            go.Bar(x=event_types, y=equity_changes, name='Equity Change'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=event_types, y=bonds_changes, name='Bonds Change'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=event_types, y=cash_changes, name='Cash Change'),
            row=1, col=2
        )
        
        # 3. Event timeline
        event_dates = [e['date'] for e in events]
        event_y_pos = list(range(len(events)))
        
        fig.add_trace(
            go.Scatter(x=event_dates, y=event_y_pos, mode='markers+text',
                      marker=dict(size=15, color='red'),
                      text=event_types, textposition='top center'),
            row=2, col=1
        )
        
        # 4. Comfort score changes
        # Find comfort scores before and after each event
        comfort_changes = []
        for event in events:
            # Find snapshots before and after the event
            before_snapshots = [s for s in self.data['portfolio_snapshots'] if s['date'] <= event['date']]
            after_snapshots = [s for s in self.data['portfolio_snapshots'] if s['date'] > event['date']]
            
            if before_snapshots and after_snapshots:
                before_comfort = before_snapshots[-1]['comfort_metrics']['comfort_score']
                after_comfort = after_snapshots[0]['comfort_metrics']['comfort_score']
                comfort_changes.append(after_comfort - before_comfort)
            else:
                comfort_changes.append(0)
        
        fig.add_trace(
            go.Bar(x=event_types, y=comfort_changes,
                  marker_color=['red' if x < 0 else 'green' for x in comfort_changes]),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Life Event Analysis",
            height=800,
            showlegend=True,
            barmode='stack'
        )
        
        return fig
    
    def generate_html_report(self):
        """Generate a comprehensive HTML report with interactive features"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dynamic Portfolio Engine Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                .life-event {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .positive {{ border-left-color: #28a745; }}
                .negative {{ border-left-color: #dc3545; }}
                select {{ padding: 5px; margin: 5px; }}
                button {{ padding: 8px 15px; margin: 5px; background-color: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }}
                button:hover {{ background-color: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Dynamic Portfolio Engine Report</h1>
                <p>Comprehensive analysis of portfolio evolution with life events and market changes</p>
            </div>
            
            <div class="section">
                <h2>📊 Portfolio Summary</h2>
                <div class="metric">
                    <strong>Initial Portfolio Value:</strong> ${self.data['client_config']['portfolio_value']:,}
                </div>
                <div class="metric">
                    <strong>Client Age:</strong> {self.data['client_config']['age']}
                </div>
                <div class="metric">
                    <strong>Risk Profile:</strong> {self.data['client_config']['risk_profile']}/5
                </div>
                <div class="metric">
                    <strong>Total Life Events:</strong> {len(self.data['life_events_log'])}
                </div>
                <div class="metric">
                    <strong>Rebalancing Events:</strong> {len(self.data['rebalancing_history'])}
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Life Events Analysis</h2>
                <select id="eventSelector" onchange="showEventDetails()">
                    <option value="">Select a life event...</option>
        """
        
        for i, event in enumerate(self.data['life_events_log']):
            html_content += f"""
                    <option value="{i}">{event['date']} - {event['type'].replace('_', ' ').title()}</option>
            """
        
        html_content += """
                </select>
                <div id="eventDetails"></div>
            </div>
            
            <div class="section">
                <h2>📈 Portfolio Evolution</h2>
                <div id="portfolioChart"></div>
            </div>
            
            <script>
                function showEventDetails() {{
                    const selector = document.getElementById('eventSelector');
                    const details = document.getElementById('eventDetails');
                    const eventIndex = selector.value;
                    
                    if (eventIndex === '') {{
                        details.innerHTML = '';
                        return;
                    }}
                    
                    const events = """ + json.dumps(self.data['life_events_log']) + """;
                    const event = events[eventIndex];
                    
                    const impactClass = event.impact_score < 0 ? 'negative' : 'positive';
                    const impactText = event.impact_score < 0 ? 'Conservative' : 'Aggressive';
                    
                    details.innerHTML = `
                        <div class="life-event ${impactClass}">
                            <h3>${event.type.replace('_', ' ').toUpperCase()}</h3>
                            <p><strong>Date:</strong> ${event.date}</p>
                            <p><strong>Description:</strong> ${event.description}</p>
                            <p><strong>Impact Score:</strong> ${event.impact_score.toFixed(2)} (${impactText})</p>
                            <p><strong>Portfolio Changes:</strong></p>
                            <ul>
                                <li>Equity: ${(event.allocation_change.equity * 100).toFixed(1)}%</li>
                                <li>Bonds: ${(event.allocation_change.bonds * 100).toFixed(1)}%</li>
                                <li>Cash: ${(event.allocation_change.cash * 100).toFixed(1)}%</li>
                            </ul>
                        </div>
                    `;
                }}
            </script>
        </body>
        </html>
        """
        
        return html_content

def create_enhanced_dashboard_demo():
    """Create and run enhanced dashboard demonstration"""
    from dynamic_portfolio_engine import DynamicPortfolioEngine
    
    # Create sample client and run simulation
    client_config = {
        'income': 250000,
        'disposable_cash': 8000,
        'allowable_var': 0.15,
        'age': 42,
        'risk_profile': 3,
        'portfolio_value': 1500000,
        'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
    }
    
    engine = DynamicPortfolioEngine(client_config)
    
    # Add life events
    life_events = [
        ('2024-03-15', 'career_change', 'Promotion to senior role - increased income stability', 0.3),
        ('2024-06-20', 'family_expansion', 'Second child born - increased expenses and need for stability', -0.4),
        ('2024-09-10', 'market_crash', 'Major market correction - client concerned about losses', -0.6),
        ('2024-12-05', 'inheritance', 'Received inheritance - increased portfolio value and flexibility', 0.2),
        ('2025-02-15', 'health_concern', 'Family health issue - need for more conservative approach', -0.3)
    ]
    
    for date, event_type, description, impact in life_events:
        engine.add_life_event(event_type, description, impact, date)
        engine.get_portfolio_snapshot()
    
    # Create enhanced dashboard
    dashboard = InteractiveDashboard(engine)
    
    # Generate all visualizations
    main_dashboard = dashboard.create_enhanced_dashboard()
    life_event_analysis = dashboard.create_life_event_analysis()
    html_report = dashboard.generate_html_report()
    
    # Save outputs
    main_dashboard.write_html("enhanced_portfolio_dashboard.html")
    
    if life_event_analysis:
        life_event_analysis.write_html("life_event_analysis.html")
    
    with open("portfolio_report.html", "w") as f:
        f.write(html_report)
    
    print("✅ Enhanced dashboard created successfully!")
    print("   - enhanced_portfolio_dashboard.html (main dashboard)")
    print("   - life_event_analysis.html (life event analysis)")
    print("   - portfolio_report.html (interactive report)")
    
    return dashboard

if __name__ == "__main__":
    print("This script is not intended to be run directly.")
    print("Please use 'run.py' to launch the application.") 