#!/usr/bin/env python
# Dynamic Comfort-Constraint Portfolio Engine
# Real-time portfolio rebalancing based on client comfort and market conditions
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicPortfolioEngine:
    """
    Dynamic portfolio engine that observes client configuration and market behavior
    to recommend portfolio adjustments ensuring comfort and risk constraints are met.
    """
    
    def __init__(self, client_config, market_data=None):
        """
        Initialize the dynamic portfolio engine.
        
        Args:
            client_config (dict): Client's current configuration including:
                - income: Current annual income
                - disposable_cash: Monthly disposable cash
                - allowable_var: Maximum Value at Risk tolerance
                - age: Current age
                - risk_profile: Risk tolerance (1-5 scale)
                - portfolio_value: Current portfolio value
                - target_allocation: Target asset allocation
            market_data (dict): Current market state including:
                - equity_volatility: Current equity market volatility
                - bond_yields: Current bond yields
                - economic_outlook: Economic outlook score (-1 to 1)
                - market_stress: Market stress indicator (0-1)
        """
        self.client_config = client_config
        self.market_data = market_data or self._get_default_market_data()
        self.rebalancing_history = []
        self.life_events_log = []
        self.portfolio_snapshots = []
        
        # Initialize portfolio state
        self.current_portfolio = {
            'equity': client_config.get('target_allocation', {}).get('equity', 0.6),
            'bonds': client_config.get('target_allocation', {}).get('bonds', 0.3),
            'cash': client_config.get('target_allocation', {}).get('cash', 0.1),
            'value': client_config.get('portfolio_value', 1000000)
        }
        
        # Comfort thresholds
        self.comfort_thresholds = {
            'max_drawdown': 0.15,
            'max_volatility': 0.12,
            'min_cash_buffer': 0.05,
            'max_equity_deviation': 0.10
        }
    
    def _get_default_market_data(self):
        """Get default market data if none provided"""
        return {
            'equity_volatility': 0.16,
            'bond_yields': 0.04,
            'economic_outlook': 0.0,
            'market_stress': 0.3,
            'equity_returns': 0.08,
            'bond_returns': 0.04
        }
    
    def add_life_event(self, event_type, description, impact_score, date=None):
        """
        Log a life event that affects portfolio decisions.
        
        Args:
            event_type (str): Type of life event (e.g., 'career_change', 'family_expansion')
            description (str): Description of the event
            impact_score (float): Impact on portfolio (-1 to 1, negative = conservative, positive = aggressive)
            date (str): Date of event (YYYY-MM-DD format)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        event = {
            'date': date,
            'type': event_type,
            'description': description,
            'impact_score': impact_score,
            'portfolio_before': self.current_portfolio.copy()
        }
        
        self.life_events_log.append(event)
        
        # Adjust portfolio based on life event
        self._adjust_for_life_event(event)
        
        # Log the event impact
        event['portfolio_after'] = self.current_portfolio.copy()
        event['allocation_change'] = {
            'equity': event['portfolio_after']['equity'] - event['portfolio_before']['equity'],
            'bonds': event['portfolio_after']['bonds'] - event['portfolio_before']['bonds'],
            'cash': event['portfolio_after']['cash'] - event['portfolio_before']['cash']
        }
    
    def _adjust_for_life_event(self, event):
        """Adjust portfolio allocation based on life event impact"""
        impact = event['impact_score']
        
        # Store current allocations before adjustment
        current_equity = self.current_portfolio['equity']
        current_bonds = self.current_portfolio['bonds']
        current_cash = self.current_portfolio['cash']
        
        # Conservative events (negative impact) -> reduce equity, increase bonds/cash
        if impact < 0:
            equity_reduction = abs(impact) * 0.15
            new_equity = max(0.2, current_equity - equity_reduction)
            equity_change = current_equity - new_equity
            
            # Distribute the reduction to bonds and cash
            self.current_portfolio['equity'] = new_equity
            self.current_portfolio['bonds'] = min(0.6, current_bonds + equity_change * 0.7)
            self.current_portfolio['cash'] = min(0.2, current_cash + equity_change * 0.3)
        
        # Aggressive events (positive impact) -> increase equity, reduce bonds/cash
        elif impact > 0:
            equity_increase = impact * 0.15
            new_equity = min(0.9, current_equity + equity_increase)
            equity_change = new_equity - current_equity
            
            # Reduce bonds and cash proportionally
            self.current_portfolio['equity'] = new_equity
            self.current_portfolio['bonds'] = max(0.1, current_bonds - equity_change * 0.8)
            self.current_portfolio['cash'] = max(0.05, current_cash - equity_change * 0.2)
        
        # Normalize allocations to ensure they sum to 1
        total = self.current_portfolio['equity'] + self.current_portfolio['bonds'] + self.current_portfolio['cash']
        self.current_portfolio['equity'] /= total
        self.current_portfolio['bonds'] /= total
        self.current_portfolio['cash'] /= total
    
    def evaluate_portfolio_comfort(self):
        """
        Evaluate current portfolio against comfort constraints.
        
        Returns:
            dict: Comfort metrics and recommendations
        """
        # Calculate current portfolio metrics
        current_volatility = (
            self.current_portfolio['equity'] * self.market_data['equity_volatility'] +
            self.current_portfolio['bonds'] * (self.market_data['bond_yields'] * 0.5)
        )
        
        # Simulate potential drawdown based on current allocation
        worst_case_drawdown = self.current_portfolio['equity'] * 0.4  # 40% equity drop
        
        # Check comfort constraints
        comfort_issues = []
        recommendations = []
        
        if current_volatility > self.comfort_thresholds['max_volatility']:
            comfort_issues.append(f"Volatility ({current_volatility:.1%}) exceeds comfort threshold")
            recommendations.append("Consider reducing equity allocation")
        
        if worst_case_drawdown > self.comfort_thresholds['max_drawdown']:
            comfort_issues.append(f"Potential drawdown ({worst_case_drawdown:.1%}) exceeds comfort threshold")
            recommendations.append("Consider increasing bond allocation")
        
        if self.current_portfolio['cash'] < self.comfort_thresholds['min_cash_buffer']:
            comfort_issues.append("Cash buffer below minimum threshold")
            recommendations.append("Consider increasing cash allocation")
        
        # Market stress adjustments
        if self.market_data['market_stress'] > 0.7:
            recommendations.append("High market stress - consider defensive positioning")
        
        return {
            'comfort_score': 1.0 - len(comfort_issues) / 4.0,
            'volatility': current_volatility,
            'potential_drawdown': worst_case_drawdown,
            'cash_buffer': self.current_portfolio['cash'],
            'issues': comfort_issues,
            'recommendations': recommendations,
            'market_stress': self.market_data['market_stress']
        }
    
    def recommend_rebalancing(self):
        """
        Generate portfolio rebalancing recommendations.
        
        Returns:
            dict: Rebalancing recommendations and rationale
        """
        comfort_metrics = self.evaluate_portfolio_comfort()
        target_allocation = self._calculate_target_allocation()
        
        # Calculate required changes
        changes = {}
        for asset in ['equity', 'bonds', 'cash']:
            current = self.current_portfolio[asset]
            target = target_allocation[asset]
            changes[asset] = target - current
        
        # Determine if rebalancing is needed
        total_deviation = sum(abs(changes[asset]) for asset in ['equity', 'bonds', 'cash'])
        rebalancing_needed = total_deviation > 0.05  # 5% threshold
        
        recommendations = {
            'rebalancing_needed': rebalancing_needed,
            'changes': changes,
            'target_allocation': target_allocation,
            'comfort_metrics': comfort_metrics,
            'rationale': self._generate_rationale(changes, comfort_metrics)
        }
        
        if rebalancing_needed:
            self._execute_rebalancing(changes)
            self.rebalancing_history.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'changes': changes,
                'comfort_score': comfort_metrics['comfort_score'],
                'market_stress': self.market_data['market_stress']
            })
        
        return recommendations
    
    def _calculate_target_allocation(self):
        """Calculate target allocation based on age, risk profile, and market conditions"""
        age = self.client_config.get('age', 40)
        risk_profile = self.client_config.get('risk_profile', 3)
        
        # Base allocation using 100-minus-age rule
        base_equity = max(0.2, min(0.9, (100 - age) / 100))
        
        # Adjust for risk profile
        risk_adjustment = (risk_profile - 3) * 0.1
        base_equity += risk_adjustment
        
        # Adjust for market conditions
        market_adjustment = 0
        if self.market_data['market_stress'] > 0.7:
            market_adjustment = -0.1  # Reduce equity in high stress
        elif self.market_data['economic_outlook'] > 0.5:
            market_adjustment = 0.05  # Increase equity in positive outlook
        
        target_equity = max(0.2, min(0.9, base_equity + market_adjustment))
        target_bonds = (1 - target_equity) * 0.8
        target_cash = (1 - target_equity) * 0.2
        
        return {
            'equity': target_equity,
            'bonds': target_bonds,
            'cash': target_cash
        }
    
    def _generate_rationale(self, changes, comfort_metrics):
        """Generate rationale for rebalancing recommendations"""
        rationale = []
        
        if changes['equity'] > 0.05:
            rationale.append("Increase equity allocation for growth potential")
        elif changes['equity'] < -0.05:
            rationale.append("Reduce equity allocation for risk management")
        
        if comfort_metrics['issues']:
            rationale.extend(comfort_metrics['recommendations'])
        
        if self.market_data['market_stress'] > 0.7:
            rationale.append("Defensive positioning due to high market stress")
        
        return rationale
    
    def _execute_rebalancing(self, changes):
        """Execute the rebalancing changes"""
        # Apply changes
        for asset, change in changes.items():
            if asset != 'value':
                self.current_portfolio[asset] += change
        
        # Normalize to ensure allocations sum to 1
        total = self.current_portfolio['equity'] + self.current_portfolio['bonds'] + self.current_portfolio['cash']
        self.current_portfolio['equity'] /= total
        self.current_portfolio['bonds'] /= total
        self.current_portfolio['cash'] /= total
    
    def update_market_data(self, new_market_data):
        """Update market data and trigger rebalancing evaluation"""
        self.market_data.update(new_market_data)
        return self.recommend_rebalancing()
    
    def get_portfolio_snapshot(self):
        """Get current portfolio snapshot"""
        comfort_metrics = self.evaluate_portfolio_comfort()
        snapshot = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio': self.current_portfolio.copy(),
            'comfort_metrics': comfort_metrics,
            'market_data': self.market_data.copy()
        }
        self.portfolio_snapshots.append(snapshot)
        return snapshot
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard showing portfolio evolution over time"""
        if not self.portfolio_snapshots:
            self.get_portfolio_snapshot()
        
        # Create time series data
        dates = [s['date'] for s in self.portfolio_snapshots]
        equity_alloc = [s['portfolio']['equity'] for s in self.portfolio_snapshots]
        bonds_alloc = [s['portfolio']['bonds'] for s in self.portfolio_snapshots]
        cash_alloc = [s['portfolio']['cash'] for s in self.portfolio_snapshots]
        comfort_scores = [s['comfort_metrics']['comfort_score'] for s in self.portfolio_snapshots]
        market_stress = [s['market_data']['market_stress'] for s in self.portfolio_snapshots]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Allocation Over Time', 'Comfort Score', 'Market Stress'),
            vertical_spacing=0.1
        )
        
        # Portfolio allocation
        fig.add_trace(
            go.Scatter(x=dates, y=equity_alloc, name='Equity', fill='tonexty', 
                      line=dict(color='#1f77b4')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=bonds_alloc, name='Bonds', fill='tonexty', 
                      line=dict(color='#ff7f0e')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=cash_alloc, name='Cash', fill='tonexty', 
                      line=dict(color='#2ca02c')),
            row=1, col=1
        )
        
        # Comfort score
        fig.add_trace(
            go.Scatter(x=dates, y=comfort_scores, name='Comfort Score', 
                      line=dict(color='#d62728', width=3)),
            row=2, col=1
        )
        
        # Market stress
        fig.add_trace(
            go.Scatter(x=dates, y=market_stress, name='Market Stress', 
                      line=dict(color='#9467bd', width=3)),
            row=3, col=1
        )
        
        # Add life events as annotations
        for event in self.life_events_log:
            fig.add_annotation(
                x=event['date'], y=0.8,
                text=f"📅 {event['type']}<br>{event['description']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0, ay=-40,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title="Dynamic Portfolio Engine Dashboard",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Allocation %", row=1, col=1)
        fig.update_yaxes(title_text="Comfort Score", row=2, col=1)
        fig.update_yaxes(title_text="Market Stress", row=3, col=1)
        
        return fig
    
    def export_data(self):
        """Export all data for analysis"""
        return {
            'portfolio_snapshots': self.portfolio_snapshots,
            'rebalancing_history': self.rebalancing_history,
            'life_events_log': self.life_events_log,
            'client_config': self.client_config,
            'comfort_thresholds': self.comfort_thresholds
        } 