#!/usr/bin/env python
# Life Choice Optimization Engine
# Analyzes past life decisions and recommends optimal next choices
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LifeChoiceOptimizer:
    """
    Advanced life choice optimization engine that analyzes past decisions
    and recommends optimal next choices for financial, comfort, and lifestyle goals.
    """
    
    def __init__(self, portfolio_engine):
        """Initialize optimizer with portfolio engine"""
        self.portfolio_engine = portfolio_engine
        self.life_choices = []
        self.optimization_scenarios = {}
        
        # Life choice categories and their financial impacts
        self.choice_categories = {
            'career': {
                'promotion': {'income_boost': 0.15, 'stress_impact': -0.1, 'time_commitment': 0.2},
                'job_change': {'income_boost': 0.25, 'stress_impact': -0.2, 'time_commitment': 0.1},
                'part_time': {'income_boost': -0.3, 'stress_impact': 0.3, 'time_commitment': -0.4},
                'entrepreneur': {'income_boost': 0.5, 'stress_impact': -0.4, 'time_commitment': 0.6},
                'retirement': {'income_boost': -0.6, 'stress_impact': 0.5, 'time_commitment': -0.8}
            },
            'family': {
                'marriage': {'income_boost': 0.1, 'expense_impact': 0.2, 'stability_boost': 0.3},
                'children': {'income_boost': -0.1, 'expense_impact': 0.4, 'stability_boost': 0.2},
                'divorce': {'income_boost': -0.2, 'expense_impact': 0.3, 'stability_boost': -0.4},
                'elder_care': {'income_boost': -0.1, 'expense_impact': 0.3, 'stability_boost': 0.1}
            },
            'lifestyle': {
                'move_city': {'income_boost': 0.1, 'expense_impact': 0.1, 'stress_impact': -0.1},
                'buy_house': {'income_boost': 0, 'expense_impact': 0.3, 'stability_boost': 0.4},
                'travel_extensive': {'income_boost': -0.1, 'expense_impact': 0.2, 'stress_impact': 0.2},
                'downsize': {'income_boost': 0, 'expense_impact': -0.2, 'stability_boost': 0.1}
            },
            'education': {
                'advanced_degree': {'income_boost': 0.2, 'expense_impact': 0.3, 'time_commitment': 0.4},
                'certification': {'income_boost': 0.1, 'expense_impact': 0.05, 'time_commitment': 0.2},
                'skill_development': {'income_boost': 0.05, 'expense_impact': 0.02, 'time_commitment': 0.1}
            },
            'health': {
                'health_improvement': {'income_boost': 0, 'expense_impact': 0.05, 'stress_impact': 0.3},
                'medical_issue': {'income_boost': -0.1, 'expense_impact': 0.2, 'stress_impact': -0.3},
                'insurance_upgrade': {'income_boost': 0, 'expense_impact': 0.1, 'stability_boost': 0.2}
            }
        }
        
        # Optimization objectives and their weights
        self.optimization_objectives = {
            'financial_growth': {'weight': 0.4, 'description': 'Maximize wealth accumulation'},
            'comfort_stability': {'weight': 0.3, 'description': 'Maintain lifestyle comfort'},
            'risk_management': {'weight': 0.2, 'description': 'Minimize financial risk'},
            'lifestyle_quality': {'weight': 0.1, 'description': 'Enhance life satisfaction'}
        }
    
    def add_life_choice(self, category: str, choice: str, date: str, 
                        custom_impacts: Optional[Dict] = None):
        """
        Add a life choice to the optimization history.
        
        Args:
            category: Life choice category (career, family, lifestyle, etc.)
            choice: Specific choice made
            date: Date of the choice
            custom_impacts: Optional custom impact overrides
        """
        if category not in self.choice_categories:
            raise ValueError(f"Invalid category: {category}")
        
        if choice not in self.choice_categories[category]:
            raise ValueError(f"Invalid choice '{choice}' for category '{category}'")
        
        impacts = self.choice_categories[category][choice].copy()
        if custom_impacts:
            impacts.update(custom_impacts)
        
        life_choice = {
            'date': date,
            'category': category,
            'choice': choice,
            'impacts': impacts,
            'portfolio_before': self.portfolio_engine.current_portfolio.copy()
        }
        
        self.life_choices.append(life_choice)
        
        # Apply impacts to portfolio engine
        self._apply_choice_impacts(life_choice)
        
        # Get portfolio snapshot after choice
        life_choice['portfolio_after'] = self.portfolio_engine.current_portfolio.copy()
        life_choice['comfort_score'] = self.portfolio_engine.evaluate_portfolio_comfort()['comfort_score']
        
        return life_choice
    
    def _apply_choice_impacts(self, life_choice: Dict):
        """Apply life choice impacts to the portfolio engine"""
        impacts = life_choice['impacts']
        
        # Update client configuration based on impacts
        config = self.portfolio_engine.client_config
        
        if 'income_boost' in impacts:
            # Simulate income change
            current_income = config.get('income', 100000)
            income_change = current_income * impacts['income_boost']
            config['income'] = current_income + income_change
        
        if 'expense_impact' in impacts:
            # Simulate expense change
            current_disposable = config.get('disposable_cash', 5000)
            expense_change = current_disposable * impacts['expense_impact']
            config['disposable_cash'] = current_disposable - expense_change
        
        # Update portfolio based on stability and stress impacts
        if 'stability_boost' in impacts:
            # More stability -> can take more risk
            if impacts['stability_boost'] > 0:
                self.portfolio_engine.current_portfolio['equity'] = min(0.9, 
                    self.portfolio_engine.current_portfolio['equity'] + impacts['stability_boost'] * 0.1)
                self.portfolio_engine.current_portfolio['bonds'] = max(0.1, 
                    self.portfolio_engine.current_portfolio['bonds'] - impacts['stability_boost'] * 0.08)
                self.portfolio_engine.current_portfolio['cash'] = max(0.05, 
                    self.portfolio_engine.current_portfolio['cash'] - impacts['stability_boost'] * 0.02)
        
        if 'stress_impact' in impacts:
            # More stress -> more conservative
            if impacts['stress_impact'] < 0:
                self.portfolio_engine.current_portfolio['equity'] = max(0.2, 
                    self.portfolio_engine.current_portfolio['equity'] + impacts['stress_impact'] * 0.15)
                self.portfolio_engine.current_portfolio['bonds'] = min(0.6, 
                    self.portfolio_engine.current_portfolio['bonds'] - impacts['stress_impact'] * 0.12)
                self.portfolio_engine.current_portfolio['cash'] = min(0.2, 
                    self.portfolio_engine.current_portfolio['cash'] - impacts['stress_impact'] * 0.03)
        
        # Normalize portfolio
        total = (self.portfolio_engine.current_portfolio['equity'] + 
                self.portfolio_engine.current_portfolio['bonds'] + 
                self.portfolio_engine.current_portfolio['cash'])
        
        self.portfolio_engine.current_portfolio['equity'] /= total
        self.portfolio_engine.current_portfolio['bonds'] /= total
        self.portfolio_engine.current_portfolio['cash'] /= total
    
    def optimize_next_choice(self, objective: str = 'financial_growth', 
                           available_choices: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        Optimize the next life choice based on current situation and objective.
        
        Args:
            objective: Primary optimization objective
            available_choices: List of (category, choice) tuples to consider
            
        Returns:
            Dict with optimization results and recommendations
        """
        if objective not in self.optimization_objectives:
            raise ValueError(f"Invalid objective: {objective}")
        
        # Get available choices if not specified
        if available_choices is None:
            available_choices = []
            for category, choices in self.choice_categories.items():
                for choice in choices:
                    available_choices.append((category, choice))
        
        # Evaluate each choice
        choice_scores = []
        
        for category, choice in available_choices:
            # Create temporary portfolio state
            temp_portfolio = self.portfolio_engine.current_portfolio.copy()
            temp_config = self.portfolio_engine.client_config.copy()
            
            # Apply choice impacts temporarily
            impacts = self.choice_categories[category][choice]
            
            # Calculate financial impact
            financial_score = 0
            if 'income_boost' in impacts:
                financial_score += impacts['income_boost'] * 0.4
            if 'expense_impact' in impacts:
                financial_score -= impacts['expense_impact'] * 0.3
            
            # Calculate comfort impact
            comfort_score = 0
            if 'stress_impact' in impacts:
                comfort_score += impacts['stress_impact'] * 0.5
            if 'stability_boost' in impacts:
                comfort_score += impacts['stability_boost'] * 0.3
            
            # Calculate risk impact
            risk_score = 0
            if 'stability_boost' in impacts:
                risk_score += impacts['stability_boost'] * 0.4
            if 'stress_impact' in impacts:
                risk_score += impacts['stress_impact'] * 0.3
            
            # Calculate lifestyle impact
            lifestyle_score = 0
            if 'time_commitment' in impacts:
                lifestyle_score -= abs(impacts['time_commitment']) * 0.3
            if 'stability_boost' in impacts:
                lifestyle_score += impacts['stability_boost'] * 0.2
            
            # Weighted total score
            weights = self.optimization_objectives[objective]['weight']
            total_score = (financial_score * 0.4 + comfort_score * 0.3 + 
                         risk_score * 0.2 + lifestyle_score * 0.1)
            
            choice_scores.append({
                'category': category,
                'choice': choice,
                'financial_score': financial_score,
                'comfort_score': comfort_score,
                'risk_score': risk_score,
                'lifestyle_score': lifestyle_score,
                'total_score': total_score,
                'impacts': impacts
            })
        
        # Sort by total score
        choice_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Get top recommendations
        top_recommendations = choice_scores[:5]
        
        return {
            'objective': objective,
            'objective_description': self.optimization_objectives[objective]['description'],
            'current_situation': {
                'portfolio': self.portfolio_engine.current_portfolio.copy(),
                'comfort_score': self.portfolio_engine.evaluate_portfolio_comfort()['comfort_score'],
                'life_choices_count': len(self.life_choices)
            },
            'recommendations': top_recommendations,
            'best_choice': top_recommendations[0] if top_recommendations else None
        }
    
    def create_optimization_dashboard(self) -> go.Figure:
        """Create an interactive dashboard for life choice optimization"""
        if not self.life_choices:
            # Create empty dashboard
            fig = go.Figure()
            fig.add_annotation(
                text="No life choices recorded yet.<br>Add life choices to see optimization analysis.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Life Choice Optimization Dashboard")
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Life Choices Timeline', 'Portfolio Evolution',
                'Choice Impact Analysis', 'Comfort Score Over Time',
                'Category Distribution', 'Optimization Recommendations'
            ),
            vertical_spacing=0.1
        )
        
        # 1. Life Choices Timeline
        dates = [choice['date'] for choice in self.life_choices]
        categories = [choice['category'] for choice in self.life_choices]
        choices = [choice['choice'] for choice in self.life_choices]
        
        # Color code by category
        category_colors = {
            'career': '#1f77b4', 'family': '#ff7f0e', 'lifestyle': '#2ca02c',
            'education': '#d62728', 'health': '#9467bd'
        }
        
        colors = [category_colors.get(cat, '#7f7f7f') for cat in categories]
        
        fig.add_trace(
            go.Scatter(x=dates, y=list(range(len(dates))), mode='markers+text',
                      marker=dict(size=15, color=colors),
                      text=[f"{cat}: {choice}" for cat, choice in zip(categories, choices)],
                      textposition='top center', name='Life Choices'),
            row=1, col=1
        )
        
        # 2. Portfolio Evolution
        equity_alloc = [choice['portfolio_after']['equity'] for choice in self.life_choices]
        bonds_alloc = [choice['portfolio_after']['bonds'] for choice in self.life_choices]
        cash_alloc = [choice['portfolio_after']['cash'] for choice in self.life_choices]
        
        fig.add_trace(
            go.Scatter(x=dates, y=equity_alloc, name='Equity', fill='tonexty',
                      line=dict(color='#1f77b4')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=bonds_alloc, name='Bonds', fill='tonexty',
                      line=dict(color='#ff7f0e')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=cash_alloc, name='Cash', fill='tonexty',
                      line=dict(color='#2ca02c')),
            row=1, col=2
        )
        
        # 3. Choice Impact Analysis
        financial_impacts = [choice['impacts'].get('income_boost', 0) for choice in self.life_choices]
        expense_impacts = [choice['impacts'].get('expense_impact', 0) for choice in self.life_choices]
        stress_impacts = [choice['impacts'].get('stress_impact', 0) for choice in self.life_choices]
        
        fig.add_trace(
            go.Bar(x=dates, y=financial_impacts, name='Income Impact',
                  marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=dates, y=expense_impacts, name='Expense Impact',
                  marker_color='red'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=dates, y=stress_impacts, name='Stress Impact',
                  marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Comfort Score Over Time
        comfort_scores = [choice['comfort_score'] for choice in self.life_choices]
        
        fig.add_trace(
            go.Scatter(x=dates, y=comfort_scores, name='Comfort Score',
                      line=dict(color='#d62728', width=3)),
            row=2, col=2
        )
        
        # 5. Category Distribution
        category_counts = {}
        for choice in self.life_choices:
            cat = choice['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(category_counts.keys()), values=list(category_counts.values()),
                  name='Category Distribution'),
            row=3, col=1
        )
        
        # 6. Optimization Recommendations
        # Run optimization for different objectives
        objectives = list(self.optimization_objectives.keys())
        best_scores = []
        
        for objective in objectives:
            result = self.optimize_next_choice(objective)
            if result['best_choice']:
                best_scores.append(result['best_choice']['total_score'])
            else:
                best_scores.append(0)
        
        fig.add_trace(
            go.Bar(x=objectives, y=best_scores, name='Best Choice Score',
                  marker_color='purple'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Life Choice Optimization Dashboard",
            height=1000,
            showlegend=True,
            barmode='group'
        )
        
        return fig
    
    def generate_optimization_report(self, objective: str = 'financial_growth') -> str:
        """Generate a detailed optimization report"""
        result = self.optimize_next_choice(objective)
        
        report = f"""
# Life Choice Optimization Report

## Current Situation
- **Portfolio Value**: ${self.portfolio_engine.client_config.get('portfolio_value', 0):,}
- **Comfort Score**: {result['current_situation']['comfort_score']:.2f}
- **Life Choices Made**: {result['current_situation']['life_choices_count']}
- **Current Allocation**: 
  - Equity: {result['current_situation']['portfolio']['equity']:.1%}
  - Bonds: {result['current_situation']['portfolio']['bonds']:.1%}
  - Cash: {result['current_situation']['portfolio']['cash']:.1%}

## Optimization Objective
**{result['objective'].replace('_', ' ').title()}**: {result['objective_description']}

## Top Recommendations

"""
        
        for i, rec in enumerate(result['recommendations'][:3], 1):
            report += f"""
### {i}. {rec['choice'].replace('_', ' ').title()} ({rec['category'].title()})
- **Total Score**: {rec['total_score']:.3f}
- **Financial Impact**: {rec['financial_score']:+.3f}
- **Comfort Impact**: {rec['comfort_score']:+.3f}
- **Risk Impact**: {rec['risk_score']:+.3f}
- **Lifestyle Impact**: {rec['lifestyle_score']:+.3f}

**Expected Impacts:**
"""
            for impact, value in rec['impacts'].items():
                report += f"- {impact.replace('_', ' ').title()}: {value:+.1%}\n"
            report += "\n"
        
        return report
    
    def export_optimization_data(self) -> Dict:
        """Export all optimization data for analysis"""
        return {
            'life_choices': self.life_choices,
            'choice_categories': self.choice_categories,
            'optimization_objectives': self.optimization_objectives,
            'current_portfolio': self.portfolio_engine.current_portfolio.copy(),
            'client_config': self.portfolio_engine.client_config.copy()
        } 