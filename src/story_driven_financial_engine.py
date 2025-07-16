#!/usr/bin/env python
# Story-Driven Financial Decision Engine
# Creates branching narratives where each decision affects future possibilities
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class DecisionNode:
    """Represents a decision point in the financial journey"""
    timestamp: datetime
    event_type: str
    description: str
    available_options: List[Dict[str, Any]]
    optimal_choice: Dict[str, Any]
    chosen_path: Optional[Dict[str, Any]] = None
    closed_paths: List[Dict[str, Any]] = None
    portfolio_impact: Dict[str, float] = None
    life_impact: Dict[str, float] = None
    
    def __post_init__(self):
        if self.closed_paths is None:
            self.closed_paths = []
        if self.portfolio_impact is None:
            self.portfolio_impact = {}
        if self.life_impact is None:
            self.life_impact = {}

@dataclass
class StoryReport:
    """Individual report for each timestamped event"""
    timestamp: datetime
    event_title: str
    current_situation: Dict[str, Any]
    available_choices: List[Dict[str, Any]]
    optimal_recommendation: Dict[str, Any]
    rationale: str
    future_implications: List[str]
    portfolio_snapshot: Dict[str, Any]
    closed_opportunities: List[str]

class StoryDrivenFinancialEngine:
    """Engine that creates branching financial narratives"""
    
    def __init__(self, client_config: Dict[str, Any]):
        self.client_config = client_config
        self.decision_timeline: List[DecisionNode] = []
        self.current_portfolio = {
            'equity': 0.6,
            'bonds': 0.3,
            'cash': 0.1,
            'value': client_config.get('portfolio_value', 500000)
        }
        self.life_state = {
            'age': client_config.get('age', 45),
            'career_stage': 'mid_career',
            'family_status': 'married',
            'financial_stress': 0.3,
            'risk_tolerance': client_config.get('risk_tolerance', 0.6)
        }
        self.closed_opportunities = []
        self.story_reports = []
        
    def create_decision_scenario(self, event_type: str, description: str, 
                               available_options: List[Dict], timestamp: datetime) -> DecisionNode:
        """Create a new decision scenario"""
        
        # Determine optimal choice based on current state
        optimal_choice = self._calculate_optimal_choice(available_options)
        
        # Determine which paths are closed based on previous decisions
        closed_paths = self._determine_closed_paths(event_type)
        
        node = DecisionNode(
            timestamp=timestamp,
            event_type=event_type,
            description=description,
            available_options=available_options,
            optimal_choice=optimal_choice,
            closed_paths=closed_paths
        )
        
        self.decision_timeline.append(node)
        return node
    
    def _calculate_optimal_choice(self, options: List[Dict]) -> Dict[str, Any]:
        """Calculate the optimal choice based on current life and portfolio state"""
        best_score = -float('inf')
        optimal_choice = options[0]
        
        for option in options:
            score = self._evaluate_option(option)
            if score > best_score:
                best_score = score
                optimal_choice = option
        
        return optimal_choice
    
    def _evaluate_option(self, option: Dict[str, Any]) -> float:
        """Evaluate an option based on current state"""
        score = 0.0
        
        # Portfolio impact scoring
        if 'portfolio_impact' in option:
            portfolio_impact = option['portfolio_impact']
            if 'risk_reduction' in portfolio_impact:
                if self.life_state['financial_stress'] > 0.5:
                    score += 2.0  # High stress -> prefer risk reduction
                else:
                    score += 0.5
            
            if 'equity_increase' in portfolio_impact:
                if self.life_state['risk_tolerance'] > 0.7:
                    score += 1.5  # High risk tolerance -> prefer equity
                else:
                    score += 0.3
        
        # Life impact scoring
        if 'life_impact' in option:
            life_impact = option['life_impact']
            if 'stress_reduction' in life_impact:
                score += 1.0  # Always good to reduce stress
            if 'income_increase' in life_impact:
                score += 1.5  # Income increase is generally positive
        
        # Risk-adjusted scoring
        if 'risk_level' in option:
            risk_level = option['risk_level']
            if risk_level == 'low' and self.life_state['financial_stress'] > 0.6:
                score += 1.0
            elif risk_level == 'high' and self.life_state['risk_tolerance'] > 0.8:
                score += 1.0
        
        return score
    
    def _determine_closed_paths(self, event_type: str) -> List[Dict[str, Any]]:
        """Determine which paths are closed based on previous decisions"""
        closed_paths = []
        
        # Check previous decisions that affect current options
        for node in self.decision_timeline:
            if node.chosen_path:
                # If they chose a conservative path, aggressive options might be closed
                if 'conservative' in node.chosen_path.get('type', ''):
                    closed_paths.extend([
                        {'type': 'aggressive', 'reason': 'Previous conservative choice'},
                        {'type': 'high_risk', 'reason': 'Previous conservative choice'}
                    ])
                
                # If they chose to reduce risk, high-risk options might be closed
                if 'risk_reduction' in node.chosen_path.get('portfolio_impact', {}):
                    closed_paths.extend([
                        {'type': 'high_equity', 'reason': 'Previous risk reduction'},
                        {'type': 'aggressive_growth', 'reason': 'Previous risk reduction'}
                    ])
        
        return closed_paths
    
    def make_decision(self, node_index: int, choice: Dict[str, Any]) -> StoryReport:
        """Make a decision and generate a report"""
        if node_index >= len(self.decision_timeline):
            raise ValueError("Invalid node index")
        
        node = self.decision_timeline[node_index]
        node.chosen_path = choice
        
        # Update portfolio and life state based on choice
        self._update_state_from_choice(choice)
        
        # Generate report
        report = self._generate_story_report(node)
        self.story_reports.append(report)
        
        return report
    
    def _update_state_from_choice(self, choice: Dict[str, Any]):
        """Update portfolio and life state based on chosen option"""
        if 'portfolio_impact' in choice:
            impact = choice['portfolio_impact']
            if 'equity_increase' in impact:
                self.current_portfolio['equity'] = min(0.9, 
                    self.current_portfolio['equity'] + impact['equity_increase'])
            if 'risk_reduction' in impact:
                self.current_portfolio['equity'] = max(0.2, 
                    self.current_portfolio['equity'] - impact['risk_reduction'])
                self.current_portfolio['bonds'] = min(0.6, 
                    self.current_portfolio['bonds'] + impact['risk_reduction'])
        
        if 'life_impact' in choice:
            impact = choice['life_impact']
            if 'stress_reduction' in impact:
                self.life_state['financial_stress'] = max(0.0, 
                    self.life_state['financial_stress'] - impact['stress_reduction'])
            if 'income_increase' in impact:
                self.current_portfolio['value'] *= (1 + impact['income_increase'])
    
    def _generate_story_report(self, node: DecisionNode) -> StoryReport:
        """Generate a detailed report for a decision node"""
        
        # Calculate future implications
        future_implications = self._calculate_future_implications(node)
        
        # Generate rationale
        rationale = self._generate_rationale(node)
        
        # Determine closed opportunities
        closed_opportunities = [path['reason'] for path in node.closed_paths]
        
        return StoryReport(
            timestamp=node.timestamp,
            event_title=node.event_type.replace('_', ' ').title(),
            current_situation={
                'portfolio': self.current_portfolio.copy(),
                'life_state': self.life_state.copy(),
                'financial_stress': self.life_state['financial_stress']
            },
            available_choices=node.available_options,
            optimal_recommendation=node.optimal_choice,
            rationale=rationale,
            future_implications=future_implications,
            portfolio_snapshot=self.current_portfolio.copy(),
            closed_opportunities=closed_opportunities
        )
    
    def _calculate_future_implications(self, node: DecisionNode) -> List[str]:
        """Calculate future implications of the chosen path"""
        implications = []
        
        if node.chosen_path:
            choice = node.chosen_path
            
            if 'portfolio_impact' in choice:
                impact = choice['portfolio_impact']
                if 'equity_increase' in impact:
                    implications.append("Higher potential returns but increased volatility")
                if 'risk_reduction' in impact:
                    implications.append("Lower volatility but potentially reduced returns")
            
            if 'life_impact' in choice:
                impact = choice['life_impact']
                if 'stress_reduction' in impact:
                    implications.append("Improved financial peace of mind")
                if 'income_increase' in impact:
                    implications.append("Enhanced financial capacity for future goals")
        
        return implications
    
    def _generate_rationale(self, node: DecisionNode) -> str:
        """Generate rationale for the optimal choice"""
        if not node.optimal_choice:
            return "No optimal choice available"
        
        choice = node.optimal_choice
        rationale_parts = []
        
        if 'portfolio_impact' in choice:
            impact = choice['portfolio_impact']
            if 'risk_reduction' in impact:
                rationale_parts.append("Reduces portfolio risk given current stress levels")
            if 'equity_increase' in impact:
                rationale_parts.append("Increases growth potential aligned with risk tolerance")
        
        if 'life_impact' in choice:
            impact = choice['life_impact']
            if 'stress_reduction' in impact:
                rationale_parts.append("Reduces financial stress and improves well-being")
            if 'income_increase' in impact:
                rationale_parts.append("Enhances income potential for future goals")
        
        if not rationale_parts:
            rationale_parts.append("Balanced approach considering current circumstances")
        
        return " ".join(rationale_parts)
    
    def create_story_visualization(self) -> go.Figure:
        """Create a story-driven visualization showing decision branches"""
        
        # Create timeline data
        timestamps = [node.timestamp for node in self.decision_timeline]
        event_types = [node.event_type for node in self.decision_timeline]
        optimal_choices = [node.optimal_choice.get('type', 'unknown') for node in self.decision_timeline]
        chosen_paths = [node.chosen_path.get('type', 'none') if node.chosen_path else 'none' 
                       for node in self.decision_timeline]
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Decision Timeline', 'Portfolio Evolution', 'Life State Changes'),
            vertical_spacing=0.1
        )
        
        # Decision timeline
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[i for i in range(len(timestamps))],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=['red' if chosen == optimal else 'grey' 
                           for chosen, optimal in zip(chosen_paths, optimal_choices)]
                ),
                text=event_types,
                textposition='top center',
                name='Decision Points'
            ),
            row=1, col=1
        )
        
        # Portfolio evolution
        portfolio_values = []
        equity_allocations = []
        for node in self.decision_timeline:
            if node.chosen_path and 'portfolio_impact' in node.chosen_path:
                # Simulate portfolio changes
                portfolio_values.append(self.current_portfolio['value'])
                equity_allocations.append(self.current_portfolio['equity'])
            else:
                portfolio_values.append(self.current_portfolio['value'])
                equity_allocations.append(self.current_portfolio['equity'])
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=portfolio_values,
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity_allocations,
                mode='lines+markers',
                name='Equity Allocation',
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
        
        # Life state changes
        stress_levels = [self.life_state['financial_stress']] * len(timestamps)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=stress_levels,
                mode='lines+markers',
                name='Financial Stress',
                line=dict(color='red', width=3)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Story-Driven Financial Journey",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        total_decisions = len(self.decision_timeline)
        optimal_choices_made = sum(1 for node in self.decision_timeline 
                                 if node.chosen_path == node.optimal_choice)
        
        return {
            'total_decisions': total_decisions,
            'optimal_choices_made': optimal_choices_made,
            'optimal_choice_percentage': (optimal_choices_made / total_decisions * 100) if total_decisions > 0 else 0,
            'final_portfolio': self.current_portfolio.copy(),
            'final_life_state': self.life_state.copy(),
            'closed_opportunities': self.closed_opportunities,
            'decision_timeline': [asdict(node) for node in self.decision_timeline],
            'story_reports': [asdict(report) for report in self.story_reports]
        }

def create_sample_story():
    """Create a sample story-driven financial journey"""
    
    client_config = {
        'age': 45,
        'portfolio_value': 500000,
        'risk_tolerance': 0.6
    }
    
    engine = StoryDrivenFinancialEngine(client_config)
    
    # Decision 1: Career Advancement Opportunity
    career_options = [
        {
            'type': 'aggressive',
            'description': 'Take the promotion with higher risk',
            'portfolio_impact': {'equity_increase': 0.15},
            'life_impact': {'income_increase': 0.2},
            'risk_level': 'high'
        },
        {
            'type': 'conservative',
            'description': 'Stay in current role',
            'portfolio_impact': {'risk_reduction': 0.1},
            'life_impact': {'stress_reduction': 0.1},
            'risk_level': 'low'
        },
        {
            'type': 'balanced',
            'description': 'Negotiate a middle ground',
            'portfolio_impact': {'equity_increase': 0.05},
            'life_impact': {'income_increase': 0.1, 'stress_reduction': 0.05},
            'risk_level': 'medium'
        }
    ]
    
    node1 = engine.create_decision_scenario(
        'career_advancement',
        'You have been offered a promotion with higher pay but more responsibility',
        career_options,
        datetime(2021, 3, 15)
    )
    
    # Make the optimal choice
    report1 = engine.make_decision(0, node1.optimal_choice)
    
    # Decision 2: Family Planning
    family_options = [
        {
            'type': 'conservative',
            'description': 'Increase emergency fund and reduce risk',
            'portfolio_impact': {'risk_reduction': 0.2},
            'life_impact': {'stress_reduction': 0.2},
            'risk_level': 'low'
        },
        {
            'type': 'maintain',
            'description': 'Keep current allocation',
            'portfolio_impact': {},
            'life_impact': {},
            'risk_level': 'medium'
        }
    ]
    
    node2 = engine.create_decision_scenario(
        'family_planning',
        'You are expecting your first child and need to plan for increased expenses',
        family_options,
        datetime(2022, 6, 1)
    )
    
    # Make the optimal choice
    report2 = engine.make_decision(1, node2.optimal_choice)
    
    return engine, [report1, report2]

if __name__ == "__main__":
    engine, reports = create_sample_story()
    
    print("🎮 Story-Driven Financial Journey")
    print("=" * 40)
    
    for i, report in enumerate(reports, 1):
        print(f"\n📅 Event {i}: {report.event_title}")
        print(f"   Timestamp: {report.timestamp.strftime('%Y-%m-%d')}")
        print(f"   Optimal Choice: {report.optimal_recommendation['description']}")
        print(f"   Rationale: {report.rationale}")
        print(f"   Future Implications: {', '.join(report.future_implications)}")
    
    summary = engine.generate_summary_report()
    print(f"\n📊 Summary:")
    print(f"   - Total Decisions: {summary['total_decisions']}")
    print(f"   - Optimal Choices Made: {summary['optimal_choices_made']}")
    print(f"   - Optimal Choice Percentage: {summary['optimal_choice_percentage']:.1f}%") 