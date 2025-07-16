#!/usr/bin/env python
"""
Timeline Grid Visualizer - Tetris-Style Financial Scenario Timeline
Author: ChatGPT 2025-01-27

Creates a visual timeline grid showing financial scenarios as blocks that 
get highlighted (optimal) or greyed out (non-feasible) as time progresses.
Like a story-driven game showing the customer's financial journey.

Features:
- Tetris-style grid layout with scenario blocks
- Real-time timeline progression
- Color-coded scenario states (optimal, feasible, stressed, ruled-out)
- Interactive hover details
- Story narrative integration
- Portfolio rebalancing timeline
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import itertools
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineGridVisualizer:
    """
    Creates a Tetris-style timeline grid showing financial scenarios
    and their evolution over time with rebalancing recommendations.
    """
    
    def __init__(self, config_path: str = "config/ips_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_time = 0  # Current year in timeline
        self.max_years = self.config.get('YEARS', 40)
        
        # Grid dimensions
        self.grid_width = 8  # Number of scenario columns
        self.grid_height = 6  # Number of scenario rows
        
        # Scenario states
        self.scenarios = {}
        self.scenario_grid = {}  # Maps grid positions to scenarios
        self.scenario_timeline = {}  # Timeline of scenario states
        self.story_events = []  # Story narrative events
        self.rebalancing_timeline = []  # Portfolio rebalancing events
        
        # Visual settings
        self.colors = {
            'optimal': '#00FF00',      # Bright green - best paths
            'feasible': '#4CAF50',     # Green - good paths  
            'stressed': '#FF9800',     # Orange - stressed but possible
            'ruled_out': '#757575',    # Grey - no longer possible
            'unknown': '#E0E0E0',      # Light grey - future scenarios
            'current': '#2196F3'       # Blue - current active scenario
        }
        
        logger.info("Initialized TimelineGridVisualizer")
    
    def _load_config(self) -> Dict:
        """Load the IPS configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def initialize_scenario_grid(self, client_scenarios: Dict = None):
        """Initialize the scenario grid with all possible configurations"""
        logger.info("Initializing scenario grid...")
        
        # Generate all possible scenarios from config
        factor_space = self.config.get('FACTOR_SPACE', {})
        scenarios = self._generate_scenario_combinations(factor_space)
        
        # Arrange scenarios in grid layout
        total_scenarios = len(scenarios)
        grid_size = self.grid_width * self.grid_height
        
        for i, scenario in enumerate(scenarios[:grid_size]):  # Limit to grid size
            row = i // self.grid_width
            col = i % self.grid_width
            
            scenario_id = f"S{row:02d}_{col:02d}"
            
            # Create scenario block
            scenario_block = {
                'id': scenario_id,
                'position': (row, col),
                'config': scenario,
                'state': 'unknown',  # Initial state
                'probability': 1.0 / total_scenarios,  # Equal probability initially
                'timeline_states': {},  # States over time
                'rebalancing_events': [],  # Portfolio changes
                'story_impact': '',  # Narrative description
                'feasibility_score': 1.0,
                'stress_level': 0.0
            }
            
            self.scenarios[scenario_id] = scenario_block
            self.scenario_grid[(row, col)] = scenario_id
        
        # Initialize timeline for all scenarios
        for year in range(self.max_years):
            for scenario_id in self.scenarios:
                self.scenarios[scenario_id]['timeline_states'][year] = 'unknown'
        
        logger.info(f"Initialized {len(self.scenarios)} scenarios in {self.grid_width}x{self.grid_height} grid")
    
    def _generate_scenario_combinations(self, factor_space: Dict) -> List[Dict]:
        """Generate all combinations of scenario factors"""
        keys = list(factor_space.keys())
        values = list(factor_space.values())
        
        scenarios = []
        for combination in itertools.product(*values):
            scenario = dict(zip(keys, combination))
            scenarios.append(scenario)
        
        return scenarios
    
    def update_timeline(self, new_year: int, client_events: List = None, 
                       scenario_updates: Dict = None):
        """Update the timeline to a new year and update scenario states"""
        logger.info(f"Updating timeline to year {new_year}")
        
        self.current_time = new_year
        
        # Process client events that affect scenarios
        if client_events:
            self._process_client_events(client_events, new_year)
        
        # Update scenario states based on new information
        if scenario_updates:
            self._update_scenario_states(scenario_updates, new_year)
        
        # Calculate optimal paths from current position
        self._calculate_optimal_paths(new_year)
        
        # Generate story narrative for this year
        self._generate_story_narrative(new_year)
        
        # Generate rebalancing recommendations
        self._generate_rebalancing_recommendations(new_year)
    
    def _process_client_events(self, events: List, year: int):
        """Process client life events and their impact on scenarios"""
        for event in events:
            logger.info(f"Processing event: {event.get('event_type', 'unknown')}")
            
            # Determine which scenarios are affected
            affected_scenarios = self._get_affected_scenarios(event)
            
            for scenario_id in affected_scenarios:
                scenario = self.scenarios[scenario_id]
                
                # Update scenario based on event
                if event.get('cash_flow_impact') == 'negative':
                    scenario['stress_level'] += 0.1
                    if scenario['stress_level'] > 0.5:
                        scenario['timeline_states'][year] = 'stressed'
                    if scenario['stress_level'] > 0.8:
                        scenario['timeline_states'][year] = 'ruled_out'
                        scenario['feasibility_score'] *= 0.5
                
                # Add rebalancing event if needed
                if scenario['stress_level'] > 0.3:
                    rebalancing_event = {
                        'year': year,
                        'reason': f"High stress from {event.get('event_type')}",
                        'recommended_allocation': self._calculate_stress_allocation(scenario),
                        'scenario_id': scenario_id
                    }
                    scenario['rebalancing_events'].append(rebalancing_event)
                    self.rebalancing_timeline.append(rebalancing_event)
    
    def _get_affected_scenarios(self, event: Dict) -> List[str]:
        """Determine which scenarios are affected by an event"""
        affected = []
        event_type = event.get('event_type', '')
        
        for scenario_id, scenario in self.scenarios.items():
            config = scenario['config']
            
            # Check if scenario configuration is compatible with event
            if event_type == 'education':
                # Education events affect scenarios with education paths
                if 'ED_PATH' in config:
                    affected.append(scenario_id)
            elif event_type == 'work':
                # Work events affect scenarios with work arrangements
                if 'HEL_WORK' in config:
                    affected.append(scenario_id)
            elif event_type == 'housing':
                # Housing events affect scenarios with financial stress
                affected.append(scenario_id)  # Affects all scenarios
            else:
                # Other events affect all scenarios
                affected.append(scenario_id)
        
        return affected
    
    def _update_scenario_states(self, updates: Dict, year: int):
        """Update scenario states based on analysis results"""
        for scenario_id, update_data in updates.items():
            if scenario_id in self.scenarios:
                scenario = self.scenarios[scenario_id]
                
                # Update feasibility
                if 'feasibility_score' in update_data:
                    scenario['feasibility_score'] = update_data['feasibility_score']
                
                # Update stress level
                if 'stress_level' in update_data:
                    scenario['stress_level'] = update_data['stress_level']
                
                # Update timeline state
                if 'state' in update_data:
                    scenario['timeline_states'][year] = update_data['state']
                
                # Update probability
                if 'probability' in update_data:
                    scenario['probability'] = update_data['probability']
    
    def _calculate_optimal_paths(self, year: int):
        """Calculate and highlight optimal scenario paths"""
        # Sort scenarios by feasibility and probability
        sorted_scenarios = sorted(
            self.scenarios.items(),
            key=lambda x: (x[1]['feasibility_score'] * x[1]['probability']),
            reverse=True
        )
        
        # Mark top scenarios as optimal
        top_count = min(3, len(sorted_scenarios))  # Top 3 scenarios
        
        for i, (scenario_id, scenario) in enumerate(sorted_scenarios):
            if i < top_count and scenario['stress_level'] < 0.5:
                scenario['timeline_states'][year] = 'optimal'
            elif scenario['stress_level'] < 0.3:
                scenario['timeline_states'][year] = 'feasible'
            elif scenario['stress_level'] < 0.6:
                scenario['timeline_states'][year] = 'stressed'
            else:
                scenario['timeline_states'][year] = 'ruled_out'
    
    def _generate_story_narrative(self, year: int):
        """Generate story narrative for the current year"""
        # Count scenario states
        state_counts = {}
        for scenario in self.scenarios.values():
            state = scenario['timeline_states'].get(year, 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Generate narrative based on states
        optimal_count = state_counts.get('optimal', 0)
        stressed_count = state_counts.get('stressed', 0)
        ruled_out_count = state_counts.get('ruled_out', 0)
        
        if optimal_count > 0:
            narrative = f"Year {year}: {optimal_count} optimal paths identified! "
        else:
            narrative = f"Year {year}: Challenging period ahead. "
        
        if stressed_count > 0:
            narrative += f"{stressed_count} scenarios showing stress. "
        
        if ruled_out_count > 0:
            narrative += f"{ruled_out_count} paths no longer feasible. "
        
        narrative += "Portfolio rebalancing recommended to maintain stability."
        
        story_event = {
            'year': year,
            'narrative': narrative,
            'state_counts': state_counts,
            'recommended_action': self._get_recommended_action(state_counts)
        }
        
        self.story_events.append(story_event)
    
    def _get_recommended_action(self, state_counts: Dict) -> str:
        """Get recommended action based on scenario states"""
        optimal_count = state_counts.get('optimal', 0)
        stressed_count = state_counts.get('stressed', 0)
        ruled_out_count = state_counts.get('ruled_out', 0)
        
        if optimal_count >= 3:
            return "Continue with current strategy"
        elif stressed_count > optimal_count:
            return "Consider conservative portfolio rebalancing"
        elif ruled_out_count > len(state_counts) * 0.5:
            return "Major strategy revision needed"
        else:
            return "Monitor closely and prepare for adjustments"
    
    def _generate_rebalancing_recommendations(self, year: int):
        """Generate portfolio rebalancing recommendations for current year"""
        # Analyze current scenario states
        stressed_scenarios = [
            s for s in self.scenarios.values()
            if s['timeline_states'].get(year) == 'stressed'
        ]
        
        if len(stressed_scenarios) > len(self.scenarios) * 0.3:  # >30% stressed
            # Recommend conservative rebalancing
            rebalancing = {
                'year': year,
                'reason': 'High stress across multiple scenarios',
                'recommended_allocation': {
                    'Cash': 0.25,
                    'Bonds': 0.55,
                    'Equities': 0.15,
                    'Alts': 0.05
                },
                'risk_reduction': True,
                'confidence': 0.85
            }
            self.rebalancing_timeline.append(rebalancing)
    
    def _calculate_stress_allocation(self, scenario: Dict) -> Dict:
        """Calculate stress-adjusted portfolio allocation"""
        stress_level = scenario['stress_level']
        
        # More conservative allocation for higher stress
        if stress_level > 0.6:
            return {'Cash': 0.30, 'Bonds': 0.60, 'Equities': 0.10, 'Alts': 0.00}
        elif stress_level > 0.3:
            return {'Cash': 0.20, 'Bonds': 0.50, 'Equities': 0.25, 'Alts': 0.05}
        else:
            # Default allocation from config
            risk_band = scenario['config'].get('RISK_BAND', 2)
            return self.config.get('RISK_SPLITS', {}).get(str(risk_band), {})
    
    def create_timeline_grid_visualization(self, output_path: str = None) -> str:
        """Create the main timeline grid visualization"""
        logger.info("Creating timeline grid visualization...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Scenario Grid (Current State)', 'Timeline Progression',
                'Stress Levels Over Time', 'Portfolio Rebalancing Events',
                'Feasibility Scores', 'Story Narrative'
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. Main scenario grid (top-left)
        self._add_scenario_grid(fig, row=1, col=1)
        
        # 2. Timeline progression (top-right)
        self._add_timeline_progression(fig, row=1, col=2)
        
        # 3. Stress levels over time (middle-left)
        self._add_stress_timeline(fig, row=2, col=1)
        
        # 4. Rebalancing events (middle-right)
        self._add_rebalancing_events(fig, row=2, col=2)
        
        # 5. Feasibility scores (bottom-left)
        self._add_feasibility_scores(fig, row=3, col=1)
        
        # 6. Story narrative table (bottom-right)
        self._add_story_narrative(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"Financial Life Timeline Grid - Year {self.current_time}",
            height=1200,
            width=1600,
            showlegend=True,
            font=dict(size=12)
        )
        
        # Save to HTML
        if output_path is None:
            output_path = f"timeline_grid_year_{self.current_time}.html"
        
        fig.write_html(output_path)
        logger.info(f"Timeline grid visualization saved to: {output_path}")
        
        return output_path
    
    def _add_scenario_grid(self, fig, row: int, col: int):
        """Add the main scenario grid (Tetris-style)"""
        # Create grid data
        grid_data = np.zeros((self.grid_height, self.grid_width))
        hover_text = []
        
        for i in range(self.grid_height):
            hover_row = []
            for j in range(self.grid_width):
                if (i, j) in self.scenario_grid:
                    scenario_id = self.scenario_grid[(i, j)]
                    scenario = self.scenarios[scenario_id]
                    
                    # Color based on current state
                    current_state = scenario['timeline_states'].get(self.current_time, 'unknown')
                    state_values = {
                        'optimal': 4, 'feasible': 3, 'stressed': 2, 
                        'ruled_out': 1, 'unknown': 0
                    }
                    grid_data[i, j] = state_values.get(current_state, 0)
                    
                    # Create hover text
                    config_text = "<br>".join([f"{k}: {v}" for k, v in scenario['config'].items()])
                    hover_info = (
                        f"Scenario: {scenario_id}<br>"
                        f"State: {current_state}<br>"
                        f"Stress: {scenario['stress_level']:.2f}<br>"
                        f"Feasibility: {scenario['feasibility_score']:.2f}<br>"
                        f"Probability: {scenario['probability']:.3f}<br>"
                        f"Configuration:<br>{config_text}"
                    )
                    hover_row.append(hover_info)
                else:
                    hover_row.append("Empty")
            hover_text.append(hover_row)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=grid_data,
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                colorscale=[
                    [0, self.colors['unknown']],
                    [0.25, self.colors['ruled_out']],
                    [0.5, self.colors['stressed']],
                    [0.75, self.colors['feasible']],
                    [1, self.colors['optimal']]
                ],
                showscale=True,
                colorbar=dict(
                    title="Scenario State",
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=["Unknown", "Ruled Out", "Stressed", "Feasible", "Optimal"]
                )
            ),
            row=row, col=col
        )
    
    def _add_timeline_progression(self, fig, row: int, col: int):
        """Add timeline progression chart"""
        years = list(range(min(self.current_time + 1, self.max_years)))
        
        # Count states over time
        state_counts_over_time = {state: [] for state in ['optimal', 'feasible', 'stressed', 'ruled_out']}
        
        for year in years:
            year_counts = {state: 0 for state in state_counts_over_time.keys()}
            for scenario in self.scenarios.values():
                state = scenario['timeline_states'].get(year, 'unknown')
                if state in year_counts:
                    year_counts[state] += 1
            
            for state in state_counts_over_time:
                state_counts_over_time[state].append(year_counts[state])
        
        # Add traces for each state
        for state, counts in state_counts_over_time.items():
            if any(counts):  # Only add if there are non-zero counts
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=counts,
                        mode='lines+markers',
                        name=state.title(),
                        line=dict(color=self.colors[state]),
                        fill='tonexty' if state != 'optimal' else None
                    ),
                    row=row, col=col
                )
    
    def _add_stress_timeline(self, fig, row: int, col: int):
        """Add stress levels timeline"""
        # Calculate average stress over time
        years = list(range(min(self.current_time + 1, self.max_years)))
        avg_stress = []
        max_stress = []
        
        for year in years:
            year_stress = [
                s['stress_level'] for s in self.scenarios.values()
                if year in s['timeline_states']
            ]
            if year_stress:
                avg_stress.append(np.mean(year_stress))
                max_stress.append(np.max(year_stress))
            else:
                avg_stress.append(0)
                max_stress.append(0)
        
        # Add stress traces
        fig.add_trace(
            go.Scatter(
                x=years, y=avg_stress,
                mode='lines+markers',
                name='Average Stress',
                line=dict(color='orange')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=years, y=max_stress,
                mode='lines+markers',
                name='Maximum Stress',
                line=dict(color='red', dash='dash')
            ),
            row=row, col=col
        )
    
    def _add_rebalancing_events(self, fig, row: int, col: int):
        """Add rebalancing events chart"""
        if not self.rebalancing_timeline:
            return
        
        years = [event['year'] for event in self.rebalancing_timeline]
        reasons = [event['reason'] for event in self.rebalancing_timeline]
        
        fig.add_trace(
            go.Bar(
                x=years,
                y=[1] * len(years),  # All bars same height
                text=reasons,
                textposition="outside",
                name='Rebalancing Events',
                marker_color='blue'
            ),
            row=row, col=col
        )
    
    def _add_feasibility_scores(self, fig, row: int, col: int):
        """Add feasibility scores over time"""
        years = list(range(min(self.current_time + 1, self.max_years)))
        avg_feasibility = []
        
        for year in years:
            feasibility_scores = [s['feasibility_score'] for s in self.scenarios.values()]
            avg_feasibility.append(np.mean(feasibility_scores))
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=avg_feasibility,
                mode='lines+markers',
                name='Average Feasibility',
                line=dict(color='green'),
                fill='tozeroy'
            ),
            row=row, col=col
        )
    
    def _add_story_narrative(self, fig, row: int, col: int):
        """Add story narrative table"""
        if not self.story_events:
            return
        
        # Create table data
        years = [event['year'] for event in self.story_events[-5:]]  # Last 5 events
        narratives = [event['narrative'] for event in self.story_events[-5:]]
        actions = [event['recommended_action'] for event in self.story_events[-5:]]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Year', 'Story', 'Recommended Action'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=[years, narratives, actions],
                    fill_color='lightgrey',
                    align='left',
                    height=30
                )
            ),
            row=row, col=col
        )
    
    def export_timeline_data(self, output_path: str = None) -> str:
        """Export timeline data to JSON"""
        if output_path is None:
            output_path = f"timeline_data_year_{self.current_time}.json"
        
        export_data = {
            'current_time': self.current_time,
            'scenarios': {
                scenario_id: {
                    'config': scenario['config'],
                    'current_state': scenario['timeline_states'].get(self.current_time, 'unknown'),
                    'stress_level': scenario['stress_level'],
                    'feasibility_score': scenario['feasibility_score'],
                    'probability': scenario['probability'],
                    'timeline_states': scenario['timeline_states'],
                    'rebalancing_events': scenario['rebalancing_events']
                }
                for scenario_id, scenario in self.scenarios.items()
            },
            'story_events': self.story_events,
            'rebalancing_timeline': self.rebalancing_timeline,
            'grid_layout': {
                'width': self.grid_width,
                'height': self.grid_height,
                'scenario_positions': {
                    scenario_id: scenario['position']
                    for scenario_id, scenario in self.scenarios.items()
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Timeline data exported to: {output_path}")
        return output_path

def demo_timeline_grid():
    """Demo the timeline grid visualizer"""
    print("=== Timeline Grid Visualizer Demo ===")
    
    # Initialize visualizer
    visualizer = TimelineGridVisualizer()
    
    # Initialize scenario grid
    visualizer.initialize_scenario_grid()
    
    # Simulate timeline progression
    for year in range(5):
        print(f"\n--- Year {year} ---")
        
        # Simulate client events
        if year == 1:
            events = [
                {'event_type': 'education', 'cash_flow_impact': 'negative', 'impact_amount': 110000},
                {'event_type': 'work', 'cash_flow_impact': 'positive', 'impact_amount': 70000}
            ]
        elif year == 3:
            events = [
                {'event_type': 'housing', 'cash_flow_impact': 'negative', 'impact_amount': 2500000}
            ]
        else:
            events = []
        
        # Update timeline
        visualizer.update_timeline(year, events)
        
        # Print summary
        state_counts = {}
        for scenario in visualizer.scenarios.values():
            state = scenario['timeline_states'].get(year, 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print(f"Scenario states: {state_counts}")
        
        if visualizer.story_events:
            print(f"Story: {visualizer.story_events[-1]['narrative']}")
            print(f"Action: {visualizer.story_events[-1]['recommended_action']}")
    
    # Create visualization
    output_path = visualizer.create_timeline_grid_visualization()
    print(f"\nTimeline grid visualization created: {output_path}")
    
    # Export data
    data_path = visualizer.export_timeline_data()
    print(f"Timeline data exported: {data_path}")

if __name__ == "__main__":
    demo_timeline_grid() 