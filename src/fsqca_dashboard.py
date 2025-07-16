#!/usr/bin/env python
"""
fsQCA Dashboard - Path of Most Happiness Analysis
Author: ChatGPT 2025-07-16

Creates interactive dashboard for Fuzzy Set Qualitative Comparative Analysis (fsQCA)
to identify optimal paths to happiness and well-being.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Add src to path
sys.path.append('src')

class fsQCADashboard:
    """Interactive dashboard for fsQCA analysis"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.results = None
        self.fsqca_data = None
        
    def load_results(self, results: Dict[str, Any]):
        """Load processing results"""
        self.results = results
        self.fsqca_data = results.get('fsqca_analysis', {})
        
    def create_comprehensive_dashboard(self) -> go.Figure:
        """Create comprehensive fsQCA dashboard"""
        if not self.results:
            return self._create_empty_dashboard()
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Happiness Estimate', 'Condition Analysis',
                'Optimal Paths', 'Solution Coverage',
                'Event Timeline', 'Event Distribution',
                'Path Recommendations', 'Client Profile'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "sunburst"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Add all components
        self._add_happiness_indicator(fig, 1, 1)
        self._add_condition_analysis(fig, 1, 2)
        self._add_optimal_paths_sunburst(fig, 2, 1)
        self._add_solution_coverage(fig, 2, 2)
        self._add_event_timeline(fig, 3, 1)
        self._add_event_distribution(fig, 3, 2)
        self._add_path_recommendations_table(fig, 4, 1)
        self._add_client_profile_indicator(fig, 4, 2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'fsQCA Analysis Dashboard - {self.client_id}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1600,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _add_happiness_indicator(self, fig: go.Figure, row: int, col: int):
        """Add happiness estimate indicator"""
        happiness = self.fsqca_data.get('happiness_estimate', 0.0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=happiness * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Happiness Estimate (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=row, col=col
        )
    
    def _add_condition_analysis(self, fig: go.Figure, row: int, col: int):
        """Add condition analysis bar chart"""
        conditions = self.fsqca_data.get('conditions', {})
        
        if not conditions:
            return
        
        # Calculate average values for each condition
        condition_avgs = {}
        for condition, values in conditions.items():
            if values:
                condition_avgs[condition.replace('_', ' ').title()] = np.mean(values)
        
        fig.add_trace(
            go.Bar(
                x=list(condition_avgs.keys()),
                y=list(condition_avgs.values()),
                marker_color='lightblue',
                name='Condition Values'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Conditions", row=row, col=col)
        fig.update_yaxes(title_text="Average Value", row=row, col=col)
    
    def _add_optimal_paths_sunburst(self, fig: go.Figure, row: int, col: int):
        """Add optimal paths sunburst chart"""
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        if not optimal_paths:
            return
        
        # Create sunburst data
        ids = []
        labels = []
        parents = []
        values = []
        
        # Root
        ids.append("root")
        labels.append("Optimal Paths")
        parents.append("")
        values.append(1)
        
        for path in optimal_paths:
            path_id = path['path_id']
            solution_type = path['solution_type']
            
            # Path node
            ids.append(path_id)
            labels.append(f"{solution_type.title()} Solution")
            parents.append("root")
            values.append(path.get('coverage', 0.5))
            
            # Recommendations
            recommendations = path.get('recommendations', [])
            for i, rec in enumerate(recommendations[:3]):  # Limit to 3 recommendations
                rec_id = f"{path_id}_rec_{i}"
                ids.append(rec_id)
                labels.append(rec[:30] + "..." if len(rec) > 30 else rec)
                parents.append(path_id)
                values.append(0.3)
        
        fig.add_trace(
            go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=px.colors.qualitative.Set3)
            ),
            row=row, col=col
        )
    
    def _add_solution_coverage(self, fig: go.Figure, row: int, col: int):
        """Add solution coverage analysis"""
        analysis_results = self.fsqca_data.get('analysis_results', {})
        
        if not analysis_results:
            return
        
        solution_types = ['Complex', 'Parsimonious', 'Intermediate']
        coverage_values = [
            analysis_results.get('coverage', 0.0),
            analysis_results.get('coverage', 0.0),
            analysis_results.get('coverage', 0.0)
        ]
        consistency_values = [
            analysis_results.get('consistency', 0.0),
            analysis_results.get('consistency', 0.0),
            analysis_results.get('consistency', 0.0)
        ]
        
        fig.add_trace(
            go.Bar(
                name='Coverage',
                x=solution_types,
                y=coverage_values,
                marker_color='lightgreen'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Bar(
                name='Consistency',
                x=solution_types,
                y=consistency_values,
                marker_color='lightcoral'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Solution Type", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_event_distribution(self, fig: go.Figure, row: int, col: int):
        """Add event distribution chart"""
        events = self.results.get('events', {}).get('event_details', [])
        
        if not events:
            return
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.get('type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(event_counts.keys()),
                y=list(event_counts.values()),
                marker_color='lightsteelblue',
                name='Event Count'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Event Type", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_event_timeline(self, fig: go.Figure, row: int, col: int):
        """Add event timeline visualization"""
        events = self.results.get('events', {}).get('event_details', [])
        
        if not events:
            return
        
        # Prepare timeline data
        x_values = []  # Years ago
        y_values = []  # Event types
        text_values = []  # Hover text
        colors = []
        
        for event in events:
            years_ago = event.get('years_ago', 0)
            event_type = event.get('type', 'unknown')
            amount = event.get('amount', 0)
            description = event.get('description', '')
            
            x_values.append(years_ago)
            y_values.append(event_type)
            text_values.append(f"{description}<br>Amount: ${amount:,.0f}<br>Years ago: {years_ago}")
            
            # Color by event type
            color_map = {
                'education': 'blue',
                'work': 'green',
                'family': 'red',
                'housing': 'orange',
                'health': 'purple',
                'financial': 'brown',
                'retirement': 'gray',
                'charity': 'pink'
            }
            colors.append(color_map.get(event_type, 'black'))
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=0.7
                ),
                text=text_values,
                hoverinfo='text',
                name='Life Events'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Years Ago", row=row, col=col)
        fig.update_yaxes(title_text="Event Types", row=row, col=col)
    
    def _add_client_profile_indicator(self, fig: go.Figure, row: int, col: int):
        """Add client profile indicator"""
        client_profile = self.results.get('client_profile', {})
        
        if not client_profile:
            return
        
        age = client_profile.get('age', 35)
        life_stage = client_profile.get('life_stage', 'mid_career')
        income_level = client_profile.get('income_level', 'middle_income')
        confidence = client_profile.get('confidence', 0.5)
        
        # Create profile text
        profile_text = f"""
        Age: {age}<br>
        Life Stage: {life_stage.replace('_', ' ').title()}<br>
        Income Level: {income_level.replace('_', ' ').title()}<br>
        Profile Confidence: {confidence:.1%}
        """
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=age,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Client Profile"},
                delta={'reference': 35},
                number={'valueformat': '.0f', 'suffix': ' years'},
                gauge={
                    'axis': {'range': [18, 85]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [18, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 85], 'color': "orange"}
                    ]
                }
            ),
            row=row, col=col
        )
    
    def _add_path_recommendations_table(self, fig: go.Figure, row: int, col: int):
        """Add path recommendations table"""
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        if not optimal_paths:
            return
        
        # Prepare table data
        table_data = []
        for path in optimal_paths:
            recommendations = path.get('recommendations', [])
            for rec in recommendations:
                table_data.append([
                    path['solution_type'].title(),
                    rec,
                    f"{path.get('coverage', 0.0):.2f}",
                    f"{path.get('consistency', 0.0):.2f}"
                ])
        
        if not table_data:
            return
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Solution Type', 'Recommendation', 'Coverage', 'Consistency'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=10)
                )
            ),
            row=row, col=col
        )
    
    def create_happiness_path_visualization(self) -> go.Figure:
        """Create specialized visualization for happiness paths"""
        if not self.fsqca_data:
            return self._create_empty_dashboard()
        
        # Create sankey diagram for happiness paths
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=self._get_sankey_labels(),
                color="blue"
            ),
            link=dict(
                source=self._get_sankey_sources(),
                target=self._get_sankey_targets(),
                value=self._get_sankey_values()
            )
        )])
        
        fig.update_layout(
            title_text="Happiness Path Analysis - fsQCA Methodology",
            font_size=10,
            height=600
        )
        
        return fig
    
    def _get_sankey_labels(self) -> List[str]:
        """Get labels for sankey diagram"""
        conditions = self.fsqca_data.get('conditions', {})
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        labels = ['Start']
        
        # Add conditions
        for condition in conditions.keys():
            labels.append(condition.replace('_', ' ').title())
        
        # Add paths
        for path in optimal_paths:
            labels.append(f"Path: {path['solution_type']}")
        
        labels.append('Happiness')
        
        return labels
    
    def _get_sankey_sources(self) -> List[int]:
        """Get source indices for sankey diagram"""
        conditions = self.fsqca_data.get('conditions', {})
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        sources = []
        
        # From start to conditions
        for i in range(len(conditions)):
            sources.append(0)
        
        # From conditions to paths
        for path in optimal_paths:
            for i in range(len(conditions)):
                sources.append(i + 1)
        
        # From paths to happiness
        for i in range(len(optimal_paths)):
            sources.append(len(conditions) + 1 + i)
        
        return sources
    
    def _get_sankey_targets(self) -> List[int]:
        """Get target indices for sankey diagram"""
        conditions = self.fsqca_data.get('conditions', {})
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        targets = []
        
        # From start to conditions
        for i in range(len(conditions)):
            targets.append(i + 1)
        
        # From conditions to paths
        for path in optimal_paths:
            for i in range(len(conditions)):
                targets.append(len(conditions) + 1 + len(optimal_paths) - 1)
        
        # From paths to happiness
        for i in range(len(optimal_paths)):
            targets.append(len(conditions) + 1 + len(optimal_paths))
        
        return targets
    
    def _get_sankey_values(self) -> List[float]:
        """Get values for sankey diagram"""
        conditions = self.fsqca_data.get('conditions', {})
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        
        values = []
        
        # From start to conditions
        for condition in conditions.values():
            if condition:
                values.append(np.mean(condition))
            else:
                values.append(0.1)
        
        # From conditions to paths
        for path in optimal_paths:
            for condition in conditions.values():
                if condition:
                    values.append(np.mean(condition) * path.get('coverage', 0.5))
                else:
                    values.append(0.1)
        
        # From paths to happiness
        for path in optimal_paths:
            values.append(path.get('happiness_estimate', 0.5))
        
        return values
    
    def generate_fsqca_report(self) -> str:
        """Generate comprehensive fsQCA report"""
        if not self.fsqca_data:
            return "No fsQCA data available."
        
        report = []
        report.append("# fsQCA Analysis Report")
        report.append(f"**Client:** {self.client_id}")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Methodology
        report.append("## Methodology")
        report.append("This analysis uses Fuzzy Set Qualitative Comparative Analysis (fsQCA) to identify optimal paths to happiness and well-being.")
        report.append("")
        
        # Happiness Estimate
        happiness = self.fsqca_data.get('happiness_estimate', 0.0)
        report.append("## Happiness Estimate")
        report.append(f"**Overall Happiness Score:** {happiness:.2%}")
        report.append("")
        
        # Conditions Analysis
        conditions = self.fsqca_data.get('conditions', {})
        if conditions:
            report.append("## Condition Analysis")
            for condition, values in conditions.items():
                if values:
                    avg_value = np.mean(values)
                    report.append(f"- **{condition.replace('_', ' ').title()}:** {avg_value:.2f}")
            report.append("")
        
        # Optimal Paths
        optimal_paths = self.fsqca_data.get('optimal_paths', [])
        if optimal_paths:
            report.append("## Optimal Paths to Happiness")
            for i, path in enumerate(optimal_paths, 1):
                report.append(f"### Path {i}: {path['solution_type'].title()} Solution")
                report.append(f"- **Formula:** {path['formula']}")
                report.append(f"- **Coverage:** {path.get('coverage', 0.0):.2%}")
                report.append(f"- **Consistency:** {path.get('consistency', 0.0):.2%}")
                report.append("")
                report.append("**Recommendations:**")
                for rec in path.get('recommendations', []):
                    report.append(f"- {rec}")
                report.append("")
        
        # Analysis Results
        analysis_results = self.fsqca_data.get('analysis_results', {})
        if analysis_results:
            report.append("## Analysis Results")
            report.append(f"- **Solution Coverage:** {analysis_results.get('coverage', 0.0):.2%}")
            report.append(f"- **Solution Consistency:** {analysis_results.get('consistency', 0.0):.2%}")
            report.append("")
            
            report.append("### Solutions")
            report.append(f"- **Complex Solution:** {analysis_results.get('complex_solution', 'N/A')}")
            report.append(f"- **Parsimonious Solution:** {analysis_results.get('parsimonious_solution', 'N/A')}")
            report.append(f"- **Intermediate Solution:** {analysis_results.get('intermediate_solution', 'N/A')}")
            report.append("")
        
        # Events Summary
        events = self.results.get('events', {}).get('event_details', [])
        if events:
            report.append("## Extracted Life Events")
            report.append(f"**Total Events:** {len(events)}")
            report.append("")
            
            event_types = {}
            for event in events:
                event_type = event.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            for event_type, count in event_types.items():
                report.append(f"- **{event_type.title()}:** {count} events")
            report.append("")
        
        return "\n".join(report)
    
    def _create_empty_dashboard(self) -> go.Figure:
        """Create empty dashboard when no data is available"""
        fig = go.Figure()
        fig.add_annotation(
            text="No fsQCA data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title="fsQCA Dashboard - No Data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def save_dashboard(self, output_path: str = None):
        """Save dashboard to HTML file"""
        if output_path is None:
            output_path = f"docs/fsqca_dashboard_{self.client_id}.html"
        
        # Create comprehensive dashboard
        dashboard = self.create_comprehensive_dashboard()
        
        # Create happiness path visualization
        happiness_viz = self.create_happiness_path_visualization()
        
        # Combine into single HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>fsQCA Dashboard - {self.client_id}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>fsQCA Analysis Dashboard</h1>
            <h2>Client: {self.client_id}</h2>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Comprehensive Analysis</h3>
            <div id="dashboard"></div>
            
            <h3>Happiness Path Visualization</h3>
            <div id="happiness-viz"></div>
            
            <script>
                {dashboard.to_json()}
                Plotly.newPlot('dashboard', dashboard.data, dashboard.layout);
                
                {happiness_viz.to_json()}
                Plotly.newPlot('happiness-viz', happiness_viz.data, happiness_viz.layout);
            </script>
        </body>
        </html>
        """
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard saved to: {output_path}")
        return output_path

def main():
    """Main function for testing the fsQCA dashboard"""
    # Test with sample data
    sample_results = {
        'client_id': 'TEST_CLIENT',
        'fsqca_analysis': {
            'happiness_estimate': 0.75,
            'conditions': {
                'education_investment': [0.8, 0.9],
                'career_advancement': [0.7, 0.8],
                'family_stability': [0.6, 0.7],
                'financial_security': [0.9, 0.8]
            },
            'optimal_paths': [
                {
                    'path_id': 'path_1',
                    'solution_type': 'complex',
                    'formula': 'EDU*CAREER*FAMILY*FIN',
                    'coverage': 0.85,
                    'consistency': 0.92,
                    'recommendations': [
                        'Invest in education and skill development',
                        'Focus on career advancement',
                        'Prioritize family stability',
                        'Build financial security'
                    ]
                }
            ],
            'analysis_results': {
                'coverage': 0.85,
                'consistency': 0.92,
                'complex_solution': 'EDU*CAREER*FAMILY*FIN',
                'parsimonious_solution': 'CAREER*FIN',
                'intermediate_solution': 'EDU*CAREER*FIN'
            }
        },
        'events': {
            'event_details': [
                {'type': 'education', 'description': 'University enrollment'},
                {'type': 'work', 'description': 'Career promotion'},
                {'type': 'family', 'description': 'Family planning'}
            ]
        }
    }
    
    dashboard = fsQCADashboard('TEST_CLIENT')
    dashboard.load_results(sample_results)
    
    # Save dashboard
    output_path = dashboard.save_dashboard()
    
    # Generate report
    report = dashboard.generate_fsqca_report()
    print("\n" + "="*50)
    print("fsQCA REPORT")
    print("="*50)
    print(report)
    
    print(f"\nDashboard saved to: {output_path}")

if __name__ == "__main__":
    main() 