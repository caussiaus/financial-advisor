#!/usr/bin/env python
# Story Report Generator
# Creates timestamped reports for each decision point with optimal recommendations
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from story_driven_financial_engine import StoryDrivenFinancialEngine, create_sample_story

class StoryReportGenerator:
    """Generates comprehensive timestamped reports for story-driven financial decisions"""
    
    def __init__(self, story_engine: StoryDrivenFinancialEngine):
        """Initialize with story-driven financial engine"""
        self.engine = story_engine
        self.reports = []
        
    def generate_timestamped_reports(self) -> List[Dict[str, Any]]:
        """Generate individual reports for each timestamped event"""
        reports = []
        
        for i, node in enumerate(self.engine.decision_timeline):
            report = self._create_individual_report(node, i)
            reports.append(report)
        
        return reports
    
    def _create_individual_report(self, node, index: int) -> Dict[str, Any]:
        """Create a detailed report for a single decision node"""
        
        # Generate report data
        report = {
            'timestamp': node.timestamp.strftime('%Y-%m-%d %H:%M'),
            'event_title': node.event_type.replace('_', ' ').title(),
            'event_description': node.description,
            'current_situation': self._get_current_situation(node),
            'available_choices': self._format_choices(node.available_options, 'available'),
            'closed_opportunities': self._format_closed_paths(node.closed_paths),
            'optimal_recommendation': self._format_optimal_choice(node.optimal_choice),
            'rationale': self._generate_rationale(node),
            'future_implications': self._calculate_future_implications(node),
            'portfolio_impact': self._calculate_portfolio_impact(node),
            'life_impact': self._calculate_life_impact(node),
            'decision_metrics': self._calculate_decision_metrics(node),
            'visualization_data': self._create_visualization_data(node)
        }
        
        return report
    
    def _get_current_situation(self, node) -> Dict[str, Any]:
        """Get current situation at the time of the decision"""
        return {
            'portfolio_value': self.engine.current_portfolio['value'],
            'equity_allocation': self.engine.current_portfolio['equity'],
            'bonds_allocation': self.engine.current_portfolio['bonds'],
            'cash_allocation': self.engine.current_portfolio['cash'],
            'financial_stress': self.engine.life_state['financial_stress'],
            'risk_tolerance': self.engine.life_state['risk_tolerance'],
            'age': self.engine.life_state['age'],
            'career_stage': self.engine.life_state['career_stage'],
            'family_status': self.engine.life_state['family_status']
        }
    
    def _format_choices(self, choices: List[Dict], status: str) -> List[Dict[str, Any]]:
        """Format available choices for the report"""
        formatted_choices = []
        
        for choice in choices:
            formatted_choice = {
                'type': choice.get('type', 'unknown'),
                'description': choice.get('description', ''),
                'risk_level': choice.get('risk_level', 'medium'),
                'portfolio_impact': choice.get('portfolio_impact', {}),
                'life_impact': choice.get('life_impact', {}),
                'status': status,
                'score': self._calculate_choice_score(choice)
            }
            formatted_choices.append(formatted_choice)
        
        return formatted_choices
    
    def _format_closed_paths(self, closed_paths: List[Dict]) -> List[Dict[str, Any]]:
        """Format closed opportunities for the report"""
        formatted_closed = []
        
        for path in closed_paths:
            formatted_path = {
                'type': path.get('type', 'unknown'),
                'reason': path.get('reason', 'Previous decision'),
                'status': 'closed',
                'impact': 'This option is no longer available due to previous decisions'
            }
            formatted_closed.append(formatted_path)
        
        return formatted_closed
    
    def _format_optimal_choice(self, optimal_choice: Dict) -> Dict[str, Any]:
        """Format the optimal recommendation"""
        return {
            'type': optimal_choice.get('type', 'unknown'),
            'description': optimal_choice.get('description', ''),
            'risk_level': optimal_choice.get('risk_level', 'medium'),
            'portfolio_impact': optimal_choice.get('portfolio_impact', {}),
            'life_impact': optimal_choice.get('life_impact', {}),
            'highlighted': True,
            'rationale': self._generate_choice_rationale(optimal_choice)
        }
    
    def _calculate_choice_score(self, choice: Dict) -> float:
        """Calculate a score for a choice based on current state"""
        score = 0.0
        
        # Portfolio impact scoring
        if 'portfolio_impact' in choice:
            portfolio_impact = choice['portfolio_impact']
            if 'risk_reduction' in portfolio_impact:
                if self.engine.life_state['financial_stress'] > 0.5:
                    score += 2.0
                else:
                    score += 0.5
            
            if 'equity_increase' in portfolio_impact:
                if self.engine.life_state['risk_tolerance'] > 0.7:
                    score += 1.5
                else:
                    score += 0.3
        
        # Life impact scoring
        if 'life_impact' in choice:
            life_impact = choice['life_impact']
            if 'stress_reduction' in life_impact:
                score += 1.0
            if 'income_increase' in life_impact:
                score += 1.5
        
        return score
    
    def _generate_choice_rationale(self, choice: Dict) -> str:
        """Generate rationale for a specific choice"""
        rationale_parts = []
        
        if 'portfolio_impact' in choice:
            impact = choice['portfolio_impact']
            if 'risk_reduction' in impact:
                rationale_parts.append("Reduces portfolio risk")
            if 'equity_increase' in impact:
                rationale_parts.append("Increases growth potential")
        
        if 'life_impact' in choice:
            impact = choice['life_impact']
            if 'stress_reduction' in impact:
                rationale_parts.append("Reduces financial stress")
            if 'income_increase' in impact:
                rationale_parts.append("Increases income potential")
        
        if not rationale_parts:
            rationale_parts.append("Balanced approach")
        
        return " ".join(rationale_parts)
    
    def _generate_rationale(self, node) -> str:
        """Generate overall rationale for the optimal choice"""
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
    
    def _calculate_future_implications(self, node) -> List[str]:
        """Calculate future implications of the optimal choice"""
        implications = []
        
        if node.optimal_choice:
            choice = node.optimal_choice
            
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
    
    def _calculate_portfolio_impact(self, node) -> Dict[str, Any]:
        """Calculate portfolio impact of the optimal choice"""
        if not node.optimal_choice or 'portfolio_impact' not in node.optimal_choice:
            return {'equity_change': 0, 'risk_change': 0, 'cash_change': 0}
        
        impact = node.optimal_choice['portfolio_impact']
        
        return {
            'equity_change': impact.get('equity_increase', 0),
            'risk_change': impact.get('risk_reduction', 0),
            'cash_change': impact.get('cash_increase', 0),
            'new_allocation': self._calculate_new_allocation(impact)
        }
    
    def _calculate_new_allocation(self, impact: Dict) -> Dict[str, float]:
        """Calculate new allocation based on impact"""
        current = self.engine.current_portfolio.copy()
        
        if 'equity_increase' in impact:
            current['equity'] = min(0.9, current['equity'] + impact['equity_increase'])
        if 'risk_reduction' in impact:
            current['equity'] = max(0.2, current['equity'] - impact['risk_reduction'])
            current['bonds'] = min(0.6, current['bonds'] + impact['risk_reduction'])
        
        return current
    
    def _calculate_life_impact(self, node) -> Dict[str, Any]:
        """Calculate life impact of the optimal choice"""
        if not node.optimal_choice or 'life_impact' not in node.optimal_choice:
            return {'stress_change': 0, 'income_change': 0}
        
        impact = node.optimal_choice['life_impact']
        
        return {
            'stress_change': impact.get('stress_reduction', 0),
            'income_change': impact.get('income_increase', 0),
            'new_stress_level': max(0.0, self.engine.life_state['financial_stress'] - impact.get('stress_reduction', 0))
        }
    
    def _calculate_decision_metrics(self, node) -> Dict[str, Any]:
        """Calculate metrics for the decision"""
        total_choices = len(node.available_options)
        closed_choices = len(node.closed_paths)
        available_choices = total_choices - closed_choices
        
        return {
            'total_choices': total_choices,
            'available_choices': available_choices,
            'closed_choices': closed_choices,
            'choice_availability_ratio': available_choices / total_choices if total_choices > 0 else 0,
            'optimal_choice_score': self._calculate_choice_score(node.optimal_choice),
            'decision_complexity': 'High' if total_choices > 3 else 'Medium' if total_choices > 2 else 'Low'
        }
    
    def _create_visualization_data(self, node) -> Dict[str, Any]:
        """Create data for decision visualization"""
        return {
            'choice_distribution': {
                'available': len(node.available_options),
                'closed': len(node.closed_paths),
                'optimal': 1
            },
            'risk_distribution': {
                'low': len([c for c in node.available_options if c.get('risk_level') == 'low']),
                'medium': len([c for c in node.available_options if c.get('risk_level') == 'medium']),
                'high': len([c for c in node.available_options if c.get('risk_level') == 'high'])
            },
            'impact_distribution': {
                'portfolio_focused': len([c for c in node.available_options if 'portfolio_impact' in c]),
                'life_focused': len([c for c in node.available_options if 'life_impact' in c]),
                'balanced': len([c for c in node.available_options if 'portfolio_impact' in c and 'life_impact' in c])
            }
        }
    
    def create_individual_report_html(self, report: Dict[str, Any]) -> str:
        """Create HTML report for an individual decision"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Decision Report - {report['event_title']}</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .content {{
                    padding: 30px;
                }}
                .report-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin-bottom: 30px;
                }}
                .report-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #667eea;
                }}
                .report-card h3 {{
                    margin: 0 0 15px 0;
                    color: #333;
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    margin: 8px 0;
                    padding: 5px 0;
                    border-bottom: 1px solid #eee;
                }}
                .metric:last-child {{
                    border-bottom: none;
                }}
                .optimal-choice {{
                    background: linear-gradient(135deg, #2ca02c 0%, #1f77b4 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .closed-opportunities {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #dc3545;
                }}
                .visualization {{
                    width: 100%;
                    height: 400px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    background: #e9ecef;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Financial Decision Report</h1>
                    <p>{report['event_title']} - {report['event_description']}</p>
                </div>
                
                <div class="content">
                    <div class="timestamp">
                        📅 Decision Date: {report['timestamp']}
                    </div>
                    
                    <div class="report-grid">
                        <div class="report-card">
                            <h3>📈 Current Situation</h3>
                            <div class="metric">
                                <span>Portfolio Value:</span>
                                <strong>${report['current_situation']['portfolio_value']:,.0f}</strong>
                            </div>
                            <div class="metric">
                                <span>Equity Allocation:</span>
                                <strong>{report['current_situation']['equity_allocation']:.1%}</strong>
                            </div>
                            <div class="metric">
                                <span>Financial Stress:</span>
                                <strong>{report['current_situation']['financial_stress']:.1%}</strong>
                            </div>
                            <div class="metric">
                                <span>Risk Tolerance:</span>
                                <strong>{report['current_situation']['risk_tolerance']:.1%}</strong>
                            </div>
                        </div>
                        
                        <div class="report-card">
                            <h3>🎯 Decision Metrics</h3>
                            <div class="metric">
                                <span>Available Choices:</span>
                                <strong>{report['decision_metrics']['available_choices']}</strong>
                            </div>
                            <div class="metric">
                                <span>Closed Opportunities:</span>
                                <strong>{report['decision_metrics']['closed_choices']}</strong>
                            </div>
                            <div class="metric">
                                <span>Choice Availability:</span>
                                <strong>{report['decision_metrics']['choice_availability_ratio']:.1%}</strong>
                            </div>
                            <div class="metric">
                                <span>Decision Complexity:</span>
                                <strong>{report['decision_metrics']['decision_complexity']}</strong>
                            </div>
                        </div>
                    </div>
                    
                    <div class="optimal-choice">
                        <h3>⭐ Optimal Recommendation</h3>
                        <p><strong>{report['optimal_recommendation']['description']}</strong></p>
                        <p><em>Rationale: {report['rationale']}</em></p>
                        <div style="margin-top: 15px;">
                            <strong>Future Implications:</strong>
                            <ul>
                                {''.join([f'<li>{implication}</li>' for implication in report['future_implications']])}
                            </ul>
                        </div>
                    </div>
                    
                    <div class="closed-opportunities">
                        <h3>🚫 Closed Opportunities</h3>
                        <p>The following options are no longer available due to previous decisions:</p>
                        <ul>
                            {''.join([f'<li><strong>{path["type"].title()}</strong>: {path["reason"]}</li>' for path in report['closed_opportunities']])}
                        </ul>
                    </div>
                    
                    <div class="visualization">
                        <h3>📊 Decision Visualization</h3>
                        <div id="decision-chart"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // Decision visualization data
                const decisionData = {json.dumps(report['visualization_data'])};
                
                // Create D3.js visualization
                const margin = {{top: 20, right: 20, bottom: 20, left: 20}};
                const width = 1160 - margin.left - margin.right;
                const height = 360 - margin.top - margin.bottom;
                
                const svg = d3.select("#decision-chart")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
                
                // Create choice distribution chart
                const choiceData = [
                    {{label: 'Available', value: decisionData.choice_distribution.available, color: '#d62728'}},
                    {{label: 'Closed', value: decisionData.choice_distribution.closed, color: '#7f7f7f'}},
                    {{label: 'Optimal', value: decisionData.choice_distribution.optimal, color: '#2ca02c'}}
                ];
                
                const radius = Math.min(width, height) / 2;
                const arc = d3.arc().innerRadius(0).outerRadius(radius);
                const pie = d3.pie().value(d => d.value);
                
                const g = svg.append("g")
                    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");
                
                g.selectAll("path")
                    .data(pie(choiceData))
                    .enter()
                    .append("path")
                    .attr("d", arc)
                    .attr("fill", d => d.data.color)
                    .attr("stroke", "white")
                    .style("stroke-width", "2px");
                
                // Add labels
                g.selectAll("text")
                    .data(pie(choiceData))
                    .enter()
                    .append("text")
                    .text(d => d.data.label + ": " + d.data.value)
                    .attr("transform", d => "translate(" + arc.centroid(d) + ")")
                    .style("text-anchor", "middle")
                    .style("font-size", "12px")
                    .style("fill", "white");
                
                console.log("Decision report visualization loaded");
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def generate_all_reports_html(self, output_dir: str = "docs/reports") -> List[str]:
        """Generate HTML reports for all decisions"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        reports = self.generate_timestamped_reports()
        html_files = []
        
        for i, report in enumerate(reports):
            html_content = self.create_individual_report_html(report)
            filename = f"{output_dir}/decision_report_{i+1}_{report['event_title'].replace(' ', '_').lower()}.html"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            html_files.append(filename)
        
        return html_files

def create_story_reports_demo():
    """Create a demo of the story report generator"""
    print("📊 Creating Story-Driven Report Generator Demo")
    print("=" * 50)
    
    # Create sample story
    engine, reports = create_sample_story()
    
    # Create report generator
    report_generator = StoryReportGenerator(engine)
    
    # Generate all reports
    html_files = report_generator.generate_all_reports_html()
    
    print("✅ Story reports generated:")
    for i, filename in enumerate(html_files, 1):
        print(f"   Report {i}: {filename}")
    
    # Print summary
    all_reports = report_generator.generate_timestamped_reports()
    print(f"\n📋 Report Summary:")
    print(f"   - Total Reports: {len(all_reports)}")
    print(f"   - Total Decisions: {len(engine.decision_timeline)}")
    print(f"   - Optimal Choices: {sum(1 for node in engine.decision_timeline if node.chosen_path == node.optimal_choice)}")
    
    return html_files

if __name__ == "__main__":
    create_story_reports_demo() 