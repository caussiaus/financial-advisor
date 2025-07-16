#!/usr/bin/env python
# Story-Driven Parallel Sets Visualizer
# Shows branching narratives with red (feasible) and grey (closed) paths
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from story_driven_financial_engine import StoryDrivenFinancialEngine, create_sample_story

class StoryDrivenParallelSets:
    """Parallel sets visualization for story-driven financial decisions"""
    
    def __init__(self, story_engine: StoryDrivenFinancialEngine):
        """Initialize with story-driven financial engine"""
        self.engine = story_engine
        self.story_data = self._create_story_data()
        
    def _create_story_data(self):
        """Create data structure for story-driven parallel sets"""
        story_data = {
            'decision_points': [],
            'available_paths': [],
            'closed_paths': [],
            'optimal_choices': [],
            'chosen_paths': [],
            'portfolio_impacts': [],
            'life_impacts': []
        }
        
        for node in self.engine.decision_timeline:
            # Decision point
            story_data['decision_points'].append({
                'timestamp': node.timestamp.strftime('%Y-%m-%d'),
                'event_type': node.event_type,
                'description': node.description
            })
            
            # Available paths (red - feasible)
            for option in node.available_options:
                story_data['available_paths'].append({
                    'decision_point': node.event_type,
                    'path_type': option.get('type', 'unknown'),
                    'description': option.get('description', ''),
                    'risk_level': option.get('risk_level', 'medium'),
                    'portfolio_impact': option.get('portfolio_impact', {}),
                    'life_impact': option.get('life_impact', {}),
                    'status': 'available'
                })
            
            # Closed paths (grey - no longer available)
            for closed_path in node.closed_paths:
                story_data['closed_paths'].append({
                    'decision_point': node.event_type,
                    'path_type': closed_path.get('type', 'unknown'),
                    'reason': closed_path.get('reason', 'Previous decision'),
                    'status': 'closed'
                })
            
            # Optimal choice
            story_data['optimal_choices'].append({
                'decision_point': node.event_type,
                'choice_type': node.optimal_choice.get('type', 'unknown'),
                'description': node.optimal_choice.get('description', ''),
                'highlighted': True
            })
            
            # Chosen path (if decision was made)
            if node.chosen_path:
                story_data['chosen_paths'].append({
                    'decision_point': node.event_type,
                    'chosen_type': node.chosen_path.get('type', 'unknown'),
                    'optimal_type': node.optimal_choice.get('type', 'unknown'),
                    'was_optimal': node.chosen_path == node.optimal_choice
                })
            
            # Portfolio impacts
            if node.chosen_path and 'portfolio_impact' in node.chosen_path:
                story_data['portfolio_impacts'].append({
                    'decision_point': node.event_type,
                    'equity_change': node.chosen_path['portfolio_impact'].get('equity_increase', 0),
                    'risk_change': node.chosen_path['portfolio_impact'].get('risk_reduction', 0),
                    'impact_type': 'positive' if node.chosen_path == node.optimal_choice else 'suboptimal'
                })
            
            # Life impacts
            if node.chosen_path and 'life_impact' in node.chosen_path:
                story_data['life_impacts'].append({
                    'decision_point': node.event_type,
                    'stress_change': node.chosen_path['life_impact'].get('stress_reduction', 0),
                    'income_change': node.chosen_path['life_impact'].get('income_increase', 0),
                    'impact_type': 'positive' if node.chosen_path == node.optimal_choice else 'suboptimal'
                })
        
        return story_data
    
    def create_story_parallel_sets(self):
        """Create parallel sets visualization for story-driven decisions"""
        
        # Create Sankey diagram showing decision flow
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = self._get_story_node_labels(),
                color = self._get_story_node_colors()
            ),
            link = dict(
                source = self._get_story_link_sources(),
                target = self._get_story_link_targets(),
                value = self._get_story_link_values(),
                color = self._get_story_link_colors()
            )
        )])
        
        fig.update_layout(
            title_text="Story-Driven Financial Journey: Decision Branches",
            font_size=10,
            height=800
        )
        
        return fig
    
    def _get_story_node_labels(self):
        """Generate node labels for story-driven Sankey diagram"""
        labels = []
        
        # Decision points
        for point in self.story_data['decision_points']:
            labels.append(f"Decision: {point['event_type'].replace('_', ' ').title()}")
        
        # Available paths
        for path in self.story_data['available_paths']:
            labels.append(f"Available: {path['path_type'].title()}")
        
        # Closed paths
        for path in self.story_data['closed_paths']:
            labels.append(f"Closed: {path['path_type'].title()}")
        
        # Optimal choices
        for choice in self.story_data['optimal_choices']:
            labels.append(f"Optimal: {choice['choice_type'].title()}")
        
        # Chosen paths
        for path in self.story_data['chosen_paths']:
            labels.append(f"Chosen: {path['chosen_type'].title()}")
        
        # Portfolio impacts
        for impact in self.story_data['portfolio_impacts']:
            labels.append(f"Portfolio: {impact['impact_type'].title()}")
        
        # Life impacts
        for impact in self.story_data['life_impacts']:
            labels.append(f"Life: {impact['impact_type'].title()}")
        
        return labels
    
    def _get_story_node_colors(self):
        """Generate node colors for story-driven Sankey diagram"""
        colors = []
        
        # Decision points - blue
        for _ in self.story_data['decision_points']:
            colors.append('#1f77b4')
        
        # Available paths - red (feasible)
        for _ in self.story_data['available_paths']:
            colors.append('#d62728')
        
        # Closed paths - grey (no longer available)
        for _ in self.story_data['closed_paths']:
            colors.append('#7f7f7f')
        
        # Optimal choices - green (highlighted)
        for _ in self.story_data['optimal_choices']:
            colors.append('#2ca02c')
        
        # Chosen paths - orange or red based on optimality
        for path in self.story_data['chosen_paths']:
            if path['was_optimal']:
                colors.append('#ff7f0e')  # Orange for optimal
            else:
                colors.append('#d62728')  # Red for suboptimal
        
        # Portfolio impacts - purple
        for _ in self.story_data['portfolio_impacts']:
            colors.append('#9467bd')
        
        # Life impacts - brown
        for _ in self.story_data['life_impacts']:
            colors.append('#8c564b')
        
        return colors
    
    def _get_story_link_sources(self):
        """Generate link sources for story-driven Sankey diagram"""
        sources = []
        node_index = 0
        
        # Map decision points to available paths
        decision_count = len(self.story_data['decision_points'])
        available_count = len(self.story_data['available_paths'])
        
        for i in range(decision_count):
            # Each decision point connects to its available paths
            for j in range(available_count // decision_count):  # Assume equal distribution
                sources.append(i)
        
        # Map available paths to optimal choices
        available_start = decision_count
        optimal_start = available_start + available_count + len(self.story_data['closed_paths'])
        
        for i in range(available_count):
            sources.append(available_start + i)
        
        # Map optimal choices to chosen paths
        chosen_start = optimal_start + len(self.story_data['optimal_choices'])
        
        for i in range(len(self.story_data['optimal_choices'])):
            sources.append(optimal_start + i)
        
        # Map chosen paths to impacts
        impact_start = chosen_start + len(self.story_data['chosen_paths'])
        
        for i in range(len(self.story_data['chosen_paths'])):
            sources.append(chosen_start + i)
        
        return sources
    
    def _get_story_link_targets(self):
        """Generate link targets for story-driven Sankey diagram"""
        targets = []
        
        decision_count = len(self.story_data['decision_points'])
        available_count = len(self.story_data['available_paths'])
        closed_count = len(self.story_data['closed_paths'])
        
        # Map decision points to available paths
        available_start = decision_count
        
        for i in range(decision_count):
            for j in range(available_count // decision_count):
                targets.append(available_start + i * (available_count // decision_count) + j)
        
        # Map available paths to optimal choices
        optimal_start = available_start + available_count + closed_count
        
        for i in range(available_count):
            targets.append(optimal_start + (i % len(self.story_data['optimal_choices'])))
        
        # Map optimal choices to chosen paths
        chosen_start = optimal_start + len(self.story_data['optimal_choices'])
        
        for i in range(len(self.story_data['optimal_choices'])):
            targets.append(chosen_start + i)
        
        # Map chosen paths to impacts
        impact_start = chosen_start + len(self.story_data['chosen_paths'])
        
        for i in range(len(self.story_data['chosen_paths'])):
            targets.append(impact_start + i)
        
        return targets
    
    def _get_story_link_values(self):
        """Generate link values for story-driven Sankey diagram"""
        # Equal weight for all connections
        return [1] * len(self._get_story_link_sources())
    
    def _get_story_link_colors(self):
        """Generate link colors for story-driven Sankey diagram"""
        colors = []
        
        # Decision to available paths - red (feasible)
        decision_count = len(self.story_data['decision_points'])
        available_count = len(self.story_data['available_paths'])
        
        for i in range(decision_count):
            for j in range(available_count // decision_count):
                colors.append('rgba(214, 39, 40, 0.6)')  # Red for feasible
        
        # Available to optimal - green (highlighted)
        for i in range(available_count):
            colors.append('rgba(44, 160, 44, 0.6)')  # Green for optimal
        
        # Optimal to chosen - orange/red based on optimality
        for path in self.story_data['chosen_paths']:
            if path['was_optimal']:
                colors.append('rgba(255, 127, 14, 0.6)')  # Orange for optimal
            else:
                colors.append('rgba(214, 39, 40, 0.6)')  # Red for suboptimal
        
        # Chosen to impacts - purple/brown
        for i in range(len(self.story_data['chosen_paths'])):
            colors.append('rgba(148, 103, 189, 0.6)')  # Purple for portfolio
        
        return colors
    
    def create_interactive_story_visualization(self):
        """Create interactive D3.js visualization for story-driven decisions"""
        
        # Create HTML template with D3.js
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Story-Driven Financial Journey</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
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
                .story-timeline {{
                    width: 100%;
                    height: 600px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .legend {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    margin: 20px 0;
                    flex-wrap: wrap;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: 500;
                }}
                .legend-red {{
                    background: rgba(214, 39, 40, 0.1);
                    color: #d62728;
                    border: 2px solid #d62728;
                }}
                .legend-grey {{
                    background: rgba(127, 127, 127, 0.1);
                    color: #7f7f7f;
                    border: 2px solid #7f7f7f;
                }}
                .legend-green {{
                    background: rgba(44, 160, 44, 0.1);
                    color: #2ca02c;
                    border: 2px solid #2ca02c;
                }}
                .legend-orange {{
                    background: rgba(255, 127, 14, 0.1);
                    color: #ff7f0e;
                    border: 2px solid #ff7f0e;
                }}
                .tooltip {{
                    position: absolute;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    font-size: 12px;
                    pointer-events: none;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    z-index: 1000;
                }}
                .decision-node {{
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                .decision-node:hover {{
                    transform: scale(1.1);
                }}
                .path-line {{
                    stroke-width: 3;
                    opacity: 0.7;
                    transition: opacity 0.3s ease;
                }}
                .path-line:hover {{
                    opacity: 1;
                    stroke-width: 5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎮 Story-Driven Financial Journey</h1>
                    <p>Interactive visualization of decision branches and their consequences</p>
                </div>
                
                <div class="content">
                    <div class="legend">
                        <div class="legend-item legend-red">
                            <div style="width: 12px; height: 12px; background: #d62728; border-radius: 50%;"></div>
                            Feasible Paths (Red)
                        </div>
                        <div class="legend-item legend-grey">
                            <div style="width: 12px; height: 12px; background: #7f7f7f; border-radius: 50%;"></div>
                            Closed Paths (Grey)
                        </div>
                        <div class="legend-item legend-green">
                            <div style="width: 12px; height: 12px; background: #2ca02c; border-radius: 50%;"></div>
                            Optimal Choices (Green)
                        </div>
                        <div class="legend-item legend-orange">
                            <div style="width: 12px; height: 12px; background: #ff7f0e; border-radius: 50%;"></div>
                            Chosen Paths (Orange)
                        </div>
                    </div>
                    
                    <div class="story-timeline"></div>
                </div>
            </div>
            
            <script>
                // Story data from Python
                const storyData = {json.dumps(self.story_data)};
                
                // D3.js Story-Driven Visualization
                const margin = {{top: 20, right: 20, bottom: 20, left: 20}};
                const width = 1360 - margin.left - margin.right;
                const height = 560 - margin.top - margin.bottom;
                
                const svg = d3.select(".story-timeline")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
                
                // Create decision timeline
                const decisionPoints = storyData.decision_points;
                const availablePaths = storyData.available_paths;
                const closedPaths = storyData.closed_paths;
                const optimalChoices = storyData.optimal_choices;
                const chosenPaths = storyData.chosen_paths;
                
                // Create scales
                const xScale = d3.scaleLinear()
                    .domain([0, decisionPoints.length - 1])
                    .range([0, width]);
                
                const yScale = d3.scaleLinear()
                    .domain([0, 100])
                    .range([height, 0]);
                
                // Create decision nodes
                const decisionNodes = svg.selectAll(".decision-node")
                    .data(decisionPoints)
                    .enter()
                    .append("circle")
                    .attr("class", "decision-node")
                    .attr("cx", (d, i) => xScale(i))
                    .attr("cy", height / 2)
                    .attr("r", 15)
                    .style("fill", "#1f77b4")
                    .style("stroke", "#000")
                    .style("stroke-width", 2);
                
                // Add decision labels
                svg.selectAll(".decision-label")
                    .data(decisionPoints)
                    .enter()
                    .append("text")
                    .attr("class", "decision-label")
                    .attr("x", (d, i) => xScale(i))
                    .attr("y", height / 2 + 30)
                    .attr("text-anchor", "middle")
                    .style("font-size", "12px")
                    .style("font-weight", "bold")
                    .text(d => d.event_type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase()));
                
                // Create available paths (red)
                const availablePathNodes = svg.selectAll(".available-node")
                    .data(availablePaths)
                    .enter()
                    .append("circle")
                    .attr("class", "available-node")
                    .attr("cx", (d, i) => xScale(i % decisionPoints.length) + (i % 3 - 1) * 30)
                    .attr("cy", height / 2 - 80)
                    .attr("r", 8)
                    .style("fill", "#d62728")
                    .style("stroke", "#000")
                    .style("stroke-width", 1);
                
                // Create closed paths (grey)
                const closedPathNodes = svg.selectAll(".closed-node")
                    .data(closedPaths)
                    .enter()
                    .append("circle")
                    .attr("class", "closed-node")
                    .attr("cx", (d, i) => xScale(i % decisionPoints.length) + (i % 3 - 1) * 30)
                    .attr("cy", height / 2 + 80)
                    .attr("r", 8)
                    .style("fill", "#7f7f7f")
                    .style("stroke", "#000")
                    .style("stroke-width", 1);
                
                // Create optimal choices (green)
                const optimalNodes = svg.selectAll(".optimal-node")
                    .data(optimalChoices)
                    .enter()
                    .append("circle")
                    .attr("class", "optimal-node")
                    .attr("cx", (d, i) => xScale(i) + 50)
                    .attr("cy", height / 2)
                    .attr("r", 12)
                    .style("fill", "#2ca02c")
                    .style("stroke", "#000")
                    .style("stroke-width", 2);
                
                // Create chosen paths (orange/red)
                const chosenNodes = svg.selectAll(".chosen-node")
                    .data(chosenPaths)
                    .enter()
                    .append("circle")
                    .attr("class", "chosen-node")
                    .attr("cx", (d, i) => xScale(i) + 100)
                    .attr("cy", height / 2)
                    .attr("r", 10)
                    .style("fill", d => d.was_optimal ? "#ff7f0e" : "#d62728")
                    .style("stroke", "#000")
                    .style("stroke-width", 2);
                
                // Create path connections
                const pathConnections = svg.selectAll(".path-line")
                    .data(availablePaths)
                    .enter()
                    .append("line")
                    .attr("class", "path-line")
                    .attr("x1", (d, i) => xScale(i % decisionPoints.length))
                    .attr("y1", height / 2)
                    .attr("x2", (d, i) => xScale(i % decisionPoints.length) + (i % 3 - 1) * 30)
                    .attr("y2", height / 2 - 80)
                    .style("stroke", "#d62728")
                    .style("stroke-width", 2);
                
                // Create tooltip
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);
                
                // Add hover effects
                decisionNodes.on("mouseover", function(event, d) {{
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`<strong>Decision Point</strong><br/>${{d.event_type.replace(/_/g, ' ')}}<br/>${{d.description}}`)
                        .style("left", (event.pageX + 5) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function(d) {{
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                }});
                
                availablePathNodes.on("mouseover", function(event, d) {{
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`<strong>Available Path</strong><br/>${{d.path_type}}<br/>${{d.description}}`)
                        .style("left", (event.pageX + 5) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function(d) {{
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                }});
                
                closedPathNodes.on("mouseover", function(event, d) {{
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(`<strong>Closed Path</strong><br/>${{d.path_type}}<br/>Reason: ${{d.reason}}`)
                        .style("left", (event.pageX + 5) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }})
                .on("mouseout", function(d) {{
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                }});
                
                console.log("Story-driven financial journey visualization loaded");
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def save_story_visualization_html(self, filename="story_driven_journey.html"):
        """Save the story-driven visualization as HTML file"""
        html_content = self.create_interactive_story_visualization()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename

def create_story_driven_demo():
    """Create a demo of the story-driven parallel sets visualization"""
    print("🎮 Creating Story-Driven Financial Journey Demo")
    print("=" * 50)
    
    # Create sample story
    engine, reports = create_sample_story()
    
    # Create story-driven parallel sets
    story_visualizer = StoryDrivenParallelSets(engine)
    
    # Generate visualizations
    plotly_fig = story_visualizer.create_story_parallel_sets()
    html_file = story_visualizer.save_story_visualization_html("docs/story_driven_journey.html")
    
    print("✅ Story-driven visualizations created:")
    print(f"   - Plotly Sankey: Available as figure object")
    print(f"   - D3.js Interactive: {html_file}")
    
    # Print story summary
    print("\n📖 Story Summary:")
    for i, report in enumerate(reports, 1):
        print(f"   Event {i}: {report.event_title}")
        print(f"     - Optimal: {report.optimal_recommendation['description']}")
        print(f"     - Rationale: {report.rationale}")
        print(f"     - Implications: {', '.join(report.future_implications)}")
    
    return plotly_fig, html_file

if __name__ == "__main__":
    create_story_driven_demo() 