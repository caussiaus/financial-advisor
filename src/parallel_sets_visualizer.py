#!/usr/bin/env python
# Parallel Sets Visualizer for IPS Analysis
# Shows relationships between life events, portfolio allocations, and outcomes
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ParallelSetsVisualizer:
    """Parallel sets visualization for showing relationships in IPS data"""
    
    def __init__(self, portfolio_engine=None):
        """Initialize with portfolio engine data"""
        self.engine = portfolio_engine
        self.data = portfolio_engine.export_data() if portfolio_engine else None
        
    def create_parallel_sets_data(self):
        """Create data structure for parallel sets visualization"""
        if not self.data:
            return self._create_sample_data()
        
        # Extract key dimensions for parallel sets
        sets_data = {
            'life_events': [],
            'portfolio_allocations': [],
            'market_conditions': [],
            'comfort_levels': [],
            'performance_outcomes': []
        }
        
        # Process life events
        for event in self.data.get('life_events_log', []):
            sets_data['life_events'].append({
                'category': event['type'],
                'impact': 'High' if abs(event['impact_score']) > 0.5 else 'Medium' if abs(event['impact_score']) > 0.2 else 'Low',
                'direction': 'Positive' if event['impact_score'] > 0 else 'Negative',
                'date': event['date']
            })
        
        # Process portfolio snapshots
        for snapshot in self.data.get('portfolio_snapshots', []):
            # Portfolio allocation categories
            equity_pct = snapshot['portfolio']['equity']
            if equity_pct > 0.7:
                allocation_category = 'High Equity'
            elif equity_pct > 0.5:
                allocation_category = 'Balanced'
            else:
                allocation_category = 'Conservative'
            
            # Market conditions
            market_stress = snapshot['market_data']['market_stress']
            if market_stress > 0.7:
                market_category = 'High Stress'
            elif market_stress > 0.3:
                market_category = 'Moderate Stress'
            else:
                market_category = 'Low Stress'
            
            # Comfort levels
            comfort = snapshot['comfort_metrics']['comfort_score']
            if comfort > 0.7:
                comfort_category = 'High Comfort'
            elif comfort > 0.4:
                comfort_category = 'Moderate Comfort'
            else:
                comfort_category = 'Low Comfort'
            
            # Performance outcomes (simulated)
            equity_return = snapshot['market_data']['equity_returns']
            bond_return = snapshot['market_data']['bond_returns']
            total_return = equity_return * snapshot['portfolio']['equity'] + bond_return * snapshot['portfolio']['bonds']
            
            if total_return > 0.05:
                performance_category = 'High Performance'
            elif total_return > 0:
                performance_category = 'Positive Performance'
            else:
                performance_category = 'Negative Performance'
            
            sets_data['portfolio_allocations'].append(allocation_category)
            sets_data['market_conditions'].append(market_category)
            sets_data['comfort_levels'].append(comfort_category)
            sets_data['performance_outcomes'].append(performance_category)
        
        return sets_data
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        return {
            'life_events': [
                {'category': 'Career Change', 'impact': 'High', 'direction': 'Positive'},
                {'category': 'Market Crash', 'impact': 'High', 'direction': 'Negative'},
                {'category': 'Retirement', 'impact': 'Medium', 'direction': 'Positive'},
                {'category': 'Health Issue', 'impact': 'Medium', 'direction': 'Negative'},
                {'category': 'Inheritance', 'impact': 'High', 'direction': 'Positive'},
                {'category': 'Job Loss', 'impact': 'High', 'direction': 'Negative'},
                {'category': 'Market Recovery', 'impact': 'Medium', 'direction': 'Positive'},
                {'category': 'Tax Change', 'impact': 'Low', 'direction': 'Negative'}
            ],
            'portfolio_allocations': ['High Equity', 'Conservative', 'Balanced', 'Conservative', 'High Equity', 'Conservative', 'Balanced', 'Balanced'],
            'market_conditions': ['Low Stress', 'High Stress', 'Moderate Stress', 'Low Stress', 'Low Stress', 'High Stress', 'Moderate Stress', 'Low Stress'],
            'comfort_levels': ['High Comfort', 'Low Comfort', 'Moderate Comfort', 'Low Comfort', 'High Comfort', 'Low Comfort', 'Moderate Comfort', 'High Comfort'],
            'performance_outcomes': ['High Performance', 'Negative Performance', 'Positive Performance', 'Negative Performance', 'High Performance', 'Negative Performance', 'Positive Performance', 'Positive Performance']
        }
    
    def create_parallel_sets_visualization(self):
        """Create parallel sets visualization using Plotly"""
        sets_data = self.create_parallel_sets_data()
        
        # Create Sankey diagram (closest to parallel sets in Plotly)
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = self._get_node_labels(sets_data),
                color = self._get_node_colors(sets_data)
            ),
            link = dict(
                source = self._get_link_sources(sets_data),
                target = self._get_link_targets(sets_data),
                value = self._get_link_values(sets_data),
                color = self._get_link_colors(sets_data)
            )
        )])
        
        fig.update_layout(
            title_text="IPS Analysis: Life Events → Portfolio Allocation → Market Conditions → Comfort → Performance",
            font_size=10,
            height=800
        )
        
        return fig
    
    def _get_node_labels(self, sets_data):
        """Generate node labels for Sankey diagram"""
        labels = []
        
        # Life events
        for event in sets_data['life_events']:
            labels.append(f"{event['category']} ({event['impact']})")
        
        # Portfolio allocations
        labels.extend(list(set(sets_data['portfolio_allocations'])))
        
        # Market conditions
        labels.extend(list(set(sets_data['market_conditions'])))
        
        # Comfort levels
        labels.extend(list(set(sets_data['comfort_levels'])))
        
        # Performance outcomes
        labels.extend(list(set(sets_data['performance_outcomes'])))
        
        return labels
    
    def _get_node_colors(self, sets_data):
        """Generate node colors for Sankey diagram"""
        colors = []
        
        # Life events colors
        for event in sets_data['life_events']:
            if event['direction'] == 'Positive':
                colors.append('lightgreen')
            else:
                colors.append('lightcoral')
        
        # Portfolio allocation colors
        allocation_colors = {'High Equity': '#ff7f0e', 'Balanced': '#2ca02c', 'Conservative': '#1f77b4'}
        for alloc in list(set(sets_data['portfolio_allocations'])):
            colors.append(allocation_colors.get(alloc, '#7f7f7f'))
        
        # Market condition colors
        market_colors = {'High Stress': '#d62728', 'Moderate Stress': '#ff7f0e', 'Low Stress': '#2ca02c'}
        for market in list(set(sets_data['market_conditions'])):
            colors.append(market_colors.get(market, '#7f7f7f'))
        
        # Comfort level colors
        comfort_colors = {'High Comfort': '#2ca02c', 'Moderate Comfort': '#ff7f0e', 'Low Comfort': '#d62728'}
        for comfort in list(set(sets_data['comfort_levels'])):
            colors.append(comfort_colors.get(comfort, '#7f7f7f'))
        
        # Performance colors
        perf_colors = {'High Performance': '#2ca02c', 'Positive Performance': '#1f77b4', 'Negative Performance': '#d62728'}
        for perf in list(set(sets_data['performance_outcomes'])):
            colors.append(perf_colors.get(perf, '#7f7f7f'))
        
        return colors
    
    def _get_link_sources(self, sets_data):
        """Generate link sources for Sankey diagram"""
        sources = []
        node_index = 0
        
        # Map life events to portfolio allocations
        for i, event in enumerate(sets_data['life_events']):
            sources.append(i)
        
        # Map portfolio allocations to market conditions
        allocation_start = len(sets_data['life_events'])
        allocation_indices = {}
        for i, alloc in enumerate(sets_data['portfolio_allocations']):
            if alloc not in allocation_indices:
                allocation_indices[alloc] = allocation_start + len(allocation_indices)
            sources.append(allocation_indices[alloc])
        
        # Map market conditions to comfort levels
        market_start = allocation_start + len(set(sets_data['portfolio_allocations']))
        market_indices = {}
        for i, market in enumerate(sets_data['market_conditions']):
            if market not in market_indices:
                market_indices[market] = market_start + len(market_indices)
            sources.append(market_indices[market])
        
        # Map comfort levels to performance outcomes
        comfort_start = market_start + len(set(sets_data['market_conditions']))
        comfort_indices = {}
        for i, comfort in enumerate(sets_data['comfort_levels']):
            if comfort not in comfort_indices:
                comfort_indices[comfort] = comfort_start + len(comfort_indices)
            sources.append(comfort_indices[comfort])
        
        return sources
    
    def _get_link_targets(self, sets_data):
        """Generate link targets for Sankey diagram"""
        targets = []
        
        # Map life events to portfolio allocations
        allocation_start = len(sets_data['life_events'])
        allocation_indices = {}
        for i, alloc in enumerate(sets_data['portfolio_allocations']):
            if alloc not in allocation_indices:
                allocation_indices[alloc] = allocation_start + len(allocation_indices)
            targets.append(allocation_indices[alloc])
        
        # Map portfolio allocations to market conditions
        market_start = allocation_start + len(set(sets_data['portfolio_allocations']))
        market_indices = {}
        for i, market in enumerate(sets_data['market_conditions']):
            if market not in market_indices:
                market_indices[market] = market_start + len(market_indices)
            targets.append(market_indices[market])
        
        # Map market conditions to comfort levels
        comfort_start = market_start + len(set(sets_data['market_conditions']))
        comfort_indices = {}
        for i, comfort in enumerate(sets_data['comfort_levels']):
            if comfort not in comfort_indices:
                comfort_indices[comfort] = comfort_start + len(comfort_indices)
            targets.append(comfort_indices[comfort])
        
        # Map comfort levels to performance outcomes
        perf_start = comfort_start + len(set(sets_data['comfort_levels']))
        perf_indices = {}
        for i, perf in enumerate(sets_data['performance_outcomes']):
            if perf not in perf_indices:
                perf_indices[perf] = perf_start + len(perf_indices)
            targets.append(perf_indices[perf])
        
        return targets
    
    def _get_link_values(self, sets_data):
        """Generate link values for Sankey diagram"""
        # Equal weight for all connections for simplicity
        return [1] * len(self._get_link_sources(sets_data))
    
    def _get_link_colors(self, sets_data):
        """Generate link colors for Sankey diagram"""
        colors = []
        
        # Color based on life event impact
        for event in sets_data['life_events']:
            if event['direction'] == 'Positive':
                colors.append('rgba(144, 238, 144, 0.3)')
            else:
                colors.append('rgba(240, 128, 128, 0.3)')
        
        # Neutral colors for other connections
        remaining_links = len(self._get_link_sources(sets_data)) - len(sets_data['life_events'])
        colors.extend(['rgba(200, 200, 200, 0.3)'] * remaining_links)
        
        return colors
    
    def create_interactive_parallel_sets(self):
        """Create interactive parallel sets with D3.js integration"""
        sets_data = self.create_parallel_sets_data()
        
        # Create HTML template with D3.js
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IPS Parallel Sets Analysis</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .parallel-sets {{
                    width: 100%;
                    height: 600px;
                }}
                .axis {{
                    font-size: 12px;
                }}
                .axis path,
                .axis line {{
                    fill: none;
                    stroke: #000;
                    shape-rendering: crispEdges;
                }}
                .ribbon {{
                    fill-opacity: 0.7;
                    stroke: #000;
                    stroke-width: 0.5px;
                }}
                .ribbon:hover {{
                    fill-opacity: 0.9;
                }}
                .tooltip {{
                    position: absolute;
                    background: rgba(0,0,0,0.8);
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    pointer-events: none;
                }}
                .controls {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 5px;
                }}
                .filter-btn {{
                    margin: 5px;
                    padding: 8px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    background: #007bff;
                    color: white;
                }}
                .filter-btn:hover {{
                    background: #0056b3;
                }}
                .filter-btn.active {{
                    background: #28a745;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>IPS Analysis: Parallel Sets Visualization</h1>
                <p>This visualization shows the relationships between life events, portfolio allocations, market conditions, comfort levels, and performance outcomes.</p>
                
                <div class="controls">
                    <h3>Filters:</h3>
                    <button class="filter-btn active" data-filter="all">Show All</button>
                    <button class="filter-btn" data-filter="positive">Positive Events Only</button>
                    <button class="filter-btn" data-filter="negative">Negative Events Only</button>
                    <button class="filter-btn" data-filter="high-impact">High Impact Only</button>
                </div>
                
                <div class="parallel-sets"></div>
            </div>
            
            <script>
                // Data from Python
                const setsData = {json.dumps(sets_data)};
                
                // D3.js Parallel Sets Implementation
                const margin = {{top: 20, right: 20, bottom: 20, left: 20}};
                const width = 1160 - margin.left - margin.right;
                const height = 560 - margin.top - margin.bottom;
                
                const svg = d3.select(".parallel-sets")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
                
                // Create dimensions
                const dimensions = [
                    'life_events',
                    'portfolio_allocations', 
                    'market_conditions',
                    'comfort_levels',
                    'performance_outcomes'
                ];
                
                // Create scales for each dimension
                const scales = {{}};
                dimensions.forEach((dim, i) => {{
                    const uniqueValues = [...new Set(setsData[dim].map(d => 
                        typeof d === 'object' ? d.category : d
                    ))];
                    scales[dim] = d3.scalePoint()
                        .domain(uniqueValues)
                        .range([0, height]);
                }});
                
                // Create axes
                const axes = svg.selectAll(".axis")
                    .data(dimensions)
                    .enter()
                    .append("g")
                    .attr("class", "axis")
                    .attr("transform", (d, i) => "translate(" + (i * width / (dimensions.length - 1)) + ",0)");
                
                axes.append("line")
                    .attr("y1", 0)
                    .attr("y2", height);
                
                axes.append("text")
                    .attr("x", -6)
                    .attr("y", height + 6)
                    .attr("text-anchor", "end")
                    .text(d => d.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                
                // Add category labels
                dimensions.forEach((dim, i) => {{
                    const uniqueValues = [...new Set(setsData[dim].map(d => 
                        typeof d === 'object' ? d.category : d
                    ))];
                    
                    const axis = svg.selectAll(".axis").filter((d, j) => j === i);
                    
                    axis.selectAll(".category")
                        .data(uniqueValues)
                        .enter()
                        .append("text")
                        .attr("class", "category")
                        .attr("x", -6)
                        .attr("y", d => scales[dim](d))
                        .attr("text-anchor", "end")
                        .text(d => d);
                }});
                
                // Create tooltip
                const tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);
                
                // Add filter functionality
                d3.selectAll(".filter-btn").on("click", function() {{
                    const filter = d3.select(this).attr("data-filter");
                    
                    // Update active button
                    d3.selectAll(".filter-btn").classed("active", false);
                    d3.select(this).classed("active", true);
                    
                    // Apply filter logic here
                    console.log("Filter applied:", filter);
                }});
                
                // Add sample ribbons (simplified for demo)
                const ribbonData = [];
                for (let i = 0; i < Math.min(setsData.life_events.length, 5); i++) {{
                    ribbonData.push({{
                        source: setsData.life_events[i].category,
                        target: setsData.portfolio_allocations[i],
                        value: 1,
                        color: setsData.life_events[i].direction === 'Positive' ? '#2ca02c' : '#d62728'
                    }});
                }}
                
                // Draw sample ribbons
                ribbonData.forEach((ribbon, i) => {{
                    const x1 = 0;
                    const y1 = scales.life_events(ribbon.source);
                    const x2 = width / (dimensions.length - 1);
                    const y2 = scales.portfolio_allocations(ribbon.target);
                    
                    const line = d3.line()
                        .x(d => d.x)
                        .y(d => d.y)
                        .curve(d3.curveBasis);
                    
                    const pathData = [
                        {{x: x1, y: y1}},
                        {{x: x2, y: y2}}
                    ];
                    
                    svg.append("path")
                        .datum(pathData)
                        .attr("class", "ribbon")
                        .attr("d", line)
                        .style("stroke", ribbon.color)
                        .style("stroke-width", 3)
                        .style("fill", "none")
                        .on("mouseover", function(event, d) {{
                            tooltip.transition()
                                .duration(200)
                                .style("opacity", .9);
                            tooltip.html(`Source: ${{ribbon.source}}<br/>Target: ${{ribbon.target}}<br/>Value: ${{ribbon.value}}`)
                                .style("left", (event.pageX + 5) + "px")
                                .style("top", (event.pageY - 28) + "px");
                        }})
                        .on("mouseout", function(d) {{
                            tooltip.transition()
                                .duration(500)
                                .style("opacity", 0);
                        }});
                }});
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def save_parallel_sets_html(self, filename="parallel_sets_visualization.html"):
        """Save the parallel sets visualization as HTML file"""
        html_content = self.create_interactive_parallel_sets()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename

def create_parallel_sets_demo():
    """Create a demo of the parallel sets visualizer"""
    visualizer = ParallelSetsVisualizer()
    
    # Create both Plotly and D3.js versions
    plotly_fig = visualizer.create_parallel_sets_visualization()
    html_file = visualizer.save_parallel_sets_html("docs/parallel_sets_visualization.html")
    
    print(f"Parallel sets visualization created:")
    print(f"- Plotly version: Available as figure object")
    print(f"- D3.js version: Saved to {html_file}")
    
    return plotly_fig, html_file

if __name__ == "__main__":
    create_parallel_sets_demo() 