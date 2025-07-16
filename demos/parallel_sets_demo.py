#!/usr/bin/env python
# Parallel Sets Demo for IPS Analysis
# Demonstrates the parallel sets visualization with sample data
# Author: ChatGPT 2025-01-16

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parallel_sets_visualizer import ParallelSetsVisualizer
from src.dynamic_portfolio_engine import DynamicPortfolioEngine
from src.realistic_life_events_generator import RealisticLifeEventsGenerator
import json
import pandas as pd
from datetime import datetime, timedelta

def create_sample_ips_data():
    """Create sample IPS data for demonstration"""
    
    # Initialize portfolio engine with sample data
    client_config = {
        'age': 45,
        'retirement_age': 65,
        'portfolio_value': 500000,
        'annual_contribution': 20000,
        'risk_tolerance': 0.6,
        'investment_horizon': 20,
        'income_stability': 0.8,
        'liquidity_needs': 0.3
    }
    
    # Create life events generator
    life_events_gen = RealisticLifeEventsGenerator("sample_client", 2020)
    
    # Generate realistic life events
    life_events = life_events_gen.generate_complete_life_journey()
    
    # Create portfolio engine
    portfolio_engine = DynamicPortfolioEngine(client_config)
    
    # Simulate portfolio evolution with life events
    portfolio_engine.simulate_portfolio_evolution(life_events)
    
    return portfolio_engine

def demonstrate_parallel_sets():
    """Demonstrate the parallel sets visualization"""
    print("🚀 Creating Parallel Sets Visualization for IPS Analysis")
    print("=" * 60)
    
    # Create sample data
    print("📊 Generating sample IPS data...")
    portfolio_engine = create_sample_ips_data()
    
    # Create parallel sets visualizer
    print("🎨 Creating parallel sets visualizer...")
    visualizer = ParallelSetsVisualizer(portfolio_engine)
    
    # Create both visualization types
    print("📈 Creating Plotly Sankey diagram...")
    plotly_fig = visualizer.create_parallel_sets_visualization()
    
    print("🌐 Creating interactive D3.js visualization...")
    html_file = visualizer.save_parallel_sets_html("docs/parallel_sets_ips_analysis.html")
    
    # Display data summary
    sets_data = visualizer.create_parallel_sets_data()
    print("\n📋 Data Summary:")
    print(f"- Life Events: {len(sets_data['life_events'])} events")
    print(f"- Portfolio Allocations: {len(set(sets_data['portfolio_allocations']))} categories")
    print(f"- Market Conditions: {len(set(sets_data['market_conditions']))} categories")
    print(f"- Comfort Levels: {len(set(sets_data['comfort_levels']))} categories")
    print(f"- Performance Outcomes: {len(set(sets_data['performance_outcomes']))} categories")
    
    # Show sample relationships
    print("\n🔗 Sample Relationships:")
    for i, event in enumerate(sets_data['life_events'][:3]):
        print(f"  {event['category']} ({event['impact']}, {event['direction']}) → "
              f"{sets_data['portfolio_allocations'][i]} → "
              f"{sets_data['market_conditions'][i]} → "
              f"{sets_data['comfort_levels'][i]} → "
              f"{sets_data['performance_outcomes'][i]}")
    
    print(f"\n✅ Visualizations created successfully!")
    print(f"   - Plotly figure: Available for display")
    print(f"   - D3.js HTML: {html_file}")
    
    return plotly_fig, html_file

def analyze_relationships():
    """Analyze the relationships in the parallel sets data"""
    print("\n🔍 Analyzing Relationships in IPS Data")
    print("=" * 40)
    
    portfolio_engine = create_sample_ips_data()
    visualizer = ParallelSetsVisualizer(portfolio_engine)
    sets_data = visualizer.create_parallel_sets_data()
    
    # Analyze life event impacts
    positive_events = [e for e in sets_data['life_events'] if e['direction'] == 'Positive']
    negative_events = [e for e in sets_data['life_events'] if e['direction'] == 'Negative']
    
    print(f"📈 Life Events Analysis:")
    print(f"   - Positive events: {len(positive_events)}")
    print(f"   - Negative events: {len(negative_events)}")
    print(f"   - High impact events: {len([e for e in sets_data['life_events'] if e['impact'] == 'High'])}")
    
    # Analyze portfolio allocation patterns
    allocation_counts = pd.Series(sets_data['portfolio_allocations']).value_counts()
    print(f"\n💼 Portfolio Allocation Patterns:")
    for alloc, count in allocation_counts.items():
        print(f"   - {alloc}: {count} occurrences")
    
    # Analyze market condition patterns
    market_counts = pd.Series(sets_data['market_conditions']).value_counts()
    print(f"\n📊 Market Condition Patterns:")
    for market, count in market_counts.items():
        print(f"   - {market}: {count} occurrences")
    
    # Analyze comfort level patterns
    comfort_counts = pd.Series(sets_data['comfort_levels']).value_counts()
    print(f"\n😌 Comfort Level Patterns:")
    for comfort, count in comfort_counts.items():
        print(f"   - {comfort}: {count} occurrences")
    
    # Analyze performance patterns
    perf_counts = pd.Series(sets_data['performance_outcomes']).value_counts()
    print(f"\n📈 Performance Outcome Patterns:")
    for perf, count in perf_counts.items():
        print(f"   - {perf}: {count} occurrences")

def create_enhanced_parallel_sets():
    """Create an enhanced version with more detailed analysis"""
    print("\n🎯 Creating Enhanced Parallel Sets Analysis")
    print("=" * 45)
    
    portfolio_engine = create_sample_ips_data()
    visualizer = ParallelSetsVisualizer(portfolio_engine)
    
    # Create enhanced HTML with additional analysis
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced IPS Parallel Sets Analysis</title>
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
            .analysis-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }}
            .analysis-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }}
            .analysis-card h3 {{
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
            .parallel-sets {{
                width: 100%;
                height: 600px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .controls {{
                margin: 20px 0;
                text-align: center;
            }}
            .filter-btn {{
                margin: 5px;
                padding: 10px 20px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                background: #667eea;
                color: white;
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            .filter-btn:hover {{
                background: #5a6fd8;
                transform: translateY(-2px);
            }}
            .filter-btn.active {{
                background: #28a745;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>IPS Parallel Sets Analysis</h1>
                <p>Interactive visualization of life events, portfolio allocations, and performance outcomes</p>
            </div>
            
            <div class="content">
                <div class="analysis-grid">
                    <div class="analysis-card">
                        <h3>📊 Data Overview</h3>
                        <div class="metric">
                            <span>Total Life Events:</span>
                            <strong>{len(visualizer.create_parallel_sets_data()['life_events'])}</strong>
                        </div>
                        <div class="metric">
                            <span>Portfolio Categories:</span>
                            <strong>{len(set(visualizer.create_parallel_sets_data()['portfolio_allocations']))}</strong>
                        </div>
                        <div class="metric">
                            <span>Market Conditions:</span>
                            <strong>{len(set(visualizer.create_parallel_sets_data()['market_conditions']))}</strong>
                        </div>
                        <div class="metric">
                            <span>Comfort Levels:</span>
                            <strong>{len(set(visualizer.create_parallel_sets_data()['comfort_levels']))}</strong>
                        </div>
                    </div>
                    
                    <div class="analysis-card">
                        <h3>🎯 Key Insights</h3>
                        <div class="metric">
                            <span>Positive Events:</span>
                            <strong style="color: #28a745;">{len([e for e in visualizer.create_parallel_sets_data()['life_events'] if e['direction'] == 'Positive'])}</strong>
                        </div>
                        <div class="metric">
                            <span>High Impact Events:</span>
                            <strong style="color: #dc3545;">{len([e for e in visualizer.create_parallel_sets_data()['life_events'] if e['impact'] == 'High'])}</strong>
                        </div>
                        <div class="metric">
                            <span>Conservative Allocations:</span>
                            <strong>{visualizer.create_parallel_sets_data()['portfolio_allocations'].count('Conservative')}</strong>
                        </div>
                        <div class="metric">
                            <span>High Performance:</span>
                            <strong style="color: #28a745;">{visualizer.create_parallel_sets_data()['performance_outcomes'].count('High Performance')}</strong>
                        </div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="filter-btn active" data-filter="all">Show All</button>
                    <button class="filter-btn" data-filter="positive">Positive Events</button>
                    <button class="filter-btn" data-filter="negative">Negative Events</button>
                    <button class="filter-btn" data-filter="high-impact">High Impact</button>
                    <button class="filter-btn" data-filter="conservative">Conservative</button>
                </div>
                
                <div class="parallel-sets"></div>
            </div>
        </div>
        
        <script>
            // Enhanced D3.js implementation would go here
            // This is a placeholder for the full implementation
            console.log("Enhanced parallel sets visualization loaded");
            
            // Add interactive functionality
            d3.selectAll(".filter-btn").on("click", function() {{
                const filter = d3.select(this).attr("data-filter");
                
                // Update active button
                d3.selectAll(".filter-btn").classed("active", false);
                d3.select(this).classed("active", true);
                
                console.log("Filter applied:", filter);
                // Implement filter logic here
            }});
        </script>
    </body>
    </html>
    """
    
    # Save enhanced HTML
    with open("docs/enhanced_parallel_sets.html", 'w', encoding='utf-8') as f:
        f.write(enhanced_html)
    
    print(f"✅ Enhanced parallel sets visualization saved to: docs/enhanced_parallel_sets.html")
    return enhanced_html

if __name__ == "__main__":
    print("🎨 IPS Parallel Sets Visualization Demo")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_parallel_sets()
    analyze_relationships()
    create_enhanced_parallel_sets()
    
    print("\n🎉 Demo completed successfully!")
    print("📁 Check the 'docs/' folder for generated visualizations:")
    print("   - parallel_sets_ips_analysis.html")
    print("   - enhanced_parallel_sets.html") 