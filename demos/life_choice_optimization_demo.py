#!/usr/bin/env python
# Life Choice Optimization Demo
# Demonstrates the complete optimization system with toggle interface
# Author: ChatGPT 2025-01-16

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

from dynamic_portfolio_engine import DynamicPortfolioEngine
from life_choice_optimizer import LifeChoiceOptimizer
from enhanced_dashboard_with_optimization import EnhancedDashboardWithOptimization

def create_sample_life_path():
    """Create a sample life path with various choices"""
    return [
        # Career choices
        ('career', 'promotion', '2022-03-15', 'Got promoted to senior manager'),
        ('career', 'job_change', '2023-01-20', 'Switched to higher-paying company'),
        
        # Family choices
        ('family', 'marriage', '2022-08-10', 'Got married'),
        ('family', 'children', '2023-06-15', 'First child born'),
        
        # Lifestyle choices
        ('lifestyle', 'buy_house', '2023-09-05', 'Purchased first home'),
        ('lifestyle', 'move_city', '2024-02-20', 'Moved to better school district'),
        
        # Education choices
        ('education', 'certification', '2022-11-30', 'Obtained professional certification'),
        ('education', 'skill_development', '2023-12-10', 'Completed advanced training'),
        
        # Health choices
        ('health', 'health_improvement', '2023-04-15', 'Started fitness program'),
        ('health', 'insurance_upgrade', '2024-01-05', 'Upgraded health insurance')
    ]

def run_life_choice_optimization_demo():
    """Run the complete life choice optimization demonstration"""
    print("🎯 Life Choice Optimization System Demo")
    print("=" * 60)
    
    # Initialize portfolio engine
    client_config = {
        'income': 250000,
        'disposable_cash': 8000,
        'allowable_var': 0.15,
        'age': 42,
        'risk_profile': 3,
        'portfolio_value': 1500000,
        'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
    }
    
    portfolio_engine = DynamicPortfolioEngine(client_config)
    optimizer = LifeChoiceOptimizer(portfolio_engine)
    
    print(f"📊 Initial Client Configuration:")
    print(f"   Age: {client_config['age']}")
    print(f"   Portfolio Value: ${client_config['portfolio_value']:,}")
    print(f"   Risk Profile: {client_config['risk_profile']}/5")
    print()
    
    # Add life choices to create a realistic life path
    life_path = create_sample_life_path()
    
    print("📝 Building Life Path (Series of Life Choices)...")
    print("   This simulates your life decisions like a series of coin tosses")
    print()
    
    for i, (category, choice, date, description) in enumerate(life_path, 1):
        print(f"   {i:2d}. {date} - {category.title()}: {choice.replace('_', ' ').title()}")
        print(f"       {description}")
        
        # Add the life choice
        result = optimizer.add_life_choice(category, choice, date)
        
        # Show impact
        comfort_score = result['comfort_score']
        portfolio = result['portfolio_after']
        print(f"       → Comfort Score: {comfort_score:.2f}")
        print(f"       → Portfolio: Equity {portfolio['equity']:.1%}, "
              f"Bonds {portfolio['bonds']:.1%}, Cash {portfolio['cash']:.1%}")
        print()
    
    # Show current situation
    current_situation = optimizer.optimize_next_choice('financial_growth')['current_situation']
    print(f"📈 Current Situation After Life Path:")
    print(f"   Portfolio Value: ${portfolio_engine.client_config['income'] * 6:,.0f} (6x income)")
    print(f"   Comfort Score: {current_situation['comfort_score']:.2f}")
    print(f"   Life Choices Made: {current_situation['life_choices_count']}")
    print(f"   Current Allocation: Equity {current_situation['portfolio']['equity']:.1%}, "
          f"Bonds {current_situation['portfolio']['bonds']:.1%}, "
          f"Cash {current_situation['portfolio']['cash']:.1%}")
    print()
    
    # Run optimization for different objectives
    print("🎯 Optimization Analysis - What Should You Do Next?")
    print("   Based on your life path so far, here are the optimal next choices:")
    print()
    
    objectives = {
        'financial_growth': 'Maximize wealth accumulation',
        'comfort_stability': 'Maintain lifestyle comfort',
        'risk_management': 'Minimize financial risk',
        'lifestyle_quality': 'Enhance life satisfaction'
    }
    
    for objective, description in objectives.items():
        result = optimizer.optimize_next_choice(objective)
        print(f"📊 {objective.replace('_', ' ').title()}: {description}")
        
        if result['best_choice']:
            best = result['best_choice']
            print(f"   🏆 Best Next Choice: {best['choice'].replace('_', ' ').title()} ({best['category']})")
            print(f"      Total Score: {best['total_score']:.3f}")
            print(f"      Financial Impact: {best['financial_score']:+.3f}")
            print(f"      Comfort Impact: {best['comfort_score']:+.3f}")
            print(f"      Risk Impact: {best['risk_score']:+.3f}")
            print(f"      Lifestyle Impact: {best['lifestyle_score']:+.3f}")
            
            # Show expected impacts
            impacts = best['impacts']
            impact_desc = []
            if 'income_boost' in impacts:
                impact_desc.append(f"Income: {impacts['income_boost']:+.1%}")
            if 'expense_impact' in impacts:
                impact_desc.append(f"Expenses: {impacts['expense_impact']:+.1%}")
            if 'stress_impact' in impacts:
                impact_desc.append(f"Stress: {impacts['stress_impact']:+.1%}")
            if 'stability_boost' in impacts:
                impact_desc.append(f"Stability: {impacts['stability_boost']:+.1%}")
            
            print(f"      Expected: {', '.join(impact_desc)}")
        print()
    
    # Create enhanced dashboard with optimization
    print("📊 Creating Enhanced Dashboard with Optimization Toggle...")
    dashboard = EnhancedDashboardWithOptimization(portfolio_engine)
    
    # Generate all visualizations
    main_dashboard = dashboard.create_enhanced_dashboard()
    optimization_dashboard = optimizer.create_optimization_dashboard()
    interactive_html = dashboard.generate_interactive_html()
    
    # Save outputs
    main_dashboard.write_html("life_choice_optimization_dashboard.html")
    optimization_dashboard.write_html("optimization_analysis_dashboard.html")
    
    with open("interactive_life_choice_dashboard.html", "w") as f:
        f.write(interactive_html)
    
    # Generate detailed report
    report = optimizer.generate_optimization_report('financial_growth')
    with open("life_choice_optimization_report.md", "w") as f:
        f.write(report)
    
    # Export data for analysis
    data = optimizer.export_optimization_data()
    with open("life_choice_optimization_data.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    print("✅ Life Choice Optimization Demo Completed!")
    print()
    print("📁 Generated Files:")
    print("   - life_choice_optimization_dashboard.html (main dashboard with toggle)")
    print("   - optimization_analysis_dashboard.html (detailed optimization analysis)")
    print("   - interactive_life_choice_dashboard.html (interactive interface)")
    print("   - life_choice_optimization_report.md (detailed report)")
    print("   - life_choice_optimization_data.json (complete data export)")
    print()
    print("🎯 Key Features Demonstrated:")
    print("   ✅ Life path analysis (series of coin tosses)")
    print("   ✅ Optimization toggle in dashboard")
    print("   ✅ Life choice interface with dropdowns")
    print("   ✅ Multi-objective optimization (financial, comfort, risk, lifestyle)")
    print("   ✅ Real-time recommendations based on current situation")
    print("   ✅ Interactive visualizations with hover functionality")
    print()
    print("💡 How to Use:")
    print("   1. Open interactive_life_choice_dashboard.html")
    print("   2. Toggle 'Optimization Mode' in the right panel")
    print("   3. Enter your life choices using the dropdown interface")
    print("   4. See real-time optimization recommendations")
    print("   5. Explore different optimization objectives")
    
    return {
        'portfolio_engine': portfolio_engine,
        'optimizer': optimizer,
        'dashboard': dashboard,
        'main_dashboard': main_dashboard,
        'optimization_dashboard': optimization_dashboard,
        'interactive_html': interactive_html,
        'report': report
    }

def demonstrate_optimization_scenarios():
    """Demonstrate different optimization scenarios"""
    print("\n🎯 Optimization Scenarios Demonstration")
    print("=" * 50)
    
    # Create different client scenarios
    scenarios = [
        {
            'name': 'Young Professional',
            'config': {'age': 28, 'income': 120000, 'risk_profile': 4, 'portfolio_value': 300000}
        },
        {
            'name': 'Mid-Career Family',
            'config': {'age': 42, 'income': 250000, 'risk_profile': 3, 'portfolio_value': 1500000}
        },
        {
            'name': 'Pre-Retirement',
            'config': {'age': 58, 'income': 180000, 'risk_profile': 2, 'portfolio_value': 2500000}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n👤 {scenario['name']} Scenario:")
        
        # Create portfolio engine for this scenario
        client_config = {
            'income': scenario['config']['income'],
            'disposable_cash': scenario['config']['income'] * 0.03,
            'allowable_var': 0.15,
            'age': scenario['config']['age'],
            'risk_profile': scenario['config']['risk_profile'],
            'portfolio_value': scenario['config']['portfolio_value'],
            'target_allocation': {'equity': 0.6, 'bonds': 0.3, 'cash': 0.1}
        }
        
        portfolio_engine = DynamicPortfolioEngine(client_config)
        optimizer = LifeChoiceOptimizer(portfolio_engine)
        
        # Add some life choices
        sample_choices = [
            ('career', 'promotion', '2023-01-15'),
            ('family', 'marriage', '2023-06-20'),
            ('lifestyle', 'buy_house', '2024-03-10')
        ]
        
        for category, choice, date in sample_choices:
            optimizer.add_life_choice(category, choice, date)
        
        # Run optimization
        result = optimizer.optimize_next_choice('financial_growth')
        
        if result['best_choice']:
            best = result['best_choice']
            print(f"   🎯 Best Next Choice: {best['choice'].replace('_', ' ').title()}")
            print(f"   📊 Score: {best['total_score']:.3f}")
            print(f"   💰 Financial Impact: {best['financial_score']:+.3f}")
            print(f"   😌 Comfort Impact: {best['comfort_score']:+.3f}")
    
    print("\n✅ Optimization scenarios completed!")

if __name__ == "__main__":
    # Run main demonstration
    results = run_life_choice_optimization_demo()
    
    # Run additional scenarios
    demonstrate_optimization_scenarios()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("   Open the HTML files to explore the interactive dashboards")
    print("   Use the toggle in the right corner to enable optimization mode")
    print("   Enter your life choices to see personalized recommendations") 