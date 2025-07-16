#!/usr/bin/env python
# Full Application Launcher
# Runs the complete life choice optimization system
# Author: ChatGPT 2025-01-16

import sys
import os
import subprocess
import webbrowser
from pathlib import Path

def run_full_application():
    """Run the complete life choice optimization application"""
    print("🚀 Launching Full Life Choice Optimization Application")
    print("=" * 60)
    
    try:
        # Add src to path
        sys.path.append('src')
        
        # Import required modules
        from dynamic_portfolio_engine import DynamicPortfolioEngine
        from life_choice_optimizer import LifeChoiceOptimizer
        from enhanced_dashboard_with_optimization import EnhancedDashboardWithOptimization
        
        print("✅ Successfully imported all modules")
        
        # Create portfolio engine
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
        
        print("✅ Portfolio engine and optimizer initialized")
        
        # Add sample life choices to demonstrate the system
        sample_choices = [
            ('career', 'promotion', '2023-01-15', 'Got promoted to senior manager'),
            ('family', 'marriage', '2023-06-20', 'Got married'),
            ('lifestyle', 'buy_house', '2024-03-10', 'Purchased first home'),
            ('education', 'certification', '2024-09-05', 'Obtained professional certification'),
            ('health', 'health_improvement', '2024-12-01', 'Started fitness program')
        ]
        
        print("\n📝 Adding sample life choices to demonstrate the system...")
        for category, choice, date, description in sample_choices:
            result = optimizer.add_life_choice(category, choice, date)
            print(f"   ✅ {category}: {choice} on {date}")
            print(f"      Comfort Score: {result['comfort_score']:.2f}")
        
        # Create enhanced dashboard
        print("\n📊 Creating enhanced dashboard with optimization toggle...")
        dashboard = EnhancedDashboardWithOptimization(portfolio_engine)
        
        # Generate all visualizations
        main_dashboard = dashboard.create_enhanced_dashboard()
        optimization_dashboard = optimizer.create_optimization_dashboard()
        interactive_html = dashboard.generate_interactive_html()
        
        # Save outputs
        main_dashboard.write_html("full_app_dashboard.html")
        optimization_dashboard.write_html("full_app_optimization.html")
        
        with open("full_app_interactive.html", "w") as f:
            f.write(interactive_html)
        
        # Generate optimization report
        report = optimizer.generate_optimization_report('financial_growth')
        with open("full_app_report.md", "w") as f:
            f.write(report)
        
        print("\n✅ Application successfully generated!")
        print("\n📁 Generated Files:")
        print("   - full_app_dashboard.html (main dashboard)")
        print("   - full_app_optimization.html (optimization analysis)")
        print("   - full_app_interactive.html (interactive interface)")
        print("   - full_app_report.md (detailed report)")
        
        # Try to open the interactive dashboard in browser
        try:
            interactive_path = Path("full_app_interactive.html").absolute()
            print(f"\n🌐 Opening interactive dashboard in browser...")
            print(f"   File location: {interactive_path}")
            webbrowser.open(f"file://{interactive_path}")
        except Exception as e:
            print(f"   ⚠️  Could not open browser automatically: {e}")
            print(f"   📂 Please open full_app_interactive.html manually")
        
        # Show optimization results
        print("\n🎯 Current Optimization Results:")
        objectives = ['financial_growth', 'comfort_stability', 'risk_management', 'lifestyle_quality']
        
        for objective in objectives:
            result = optimizer.optimize_next_choice(objective)
            if result['best_choice']:
                best = result['best_choice']
                print(f"\n📊 {objective.replace('_', ' ').title()}:")
                print(f"   🏆 Best Choice: {best['choice'].replace('_', ' ').title()} ({best['category']})")
                print(f"   📈 Score: {best['total_score']:.3f}")
                print(f"   💰 Financial: {best['financial_score']:+.3f}")
                print(f"   😌 Comfort: {best['comfort_score']:+.3f}")
                print(f"   🛡️  Risk: {best['risk_score']:+.3f}")
                print(f"   🌟 Lifestyle: {best['lifestyle_score']:+.3f}")
        
        print("\n🎉 Full application is ready for testing!")
        print("\n💡 How to Test:")
        print("   1. Open full_app_interactive.html in your browser")
        print("   2. Toggle 'Optimization Mode' in the right panel")
        print("   3. Enter additional life choices using the dropdowns")
        print("   4. See real-time optimization recommendations")
        print("   5. Explore different optimization objectives")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running application: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_application()
    if success:
        print("\n✅ Application completed successfully!")
    else:
        print("\n❌ Application failed to run") 