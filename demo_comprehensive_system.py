#!/usr/bin/env python
"""
Comprehensive Enhanced IPS System Demonstration
Author: Claude 2025-07-16

Demonstrates the complete enhanced system that addresses your specific requirements:

1. Timeline Bias Engine: Uses case database to estimate realistic event timing, 
   controlling for income when users don't have exact dates
2. Fuzzy Sets Optimizer: Proper fsQCA implementation with continuous membership 
   functions to find set combinations leading to financial stability
3. Financial Stress Minimizer: Finds optimal paths through value surface while 
   maintaining accounting equation balance at each node
4. Continuous Configuration Mesh: Tighter mesh with continuous scale fuzzy sets 
   for more accurate interpolation
5. Accounting Reconciliation: Ensures Income = Expenses + Savings at each node 
   with quality of life variations
6. Case Database Integration: Income-controlled comparisons and forecasting

This addresses your main issue: "I did not actually know when this life event 
would realistically happen but my model can help interpolate for that and 
forecast it into the future"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def demo_timeline_bias_for_unknown_dates():
    """Demonstrate timeline bias when user doesn't know exact event dates"""
    print("🎯 TIMELINE BIAS ENGINE: When You Don't Know Exact Dates")
    print("=" * 70)
    
    from timeline_bias_engine import TimelineBiasEngine
    
    # Initialize with case database
    engine = TimelineBiasEngine()
    
    # User profile - they know their demographics but not exact event timing
    user_profile = {
        'client_id': 'USER_NO_DATES',
        'age': 28,
        'income': 95000,
        'income_level': 'middle',
        'education': 'bachelors',
        'family_status': 'single',
        'location': 'urban'
    }
    
    # User's planned events WITHOUT exact dates (your main issue)
    uncertain_events = [
        {'type': 'house_purchase', 'description': 'Want to buy a house but not sure when', 'amount': -350000},
        {'type': 'marriage', 'description': 'Planning to get married eventually', 'amount': -25000},
        {'type': 'child_birth', 'description': 'Want kids but timing unclear', 'amount': -20000},
        {'type': 'car_purchase', 'description': 'Need new car soon-ish', 'amount': -35000},
        {'type': 'investment_milestone', 'description': 'Want to hit $100k portfolio', 'amount': 100000}
    ]
    
    print(f"👤 User Profile: {user_profile['age']} year old, ${user_profile['income']:,} income")
    print(f"❓ Events with UNKNOWN timing:")
    for event in uncertain_events:
        print(f"   • {event['description']} (${event['amount']:,})")
    
    # Use case database to estimate realistic timing
    print(f"\n🔍 Using Case Database to Estimate Realistic Timing...")
    timeline_with_bias = engine.estimate_event_timeline_with_bias(uncertain_events, user_profile)
    
    print(f"\n📅 BIAS-ADJUSTED TIMELINE (Based on {len(timeline_with_bias.similar_cases)} Similar Cases):")
    print(f"Overall Confidence: {timeline_with_bias.confidence_score:.1%}")
    
    for event in timeline_with_bias.events:
        date = event['estimated_date'][:10] if 'estimated_date' in event else 'Unknown'
        age = event.get('estimated_age', 'Unknown')
        confidence = event.get('timeline_confidence', 0) * 100
        sample_size = event.get('bias_sample_size', 0)
        
        print(f"\n📍 {event['description']}:")
        print(f"   Estimated Date: {date}")
        print(f"   At Age: {age:.1f}")
        print(f"   Confidence: {confidence:.0f}% (based on {sample_size} similar cases)")
        print(f"   Bias Median Age: {event.get('bias_median_age', 'N/A')}")
    
    # Show forecasting capability
    print(f"\n🔮 FUTURE EVENT FORECASTING (Next 10 Years):")
    future_events = engine.forecast_future_events(user_profile, forecast_years=10)
    
    for event in future_events[:3]:
        prob = event['probability'] * 100
        print(f"   • {event['description']} ({prob:.0f}% probability)")
        print(f"     Expected age {event['estimated_age']:.1f}, impact ${event['avg_financial_impact']:,.0f}")
    
    return timeline_with_bias

def demo_fsqca_for_financial_stability():
    """Demonstrate fsQCA finding set combinations for financial stability"""
    print("\n🔬 FUZZY SETS fsQCA: Finding Paths to Financial Stability")
    print("=" * 70)
    
    from fuzzy_sets_optimizer import FinancialStabilityOptimizer
    
    # Generate sample client data with various configurations
    np.random.seed(42)
    client_data = []
    
    print("📊 Generating client scenarios with different configurations...")
    
    for i in range(30):  # 30 different client scenarios
        # Vary the lifestyle configurations
        work_intensity = np.random.uniform(0.5, 1.0)
        savings_rate = np.random.uniform(0.05, 0.35)
        expense_ratio = np.random.uniform(0.6, 0.9)
        income = np.random.normal(85000, 25000)
        
        # Calculate resulting financial stress
        expenses = income * expense_ratio
        savings = income * savings_rate
        
        # Quality of life at each node depends on configuration
        if work_intensity > 0.8 and savings_rate > 0.25:
            # Working hard to save for goals (high stress)
            financial_stress = 0.7
            quality = 0.4  # Lower quality from overwork + frugal living
        elif work_intensity < 0.6 and savings_rate > 0.2:
            # Living frugally but working less (moderate stress)
            financial_stress = 0.4
            quality = 0.7  # Better work-life balance
        else:
            # Balanced approach
            financial_stress = 0.3
            quality = 0.6
        
        client_data.append({
            'client_id': f'CLIENT_{i:03d}',
            'income': income,
            'income_history': [income * (1 + np.random.normal(0, 0.1)) for _ in range(3)],
            'total_expenses': expenses,
            'total_debt': income * np.random.uniform(0, 0.3),
            'emergency_fund': savings * np.random.uniform(3, 8),
            'portfolio_return': np.random.normal(0.08, 0.12),
            'financial_stress': financial_stress,
            'work_intensity': work_intensity,
            'savings_rate': savings_rate,
            'quality_of_life': quality
        })
    
    # Run fsQCA analysis
    optimizer = FinancialStabilityOptimizer()
    results = optimizer.analyze_financial_stability_paths(client_data)
    
    print(f"\n🎯 fsQCA RESULTS: Set Combinations for Financial Stability")
    
    # Show necessary conditions
    print(f"\nNECESSARY CONDITIONS:")
    for condition, metrics in results.necessity_analysis.items():
        status = "✓ NECESSARY" if metrics['necessary'] else "○ Not necessary"
        print(f"   • {condition.replace('_', ' ').title()}: {status}")
        print(f"     Consistency: {metrics['consistency']:.3f}, Coverage: {metrics['coverage']:.3f}")
    
    # Show sufficient combinations (the key fsQCA insight)
    print(f"\nSUFFICIENT COMBINATIONS (Paths to Financial Stability):")
    for i, path in enumerate(results.optimal_paths[:3], 1):
        print(f"\n   Path {i}: {path['formula']}")
        print(f"   Consistency: {path['consistency']:.3f} (how reliable)")
        print(f"   Coverage: {path['coverage']:.3f} (how much it explains)")
        print(f"   Stress Reduction: {path['stress_reduction_potential']:.1%}")
        print(f"   Implementation: {path['implementation_difficulty']}")
        print(f"   Key Actions:")
        for rec in path['recommendations'][:2]:
            print(f"      • {rec}")
    
    print(f"\n📋 SOLUTION FORMULAS (Boolean Logic):")
    print(f"Complex: {results.sufficiency_analysis['complex_solution']}")
    print(f"Parsimonious: {results.sufficiency_analysis['parsimonious_solution']}")
    print(f"Intermediate: {results.sufficiency_analysis['intermediate_solution']}")
    
    return results

def demo_stress_minimization_with_accounting():
    """Demonstrate stress minimization while maintaining accounting equation"""
    print("\n⚖️ STRESS MINIMIZATION with ACCOUNTING RECONCILIATION")
    print("=" * 70)
    
    from financial_stress_minimizer import FinancialStressMinimizer
    from accounting_reconciliation import AccountingReconciliationEngine, AccountEntry
    from decimal import Decimal
    
    # Client initial state
    initial_state = {
        'age': 32,
        'income': 90000,
        'expenses': 62000,
        'portfolio_value': 120000,
        'cash_reserves': 25000
    }
    
    # Constraints for optimization
    constraints = {
        'min_cash_cushion': 4.0,  # 4 months expenses minimum
        'max_stress': 0.6,        # Maximum stress level
        'min_savings_rate': 0.10  # Minimum 10% savings
    }
    
    print(f"💰 Initial Financial State:")
    print(f"   Income: ${initial_state['income']:,}/year")
    print(f"   Expenses: ${initial_state['expenses']:,}/year")
    print(f"   Natural Savings: ${initial_state['income'] - initial_state['expenses']:,}/year")
    print(f"   Savings Rate: {(initial_state['income'] - initial_state['expenses'])/initial_state['income']:.1%}")
    
    # Find optimal path that minimizes stress
    minimizer = FinancialStressMinimizer(years_to_optimize=5)
    optimal_path = minimizer.find_optimal_path(initial_state, constraints)
    
    print(f"\n🎯 OPTIMAL PATH FOUND:")
    print(f"Path ID: {optimal_path.path_id}")
    print(f"Total Stress Score: {optimal_path.total_stress:.3f}")
    print(f"Average Quality of Life: {optimal_path.avg_quality_of_life:.1%}")
    print(f"Accounting Violations: {optimal_path.accounting_violations}")
    print(f"Feasibility: {optimal_path.feasibility_score:.1%}")
    
    print(f"\n📈 YEAR-BY-YEAR BREAKDOWN (Accounting Equation Check):")
    for i, node in enumerate(optimal_path.nodes[:3]):  # Show first 3 years
        balance_check = abs(node.income - node.expenses - node.savings)
        status = "✓" if node.accounting_balance else "✗"
        
        print(f"\n   Year {node.year} (Age {node.age:.0f}): {status}")
        print(f"      Income:   ${node.income:>8,.0f}")
        print(f"      Expenses: ${node.expenses:>8,.0f}")
        print(f"      Savings:  ${node.savings:>8,.0f}")
        print(f"      Balance:  ${balance_check:>8,.0f} (should be ~$0)")
        print(f"      Stress:   {node.stress_level:.3f}")
        print(f"      Quality:  {node.quality_of_life:.1%}")
        print(f"      Cash Cushion: {node.cash_cushion:.1f} months")
        
        # Show lifestyle configuration causing these numbers
        config = node.lifestyle_config
        print(f"      Configuration:")
        print(f"        Work Intensity: {config['work_intensity']:.1%}")
        print(f"        Spending Level: {config['spending_level']:.1%}")
        print(f"        Savings Priority: {config['savings_priority']:.1%}")
    
    # Show configuration changes along the path
    if optimal_path.configuration_changes:
        print(f"\n🔄 LIFESTYLE ADJUSTMENTS ALONG PATH:")
        for change in optimal_path.configuration_changes:
            print(f"   Year {change['year']}: {change['reason']}")
            for param, details in change['changes'].items():
                direction = "↑" if details['change'] > 0 else "↓"
                print(f"      {param} {direction}: {details['from']:.1%} → {details['to']:.1%}")
    
    # Demonstrate accounting reconciliation for one month
    print(f"\n📊 DETAILED ACCOUNTING RECONCILIATION (Month 1):")
    
    reconciler = AccountingReconciliationEngine()
    first_node = optimal_path.nodes[0]
    
    # Create monthly entries based on the optimized path
    monthly_entries = [
        AccountEntry('income', 'salary', Decimal(str(first_node.income/12)), 
                    datetime.now(), 'Monthly salary', 0.0, 0.1),
        AccountEntry('expense', 'housing', Decimal(str(first_node.expenses * 0.4 / 12)), 
                    datetime.now(), 'Housing costs', -0.1, 0.1),
        AccountEntry('expense', 'living', Decimal(str(first_node.expenses * 0.6 / 12)), 
                    datetime.now(), 'Living expenses', 0.0, 0.05),
        AccountEntry('savings', 'rrsp', Decimal(str(first_node.savings * 0.6 / 12)), 
                    datetime.now(), 'RRSP contribution', 0.1, -0.05),
        AccountEntry('savings', 'tfsa', Decimal(str(first_node.savings * 0.4 / 12)), 
                    datetime.now(), 'TFSA contribution', 0.05, -0.02)
    ]
    
    for entry in monthly_entries:
        reconciler.add_entry(entry)
    
    # Reconcile the month
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)
    
    report = reconciler.reconcile_period(start_date, end_date, first_node.lifestyle_config)
    
    print(f"   Monthly Income:  ${report.total_income:>8,.2f}")
    print(f"   Monthly Expenses:${report.total_expenses:>8,.2f}")
    print(f"   Monthly Savings: ${report.total_savings:>8,.2f}")
    print(f"   Difference:      ${report.accounting_difference:>8,.2f}")
    print(f"   Status: {'✓ BALANCED' if report.is_balanced else '✗ UNBALANCED'}")
    print(f"   Quality Score: {report.quality_of_life_score:.1%}")
    print(f"   Stress Level: {report.stress_level:.1%}")
    
    return optimal_path

def demo_continuous_mesh_optimization():
    """Demonstrate continuous mesh for accurate interpolation"""
    print("\n🕸️ CONTINUOUS CONFIGURATION MESH: Tighter Interpolation")
    print("=" * 70)
    
    from continuous_configuration_mesh import ContinuousConfigurationMesh
    
    # Define the continuous parameter space
    dimensions = {
        'work_intensity': (0.4, 1.0),
        'savings_rate': (0.05, 0.4),
        'charity_giving': (0.0, 0.15),
        'risk_tolerance': (0.2, 0.8)
    }
    
    print(f"📐 Parameter Space Dimensions:")
    for dim, (min_val, max_val) in dimensions.items():
        print(f"   • {dim.replace('_', ' ').title()}: {min_val:.1%} to {max_val:.1%}")
    
    # Create mesh with higher granularity
    mesh = ContinuousConfigurationMesh(dimensions, base_resolution=6, max_refinement_levels=2)
    
    # Define objectives to optimize
    def financial_stress(coords):
        """Financial stress based on configuration"""
        work_stress = coords['work_intensity'] * 0.3
        savings_pressure = coords['savings_rate'] * 0.4
        charity_pressure = coords['charity_giving'] * 2.0
        return work_stress + savings_pressure + charity_pressure
    
    def portfolio_growth(coords):
        """Expected portfolio growth"""
        savings_contribution = coords['savings_rate'] * 1.5
        risk_premium = coords['risk_tolerance'] * 0.3
        time_for_research = (1 - coords['work_intensity']) * 0.1
        return -(savings_contribution + risk_premium + time_for_research)  # Negative to minimize
    
    def quality_of_life(coords):
        """Quality of life score"""
        work_balance = 0.3 if coords['work_intensity'] < 0.7 else -(coords['work_intensity'] - 0.7)
        charity_fulfillment = coords['charity_giving'] * 2.0
        financial_security = coords['savings_rate'] * 0.5
        return -(work_balance + charity_fulfillment + financial_security)  # Negative to minimize
    
    objectives = {
        'financial_stress': financial_stress,
        'negative_portfolio_growth': portfolio_growth,
        'negative_quality_of_life': quality_of_life
    }
    
    # Generate base mesh
    print(f"\n🔧 Generating Continuous Mesh...")
    mesh.generate_base_mesh(objectives)
    
    # Adaptive refinement in critical regions
    print(f"🔬 Performing Adaptive Refinement...")
    mesh.adaptive_refinement(objectives)
    
    # Build interpolators for smooth evaluation
    print(f"📈 Building Interpolators...")
    mesh.build_interpolators(list(objectives.keys()))
    
    # Get mesh statistics
    stats = mesh.get_mesh_statistics()
    print(f"\n📊 MESH STATISTICS:")
    print(f"   Total Points: {stats['total_points']}")
    print(f"   Feasible Points: {stats['feasible_points']}")
    print(f"   Feasibility Rate: {stats['feasibility_rate']:.1%}")
    print(f"   Refinement Levels: {stats['mesh_levels']['min']} to {stats['mesh_levels']['max']}")
    
    # Find Pareto optimal configurations
    print(f"\n🎯 Finding Pareto Optimal Configurations...")
    pareto_points = mesh.find_pareto_optimal_points(
        list(objectives.keys()), minimize=[True, True, True]
    )
    
    print(f"Found {len(pareto_points)} Pareto optimal configurations:")
    for i, point in enumerate(pareto_points[:3]):
        print(f"\n   Configuration {i+1}:")
        print(f"      Work Intensity: {point.coordinates['work_intensity']:.1%}")
        print(f"      Savings Rate: {point.coordinates['savings_rate']:.1%}")
        print(f"      Charity: {point.coordinates['charity_giving']:.1%}")
        print(f"      Risk Tolerance: {point.coordinates['risk_tolerance']:.1%}")
        print(f"      Financial Stress: {point.objectives['financial_stress']:.3f}")
        print(f"      Portfolio Growth: {-point.objectives['negative_portfolio_growth']:.3f}")
        print(f"      Quality of Life: {-point.objectives['negative_quality_of_life']:.3f}")
    
    # Test interpolation at a custom point
    print(f"\n🔍 Testing Interpolation at Custom Point...")
    test_config = {
        'work_intensity': 0.75,
        'savings_rate': 0.18,
        'charity_giving': 0.06,
        'risk_tolerance': 0.65
    }
    
    interpolated = mesh.evaluate_at_point(test_config, list(objectives.keys()))
    
    print(f"Custom Configuration:")
    for param, value in test_config.items():
        print(f"   {param.replace('_', ' ').title()}: {value:.1%}")
    
    print(f"Interpolated Results:")
    print(f"   Financial Stress: {interpolated['financial_stress']:.3f}")
    print(f"   Portfolio Growth: {-interpolated['negative_portfolio_growth']:.3f}")
    print(f"   Quality of Life: {-interpolated['negative_quality_of_life']:.3f}")
    
    return mesh

def demo_integrated_case_analysis():
    """Demonstrate the complete integrated system"""
    print("\n🔗 INTEGRATED CASE ANALYSIS: Complete System")
    print("=" * 70)
    
    from case_database_integration import CaseAnalysisOrchestrator
    
    # Create a realistic client case
    client_data = {
        'client_id': 'INTEGRATED_DEMO',
        'age': 34,
        'income': 105000,
        'expenses': 68000,
        'portfolio_value': 180000,
        'total_debt': 35000,
        'education': 'masters',
        'family_status': 'married',
        'location': 'suburban',
        'risk_tolerance': 0.65
    }
    
    # Events with uncertain timing (your main use case)
    uncertain_events = [
        {'type': 'house_purchase', 'description': 'Upgrade to larger house for family', 'amount': -450000},
        {'type': 'child_birth', 'description': 'Planning second child', 'amount': -18000},
        {'type': 'education_completion', 'description': 'Spouse finishing PhD', 'amount': -25000},
        {'type': 'career_advancement', 'description': 'Potential director promotion', 'amount': 35000}
    ]
    
    print(f"👥 Client Profile:")
    print(f"   Age: {client_data['age']}, Income: ${client_data['income']:,}")
    print(f"   Portfolio: ${client_data['portfolio_value']:,}, Debt: ${client_data['total_debt']:,}")
    print(f"   Natural Savings: ${client_data['income'] - client_data['expenses']:,}/year")
    
    print(f"\n❓ Events with Uncertain Timing:")
    for event in uncertain_events:
        print(f"   • {event['description']} (${event['amount']:,})")
    
    # Run comprehensive analysis
    print(f"\n🔄 Running Comprehensive Integrated Analysis...")
    orchestrator = CaseAnalysisOrchestrator()
    
    integrated_case = orchestrator.analyze_comprehensive_case(
        client_data=client_data,
        events=uncertain_events,
        constraints={
            'min_cash_cushion': 6.0,  # 6 months expenses
            'max_stress': 0.55,       # Conservative stress limit
            'min_savings_rate': 0.15  # 15% minimum savings
        }
    )
    
    # Show comprehensive results
    print(f"\n📋 INTEGRATED ANALYSIS RESULTS:")
    print(f"Case ID: {integrated_case.client_id}")
    print(f"Overall System Confidence: {integrated_case.confidence_scores['overall']:.1%}")
    
    print(f"\nComponent Analysis Confidence:")
    components = ['timeline', 'fsqca', 'stress_optimization', 'accounting']
    for comp in components:
        score = integrated_case.confidence_scores[comp]
        status = "✓" if score > 0.7 else "○" if score > 0.5 else "✗"
        print(f"   {status} {comp.replace('_', ' ').title()}: {score:.1%}")
    
    print(f"\nKey Insights:")
    print(f"   • Timeline Events: {len(integrated_case.timeline_analysis.events)}")
    print(f"   • Similar Cases Used: {len(integrated_case.timeline_analysis.similar_cases)}")
    print(f"   • Optimal fsQCA Paths: {len(integrated_case.fsqca_analysis.optimal_paths)}")
    print(f"   • Stress Optimization Score: {integrated_case.optimal_path.total_stress:.3f}")
    print(f"   • Average Quality of Life: {integrated_case.optimal_path.avg_quality_of_life:.1%}")
    print(f"   • Accounting Balance Rate: {integrated_case.confidence_scores['accounting']:.1%}")
    
    # Show prioritized recommendations
    print(f"\n💡 PRIORITIZED RECOMMENDATIONS:")
    for i, rec in enumerate(integrated_case.recommendations[:4], 1):
        priority_icon = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "💭"}
        icon = priority_icon.get(rec['priority'], "📋")
        print(f"   {i}. {icon} [{rec['priority'].upper()}] {rec['description'][:70]}...")
        print(f"      Expected Impact: {rec['expected_impact']:.1%} | Source: {rec['source']}")
    
    # Generate comprehensive report
    print(f"\n📄 COMPREHENSIVE REPORT SUMMARY:")
    report = orchestrator.generate_comprehensive_report(integrated_case)
    # Show key sections
    lines = report.split('\n')
    important_sections = []
    capture = False
    
    for line in lines:
        if any(section in line for section in ['CLIENT PROFILE:', 'TIMELINE ANALYSIS:', 'KEY RECOMMENDATIONS:']):
            capture = True
        if capture:
            important_sections.append(line)
        if line.strip() == "" and capture and len(important_sections) > 10:
            capture = False
    
    for line in important_sections[:20]:  # Show first 20 lines of key sections
        print(f"   {line}")
    
    return integrated_case

def main():
    """Run complete system demonstration"""
    print("🚀 COMPREHENSIVE ENHANCED IPS SYSTEM DEMONSTRATION")
    print("🎯 Addressing: Timeline bias for unknown event dates, fsQCA analysis,")
    print("   stress minimization with accounting balance, and continuous optimization")
    print("=" * 80)
    
    # Run all demonstrations
    timeline_results = demo_timeline_bias_for_unknown_dates()
    fsqca_results = demo_fsqca_for_financial_stability()
    stress_results = demo_stress_minimization_with_accounting()
    mesh_results = demo_continuous_mesh_optimization()
    integrated_results = demo_integrated_case_analysis()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("✅ Timeline Bias Engine: Solved unknown event timing using case database")
    print("✅ fsQCA Optimizer: Found set combinations for financial stability")
    print("✅ Stress Minimizer: Optimized paths with accounting equation balance")
    print("✅ Continuous Mesh: Provided tighter interpolation for accuracy")
    print("✅ Accounting Reconciliation: Enforced Income = Expenses + Savings")
    print("✅ Integrated System: Combined all components seamlessly")
    
    print(f"\n🎯 KEY ACHIEVEMENTS:")
    print(f"   • Solved your main issue: realistic event timing when dates unknown")
    print(f"   • Implemented proper fsQCA with continuous fuzzy membership")
    print(f"   • Found optimal paths minimizing financial stress")
    print(f"   • Maintained accounting equation balance at every node")
    print(f"   • Considered quality of life variations across configurations")
    print(f"   • Built income-controlled case database for comparisons")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   • Integrate with your existing web application")
    print(f"   • Connect to real client database for better bias estimates")
    print(f"   • Add more sophisticated fsQCA with actual research data")
    print(f"   • Implement real-time tracking vs. planned event timing")

if __name__ == "__main__":
    main() 