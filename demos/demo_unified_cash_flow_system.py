#!/usr/bin/env python3
"""
Unified Cash Flow System Demo

This script demonstrates the complete unified cash flow system that:
1. Incorporates time uncertainty mesh for event timing/amount uncertainty
2. Integrates with accounting system using case-specific account names
3. Provides real-time cash flow tracking and state monitoring
4. Ensures accounting consistency at any point in time
5. Includes comprehensive debugging and sense-checking
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent, demo_unified_cash_flow_model
from src.accounting_debugger import AccountingDebugger, demo_accounting_debugger
from src.time_uncertainty_mesh import TimeUncertaintyMeshEngine
from src.accounting_reconciliation import AccountingReconciliationEngine

def convert_numpy_to_lists(obj):
    """Recursively convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    else:
        return obj

def run_comprehensive_demo():
    """Run comprehensive demo of the unified cash flow system"""
    print("🎯 Unified Cash Flow System - Comprehensive Demo")
    print("=" * 60)
    
    # Step 1: Initialize the unified cash flow model
    print("\n📋 Step 1: Initializing Unified Cash Flow Model")
    print("-" * 50)
    
    initial_state = {
        'total_wealth': 764560.97,  # From Case_1 analysis
        'cash': 764560.97 * 0.0892,  # 8.92% cash allocation
        'investments': 764560.97 * 0.9554,  # 95.54% investments
        'income': 150000,  # Annual salary
        'expenses': 60000   # Annual expenses
    }
    
    model = UnifiedCashFlowModel(initial_state)
    print(f"✅ Model initialized with ${initial_state['total_wealth']:,.2f} total wealth")
    
    # Step 2: Add case-specific cash flow events
    print("\n📋 Step 2: Adding Case-Specific Cash Flow Events")
    print("-" * 50)
    
    case_events = model.create_case_events_from_analysis()
    for event in case_events:
        model.add_cash_flow_event(event)
        print(f"  ➕ {event.event_id}: {event.description} (${event.amount:,.2f})")
    
    print(f"✅ Added {len(case_events)} case-specific events")
    
    # Step 3: Initialize time uncertainty mesh
    print("\n📋 Step 3: Initializing Time Uncertainty Mesh")
    print("-" * 50)
    
    mesh_data, risk_analysis = model.initialize_time_uncertainty_mesh(
        num_scenarios=5000,  # Reduced for demo
        time_horizon_years=5
    )
    
    print(f"✅ Time uncertainty mesh initialized")
    print(f"  📊 Scenarios: {len(mesh_data.get('scenarios', []))}")
    print(f"  🎯 Risk metrics: {len(risk_analysis.get('risk_metrics', {}))}")
    
    # Step 4: Simulate cash flows over time
    print("\n📋 Step 4: Simulating Cash Flows Over Time")
    print("-" * 50)
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    states = model.simulate_cash_flows_over_time(start_date, end_date)
    
    print(f"✅ Simulated {len(states)} cash flow states")
    print(f"  📅 Period: {start_date.date()} to {end_date.date()}")
    print(f"  📊 States: {len(states)}")
    
    # Step 5: Get cash flow summary
    print("\n📋 Step 5: Cash Flow Summary")
    print("-" * 50)
    
    summary = model.get_cash_flow_summary()
    
    print("📊 Financial Summary:")
    financial_summary = summary.get('financial_summary', {})
    print(f"  💰 Initial Net Worth: ${financial_summary.get('initial_net_worth', 0):,.2f}")
    print(f"  💰 Final Net Worth: ${financial_summary.get('final_net_worth', 0):,.2f}")
    print(f"  📈 Net Worth Change: ${financial_summary.get('net_worth_change', 0):,.2f}")
    print(f"  📊 Growth Rate: {financial_summary.get('net_worth_growth_rate', 0):.2%}")
    print(f"  💧 Liquidity Ratio: {financial_summary.get('final_liquidity_ratio', 0):.2%}")
    print(f"  ⚠️ Stress Level: {financial_summary.get('final_stress_level', 0):.2%}")
    
    print("\n📊 Cash Flow Analysis:")
    cash_flow_analysis = summary.get('cash_flow_analysis', {})
    print(f"  📋 Total Events: {cash_flow_analysis.get('total_events_processed', 0)}")
    print(f"  ➕ Positive Events: {cash_flow_analysis.get('positive_events', 0)}")
    print(f"  ➖ Negative Events: {cash_flow_analysis.get('negative_events', 0)}")
    print(f"  💰 Total Impact: ${cash_flow_analysis.get('total_cash_flow_impact', 0):,.2f}")
    print(f"  📊 Avg Monthly CF: ${cash_flow_analysis.get('average_monthly_cash_flow', 0):,.2f}")
    
    # Step 6: Initialize accounting debugger
    print("\n📋 Step 6: Initializing Accounting Debugger")
    print("-" * 50)
    
    debugger = AccountingDebugger(model.accounting_engine, model)
    print("✅ Accounting debugger initialized")
    
    # Step 7: Run comprehensive validation
    print("\n📋 Step 7: Running Comprehensive Validation")
    print("-" * 50)
    
    validation_result = debugger.validate_accounting_state()
    
    print(f"Validation Result: {'✅ PASSED' if validation_result.is_valid else '❌ FAILED'}")
    print(f"  📊 Balance Sheet: {'✅' if validation_result.balance_sheet_check else '❌'}")
    print(f"  📊 Cash Flow: {'✅' if validation_result.cash_flow_check else '❌'}")
    print(f"  📊 Double Entry: {'✅' if validation_result.double_entry_check else '❌'}")
    print(f"  📊 Constraints: {'✅' if validation_result.constraint_check else '❌'}")
    print(f"  📊 Stress Test: {'✅' if validation_result.stress_test_passed else '❌'}")
    
    if validation_result.errors:
        print(f"\n❌ Errors ({len(validation_result.errors)}):")
        for error in validation_result.errors:
            print(f"  • {error}")
    
    if validation_result.warnings:
        print(f"\n⚠️ Warnings ({len(validation_result.warnings)}):")
        for warning in validation_result.warnings:
            print(f"  • {warning}")
    
    # Step 8: Calculate and display accounting metrics
    print("\n📋 Step 8: Accounting Metrics")
    print("-" * 50)
    
    metrics = debugger.calculate_accounting_metrics()
    
    print("📊 Key Metrics:")
    print(f"  💰 Total Assets: ${metrics.total_assets:,.2f}")
    print(f"  💳 Total Liabilities: ${metrics.total_liabilities:,.2f}")
    print(f"  💎 Net Worth: ${metrics.net_worth:,.2f}")
    print(f"  💧 Liquidity Ratio: {metrics.liquidity_ratio:.2%}")
    print(f"  📊 Debt to Asset Ratio: {metrics.debt_to_asset_ratio:.2%}")
    print(f"  📈 Cash Flow Coverage: {metrics.cash_flow_coverage:.2f}")
    print(f"  ⚠️ Stress Level: {metrics.stress_level:.2%}")
    
    # Step 9: Monitor cash flow consistency
    print("\n📋 Step 9: Cash Flow Consistency Monitoring")
    print("-" * 50)
    
    consistency_report = debugger.monitor_cash_flow_consistency(time_period_days=30)
    
    cash_flow_analysis = consistency_report.get('cash_flow_analysis', {})
    print(f"📊 Cash Flow Analysis (30 days):")
    print(f"  📅 Total Days: {cash_flow_analysis.get('total_states', 0)}")
    print(f"  ➕ Positive Days: {cash_flow_analysis.get('positive_cash_flow_days', 0)}")
    print(f"  ➖ Negative Days: {cash_flow_analysis.get('negative_cash_flow_days', 0)}")
    print(f"  📊 Avg Daily CF: ${cash_flow_analysis.get('average_daily_cash_flow', 0):,.2f}")
    print(f"  📈 CF Volatility: ${cash_flow_analysis.get('cash_flow_volatility', 0):,.2f}")
    
    stress_analysis = consistency_report.get('stress_analysis', {})
    print(f"⚠️ Stress Analysis:")
    print(f"  📊 Avg Stress: {stress_analysis.get('average_stress_level', 0):.2%}")
    print(f"  📈 Peak Stress: {stress_analysis.get('peak_stress_level', 0):.2%}")
    print(f"  📊 Stress Volatility: {stress_analysis.get('stress_volatility', 0):.2%}")
    
    # Step 10: Generate comprehensive report
    print("\n📋 Step 10: Generating Comprehensive Report")
    print("-" * 50)
    
    report = debugger.generate_accounting_report("comprehensive_cash_flow_report.json")
    
    validation_summary = report.get('validation_summary', {})
    print(f"📊 Validation Summary:")
    print(f"  ✅ Overall Valid: {validation_summary.get('overall_valid', False)}")
    print(f"  ❌ Total Errors: {validation_summary.get('total_errors', 0)}")
    print(f"  ⚠️ Total Warnings: {validation_summary.get('total_warnings', 0)}")
    print(f"  📊 Checks Passed: {validation_summary.get('checks_passed', 0)}/{validation_summary.get('checks_total', 0)}")
    
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    # Step 11: Export data
    print("\n📋 Step 11: Exporting Data")
    print("-" * 50)
    
    model.export_cash_flow_data("unified_cash_flow_data.json")
    
    # Export mesh data
    with open("time_uncertainty_mesh_data.json", "w") as f:
        json.dump(convert_numpy_to_lists(mesh_data), f, indent=2)
    
    # Export risk analysis
    with open("risk_analysis_data.json", "w") as f:
        json.dump(convert_numpy_to_lists(risk_analysis), f, indent=2)
    
    print("✅ Data exported:")
    print("  📄 unified_cash_flow_data.json")
    print("  📄 comprehensive_cash_flow_report.json")
    print("  📄 time_uncertainty_mesh_data.json")
    print("  📄 risk_analysis_data.json")
    
    # Step 12: Final summary
    print("\n📋 Step 12: Final Summary")
    print("-" * 50)
    
    print("🎯 Unified Cash Flow System Demo Complete!")
    print(f"✅ Successfully integrated time uncertainty mesh with accounting system")
    print(f"✅ Processed {len(case_events)} cash flow events")
    print(f"✅ Simulated {len(states)} cash flow states")
    print(f"✅ Validated accounting system: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    print(f"✅ Generated comprehensive reports and data exports")
    
    return {
        'model': model,
        'debugger': debugger,
        'validation_result': validation_result,
        'metrics': metrics,
        'summary': summary,
        'consistency_report': consistency_report,
        'mesh_data': mesh_data,
        'risk_analysis': risk_analysis
    }

def run_quick_validation():
    """Run a quick validation check"""
    print("\n🔍 Quick Validation Check")
    print("-" * 30)
    
    # Initialize with minimal setup
    initial_state = {'total_wealth': 1000000}
    model = UnifiedCashFlowModel(initial_state)
    
    # Add a simple event
    simple_event = CashFlowEvent(
        event_id="test_event",
        description="Test cash flow event",
        estimated_date="2024-01-01",
        amount=10000,
        source_account="cash_checking",
        target_account="investments_stocks",
        event_type="transfer"
    )
    model.add_cash_flow_event(simple_event)
    
    # Create debugger
    debugger = AccountingDebugger(model.accounting_engine, model)
    
    # Run validation
    validation_result = debugger.validate_accounting_state()
    metrics = debugger.calculate_accounting_metrics()
    
    print(f"Quick Validation: {'✅ PASSED' if validation_result.is_valid else '❌ FAILED'}")
    print(f"Net Worth: ${metrics.net_worth:,.2f}")
    print(f"Liquidity: {metrics.liquidity_ratio:.2%}")
    print(f"Stress Level: {metrics.stress_level:.2%}")
    
    return validation_result.is_valid

def test_no_data():
    print("\n=== TEST: No Data (Empty Events) ===")
    initial_state = {'total_wealth': 100000}
    model = UnifiedCashFlowModel(initial_state)
    # No events added
    states = model.simulate_cash_flows_over_time(datetime(2020, 1, 1), datetime(2020, 12, 31))
    summary = model.get_cash_flow_summary()
    print("Summary:", summary)
    debugger = AccountingDebugger(model.accounting_engine, model)
    validation = debugger.validate_accounting_state()
    print("Validation:", validation)
    print("---\n")

def test_safe_data():
    print("\n=== TEST: Safe Data (No Cycles) ===")
    initial_state = {'total_wealth': 100000}
    model = UnifiedCashFlowModel(initial_state)
    # Add salary income event
    salary_event = CashFlowEvent(
        event_id="salary_income",
        description="Monthly salary",
        estimated_date="2020-01-15",
        amount=5000,
        source_account="salary",
        target_account="cash_checking",
        event_type="income"
    )
    # Add living expense event
    expense_event = CashFlowEvent(
        event_id="living_expense",
        description="Monthly living expense",
        estimated_date="2020-01-20",
        amount=-3000,
        source_account="cash_checking",
        target_account="living_expenses",
        event_type="expense"
    )
    model.add_cash_flow_event(salary_event)
    model.add_cash_flow_event(expense_event)
    states = model.simulate_cash_flows_over_time(datetime(2020, 1, 1), datetime(2020, 12, 31))
    summary = model.get_cash_flow_summary()
    print("Summary:", summary)
    debugger = AccountingDebugger(model.accounting_engine, model)
    validation = debugger.validate_accounting_state()
    print("Validation:", validation)
    print("---\n")

def test_constraint_propagation():
    print("\n=== TEST: Constraint Propagation (Overdraft/Borrowing) ===")
    initial_state = {'total_wealth': 10000}
    model = UnifiedCashFlowModel(initial_state)
    # Add a large expense to force overdraft
    big_expense = CashFlowEvent(
        event_id="big_expense",
        description="Large one-time expense",
        estimated_date="2020-02-01",
        amount=-20000,
        source_account="cash_checking",
        target_account="living_expenses",
        event_type="expense"
    )
    model.add_cash_flow_event(big_expense)
    states = model.simulate_cash_flows_over_time(datetime(2020, 1, 1), datetime(2020, 12, 31))
    summary = model.get_cash_flow_summary()
    print("Summary:", summary)
    debugger = AccountingDebugger(model.accounting_engine, model)
    validation = debugger.validate_accounting_state()
    print("Validation:", validation)
    # Show interest expense
    interest = model.accounting_engine.get_account_balance('interest_expense')
    print(f"Interest expense accrued: {interest}")
    print("---\n")

def main():
    """Main demo function"""
    print("🎯 Unified Cash Flow System Demo")
    print("=" * 60)
    
    try:
        # Run comprehensive demo
        results = run_comprehensive_demo()
        
        # Run quick validation
        quick_valid = run_quick_validation()
        
        # Run new tests
        test_no_data()
        test_safe_data()
        test_constraint_propagation()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"✅ Comprehensive demo: COMPLETE")
        print(f"✅ Quick validation: {'PASSED' if quick_valid else 'FAILED'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 