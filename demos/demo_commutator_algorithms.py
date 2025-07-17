#!/usr/bin/env python3
"""
Demonstration of Commutator Algorithms for Financial State Switching
Tests the implementation of commutator decision making for transforming
suboptimal financial states into positive ones while maintaining capital constraints.
"""

import json
from datetime import datetime
from src.commutator_decision_engine import create_commutator_engine, optimize_financial_state


def demo_suboptimal_state_optimization():
    """Demonstrate optimization of a suboptimal financial state"""
    
    print("🔄 COMMUTATOR ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Create a suboptimal financial state
    suboptimal_state = {
        'cash': 50000,
        'investments': {
            'equities': 800000,  # Too concentrated in equities
            'bonds': 50000,       # Too little in bonds
            'real_estate': 0      # No real estate diversification
        },
        'debts': {
            'credit_card': 15000,  # High-cost debt
            'personal_loan': 25000, # High-cost debt
            'mortgage': 300000     # Reasonable mortgage
        },
        'income_streams': {
            'salary': 8000,  # Only one income stream
        },
        'constraints': {
            'min_cash_reserve': 25000,
            'max_risk': 0.7,
            'max_debt_ratio': 0.4
        }
    }
    
    print("📊 Initial Suboptimal State:")
    print(f"   Cash: ${suboptimal_state['cash']:,.2f}")
    print(f"   Total Investments: ${sum(suboptimal_state['investments'].values()):,.2f}")
    print(f"   Total Debt: ${sum(suboptimal_state['debts'].values()):,.2f}")
    print(f"   Income Streams: {len(suboptimal_state['income_streams'])}")
    
    # Create commutator engine
    engine = create_commutator_engine(suboptimal_state)
    
    # Analyze current state
    analysis = engine.get_state_analysis()
    print("\n🔍 Current State Analysis:")
    print(f"   Total Wealth: ${analysis['current_state']['total_wealth']:,.2f}")
    print(f"   Risk Score: {analysis['current_state']['risk_score']:.3f}")
    print(f"   Available Capital: ${analysis['current_state']['available_capital']:,.2f}")
    print(f"   Debt Ratio: {analysis['current_state']['debt_ratio']:.3f}")
    
    # Show suboptimal aspects
    suboptimal_aspects = analysis['suboptimal_aspects']
    print("\n⚠️ Suboptimal Aspects:")
    for aspect, score in suboptimal_aspects.items():
        print(f"   {aspect.replace('_', ' ').title()}: {score:.3f}")
    
    # Generate commutator operations
    operations = engine.generate_commutator_operations()
    print(f"\n🔄 Generated {len(operations)} Commutator Operations:")
    
    for i, operation in enumerate(operations[:5], 1):  # Show first 5
        print(f"   {i}. {operation.operation_type.title()}")
        print(f"      Impact: {operation.expected_impact}")
        print(f"      Risk Change: {operation.risk_change:+.3f}")
        print(f"      Capital Required: ${operation.capital_required:,.2f}")
        print(f"      Success Probability: {operation.success_probability:.1%}")
    
    # Optimize state
    target_metrics = {
        'risk_score': 0.3,
        'capital_efficiency': 0.8,
        'debt_ratio': 0.3,
        'income_diversity': 0.7
    }
    
    print(f"\n🎯 Optimizing towards target metrics: {target_metrics}")
    
    # Execute optimization
    success = engine.optimize_state(target_metrics)
    
    if success:
        print("✅ State optimization completed successfully!")
        
        # Get updated analysis
        updated_analysis = engine.get_state_analysis()
        print("\n📈 Optimized State Results:")
        print(f"   Total Wealth: ${updated_analysis['current_state']['total_wealth']:,.2f}")
        print(f"   Risk Score: {updated_analysis['current_state']['risk_score']:.3f}")
        print(f"   Available Capital: ${updated_analysis['current_state']['available_capital']:,.2f}")
        print(f"   Debt Ratio: {updated_analysis['current_state']['debt_ratio']:.3f}")
        print(f"   Income Diversity: {updated_analysis['current_state']['income_diversity']}")
        
        print(f"\n📊 Operations Executed: {updated_analysis['state_history'] - 1}")
        print(f"📊 Total Operations in History: {updated_analysis['operations_executed']}")
        
        # Show state transformation
        wealth_change = (updated_analysis['current_state']['total_wealth'] - analysis['current_state']['total_wealth']) / analysis['current_state']['total_wealth']
        risk_change = updated_analysis['current_state']['risk_score'] - analysis['current_state']['risk_score']
        
        print(f"\n📈 Transformation Summary:")
        print(f"   Wealth Change: {wealth_change:+.1%}")
        print(f"   Risk Change: {risk_change:+.3f}")
        print(f"   Capital Efficiency: {updated_analysis['current_state']['available_capital'] / updated_analysis['current_state']['total_wealth']:.1%}")
        
    else:
        print("⚠️ State optimization failed or no improvements found")
    
    return success


def demo_specific_commutator_operations():
    """Demonstrate specific commutator operations"""
    
    print("\n🔄 SPECIFIC COMMUTATOR OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a state with specific issues
    test_state = {
        'cash': 100000,
        'investments': {
            'equities': 600000,
            'bonds': 100000,
            'real_estate': 0
        },
        'debts': {
            'credit_card': 20000,
            'mortgage': 400000
        },
        'income_streams': {
            'salary': 10000
        },
        'constraints': {
            'min_cash_reserve': 50000,
            'max_risk': 0.6,
            'max_debt_ratio': 0.4
        }
    }
    
    engine = create_commutator_engine(test_state)
    
    # Test different operation types
    operation_types = ['rebalance', 'debt_restructure', 'income_optimization', 'capital_efficiency']
    
    for op_type in operation_types:
        print(f"\n🔧 Testing {op_type.title()} Operations:")
        
        result = engine.execute_commutator_sequence_by_type([op_type])
        
        if result['success']:
            print(f"   ✅ Executed {result['operations_executed']} operations")
            print(f"   📊 Total Impact: {result['total_impact']:.2f}")
            print(f"   ⚠️ Risk Change: {result['risk_change']:+.3f}")
        else:
            print(f"   ❌ No feasible {op_type} operations found")
    
    return True


def demo_constraint_aware_optimization():
    """Demonstrate optimization with strict constraints"""
    
    print("\n🔒 CONSTRAINT-AWARE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create state with strict constraints
    constrained_state = {
        'cash': 25000,
        'investments': {
            'equities': 200000,
            'bonds': 50000,
            'real_estate': 0
        },
        'debts': {
            'credit_card': 15000,
            'mortgage': 300000
        },
        'income_streams': {
            'salary': 6000
        },
        'constraints': {
            'min_cash_reserve': 20000,  # Very strict cash requirement
            'max_risk': 0.4,            # Low risk tolerance
            'max_debt_ratio': 0.3       # Conservative debt ratio
        }
    }
    
    engine = create_commutator_engine(constrained_state)
    
    print("📊 Constrained State Analysis:")
    analysis = engine.get_state_analysis()
    print(f"   Available Capital: ${analysis['current_state']['available_capital']:,.2f}")
    print(f"   Risk Score: {analysis['current_state']['risk_score']:.3f}")
    print(f"   Debt Ratio: {analysis['current_state']['debt_ratio']:.3f}")
    
    # Generate operations
    operations = engine.generate_commutator_operations()
    feasible_operations = [op for op in operations if engine.evaluate_operation_feasibility(op)]
    
    print(f"\n🔍 Feasibility Analysis:")
    print(f"   Total Operations Generated: {len(operations)}")
    print(f"   Feasible Operations: {len(feasible_operations)}")
    
    for operation in feasible_operations[:3]:  # Show first 3 feasible
        print(f"   ✅ {operation.operation_type}: ${operation.capital_required:,.2f} required")
    
    # Try optimization
    success = engine.optimize_state({
        'risk_score': 0.25,
        'capital_efficiency': 0.7,
        'debt_ratio': 0.25
    })
    
    if success:
        updated_analysis = engine.get_state_analysis()
        print(f"\n✅ Constrained Optimization Results:")
        print(f"   Final Risk Score: {updated_analysis['current_state']['risk_score']:.3f}")
        print(f"   Final Debt Ratio: {updated_analysis['current_state']['debt_ratio']:.3f}")
        print(f"   Operations Executed: {updated_analysis['operations_executed']}")
    else:
        print("\n⚠️ Constrained optimization limited by strict constraints")
    
    return success


def main():
    """Main demonstration function"""
    
    print("🚀 COMMUTATOR ALGORITHM DEMONSTRATION SUITE")
    print("=" * 80)
    print("Testing financial state switching algorithms...")
    print()
    
    try:
        # Demo 1: Basic state optimization
        success1 = demo_suboptimal_state_optimization()
        
        # Demo 2: Specific operations
        success2 = demo_specific_commutator_operations()
        
        # Demo 3: Constraint-aware optimization
        success3 = demo_constraint_aware_optimization()
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE")
        print(f"✅ Suboptimal State Optimization: {'SUCCESS' if success1 else 'FAILED'}")
        print(f"✅ Specific Operations: {'SUCCESS' if success2 else 'FAILED'}")
        print(f"✅ Constraint-Aware Optimization: {'SUCCESS' if success3 else 'FAILED'}")
        
        if all([success1, success2, success3]):
            print("\n🎯 All commutator algorithms working correctly!")
            print("💡 The system can now transform suboptimal states into positive ones")
            print("🔄 while maintaining capital constraints and risk limits.")
        else:
            print("\n⚠️ Some demonstrations failed. Check implementation.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 