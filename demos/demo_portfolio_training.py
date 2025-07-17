#!/usr/bin/env python3
"""
Demo: Portfolio Training with Stochastic Gradient Descent

This demo showcases the portfolio training engine that uses stochastic gradient descent
to optimize portfolio composition changes across multiple mesh engines.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.portfolio_training_engine import PortfolioTrainingEngine, PortfolioWeights, TrainingScenario


def demo_portfolio_training():
    """Demonstrate portfolio training with stochastic gradient descent"""
    print("=" * 80)
    print("DEMO: Portfolio Training with Stochastic Gradient Descent")
    print("=" * 80)
    
    # Initialize the portfolio training engine
    print("\n🚀 Initializing Portfolio Training Engine...")
    training_engine = PortfolioTrainingEngine(learning_rate=0.01, batch_size=32)
    
    # Initialize mesh engines
    initial_financial_state = {
        'cash': 200000,
        'bonds': 300000,
        'stocks': 400000,
        'real_estate': 100000,
        'total_wealth': 1000000
    }
    
    training_engine.initialize_mesh_engines(initial_financial_state)
    
    # Generate training scenarios
    print("\n📊 Generating Training Scenarios...")
    scenarios = training_engine.generate_training_scenarios(num_scenarios=200)
    
    print(f"✅ Generated {len(scenarios)} diverse training scenarios")
    print("   - Initial wealth: $100K - $2M")
    print("   - Time horizons: 5-20 years")
    print("   - Risk tolerance: 0.1 - 0.9")
    print("   - Market volatility: 15% - 35%")
    print("   - Target returns: 5% - 12%")
    
    # Run portfolio optimization training
    print("\n🎯 Starting Portfolio Optimization Training...")
    print("   - Algorithm: Stochastic Gradient Descent")
    print("   - Learning rate: 0.01")
    print("   - Batch size: 32")
    print("   - Epochs: 100")
    
    training_results = training_engine.train_portfolio_optimization(
        scenarios=scenarios,
        num_epochs=100
    )
    
    # Display training results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    
    optimal_weights = training_results['optimal_weights']
    print(f"\n📈 Optimal Portfolio Weights:")
    print(f"   Cash: {optimal_weights['cash']:.1%}")
    print(f"   Bonds: {optimal_weights['bonds']:.1%}")
    print(f"   Stocks: {optimal_weights['stocks']:.1%}")
    print(f"   Real Estate: {optimal_weights['real_estate']:.1%}")
    
    print(f"\n📊 Final Metrics:")
    print(f"   Final Loss: {training_results['final_loss']:.6f}")
    print(f"   Expected Return: {training_results['final_return']:.1%}")
    print(f"   Portfolio Volatility: {training_results['final_volatility']:.1%}")
    
    # Run mesh integration test
    print("\n🔄 Running Mesh Integration Test...")
    integration_results = training_engine.run_mesh_integration_test(training_results)
    
    print(f"\n📋 Mesh Integration Results:")
    if 'mesh_test' in integration_results:
        mesh_test = integration_results['mesh_test']
        print(f"   Mesh Status: {mesh_test.get('mesh_status', 'N/A')}")
        print(f"   Payment Options: {mesh_test.get('payment_options_count', 0)}")
    
    if 'accounting_test' in integration_results:
        accounting_test = integration_results['accounting_test']
        print(f"   Total Assets: ${accounting_test.get('total_assets', 0):,.2f}")
        print(f"   Net Worth: ${accounting_test.get('net_worth', 0):,.2f}")
    
    # Export training results
    print("\n📁 Exporting Training Results...")
    training_engine.export_training_results(training_results, "portfolio_training_results.json")
    
    # Get training statistics
    stats = training_engine.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total Entries: {stats['total_entries']}")
    print(f"   Flow Items: {stats['flow_items']}")
    print(f"   Balance Items: {stats['balance_items']}")
    print(f"   Total Amount: ${stats['total_amount']}")
    
    return training_engine, training_results, integration_results


def demo_training_visualization(training_results):
    """Demonstrate training visualization"""
    print("\n" + "=" * 80)
    print("TRAINING VISUALIZATION")
    print("=" * 80)
    
    # Create visualization of training progress
    training_history = training_results['training_history']
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    epochs = range(1, len(training_history['losses']) + 1)
    ax1.plot(epochs, training_history['losses'], 'b-', linewidth=2)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Portfolio Returns
    ax2.plot(epochs, training_history['returns'], 'g-', linewidth=2)
    ax2.set_title('Portfolio Expected Return Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Expected Return')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Portfolio Volatility
    ax3.plot(epochs, training_history['volatilities'], 'r-', linewidth=2)
    ax3.set_title('Portfolio Volatility Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Volatility')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Portfolio Weights
    optimal_weights = training_results['optimal_weights']
    asset_classes = list(optimal_weights.keys())
    weights = list(optimal_weights.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax4.pie(weights, labels=asset_classes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Optimal Portfolio Allocation')
    
    plt.tight_layout()
    plt.savefig('portfolio_training_visualization.png', dpi=300, bbox_inches='tight')
    print("📊 Training visualization saved as 'portfolio_training_visualization.png'")
    
    # Create detailed analysis
    print("\n📈 Detailed Training Analysis:")
    
    # Loss analysis
    final_loss = training_results['final_loss']
    initial_loss = training_history['losses'][0]
    loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"   Loss Improvement: {loss_improvement:.1f}%")
    
    # Return analysis
    final_return = training_results['final_return']
    print(f"   Final Expected Return: {final_return:.1%}")
    
    # Volatility analysis
    final_volatility = training_results['final_volatility']
    sharpe_ratio = final_return / final_volatility if final_volatility > 0 else 0
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Weight analysis
    print(f"\n📊 Portfolio Weight Analysis:")
    for asset, weight in optimal_weights.items():
        print(f"   {asset.title()}: {weight:.1%}")
    
    plt.show()


def demo_scenario_analysis(training_engine, scenarios):
    """Demonstrate scenario analysis"""
    print("\n" + "=" * 80)
    print("SCENARIO ANALYSIS")
    print("=" * 80)
    
    # Analyze scenario distribution
    print("\n📊 Training Scenario Distribution:")
    
    # Wealth distribution
    wealths = [s.initial_wealth for s in scenarios]
    print(f"   Wealth Range: ${min(wealths):,.0f} - ${max(wealths):,.0f}")
    print(f"   Average Wealth: ${np.mean(wealths):,.0f}")
    
    # Risk tolerance distribution
    risk_tolerances = [s.risk_tolerance for s in scenarios]
    print(f"   Risk Tolerance Range: {min(risk_tolerances):.1f} - {max(risk_tolerances):.1f}")
    print(f"   Average Risk Tolerance: {np.mean(risk_tolerances):.2f}")
    
    # Target return distribution
    target_returns = [s.target_return for s in scenarios]
    print(f"   Target Return Range: {min(target_returns):.1%} - {max(target_returns):.1%}")
    print(f"   Average Target Return: {np.mean(target_returns):.1%}")
    
    # Market volatility distribution
    volatilities = [s.market_volatility for s in scenarios]
    print(f"   Market Volatility Range: {min(volatilities):.1%} - {max(volatilities):.1%}")
    print(f"   Average Market Volatility: {np.mean(volatilities):.1%}")
    
    # Create scenario analysis visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Wealth distribution
    ax1.hist(wealths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Initial Wealth Distribution')
    ax1.set_xlabel('Initial Wealth ($)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Risk tolerance distribution
    ax2.hist(risk_tolerances, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Risk Tolerance Distribution')
    ax2.set_xlabel('Risk Tolerance')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Target return distribution
    ax3.hist(target_returns, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Target Return Distribution')
    ax3.set_xlabel('Target Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Market volatility distribution
    ax4.hist(volatilities, bins=20, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_title('Market Volatility Distribution')
    ax4.set_xlabel('Market Volatility')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scenario_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Scenario analysis saved as 'scenario_analysis.png'")
    
    plt.show()


def demo_mesh_engine_comparison(training_engine, integration_results):
    """Demonstrate comparison across different mesh engines"""
    print("\n" + "=" * 80)
    print("MESH ENGINE COMPARISON")
    print("=" * 80)
    
    print("\n🔍 Comparing Mesh Engine Performance:")
    
    # Stochastic mesh engine
    if training_engine.stochastic_mesh:
        print("   ✅ Stochastic Mesh Engine: Initialized")
        mesh_status = training_engine.stochastic_mesh.get_mesh_status()
        print(f"      - Nodes: {mesh_status.get('total_nodes', 'N/A')}")
        print(f"      - Edges: {mesh_status.get('total_edges', 'N/A')}")
    else:
        print("   ❌ Stochastic Mesh Engine: Not initialized")
    
    # Time uncertainty mesh engine
    if training_engine.time_uncertainty_mesh:
        print("   ✅ Time Uncertainty Mesh Engine: Initialized")
        print(f"      - GPU Acceleration: {'Yes' if training_engine.time_uncertainty_mesh.use_gpu else 'No'}")
    else:
        print("   ❌ Time Uncertainty Mesh Engine: Not initialized")
    
    # Mesh engine layer
    if training_engine.mesh_engine_layer:
        print("   ✅ Mesh Engine Layer: Initialized")
    else:
        print("   ❌ Mesh Engine Layer: Not initialized")
    
    # Accounting engine
    if training_engine.accounting_engine:
        print("   ✅ Accounting Engine: Initialized")
        accounts = len(training_engine.accounting_engine.accounts)
        print(f"      - Accounts: {accounts}")
    else:
        print("   ❌ Accounting Engine: Not initialized")
    
    # Integration test results
    if 'mesh_test' in integration_results:
        mesh_test = integration_results['mesh_test']
        print(f"\n📊 Mesh Integration Test Results:")
        print(f"   - Payment Options: {mesh_test.get('payment_options_count', 0)}")
        print(f"   - Applied Weights: {mesh_test.get('applied_weights', {})}")
    
    if 'accounting_test' in integration_results:
        accounting_test = integration_results['accounting_test']
        print(f"\n📊 Accounting Integration Test Results:")
        print(f"   - Total Assets: ${accounting_test.get('total_assets', 0):,.2f}")
        print(f"   - Net Worth: ${accounting_test.get('net_worth', 0):,.2f}")


def main():
    """Main demo function"""
    print("🎯 Portfolio Training with Stochastic Gradient Descent Demo")
    print("=" * 80)
    print("This demo showcases:")
    print("  • Stochastic gradient descent for portfolio optimization")
    print("  • Integration with multiple mesh engines")
    print("  • Enhanced logging and accounting")
    print("  • Training visualization and analysis")
    print()
    
    try:
        # Run portfolio training
        training_engine, training_results, integration_results = demo_portfolio_training()
        
        # Run training visualization
        demo_training_visualization(training_results)
        
        # Run scenario analysis
        scenarios = training_engine.generate_training_scenarios(num_scenarios=50)  # Smaller set for analysis
        demo_scenario_analysis(training_engine, scenarios)
        
        # Run mesh engine comparison
        demo_mesh_engine_comparison(training_engine, integration_results)
        
        print("\n" + "=" * 80)
        print("✅ PORTFOLIO TRAINING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📁 Generated Files:")
        print("  • portfolio_training_results.json")
        print("  • portfolio_training_visualization.png")
        print("  • scenario_analysis.png")
        
        print("\n🎯 Key Features Demonstrated:")
        print("  • Stochastic gradient descent optimization")
        print("  • Multi-asset portfolio allocation")
        print("  • Risk-return optimization")
        print("  • Mesh engine integration")
        print("  • Enhanced accounting logging")
        print("  • Training visualization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 