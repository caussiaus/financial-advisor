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
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.portfolio_training_engine import (
    PortfolioTrainingEngine,
    PortfolioWeights,
    TrainingScenario,
)


def create_hardcoded_time_shocks():
    """Create hardcoded financial time shocks for testing"""
    base_date = datetime.now()

    shocks = [
        {
            "timestamp": base_date + timedelta(days=30),
            "type": "market_crash",
            "magnitude": -0.15,
            "description": "Market correction affecting portfolio value",
            "category": "investment",
        },
        {
            "timestamp": base_date + timedelta(days=90),
            "type": "interest_rate_increase",
            "magnitude": -0.08,
            "description": "Federal Reserve rate hike",
            "category": "interest_rate",
        },
        {
            "timestamp": base_date + timedelta(days=180),
            "type": "job_loss",
            "magnitude": -0.25,
            "description": "Unexpected job loss",
            "category": "income",
        },
        {
            "timestamp": base_date + timedelta(days=270),
            "type": "medical_emergency",
            "magnitude": 0.12,
            "description": "Medical emergency expenses",
            "category": "expense",
        },
        {
            "timestamp": base_date + timedelta(days=365),
            "type": "inheritance",
            "magnitude": 0.20,
            "description": "Unexpected inheritance",
            "category": "windfall",
        },
        {
            "timestamp": base_date + timedelta(days=450),
            "type": "home_repair",
            "magnitude": -0.10,
            "description": "Major home repair needed",
            "category": "expense",
        },
        {
            "timestamp": base_date + timedelta(days=540),
            "type": "promotion",
            "magnitude": 0.15,
            "description": "Career promotion with salary increase",
            "category": "income",
        },
        {
            "timestamp": base_date + timedelta(days=630),
            "type": "inflation_shock",
            "magnitude": -0.05,
            "description": "Unexpected inflation reducing purchasing power",
            "category": "inflation",
        },
    ]

    return shocks


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
        "cash": 200000,
        "bonds": 300000,
        "stocks": 400000,
        "real_estate": 100000,
        "total_wealth": 1000000,
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
        scenarios=scenarios, num_epochs=100
    )

    # Display training results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)

    optimal_weights = training_results["optimal_weights"]
    print(f"\n📈 Optimal Portfolio Weights:")
    print(f"   Cash: {optimal_weights['cash']:.1%}")
    print(f"   Bonds: {optimal_weights['bonds']:.1%}")
    print(f"   Stocks: {optimal_weights['stocks']:.1%}")
    print(f"   Real Estate: {optimal_weights['real_estate']:.1%}")

    print(f"\n📊 Final Metrics:")
    print(f"   Final Loss: {training_results['final_loss']:.6f}")
    print(f"   Expected Return: {training_results['final_return']:.1%}")
    print(f"   Portfolio Volatility: {training_results['final_volatility']:.1%}")

    # Run mesh integration test with hardcoded shocks
    print("\n🔄 Running Mesh Integration Test with Hardcoded Time Shocks...")
    integration_results = run_mesh_integration_with_hardcoded_shocks(
        training_engine, training_results
    )

    print(f"\n📋 Mesh Integration Results:")
    if "mesh_test" in integration_results:
        mesh_test = integration_results["mesh_test"]
        print(f"   Mesh Status: {mesh_test.get('mesh_status', 'N/A')}")
        print(f"   Payment Options: {mesh_test.get('payment_options_count', 0)}")

    if "accounting_test" in integration_results:
        accounting_test = integration_results["accounting_test"]
        print(f"   Total Assets: ${accounting_test.get('total_assets', 0):,.2f}")
        print(f"   Net Worth: ${accounting_test.get('net_worth', 0):,.2f}")

    # Export training results
    print("\n📁 Exporting Training Results...")
    training_engine.export_training_results(
        training_results, "portfolio_training_results.json"
    )

    # Get training statistics
    stats = training_engine.get_training_statistics()
    print(f"\n📊 Training Statistics:")
    print(f"   Total Entries: {stats['total_entries']}")
    print(f"   Flow Items: {stats['flow_items']}")
    print(f"   Balance Items: {stats['balance_items']}")
    print(f"   Total Amount: ${stats['total_amount']}")

    return training_engine, training_results, integration_results


def run_mesh_integration_with_hardcoded_shocks(training_engine, training_results):
    """Run mesh integration test using hardcoded time shocks"""
    try:
        print("   📅 Creating hardcoded financial time shocks...")
        shocks = create_hardcoded_time_shocks()

        print(f"   ✅ Created {len(shocks)} hardcoded time shocks:")
        for i, shock in enumerate(shocks, 1):
            print(
                f"      {i}. {shock['type']}: {shock['magnitude']:.1%} at {shock['timestamp'].strftime('%Y-%m-%d')}"
            )

        # Create a simple test scenario
        test_scenario = TrainingScenario(
            scenario_id="hardcoded_test",
            initial_wealth=1000000,
            time_horizon_years=2.0,
            risk_tolerance=0.6,
            age=35,
            income_growth_rate=0.05,
            market_volatility=0.20,
            target_return=0.08,
            constraints={
                "max_stock_allocation": 0.7,
                "min_cash_allocation": 0.1,
                "max_real_estate_allocation": 0.3,
            },
        )

        # Run the mesh integration test
        integration_results = training_engine.run_mesh_integration_test(
            training_results
        )

        # Add shock information to results
        integration_results["hardcoded_shocks"] = {
            "shock_count": len(shocks),
            "shock_types": [shock["type"] for shock in shocks],
            "total_impact": sum(shock["magnitude"] for shock in shocks),
            "shocks": shocks,
        }

        print("   ✅ Mesh integration test completed successfully")
        return integration_results

    except Exception as e:
        print(f"   ❌ Mesh integration test failed: {e}")
        return {
            "error": str(e),
            "hardcoded_shocks": {
                "shock_count": len(shocks) if "shocks" in locals() else 0,
                "error": "Integration test failed",
            },
        }


def demo_training_visualization(training_results):
    """Demonstrate training visualization"""
    print("\n" + "=" * 80)
    print("TRAINING VISUALIZATION")
    print("=" * 80)

    # Create visualization of training progress
    training_history = training_results["training_history"]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training Loss
    epochs = range(1, len(training_history["losses"]) + 1)
    ax1.plot(epochs, training_history["losses"], "b-", linewidth=2)
    ax1.set_title("Training Loss Over Time")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Portfolio Returns
    ax2.plot(epochs, training_history["returns"], "g-", linewidth=2)
    ax2.set_title("Portfolio Expected Return Over Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Expected Return")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Portfolio Volatility
    ax3.plot(epochs, training_history["volatilities"], "r-", linewidth=2)
    ax3.set_title("Portfolio Volatility Over Time")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Volatility")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Portfolio Weights
    optimal_weights = training_results["optimal_weights"]
    asset_classes = list(optimal_weights.keys())
    weights = list(optimal_weights.values())

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    ax4.pie(
        weights, labels=asset_classes, autopct="%1.1f%%", colors=colors, startangle=90
    )
    ax4.set_title("Optimal Portfolio Allocation")

    plt.tight_layout()
    plt.savefig("portfolio_training_visualization.png", dpi=300, bbox_inches="tight")
    print("📊 Training visualization saved as 'portfolio_training_visualization.png'")

    # Create detailed analysis
    print("\n📈 Detailed Training Analysis:")

    # Loss analysis
    final_loss = training_results["final_loss"]
    initial_loss = training_history["losses"][0]
    loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"   Loss Improvement: {loss_improvement:.1f}%")

    # Return analysis
    final_return = training_results["final_return"]
    print(f"   Final Expected Return: {final_return:.1%}")

    # Volatility analysis
    final_volatility = training_results["final_volatility"]
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
    print(
        f"   Risk Tolerance Range: {min(risk_tolerances):.2f} - {max(risk_tolerances):.2f}"
    )
    print(f"   Average Risk Tolerance: {np.mean(risk_tolerances):.2f}")

    # Age distribution
    ages = [s.age for s in scenarios]
    print(f"   Age Range: {min(ages)} - {max(ages)} years")
    print(f"   Average Age: {np.mean(ages):.1f} years")

    # Target return distribution
    target_returns = [s.target_return for s in scenarios]
    print(
        f"   Target Return Range: {min(target_returns):.1%} - {max(target_returns):.1%}"
    )
    print(f"   Average Target Return: {np.mean(target_returns):.1%}")


def demo_mesh_engine_comparison(training_engine, integration_results):
    """Demonstrate mesh engine comparison"""
    print("\n" + "=" * 80)
    print("MESH ENGINE COMPARISON")
    print("=" * 80)

    if "hardcoded_shocks" in integration_results:
        shocks = integration_results["hardcoded_shocks"]
        print(f"\n📅 Hardcoded Time Shocks Analysis:")
        print(f"   Total Shocks: {shocks['shock_count']}")
        if "shock_types" in shocks:
            print(f"   Shock Types: {', '.join(shocks['shock_types'])}")
        if "total_impact" in shocks:
            print(f"   Total Impact: {shocks['total_impact']:.1%}")

        if "shocks" in shocks:
            print(f"\n📊 Individual Shock Details:")
            for i, shock in enumerate(shocks["shocks"], 1):
                print(f"   {i}. {shock['type'].replace('_', ' ').title()}")
                print(f"      Date: {shock['timestamp'].strftime('%Y-%m-%d')}")
                print(f"      Impact: {shock['magnitude']:.1%}")
                print(f"      Category: {shock['category']}")
                print(f"      Description: {shock['description']}")
                print()


def main():
    """Main demo function"""
    try:
        print("🎯 Starting Portfolio Training Demo with Hardcoded Time Shocks")

        # Run the main demo
        training_engine, training_results, integration_results = (
            demo_portfolio_training()
        )

        # Run additional analysis
        scenarios = training_engine.generate_training_scenarios(
            num_scenarios=50
        )  # Generate fresh scenarios for analysis
        demo_scenario_analysis(training_engine, scenarios)
        demo_mesh_engine_comparison(training_engine, integration_results)

        # Create visualizations
        demo_training_visualization(training_results)

        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📁 Generated files:")
        print("   - portfolio_training_results.json")
        print("   - portfolio_training_visualization.png")
        print("\n📊 Key Results:")
        print(f"   - Optimal Portfolio: {training_results['optimal_weights']}")
        print(f"   - Final Return: {training_results['final_return']:.1%}")
        print(f"   - Final Volatility: {training_results['final_volatility']:.1%}")
        print(f"   - Training Loss: {training_results['final_loss']:.6f}")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
