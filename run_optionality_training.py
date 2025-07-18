#!/usr/bin/env python3
"""
Optionality Training Script

This script demonstrates the implementation of the optionality algorithm for
optimal path switching and stress minimization under various market conditions.

The algorithm treats "how many ways you can get from here to a safe (good) zone"
as a first-class metric of optionality or flexibility in financial state.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import logging
from pathlib import Path

# Import our training engines
from src.training.optionality_training_engine import OptionalityTrainingEngine, run_optionality_training
from src.training.optionality_integration import OptionalityIntegrationEngine, run_integrated_training


def setup_logging():
    """Setup logging for the training script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optionality_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def demonstrate_optionality_algorithm():
    """Demonstrate the optionality algorithm with examples"""
    logger = setup_logging()
    logger.info("🚀 Starting Optionality Algorithm Demonstration")
    
    # Create optionality training engine
    engine = OptionalityTrainingEngine()
    
    # Define good region criteria
    good_region_criteria = {
        'min_wealth': 200000,
        'min_cash_ratio': 0.15,
        'max_debt_ratio': 0.3,
        'max_stress': 0.4
    }
    engine.define_good_region(good_region_criteria)
    
    # Generate sample financial states
    sample_states = generate_sample_financial_states()
    
    logger.info(f"Generated {len(sample_states)} sample financial states")
    
    # Calculate optionality for each state
    optionality_results = []
    for i, state in enumerate(sample_states):
        state_id = engine._find_or_create_state(state)
        optionality = engine.calculate_optionality(state_id, engine._get_average_market_condition())
        
        optionality_results.append({
            'state_id': state_id,
            'financial_state': state,
            'optionality_score': optionality,
            'stress_level': engine._calculate_stress_level(state),
            'is_good_region': engine._is_good_region_state(state, good_region_criteria)
        })
        
        if i % 10 == 0:
            logger.info(f"Processed {i+1}/{len(sample_states)} states")
    
    # Analyze results
    analyze_optionality_results(optionality_results)
    
    return optionality_results


def generate_sample_financial_states() -> List[Dict[str, float]]:
    """Generate sample financial states for demonstration"""
    states = []
    
    # Wealth levels
    wealth_levels = [50000, 100000, 200000, 500000, 1000000]
    
    # Cash ratios
    cash_ratios = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    # Debt ratios
    debt_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    # Income levels
    income_levels = [30000, 50000, 75000, 100000, 150000]
    
    for wealth in wealth_levels:
        for cash_ratio in cash_ratios:
            for debt_ratio in debt_ratios:
                for income in income_levels:
                    # Skip impossible combinations
                    if debt_ratio > 0.8 or cash_ratio + debt_ratio > 0.9:
                        continue
                    
                    cash = wealth * cash_ratio
                    debt = wealth * debt_ratio
                    investments = wealth * (1 - cash_ratio - debt_ratio)
                    
                    state = {
                        'total_wealth': wealth,
                        'cash': cash,
                        'investments': investments,
                        'debt': debt,
                        'income': income / 12,  # Monthly
                        'expenses': income * 0.6 / 12  # Monthly
                    }
                    
                    states.append(state)
    
    return states


def analyze_optionality_results(results: List[Dict[str, Any]]):
    """Analyze and visualize optionality results"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Analyzing Optionality Results")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Extract financial state data
    financial_data = []
    for result in results:
        financial_state = result['financial_state']
        financial_data.append({
            'optionality_score': result['optionality_score'],
            'stress_level': result['stress_level'],
            'is_good_region': result['is_good_region'],
            'total_wealth': financial_state.get('total_wealth', 0),
            'cash': financial_state.get('cash', 0),
            'investments': financial_state.get('investments', 0),
            'debt': financial_state.get('debt', 0),
            'income': financial_state.get('income', 0),
            'expenses': financial_state.get('expenses', 0)
        })
    
    df = pd.DataFrame(financial_data)
    
    # Basic statistics
    logger.info(f"Total states analyzed: {len(df)}")
    logger.info(f"Average optionality score: {df['optionality_score'].mean():.3f}")
    logger.info(f"States in good region: {df['is_good_region'].sum()}")
    logger.info(f"Average stress level: {df['stress_level'].mean():.3f}")
    
    # Correlation analysis
    correlations = df[['optionality_score', 'stress_level', 'total_wealth']].corr()
    logger.info("Correlation Matrix:")
    logger.info(correlations)
    
    # Create visualizations
    create_optionality_visualizations(df)
    
    # Save detailed results
    save_analysis_results(df)


def create_optionality_visualizations(df: pd.DataFrame):
    """Create visualizations for optionality analysis"""
    logger = logging.getLogger(__name__)
    logger.info("📈 Creating Optionality Visualizations")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optionality Algorithm Analysis', fontsize=16, fontweight='bold')
    
    # 1. Optionality vs Stress
    axes[0, 0].scatter(df['stress_level'], df['optionality_score'], alpha=0.6)
    axes[0, 0].set_xlabel('Stress Level')
    axes[0, 0].set_ylabel('Optionality Score')
    axes[0, 0].set_title('Optionality vs Stress Level')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Optionality vs Wealth
    axes[0, 1].scatter(df['total_wealth'], df['optionality_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Total Wealth')
    axes[0, 1].set_ylabel('Optionality Score')
    axes[0, 1].set_title('Optionality vs Total Wealth')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Optionality Distribution
    axes[0, 2].hist(df['optionality_score'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Optionality Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Optionality Score Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Stress vs Wealth
    axes[1, 0].scatter(df['total_wealth'], df['stress_level'], alpha=0.6)
    axes[1, 0].set_xlabel('Total Wealth')
    axes[1, 0].set_ylabel('Stress Level')
    axes[1, 0].set_title('Stress Level vs Total Wealth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Good Region Analysis
    good_region_df = df[df['is_good_region']]
    not_good_region_df = df[~df['is_good_region']]
    
    axes[1, 1].scatter(good_region_df['total_wealth'], good_region_df['optionality_score'], 
                       alpha=0.7, label='Good Region', color='green')
    axes[1, 1].scatter(not_good_region_df['total_wealth'], not_good_region_df['optionality_score'], 
                       alpha=0.7, label='Not Good Region', color='red')
    axes[1, 1].set_xlabel('Total Wealth')
    axes[1, 1].set_ylabel('Optionality Score')
    axes[1, 1].set_title('Optionality by Region Classification')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cash Ratio vs Optionality
    df['cash_ratio'] = df['cash'] / df['total_wealth']
    axes[1, 2].scatter(df['cash_ratio'], df['optionality_score'], alpha=0.6)
    axes[1, 2].set_xlabel('Cash Ratio')
    axes[1, 2].set_ylabel('Optionality Score')
    axes[1, 2].set_title('Optionality vs Cash Ratio')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "optionality_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Visualizations saved to {output_dir}")


def save_analysis_results(df: pd.DataFrame):
    """Save detailed analysis results"""
    output_dir = Path("data/outputs/optionality_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(output_dir / "optionality_analysis.csv", index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_states': len(df),
        'average_optionality': df['optionality_score'].mean(),
        'median_optionality': df['optionality_score'].median(),
        'std_optionality': df['optionality_score'].std(),
        'average_stress': df['stress_level'].mean(),
        'good_region_count': df['is_good_region'].sum(),
        'good_region_percentage': (df['is_good_region'].sum() / len(df)) * 100,
        'wealth_correlation': df['optionality_score'].corr(df['total_wealth']),
        'stress_correlation': df['optionality_score'].corr(df['stress_level'])
    }
    
    with open(output_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print("📊 Analysis Results Summary:")
    print(f"   Total states analyzed: {summary_stats['total_states']}")
    print(f"   Average optionality: {summary_stats['average_optionality']:.3f}")
    print(f"   States in good region: {summary_stats['good_region_count']} ({summary_stats['good_region_percentage']:.1f}%)")
    print(f"   Wealth-optionality correlation: {summary_stats['wealth_correlation']:.3f}")
    print(f"   Stress-optionality correlation: {summary_stats['stress_correlation']:.3f}")


def run_full_training_demonstration():
    """Run full training demonstration with both optionality and integrated approaches"""
    logger = setup_logging()
    logger.info("🎯 Starting Full Training Demonstration")
    
    # Step 1: Demonstrate basic optionality algorithm
    logger.info("Step 1: Demonstrating Optionality Algorithm")
    optionality_results = demonstrate_optionality_algorithm()
    
    # Step 2: Run optionality training
    logger.info("Step 2: Running Optionality Training")
    optionality_training_result = run_optionality_training(num_scenarios=50)
    
    # Step 3: Run integrated training
    logger.info("Step 3: Running Integrated Training")
    integrated_result = run_integrated_training(num_scenarios=50)
    
    # Step 4: Compare results
    logger.info("Step 4: Comparing Results")
    compare_training_results(optionality_training_result, integrated_result)
    
    logger.info("✅ Full Training Demonstration Completed")
    
    return {
        'optionality_results': optionality_results,
        'optionality_training': optionality_training_result,
        'integrated_training': integrated_result
    }


def compare_training_results(optionality_result: Any, integrated_result: Any):
    """Compare results from different training approaches"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Comparing Training Results")
    
    print("\n" + "="*60)
    print("TRAINING RESULTS COMPARISON")
    print("="*60)
    
    # Optionality-only results
    print("\n🔍 OPTIONALITY-ONLY TRAINING:")
    print(f"   Average optionality: {optionality_result.average_optionality:.3f}")
    print(f"   Stress minimization success: {optionality_result.stress_minimization_success:.3f}")
    print(f"   Optimal paths found: {len(optionality_result.optimal_paths)}")
    print(f"   States explored: {optionality_result.num_states_explored}")
    
    # Integrated results
    print("\n🔗 INTEGRATED TRAINING:")
    combined_metrics = integrated_result.combined_insights['combined_metrics']
    print(f"   Overall success rate: {combined_metrics['overall_success_rate']:.3f}")
    print(f"   Flexibility score: {combined_metrics['flexibility_score']:.3f}")
    print(f"   Stress resilience: {combined_metrics['stress_resilience']:.3f}")
    print(f"   Optimal strategies: {len(integrated_result.optimal_strategies)}")
    
    # Mesh-specific results
    mesh_results = integrated_result.mesh_results
    print(f"   Successful recoveries: {mesh_results.successful_recoveries}")
    print(f"   Failed recoveries: {mesh_results.failed_recoveries}")
    print(f"   Average recovery time: {mesh_results.average_recovery_time:.1f} months")
    
    # Market condition performance
    optionality_market = optionality_result.market_condition_performance
    print(f"\n📈 MARKET CONDITION PERFORMANCE (Optionality):")
    for condition, performance in optionality_market.items():
        print(f"   {condition}: {performance:.3f}")
    
    print("\n" + "="*60)


def demonstrate_path_optimization():
    """Demonstrate path optimization using optionality algorithm"""
    logger = setup_logging()
    logger.info("🛤️ Demonstrating Path Optimization")
    
    # Create training engine
    engine = OptionalityTrainingEngine()
    
    # Define good region
    engine.define_good_region({
        'min_wealth': 200000,
        'min_cash_ratio': 0.15,
        'max_debt_ratio': 0.3,
        'max_stress': 0.4
    })
    
    # Create sample financial states
    sample_states = [
        {
            'total_wealth': 150000,
            'cash': 20000,
            'investments': 100000,
            'debt': 30000,
            'income': 6000,
            'expenses': 4000
        },
        {
            'total_wealth': 300000,
            'cash': 60000,
            'investments': 200000,
            'debt': 40000,
            'income': 8000,
            'expenses': 5000
        },
        {
            'total_wealth': 80000,
            'cash': 5000,
            'investments': 60000,
            'debt': 15000,
            'income': 4000,
            'expenses': 3500
        }
    ]
    
    print("\n" + "="*60)
    print("PATH OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    for i, state in enumerate(sample_states):
        print(f"\n📊 Financial State {i+1}:")
        print(f"   Total Wealth: ${state['total_wealth']:,.0f}")
        print(f"   Cash: ${state['cash']:,.0f}")
        print(f"   Investments: ${state['investments']:,.0f}")
        print(f"   Debt: ${state['debt']:,.0f}")
        
        # Find optimal paths
        state_id = engine._find_or_create_state(state)
        optimal_paths = engine.find_optimal_paths(state_id, target_optionality=0.3)
        
        print(f"   Optimal paths found: {len(optimal_paths)}")
        
        if optimal_paths:
            best_path = optimal_paths[0]
            print(f"   Best path optionality gain: {best_path.optionality_gain:.3f}")
            print(f"   Best path stress level: {best_path.total_stress:.3f}")
            print(f"   Best path time horizon: {best_path.time_horizon} months")
            print(f"   Recommended actions: {best_path.actions[:3]}...")  # Show first 3 actions
        else:
            print("   No optimal paths found")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🚀 Optionality Algorithm Training Script")
    print("="*60)
    
    # Run full demonstration
    results = run_full_training_demonstration()
    
    # Demonstrate path optimization
    demonstrate_path_optimization()
    
    print("\n✅ Training script completed successfully!")
    print("📁 Check the following directories for results:")
    print("   - data/outputs/optionality_training/")
    print("   - data/outputs/integrated_training/")
    print("   - data/outputs/optionality_analysis/")
    print("   - visualizations/") 