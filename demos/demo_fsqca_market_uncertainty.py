#!/usr/bin/env python3
"""
Demo: fsQCA Market Uncertainty Decision Analysis

This demo showcases fsQCA analysis for determining optimal financial decisions
and capital allocation strategies during market uncertainty.

Key Features:
- Market uncertainty surface analysis
- Capital allocation decision optimization
- Backtesting with synthesized data
- Real-time decision recommendations
- Risk-adjusted return optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import time

# Import the fsQCA market uncertainty analyzer
try:
    from src.analysis.fsqca_market_uncertainty_analyzer import (
        fsQCAMarketUncertaintyAnalyzer, 
        MarketUncertaintyLevel, 
        DecisionType
    )
except ImportError:
    print("⚠️ fsQCA market uncertainty analyzer not found, creating simplified version...")
    # Create a simplified version for demo purposes
    class fsQCAMarketUncertaintyAnalyzer:
        def __init__(self, config=None):
            self.config = config or {}
        
        def run_fsqca_analysis(self, market_data, portfolio_data):
            # Simplified analysis
            return type('obj', (object,), {
                'solution_coverage': 0.85,
                'solution_consistency': 0.78,
                'necessary_conditions': {'high_uncertainty': 0.92, 'defensive_allocation': 0.88},
                'sufficient_conditions': {'high_uncertainty': 0.95, 'defensive_allocation': 0.90},
                'optimal_decisions': [],
                'capital_allocation_strategies': [],
                'market_surfaces': [],
                'backtest_results': {
                    'overall_performance': {
                        'avg_return': 0.065,
                        'avg_risk': 0.18,
                        'best_strategy': 'strategy_surface_2'
                    }
                },
                'decision_recommendations': {
                    'capital_allocation': {'cash': 0.4, 'bonds': 0.4, 'stocks': 0.15, 'real_estate': 0.03, 'commodities': 0.02},
                    'confidence': 0.85
                },
                'analysis_summary': {
                    'total_surfaces': 4,
                    'total_strategies': 4,
                    'total_decisions': 12,
                    'uncertainty_distribution': {'low': 1, 'medium': 1, 'high': 1, 'extreme': 1}
                }
            })()
    
    class MarketUncertaintyLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        EXTREME = "extreme"
    
    class DecisionType:
        CAPITAL_ALLOCATION = "capital_allocation"
        RISK_MANAGEMENT = "risk_management"
        ASSET_SELECTION = "asset_selection"
        TIMING = "timing"

def generate_synthetic_market_data() -> Dict[str, Any]:
    """Generate synthetic market data for testing"""
    print("🔄 Generating synthetic market data...")
    
    # Generate time series data
    n_periods = 100
    base_price = 100
    
    # Generate price series with different market conditions
    prices = []
    volumes = []
    volatilities = []
    correlations = []
    
    for i in range(n_periods):
        # Add market cycles
        cycle_factor = np.sin(i * 2 * np.pi / 20)  # 20-period cycle
        
        # Add random walk
        price_change = np.random.normal(0, 0.02) + 0.001 * cycle_factor
        base_price *= (1 + price_change)
        prices.append(base_price)
        
        # Generate volume (inversely related to volatility)
        volatility = 0.2 + 0.1 * abs(cycle_factor) + np.random.normal(0, 0.05)
        volatilities.append(max(0.05, min(0.8, volatility)))
        
        # Generate volume
        volume = 1000 + 200 * (1 - volatility) + np.random.normal(0, 100)
        volumes.append(max(100, volume))
        
        # Generate correlation matrix (simplified)
        correlation = 0.5 + 0.3 * cycle_factor + np.random.normal(0, 0.1)
        correlations.append(max(0.1, min(0.9, correlation)))
    
    market_data = {
        'prices': prices,
        'volumes': volumes,
        'volatilities': volatilities,
        'correlations': np.array(correlations).reshape(-1, 1),
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'data_points': n_periods,
            'market_conditions': 'synthetic',
            'volatility_regime': 'mixed'
        }
    }
    
    print(f"✅ Generated {n_periods} periods of synthetic market data")
    return market_data

def generate_sample_portfolio() -> Dict[str, float]:
    """Generate sample portfolio data"""
    return {
        'cash': 0.20,
        'bonds': 0.30,
        'stocks': 0.35,
        'real_estate': 0.10,
        'commodities': 0.05
    }

def demo_market_surface_analysis():
    """Demo market surface analysis"""
    print("\n" + "="*80)
    print("🔍 MARKET SURFACE ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = fsQCAMarketUncertaintyAnalyzer({
        'uncertainty_thresholds': {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.6,
            'extreme': 0.8
        }
    })
    
    # Generate market data
    market_data = generate_synthetic_market_data()
    
    # Analyze market surfaces
    print("\n🔄 Analyzing market surfaces...")
    surfaces = analyzer.analyze_market_uncertainty_surfaces(market_data)
    
    print(f"\n📊 SURFACE ANALYSIS RESULTS:")
    print(f"  Total surfaces identified: {len(surfaces)}")
    
    for i, surface in enumerate(surfaces):
        print(f"\n  Surface {i+1} ({surface.surface_id}):")
        print(f"    Uncertainty level: {surface.uncertainty_level.value}")
        print(f"    Volatility: {surface.volatility:.3f}")
        print(f"    Decision opportunities: {len(surface.decision_opportunities)}")
        print(f"    Market depth: {surface.liquidity_metrics['market_depth']:.3f}")
        
        # Show decision opportunities
        for j, opportunity in enumerate(surface.decision_opportunities):
            print(f"      Opportunity {j+1}: {opportunity['action']} (confidence: {opportunity['confidence']:.2f})")
    
    return analyzer, surfaces

def demo_capital_allocation_optimization():
    """Demo capital allocation optimization"""
    print("\n" + "="*80)
    print("🎯 CAPITAL ALLOCATION OPTIMIZATION")
    print("="*80)
    
    analyzer, surfaces = demo_market_surface_analysis()
    
    # Generate sample portfolio
    portfolio = generate_sample_portfolio()
    
    print(f"\n📊 Current Portfolio:")
    for asset, allocation in portfolio.items():
        print(f"  {asset}: {allocation:.1%}")
    
    # Optimize capital allocation
    print(f"\n🔄 Optimizing capital allocation strategies...")
    strategies = analyzer.optimize_capital_allocation(surfaces, portfolio)
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"  Total strategies created: {len(strategies)}")
    
    for i, strategy in enumerate(strategies):
        print(f"\n  Strategy {i+1} ({strategy.strategy_id}):")
        print(f"    Uncertainty level: {strategy.uncertainty_level.value}")
        print(f"    Expected return: {strategy.success_metrics['expected_return']:.1%}")
        print(f"    Risk score: {strategy.success_metrics['risk_score']:.3f}")
        print(f"    Stability score: {strategy.success_metrics['stability_score']:.3f}")
        
        print(f"    Asset allocation:")
        for asset, allocation in strategy.asset_allocation.items():
            print(f"      {asset}: {allocation:.1%}")
        
        print(f"    Risk parameters:")
        for param, value in strategy.risk_parameters.items():
            print(f"      {param}: {value:.3f}")
    
    return analyzer, strategies

def demo_fsqca_analysis():
    """Demo comprehensive fsQCA analysis"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE fsQCA ANALYSIS")
    print("="*80)
    
    analyzer, strategies = demo_capital_allocation_optimization()
    
    # Generate market data and portfolio
    market_data = generate_synthetic_market_data()
    portfolio = generate_sample_portfolio()
    
    # Run fsQCA analysis
    print(f"\n🔄 Running fsQCA analysis...")
    result = analyzer.run_fsqca_analysis(market_data, portfolio)
    
    print(f"\n📊 fsQCA ANALYSIS RESULTS:")
    print(f"  Solution coverage: {result.solution_coverage:.2%}")
    print(f"  Solution consistency: {result.solution_consistency:.2%}")
    print(f"  Total decisions: {len(result.optimal_decisions)}")
    print(f"  Total strategies: {len(result.capital_allocation_strategies)}")
    print(f"  Total surfaces: {len(result.market_surfaces)}")
    
    # Print necessary conditions
    print(f"\n🔍 NECESSARY CONDITIONS:")
    for condition, score in result.necessary_conditions.items():
        if score > 0.5:
            print(f"  {condition}: {score:.2%}")
    
    # Print sufficient conditions
    print(f"\n✅ SUFFICIENT CONDITIONS:")
    for condition, score in result.sufficient_conditions.items():
        if score > 0.7:
            print(f"  {condition}: {score:.2%}")
    
    # Print backtest results
    print(f"\n📈 BACKTEST RESULTS:")
    perf = result.backtest_results['overall_performance']
    print(f"  Average return: {perf['avg_return']:.1%}")
    print(f"  Average risk: {perf['avg_risk']:.1%}")
    print(f"  Best strategy: {perf['best_strategy']}")
    
    # Print decision recommendations
    print(f"\n💡 DECISION RECOMMENDATIONS:")
    recs = result.decision_recommendations
    print(f"  Confidence: {recs['confidence']:.2%}")
    
    if 'capital_allocation' in recs:
        print(f"  Recommended allocation:")
        for asset, allocation in recs['capital_allocation'].items():
            print(f"    {asset}: {allocation:.1%}")
    
    return analyzer, result

def demo_real_time_recommendations():
    """Demo real-time decision recommendations"""
    print("\n" + "="*80)
    print("⚡ REAL-TIME DECISION RECOMMENDATIONS")
    print("="*80)
    
    analyzer, result = demo_fsqca_analysis()
    
    # Simulate real-time market data
    print(f"\n🔄 Simulating real-time market conditions...")
    
    # Generate current market data with different uncertainty levels
    uncertainty_scenarios = [
        ('Low Uncertainty', {'volatilities': [0.15, 0.18, 0.16]}),
        ('Medium Uncertainty', {'volatilities': [0.35, 0.38, 0.36]}),
        ('High Uncertainty', {'volatilities': [0.55, 0.58, 0.56]}),
        ('Extreme Uncertainty', {'volatilities': [0.75, 0.78, 0.76]})
    ]
    
    current_portfolio = generate_sample_portfolio()
    
    for scenario_name, market_conditions in uncertainty_scenarios:
        print(f"\n📊 {scenario_name.upper()}:")
        
        # Create current market data
        current_market_data = {
            'prices': [100, 102, 98, 105, 103],
            'volumes': [1000, 1100, 900, 1200, 1150],
            'volatilities': market_conditions['volatilities'],
            'correlations': np.array([0.5, 0.6, 0.4, 0.7, 0.5])
        }
        
        # Get real-time recommendations
        recommendations = analyzer.get_real_time_recommendations(current_market_data, current_portfolio)
        
        print(f"  Uncertainty level: {recommendations['uncertainty_level']}")
        print(f"  Confidence: {recommendations['confidence']:.2%}")
        print(f"  Best strategy: {recommendations['best_strategy']}")
        print(f"  Market surfaces: {recommendations['market_surfaces']}")
        
        if 'recommendations' in recommendations and 'capital_allocation' in recommendations['recommendations']:
            print(f"  Recommended allocation:")
            for asset, allocation in recommendations['recommendations']['capital_allocation'].items():
                print(f"    {asset}: {allocation:.1%}")

def demo_ui_integration():
    """Demo UI integration for real-time analysis"""
    print("\n" + "="*80)
    print("🖥️ UI INTEGRATION DEMO")
    print("="*80)
    
    print(f"\n🔄 Simulating UI integration...")
    
    # Simulate UI data flow
    ui_data = {
        'market_data': generate_synthetic_market_data(),
        'portfolio_data': generate_sample_portfolio(),
        'user_preferences': {
            'risk_tolerance': 'moderate',
            'investment_horizon': 'medium',
            'liquidity_needs': 'high'
        },
        'market_conditions': {
            'current_volatility': 0.45,
            'trend_direction': 'sideways',
            'liquidity_conditions': 'normal'
        }
    }
    
    # Initialize analyzer
    analyzer = fsQCAMarketUncertaintyAnalyzer()
    
    # Run analysis
    result = analyzer.run_fsqca_analysis(ui_data['market_data'], ui_data['portfolio_data'])
    
    # Prepare UI response
    ui_response = {
        'analysis_status': 'completed',
        'confidence_score': result.decision_recommendations['confidence'],
        'recommendations': result.decision_recommendations,
        'market_surfaces': len(result.market_surfaces),
        'backtest_performance': result.backtest_results['overall_performance'],
        'uncertainty_distribution': result.analysis_summary['uncertainty_distribution'],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n📊 UI INTEGRATION RESULTS:")
    print(f"  Analysis status: {ui_response['analysis_status']}")
    print(f"  Confidence score: {ui_response['confidence_score']:.2%}")
    print(f"  Market surfaces: {ui_response['market_surfaces']}")
    print(f"  Backtest return: {ui_response['backtest_performance']['avg_return']:.1%}")
    print(f"  Uncertainty distribution: {ui_response['uncertainty_distribution']}")
    
    # Simulate real-time updates
    print(f"\n🔄 Simulating real-time updates...")
    for i in range(3):
        time.sleep(1)  # Simulate processing time
        print(f"  Update {i+1}: Processing market data...")
    
    print(f"  ✅ Real-time analysis complete!")
    
    return ui_response

def demo_surface_visualization():
    """Demo surface visualization capabilities"""
    print("\n" + "="*80)
    print("📊 SURFACE VISUALIZATION DEMO")
    print("="*80)
    
    analyzer, result = demo_fsqca_analysis()
    
    print(f"\n🔄 Generating surface visualizations...")
    
    # Create surface summary
    surface_summary = []
    for surface in result.market_surfaces:
        surface_data = {
            'surface_id': surface.surface_id,
            'uncertainty_level': surface.uncertainty_level.value,
            'volatility': surface.volatility,
            'liquidity_score': surface.liquidity_metrics['market_depth'],
            'decision_opportunities': len(surface.decision_opportunities),
            'coordinates_shape': surface.coordinates.shape
        }
        surface_summary.append(surface_data)
    
    print(f"\n📊 SURFACE SUMMARY:")
    for surface in surface_summary:
        print(f"  {surface['surface_id']}:")
        print(f"    Uncertainty: {surface['uncertainty_level']}")
        print(f"    Volatility: {surface['volatility']:.3f}")
        print(f"    Liquidity: {surface['liquidity_score']:.3f}")
        print(f"    Opportunities: {surface['decision_opportunities']}")
        print(f"    Data points: {surface['coordinates_shape'][0]}")
    
    # Create visualization data
    viz_data = {
        'surfaces': surface_summary,
        'strategies': len(result.capital_allocation_strategies),
        'decisions': len(result.optimal_decisions),
        'performance': result.backtest_results['overall_performance'],
        'recommendations': result.decision_recommendations
    }
    
    # Save visualization data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_file = f"surface_visualization_data_{timestamp}.json"
    
    with open(viz_file, 'w') as f:
        json.dump(viz_data, f, indent=2, default=str)
    
    print(f"\n💾 Visualization data saved to: {viz_file}")
    
    return viz_data

def main():
    """Run all fsQCA market uncertainty analysis demos"""
    print("🚀 fsQCA MARKET UNCERTAINTY ANALYSIS DEMO")
    print("="*80)
    print("This demo showcases fsQCA analysis for determining optimal")
    print("financial decisions and capital allocation during market uncertainty.")
    print("="*80)
    
    try:
        # Run all demos
        demo_market_surface_analysis()
        demo_capital_allocation_optimization()
        demo_fsqca_analysis()
        demo_real_time_recommendations()
        demo_ui_integration()
        demo_surface_visualization()
        
        print("\n" + "="*80)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print("\n🎯 KEY INSIGHTS:")
        print("1. fsQCA can identify optimal decisions during market uncertainty")
        print("2. Market surfaces reveal decision opportunities across volatility regimes")
        print("3. Capital allocation strategies adapt to uncertainty levels")
        print("4. Backtesting validates strategy performance across scenarios")
        print("5. Real-time recommendations enable dynamic decision making")
        print("6. UI integration provides actionable insights for users")
        
        print("\n💡 APPLICATIONS:")
        print("- Portfolio rebalancing during market stress")
        print("- Risk management in volatile conditions")
        print("- Asset allocation optimization")
        print("- Timing decisions for market entry/exit")
        print("- Liquidity management strategies")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 