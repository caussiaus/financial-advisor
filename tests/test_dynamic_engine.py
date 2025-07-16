#!/usr/bin/env python
# Simple test for Dynamic Portfolio Engine
# Author: ChatGPT 2025-01-16

import sys
import os
sys.path.append('src')

from dynamic_portfolio_engine import DynamicPortfolioEngine

def test_dynamic_engine():
    """Test the dynamic portfolio engine"""
    print("🧪 Testing Dynamic Portfolio Engine")
    
    # Create client configuration
    client_config = {
        'income': 250000,
        'disposable_cash': 8000,
        'allowable_var': 0.15,
        'age': 42,
        'risk_profile': 3,
        'portfolio_value': 1500000,
        'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
    }
    
    # Initialize engine
    engine = DynamicPortfolioEngine(client_config)
    
    # Get initial snapshot
    snapshot = engine.get_portfolio_snapshot()
    print(f"✅ Initial portfolio: Equity {snapshot['portfolio']['equity']:.1%}, "
          f"Bonds {snapshot['portfolio']['bonds']:.1%}, "
          f"Cash {snapshot['portfolio']['cash']:.1%}")
    
    # Add a life event
    engine.add_life_event('career_change', 'Promotion - increased stability', 0.3, '2024-03-15')
    
    # Get updated snapshot
    snapshot = engine.get_portfolio_snapshot()
    print(f"✅ After life event: Equity {snapshot['portfolio']['equity']:.1%}, "
          f"Bonds {snapshot['portfolio']['bonds']:.1%}, "
          f"Cash {snapshot['portfolio']['cash']:.1%}")
    
    # Test market update
    market_update = {'market_stress': 0.8, 'equity_volatility': 0.25}
    recommendations = engine.update_market_data(market_update)
    
    print(f"✅ Market update recommendations: {recommendations['rebalancing_needed']}")
    
    # Export data
    data = engine.export_data()
    print(f"✅ Data exported: {len(data['portfolio_snapshots'])} snapshots, "
          f"{len(data['life_events_log'])} life events")
    
    print("🎉 Dynamic Portfolio Engine test completed successfully!")

if __name__ == "__main__":
    test_dynamic_engine() 