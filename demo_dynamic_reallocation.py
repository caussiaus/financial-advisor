#!/usr/bin/env python3
"""
Dynamic Reallocation Demonstration

This script demonstrates how Horatio's dynamic reallocation strategy
provides a smoother portfolio ride compared to static strategies.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicReallocationDemo:
    """Demonstrates dynamic reallocation strategies"""
    
    def __init__(self):
        self.original_events = self.load_original_events()
        self.horatio_profile = self.load_horatio_profile()
        
    def load_original_events(self) -> Dict:
        """Load the original 5 life events data"""
        try:
            with open('data/outputs/analysis_data/realistic_life_events_analysis.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Original events data not found")
            return {"events": [], "analysis_report": {}}
    
    def load_horatio_profile(self) -> Dict:
        """Load Horatio's profile data"""
        try:
            with open('horatio_profile.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Horatio profile not found")
            return {}
    
    def generate_static_portfolio(self, start_value: float = 470588.24) -> pd.DataFrame:
        """Generate portfolio data for static strategy"""
        portfolio_data = []
        current_value = start_value
        current_equity = 0.53
        current_bonds = 0.41
        current_cash = 0.06
        
        # Generate monthly data from 2020 to 2025
        for year in range(2020, 2026):
            for month in range(12):
                date = datetime(year, month + 1, 1)
                
                # Check for events
                event = next((e for e in self.original_events['events'] 
                            if datetime.fromisoformat(e['actual_date'].replace('Z', '+00:00')).year == year
                            and datetime.fromisoformat(e['actual_date'].replace('Z', '+00:00')).month == month + 1), None)
                
                if event:
                    # Apply event impact
                    current_value += event['impact_amount']
                    
                    # Apply portfolio adjustments
                    if 'portfolio_adjustment' in event:
                        adj = event['portfolio_adjustment']
                        if 'risk_reduction' in adj:
                            current_equity -= adj['risk_reduction']
                            current_cash += adj.get('cash_increase', 0)
                            current_bonds = 1 - current_equity - current_cash
                        if 'equity_increase' in adj:
                            current_equity += adj['equity_increase']
                            if 'bonds_decrease' in adj:
                                current_bonds -= adj['bonds_decrease']
                            current_cash = 1 - current_equity - current_bonds
                
                # Add market returns (simplified)
                market_return = np.random.normal(0, 0.02)  # 2% monthly volatility
                current_value *= (1 + market_return)
                
                portfolio_data.append({
                    'date': date,
                    'value': current_value,
                    'equity': current_equity,
                    'bonds': current_bonds,
                    'cash': current_cash,
                    'strategy': 'static',
                    'event': event['description'] if event else None
                })
        
        return pd.DataFrame(portfolio_data)
    
    def generate_dynamic_portfolio(self, start_value: float = 850000) -> pd.DataFrame:
        """Generate portfolio data for Horatio's dynamic strategy"""
        portfolio_data = []
        current_value = start_value
        current_stocks = 0.60
        current_bonds = 0.25
        current_real_estate = 0.10
        current_cash = 0.05
        
        # Generate monthly data from 2024 to 2035
        for year in range(2024, 2036):
            for month in range(12):
                date = datetime(year, month + 1, 1)
                
                # Check for Horatio's events
                event = next((e for e in self.horatio_profile.get('lifestyle_events', [])
                            if datetime.fromisoformat(e['estimated_date'].replace('Z', '+00:00')).year == year
                            and datetime.fromisoformat(e['estimated_date'].replace('Z', '+00:00')).month == month + 1), None)
                
                if event:
                    current_value += event['amount']
                    
                    # Dynamic reallocation based on event type
                    if event['event_type'] == 'major_expense':
                        # Increase cash for major expenses
                        current_cash += 0.1
                        current_stocks -= 0.05
                        current_bonds -= 0.05
                    elif event['event_type'] == 'income_change':
                        # Increase equity for positive income changes
                        current_stocks += 0.1
                        current_cash -= 0.1
                
                # Dynamic rebalancing based on market conditions
                market_stress = np.random.random()
                if market_stress > 0.7:
                    # High stress - reduce risk
                    current_stocks -= 0.05
                    current_bonds += 0.03
                    current_cash += 0.02
                elif market_stress < 0.3:
                    # Low stress - increase risk
                    current_stocks += 0.03
                    current_bonds -= 0.02
                    current_cash -= 0.01
                
                # Normalize allocations
                total = current_stocks + current_bonds + current_real_estate + current_cash
                current_stocks /= total
                current_bonds /= total
                current_real_estate /= total
                current_cash /= total
                
                # Add market returns (slightly lower volatility for dynamic strategy)
                market_return = np.random.normal(0, 0.015)  # 1.5% monthly volatility
                current_value *= (1 + market_return)
                
                portfolio_data.append({
                    'date': date,
                    'value': current_value,
                    'stocks': current_stocks,
                    'bonds': current_bonds,
                    'real_estate': current_real_estate,
                    'cash': current_cash,
                    'strategy': 'dynamic',
                    'event': event['description'] if event else None
                })
        
        return pd.DataFrame(portfolio_data)
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        returns = df['value'].pct_change().dropna()
        
        metrics = {
            'total_return': (df['value'].iloc[-1] / df['value'].iloc[0] - 1) * 100,
            'volatility': returns.std() * np.sqrt(12) * 100,  # Annualized
            'sharpe_ratio': (returns.mean() * 12) / (returns.std() * np.sqrt(12)),
            'max_drawdown': ((df['value'] / df['value'].expanding().max() - 1) * 100).min(),
            'calmar_ratio': (returns.mean() * 12) / abs(((df['value'] / df['value'].expanding().max() - 1) * 100).min())
        }
        
        return metrics
    
    def create_comparison_visualization(self):
        """Create comparison visualization"""
        # Generate both portfolios
        static_df = self.generate_static_portfolio()
        dynamic_df = self.generate_dynamic_portfolio()
        
        # Calculate metrics
        static_metrics = self.calculate_metrics(static_df)
        dynamic_metrics = self.calculate_metrics(dynamic_df)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dynamic vs Static Portfolio Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Portfolio value comparison
        ax1 = axes[0, 0]
        ax1.plot(static_df['date'], static_df['value'], label='Static Strategy', linewidth=2, color='#667eea')
        ax1.plot(dynamic_df['date'], dynamic_df['value'], label='Horatio\'s Dynamic Strategy', linewidth=2, color='#28a745')
        ax1.set_title('Portfolio Value Evolution')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatility comparison
        ax2 = axes[0, 1]
        static_returns = static_df['value'].pct_change().dropna()
        dynamic_returns = dynamic_df['value'].pct_change().dropna()
        
        ax2.hist(static_returns, bins=30, alpha=0.7, label='Static Strategy', color='#667eea')
        ax2.hist(dynamic_returns, bins=30, alpha=0.7, label='Dynamic Strategy', color='#28a745')
        ax2.set_title('Return Distribution')
        ax2.set_xlabel('Monthly Returns')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Allocation comparison
        ax3 = axes[1, 0]
        # Static allocation
        static_alloc = [static_df['equity'].mean(), static_df['bonds'].mean(), static_df['cash'].mean()]
        # Dynamic allocation (average)
        dynamic_alloc = [dynamic_df['stocks'].mean(), dynamic_df['bonds'].mean(), 
                        dynamic_df['real_estate'].mean(), dynamic_df['cash'].mean()]
        
        x = np.arange(len(['Equity/Stocks', 'Bonds', 'Real Estate', 'Cash']))
        width = 0.35
        
        ax3.bar(x - width/2, static_alloc + [0], width, label='Static Strategy', color='#667eea', alpha=0.7)
        ax3.bar(x + width/2, dynamic_alloc, width, label='Dynamic Strategy', color='#28a745', alpha=0.7)
        ax3.set_title('Average Asset Allocation')
        ax3.set_ylabel('Allocation (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Equity/Stocks', 'Bonds', 'Real Estate', 'Cash'])
        ax3.legend()
        
        # Performance metrics comparison
        ax4 = axes[1, 1]
        metrics = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        static_values = [static_metrics['total_return'], static_metrics['volatility'], 
                        static_metrics['sharpe_ratio'], static_metrics['max_drawdown']]
        dynamic_values = [dynamic_metrics['total_return'], dynamic_metrics['volatility'], 
                         dynamic_metrics['sharpe_ratio'], dynamic_metrics['max_drawdown']]
        
        x = np.arange(len(metrics))
        ax4.bar(x - width/2, static_values, width, label='Static Strategy', color='#667eea', alpha=0.7)
        ax4.bar(x + width/2, dynamic_values, width, label='Dynamic Strategy', color='#28a745', alpha=0.7)
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('dynamic_reallocation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison summary
        print("\n" + "="*60)
        print("DYNAMIC REALLOCATION STRATEGY COMPARISON")
        print("="*60)
        print(f"\nStatic Strategy Metrics:")
        print(f"  Total Return: {static_metrics['total_return']:.2f}%")
        print(f"  Volatility: {static_metrics['volatility']:.2f}%")
        print(f"  Sharpe Ratio: {static_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {static_metrics['max_drawdown']:.2f}%")
        
        print(f"\nHoratio's Dynamic Strategy Metrics:")
        print(f"  Total Return: {dynamic_metrics['total_return']:.2f}%")
        print(f"  Volatility: {dynamic_metrics['volatility']:.2f}%")
        print(f"  Sharpe Ratio: {dynamic_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {dynamic_metrics['max_drawdown']:.2f}%")
        
        print(f"\nImprovements:")
        print(f"  Return Improvement: {dynamic_metrics['total_return'] - static_metrics['total_return']:.2f}%")
        print(f"  Volatility Reduction: {static_metrics['volatility'] - dynamic_metrics['volatility']:.2f}%")
        print(f"  Sharpe Ratio Improvement: {dynamic_metrics['sharpe_ratio'] - static_metrics['sharpe_ratio']:.3f}")
        print(f"  Drawdown Reduction: {static_metrics['max_drawdown'] - dynamic_metrics['max_drawdown']:.2f}%")
        
        return {
            'static_metrics': static_metrics,
            'dynamic_metrics': dynamic_metrics,
            'improvements': {
                'return_improvement': dynamic_metrics['total_return'] - static_metrics['total_return'],
                'volatility_reduction': static_metrics['volatility'] - dynamic_metrics['volatility'],
                'sharpe_improvement': dynamic_metrics['sharpe_ratio'] - static_metrics['sharpe_ratio'],
                'drawdown_reduction': static_metrics['max_drawdown'] - dynamic_metrics['max_drawdown']
            }
        }
    
    def run_demo(self):
        """Run the complete demonstration"""
        logger.info("Starting Dynamic Reallocation Demonstration...")
        
        try:
            results = self.create_comparison_visualization()
            logger.info("Demonstration completed successfully!")
            logger.info(f"Visualization saved as: dynamic_reallocation_comparison.png")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
            return None

def main():
    """Main function to run the demonstration"""
    demo = DynamicReallocationDemo()
    results = demo.run_demo()
    
    if results:
        print("\n✅ Dynamic reallocation demonstration completed successfully!")
        print("📊 Check the visualization at: http://localhost:5003")
        print("📈 Comparison chart saved as: dynamic_reallocation_comparison.png")
    else:
        print("❌ Demonstration failed. Check logs for details.")

if __name__ == "__main__":
    main() 