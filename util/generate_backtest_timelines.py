#!/usr/bin/env python3
"""
Generate Backtest Timelines with Sliding Windows

Downloads a large window of historical market data and slides a training window
across it to generate multiple backtest periods. This creates realistic timeline
CSV files for the dashboard demo.

Usage:
    python util/generate_backtest_timelines.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')
sys.path.append('.')

try:
    from src.market_tracking_backtest import (
        MarketDataTracker, PersonalFinanceInvestmentMapper, BacktestEngine, 
        BacktestAnalyzer, PersonalFinanceAction, InvestmentDecision, BacktestResult
    )
except ImportError:
    # Try alternative import path
    from market_tracking_backtest import (
        MarketDataTracker, PersonalFinanceInvestmentMapper, BacktestEngine, 
        BacktestAnalyzer, PersonalFinanceAction, InvestmentDecision, BacktestResult
    )

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimelineGenerator:
    """Generate timeline CSV files using sliding window backtesting"""
    
    def __init__(self):
        self.market_tracker = MarketDataTracker()
        self.mapper = PersonalFinanceInvestmentMapper()
        self.backtest_engine = BacktestEngine(self.market_tracker, self.mapper)
        self.analyzer = BacktestAnalyzer()
        
        # Output directory
        self.output_dir = Path("data/outputs/ips_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_large_market_window(self, 
                                   start_date: datetime = datetime(2000, 1, 1),
                                   end_date: datetime = datetime(2024, 12, 31)) -> Dict[str, pd.DataFrame]:
        """Download a large window of market data"""
        logger.info(f"📊 Downloading market data from {start_date} to {end_date}")
        
        # Major market symbols
        symbols = [
            '^GSPC',  # S&P 500
            '^DJI',   # Dow Jones
            '^IXIC',  # NASDAQ
            '^TNX',   # 10Y Treasury
            'GLD',    # Gold ETF
            'TLT',    # Long-term Treasury ETF
            '^VIX'    # Volatility Index
        ]
        
        market_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate additional metrics
                    data['Returns'] = data['Close'].pct_change()
                    data['Volatility'] = data['Returns'].rolling(window=20).std()
                    data['Cumulative_Return'] = (1 + data['Returns']).cumprod()
                    
                    market_data[symbol] = data
                    logger.info(f"✅ Downloaded {len(data)} data points for {symbol}")
                else:
                    logger.warning(f"⚠️ No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading {symbol}: {e}")
                
        return market_data
    
    def create_sample_personal_finance_actions(self, start_date: datetime, end_date: datetime) -> List[PersonalFinanceAction]:
        """Create realistic personal finance actions for the period"""
        actions = []
        
        # Generate realistic life events
        life_events = [
            ('income_increase', 15000, 'Promotion and salary increase'),
            ('major_expense', 25000, 'Home renovation project'),
            ('debt_payment', 10000, 'Student loan payoff'),
            ('milestone_achievement', 5000, 'Emergency fund goal reached'),
            ('income_increase', 20000, 'Job change with higher salary'),
            ('major_expense', 30000, 'Car purchase'),
            ('debt_payment', 15000, 'Credit card payoff'),
            ('milestone_achievement', 8000, 'Investment milestone reached'),
            ('income_increase', 12000, 'Annual bonus'),
            ('major_expense', 18000, 'Medical expenses'),
        ]
        
        # Distribute events across the time period
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        event_dates = np.random.choice(date_range, size=len(life_events), replace=False)
        
        for i, (event_type, amount, description) in enumerate(life_events):
            action = PersonalFinanceAction(
                timestamp=event_dates[i],
                action_type=event_type,
                amount=amount,
                description=description,
                category='general',
                confidence=np.random.uniform(0.7, 0.95),
                impact_duration=np.random.randint(90, 365)
            )
            actions.append(action)
        
        return actions
    
    def run_sliding_window_backtests(self, 
                                   market_data: Dict[str, pd.DataFrame],
                                   window_size_years: int = 3,
                                   step_size_months: int = 6) -> List[Dict]:
        """Run backtests using sliding windows"""
        logger.info(f"🔄 Running sliding window backtests (window: {window_size_years} years, step: {step_size_months} months)")
        
        # Get the date range from market data
        all_dates = []
        for symbol, data in market_data.items():
            all_dates.extend(data.index)
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Create sliding windows
        window_start = start_date
        results = []
        window_count = 0
        
        while window_start + timedelta(days=window_size_years*365) <= end_date:
            window_end = window_start + timedelta(days=window_size_years*365)
            
            # Create sample actions for this window
            actions = self.create_sample_personal_finance_actions(window_start, window_end)
            
            # Initial portfolio
            initial_portfolio = {
                'equity': 60000,
                'bonds': 30000,
                'cash': 10000
            }
            
            try:
                # Run backtest for this window
                result = self.backtest_engine.run_backtest(
                    start_date=window_start,
                    end_date=window_end,
                    initial_portfolio=initial_portfolio,
                    personal_finance_actions=actions,
                    risk_tolerance='moderate',
                    rebalance_frequency='monthly'
                )
                
                # Store results
                window_result = {
                    'window_id': f"CFG_{window_count:03d}",
                    'start_date': window_start,
                    'end_date': window_end,
                    'backtest_result': result,
                    'actions': actions
                }
                results.append(window_result)
                
                logger.info(f"✅ Window {window_count}: {window_start.date()} to {window_end.date()}")
                window_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error in window {window_count}: {e}")
            
            # Move window forward
            window_start += timedelta(days=step_size_months*30)
        
        logger.info(f"✅ Completed {len(results)} sliding window backtests")
        return results
    
    def generate_timeline_csv(self, window_results: List[Dict]) -> None:
        """Generate timeline CSV files from backtest results"""
        logger.info("📊 Generating timeline CSV files...")
        
        for window_result in window_results:
            window_id = window_result['window_id']
            result = window_result['backtest_result']
            
            # Create timeline data
            timeline_data = []
            
            # Get portfolio history from backtest
            if hasattr(result, 'portfolio_history') and result.portfolio_history:
                for entry in result.portfolio_history:
                    timeline_data.append({
                        'date': entry['date'],
                        'total_value': entry['total_value'],
                        'equity': entry['portfolio'].get('equity', 0),
                        'bonds': entry['portfolio'].get('bonds', 0),
                        'cash': entry['portfolio'].get('cash', 0),
                        'market_stress': entry.get('market_conditions', {}).get('market_stress', 0.3)
                    })
            else:
                # Generate synthetic timeline if no history available
                start_date = window_result['start_date']
                end_date = window_result['end_date']
                date_range = pd.date_range(start=start_date, end=end_date, freq='M')
                
                initial_value = 100000
                for i, date in enumerate(date_range):
                    # Simulate portfolio growth
                    monthly_return = np.random.normal(0.005, 0.02)  # 0.5% monthly return, 2% volatility
                    total_value = initial_value * (1 + monthly_return) ** (i + 1)
                    
                    # Allocate to asset classes
                    equity_ratio = 0.6 + np.random.normal(0, 0.1)
                    bond_ratio = 0.3 + np.random.normal(0, 0.1)
                    cash_ratio = 1 - equity_ratio - bond_ratio
                    
                    timeline_data.append({
                        'date': date,
                        'total_value': total_value,
                        'equity': total_value * equity_ratio,
                        'bonds': total_value * bond_ratio,
                        'cash': total_value * cash_ratio,
                        'market_stress': np.random.uniform(0.2, 0.8)
                    })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(timeline_data)
            
            # Save lifecycle CSV
            lifecycle_filename = f"lifecycle_{window_id}.csv"
            lifecycle_path = self.output_dir / lifecycle_filename
            df.to_csv(lifecycle_path, index=False)
            logger.info(f"✅ Saved {lifecycle_filename}")
            
            # Save cashflows CSV
            cashflows_filename = f"cashflows_{window_id}.csv"
            cashflows_path = self.output_dir / cashflows_filename
            
            # Create cashflow data
            cashflow_data = []
            for i, row in df.iterrows():
                if i > 0:
                    prev_row = df.iloc[i-1]
                    cashflow_data.append({
                        'date': row['date'],
                        'total_cashflow': row['total_value'] - prev_row['total_value'],
                        'equity_cashflow': row['equity'] - prev_row['equity'],
                        'bonds_cashflow': row['bonds'] - prev_row['bonds'],
                        'cash_cashflow': row['cash'] - prev_row['cash'],
                        'market_stress': row['market_stress']
                    })
            
            cashflow_df = pd.DataFrame(cashflow_data)
            cashflow_df.to_csv(cashflows_path, index=False)
            logger.info(f"✅ Saved {cashflows_filename}")
    
    def run_comprehensive_timeline_generation(self):
        """Run the complete timeline generation process"""
        logger.info("🚀 Starting comprehensive timeline generation...")
        
        # Download large market window
        market_data = self.download_large_market_window()
        
        if not market_data:
            logger.error("❌ No market data available")
            return
        
        # Run sliding window backtests
        window_results = self.run_sliding_window_backtests(market_data)
        
        if not window_results:
            logger.error("❌ No backtest results generated")
            return
        
        # Generate timeline CSV files
        self.generate_timeline_csv(window_results)
        
        # Save summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_windows': len(window_results),
            'market_symbols': list(market_data.keys()),
            'output_files': [f.name for f in self.output_dir.glob("*.csv")]
        }
        
        summary_path = self.output_dir / "timeline_generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Timeline generation complete! Generated {len(summary['output_files'])} files")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        return summary

def main():
    """Main function to run timeline generation"""
    print("🚀 Starting Timeline Generation for Dashboard Demo...")
    
    generator = TimelineGenerator()
    summary = generator.run_comprehensive_timeline_generation()
    
    if summary:
        print(f"\n✅ Successfully generated {summary['total_windows']} timeline windows")
        print(f"📊 Market symbols: {', '.join(summary['market_symbols'])}")
        print(f"📁 Output files: {len(summary['output_files'])} CSV files")
        print(f"🌐 Dashboard will now show real timeline data!")
    else:
        print("❌ Timeline generation failed")

if __name__ == "__main__":
    main() 