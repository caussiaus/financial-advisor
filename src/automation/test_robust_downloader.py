"""
Test Robust Market Data Downloader

Test the robust market data downloader with a small subset of symbols
to verify functionality before running the full download.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRobustMarketDownloader:
    """Test robust market data downloader with small subset"""
    
    def __init__(self):
        self.data_dir = Path("data/market_data_test_robust")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test with a small subset of symbols
        self.test_symbols = {
            "stocks": {
                "large_cap": ["AAPL", "MSFT", "GOOGL"]
            },
            "etfs": {
                "equity": ["SPY", "QQQ", "VTI"]
            },
            "indices": {
                "major": ["^GSPC", "^DJI", "^IXIC"]
            },
            "commodities": {
                "precious_metals": ["GC=F", "SLV"]
            },
            "crypto": {
                "major": ["BTC-USD", "ETH-USD"]
            }
        }
    
    def get_yfinance_data(self, symbol: str, period: str = "6mo") -> Optional[Dict]:
        """Get data using yfinance library"""
        try:
            logger.info(f"Fetching {symbol} using yfinance...")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if not df.empty:
                # Calculate additional metrics
                df['returns'] = df['Close'].pct_change()
                volatility = df['returns'].std() * np.sqrt(252) * 100
                
                # Convert DataFrame to JSON-serializable format
                df_json = df.reset_index()
                df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d')
                
                return {
                    "symbol": symbol,
                    "data": df_json.to_dict("records"),
                    "summary": {
                        "start_date": df.index.min().isoformat(),
                        "end_date": df.index.max().isoformat(),
                        "total_days": len(df),
                        "current_price": float(df['Close'].iloc[-1]),
                        "price_change": float(df['Close'].iloc[-1] - df['Close'].iloc[0]),
                        "price_change_pct": float(((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100),
                        "avg_volume": float(df['Volume'].mean()),
                        "volatility": float(volatility),
                        "high": float(df['High'].max()),
                        "low": float(df['Low'].min()),
                        "avg_price": float(df['Close'].mean())
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get yfinance data for {symbol}: {e}")
            return None
    
    def download_test_data(self) -> Dict:
        """Download test market data"""
        
        logger.info("Starting test robust market data download...")
        
        all_results = {
            "download_date": datetime.now().isoformat(),
            "total_investment_classes": 0,
            "total_symbols": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "investment_classes": {}
        }
        
        for investment_class, categories in self.test_symbols.items():
            all_results["investment_classes"][investment_class] = {}
            
            for category, symbols in categories.items():
                logger.info(f"Downloading {investment_class} - {category} data...")
                
                results = {
                    "investment_class": investment_class,
                    "category": category,
                    "symbols": symbols,
                    "data": {},
                    "summary": {
                        "total_symbols": len(symbols),
                        "successful_downloads": 0,
                        "failed_downloads": 0,
                        "download_date": datetime.now().isoformat()
                    }
                }
                
                for i, symbol in enumerate(symbols):
                    logger.info(f"Downloading {symbol} ({i+1}/{len(symbols)})...")
                    
                    data = self.get_yfinance_data(symbol)
                    
                    if data:
                        results["data"][symbol] = data
                        results["summary"]["successful_downloads"] += 1
                        logger.info(f"✅ Successfully downloaded {symbol}")
                    else:
                        results["summary"]["failed_downloads"] += 1
                        logger.warning(f"❌ Failed to download {symbol}")
                    
                    # Rate limiting
                    time.sleep(random.uniform(1.0, 2.0))
                
                all_results["investment_classes"][investment_class][category] = results
                all_results["total_symbols"] += len(symbols)
                all_results["successful_downloads"] += results["summary"]["successful_downloads"]
                all_results["failed_downloads"] += results["summary"]["failed_downloads"]
                
                # Save category data
                category_file = self.data_dir / f"{investment_class}_{category}.json"
                with open(category_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Saved {investment_class} - {category} data to {category_file}")
            
            all_results["total_investment_classes"] += 1
        
        # Save comprehensive results
        summary_file = self.data_dir / "test_robust_market_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Test robust market data download complete! Summary saved to {summary_file}")
        
        return all_results
    
    def print_test_summary(self, results: Dict):
        """Print test market data summary"""
        
        print(f"\n{'='*80}")
        print(f" TEST ROBUST MARKET DATA DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        print(f"Download Date: {results['download_date']}")
        print(f"Total Investment Classes: {results['total_investment_classes']}")
        print(f"Total Symbols: {results['total_symbols']}")
        print(f"Successful Downloads: {results['successful_downloads']}")
        print(f"Failed Downloads: {results['failed_downloads']}")
        print(f"Success Rate: {(results['successful_downloads'] / results['total_symbols'] * 100):.1f}%")
        
        print(f"\n{'='*80}")
        print(f" TEST INVESTMENT CLASS BREAKDOWN")
        print(f"{'='*80}")
        
        for investment_class, categories in results["investment_classes"].items():
            print(f"\n📊 {investment_class.upper()}:")
            
            for category, data in categories.items():
                success_rate = (data["summary"]["successful_downloads"] / data["summary"]["total_symbols"]) * 100
                print(f"   {category}: {data['summary']['successful_downloads']}/{data['summary']['total_symbols']} ({success_rate:.1f}%)")
        
        print(f"\n{'='*80}")
        print(f" TEST DATA SAMPLES")
        print(f"{'='*80}")
        
        # Show some sample data
        for investment_class, categories in results["investment_classes"].items():
            for category, data in categories.items():
                if data["data"]:
                    first_symbol = list(data["data"].keys())[0]
                    symbol_data = data["data"][first_symbol]
                    
                    if "summary" in symbol_data:
                        summary = symbol_data["summary"]
                        print(f"📈 {first_symbol}:")
                        print(f"   Current Price: ${summary.get('current_price', 0):.2f}")
                        print(f"   Price Change: {summary.get('price_change_pct', 0):.2f}%")
                        print(f"   Volatility: {summary.get('volatility', 0):.2f}%")
                        print(f"   Days of Data: {summary.get('total_days', 0)}")
                        print(f"   High: ${summary.get('high', 0):.2f}")
                        print(f"   Low: ${summary.get('low', 0):.2f}")
                        break


def main():
    """Run test robust market data download"""
    downloader = TestRobustMarketDownloader()
    
    # Download test market data
    results = downloader.download_test_data()
    
    # Print summary
    downloader.print_test_summary(results)
    
    print(f"\n🎉 Test robust market data download completed successfully!")
    print(f"📊 Test data saved to: {downloader.data_dir}")
    print(f"📈 Ready to run full robust market data download!")


if __name__ == "__main__":
    main() 