"""
Test Market Data Downloader

Test the market data downloader with a small subset of symbols
to verify functionality before running the full download.
"""

import requests
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMarketDataDownloader:
    """Test market data downloader with small subset"""
    
    def __init__(self):
        self.data_dir = Path("data/market_data_test")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test with a small subset of symbols
        self.test_symbols = {
            "stocks": {
                "large_cap": ["AAPL", "MSFT", "GOOGL"],
                "mid_cap": ["AMD", "NFLX", "CRM"]
            },
            "etfs": {
                "equity": ["SPY", "QQQ", "VTI"],
                "bonds": ["TLT", "IEF", "BND"]
            },
            "indices": {
                "major": ["^GSPC", "^DJI", "^IXIC"]
            },
            "commodities": {
                "precious_metals": ["GC=F", "SLV"],
                "energy": ["CL=F", "NG=F"]
            },
            "crypto": {
                "major": ["BTC-USD", "ETH-USD"]
            }
        }
    
    def get_yahoo_finance_data(self, symbol: str) -> Optional[Dict]:
        """Get data from Yahoo Finance API (free, no API key required)"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "period1": int((datetime.now() - timedelta(days=30)).timestamp()),  # 30 days for testing
                "period2": int(datetime.now().timestamp()),
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,split"
            }
            
            logger.info(f"Fetching data for {symbol}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Extract price data
                timestamps = result.get("timestamp", [])
                quotes = result.get("indicators", {}).get("quote", [{}])[0]
                
                # Create DataFrame
                df = pd.DataFrame({
                    "date": pd.to_datetime(timestamps, unit="s"),
                    "open": quotes.get("open", []),
                    "high": quotes.get("high", []),
                    "low": quotes.get("low", []),
                    "close": quotes.get("close", []),
                    "volume": quotes.get("volume", [])
                })
                
                # Remove rows with NaN values
                df = df.dropna()
                
                if not df.empty:
                    return {
                        "symbol": symbol,
                        "data": df.to_dict("records"),
                        "summary": {
                            "start_date": df["date"].min().isoformat(),
                            "end_date": df["date"].max().isoformat(),
                            "total_days": len(df),
                            "current_price": df["close"].iloc[-1],
                            "price_change": df["close"].iloc[-1] - df["close"].iloc[0],
                            "price_change_pct": ((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100,
                            "avg_volume": df["volume"].mean(),
                            "volatility": df["close"].pct_change().std() * np.sqrt(252) * 100
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Yahoo Finance data for {symbol}: {e}")
            return None
    
    def download_test_data(self) -> Dict:
        """Download test market data"""
        
        logger.info("Starting test market data download...")
        
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
                
                for symbol in symbols:
                    data = self.get_yahoo_finance_data(symbol)
                    
                    if data:
                        results["data"][symbol] = data
                        results["summary"]["successful_downloads"] += 1
                        logger.info(f"✅ Successfully downloaded {symbol}")
                    else:
                        results["summary"]["failed_downloads"] += 1
                        logger.warning(f"❌ Failed to download {symbol}")
                    
                    # Rate limiting
                    time.sleep(random.uniform(0.5, 1.0))
                
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
        summary_file = self.data_dir / "test_market_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Test market data download complete! Summary saved to {summary_file}")
        
        return all_results
    
    def print_test_summary(self, results: Dict):
        """Print test market data summary"""
        
        print(f"\n{'='*80}")
        print(f" TEST MARKET DATA DOWNLOAD SUMMARY")
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
                        break


def main():
    """Run test market data download"""
    downloader = TestMarketDataDownloader()
    
    # Download test market data
    results = downloader.download_test_data()
    
    # Print summary
    downloader.print_test_summary(results)
    
    print(f"\n🎉 Test market data download completed successfully!")
    print(f"📊 Test data saved to: {downloader.data_dir}")
    print(f"📈 Ready to run full market data download!")


if __name__ == "__main__":
    main() 