"""
Robust Market Data Downloader

Download real market data using multiple free APIs with proper rate limiting
and fallback mechanisms to ensure reliable data collection.
"""

import requests
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import random
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustMarketDataDownloader:
    """Robust market data downloader with multiple API fallbacks"""
    
    def __init__(self):
        self.data_dir = Path("data/market_data_robust")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Investment classes with realistic symbols
        self.investment_classes = {
            "stocks": {
                "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JNJ", "V", "PG"],
                "mid_cap": ["AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "QCOM", "TXN", "AVGO", "ABT"],
                "small_cap": ["SNAP", "UBER", "LYFT", "SPOT", "ZM", "SQ", "SHOP", "ROKU", "PINS", "SNAP"]
            },
            "etfs": {
                "equity": ["SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "EFA", "EEM", "VTV", "VUG"],
                "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "BND", "AGG", "VCIT", "VCSH", "BSV"],
                "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "XLE", "XLB", "XLI", "XLF"]
            },
            "indices": {
                "major": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^TNX", "^TYX", "^DXY"],
                "international": ["^FTSE", "^N225", "^GDAXI", "^FCHI", "^HSI", "^BSESN", "^AXJO"]
            },
            "commodities": {
                "precious_metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
                "energy": ["CL=F", "NG=F", "HO=F", "RB=F"],
                "agriculture": ["ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F"]
            },
            "currencies": {
                "major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X"],
                "emerging": ["USDCNY=X", "USDBRL=X", "USDINR=X", "USDMXN=X", "USDRUB=X", "USDZAR=X"]
            },
            "crypto": {
                "major": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD"],
                "altcoins": ["XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "LINK-USD", "UNI-USD"]
            }
        }
    
    def get_yfinance_data(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Get data using yfinance library (most reliable free option)"""
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
    
    def get_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Get data from Alpha Vantage API with demo key"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": "demo"
            }
            
            logger.info(f"Fetching {symbol} using Alpha Vantage...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                
                # Convert to DataFrame
                records = []
                for date, values in time_series.items():
                    records.append({
                        "date": date,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    })
                
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                
                if not df.empty:
                    df['returns'] = df['close'].pct_change()
                    volatility = df['returns'].std() * np.sqrt(252) * 100
                    
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
                            "volatility": volatility,
                            "high": df["high"].max(),
                            "low": df["low"].min(),
                            "avg_price": df["close"].mean()
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Alpha Vantage data for {symbol}: {e}")
            return None
    
    def get_finnhub_data(self, symbol: str) -> Optional[Dict]:
        """Get data from Finnhub API with demo token"""
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {
                "symbol": symbol,
                "token": "demo"
            }
            
            logger.info(f"Fetching {symbol} using Finnhub...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "c" in data and data["c"] > 0:
                return {
                    "symbol": symbol,
                    "data": [{
                        "date": datetime.now().isoformat(),
                        "open": data.get("o", 0),
                        "high": data.get("h", 0),
                        "low": data.get("l", 0),
                        "close": data.get("c", 0),
                        "volume": data.get("v", 0)
                    }],
                    "summary": {
                        "current_price": data.get("c", 0),
                        "price_change": data.get("d", 0),
                        "price_change_pct": data.get("dp", 0),
                        "high": data.get("h", 0),
                        "low": data.get("l", 0),
                        "volume": data.get("v", 0),
                        "open": data.get("o", 0)
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get Finnhub data for {symbol}: {e}")
            return None
    
    def download_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Download data for a single symbol using multiple APIs with fallback"""
        
        # Try yfinance first (most reliable)
        data = self.get_yfinance_data(symbol)
        
        if not data:
            # Try Alpha Vantage as backup
            data = self.get_alpha_vantage_data(symbol)
        
        if not data:
            # Try Finnhub as last resort
            data = self.get_finnhub_data(symbol)
        
        return data
    
    def download_investment_class_data(self, investment_class: str, category: str, symbols: List[str]) -> Dict:
        """Download data for a specific investment class and category"""
        
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
            
            data = self.download_symbol_data(symbol)
            
            if data:
                results["data"][symbol] = data
                results["summary"]["successful_downloads"] += 1
                logger.info(f"✅ Successfully downloaded {symbol}")
            else:
                results["summary"]["failed_downloads"] += 1
                logger.warning(f"❌ Failed to download {symbol}")
            
            # Aggressive rate limiting to avoid API issues
            time.sleep(random.uniform(2.0, 4.0))
        
        return results
    
    def download_all_market_data(self) -> Dict:
        """Download data for all investment classes"""
        
        logger.info("Starting robust market data download...")
        
        all_results = {
            "download_date": datetime.now().isoformat(),
            "total_investment_classes": 0,
            "total_symbols": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "investment_classes": {}
        }
        
        for investment_class, categories in self.investment_classes.items():
            all_results["investment_classes"][investment_class] = {}
            
            for category, symbols in categories.items():
                results = self.download_investment_class_data(investment_class, category, symbols)
                
                all_results["investment_classes"][investment_class][category] = results
                all_results["total_symbols"] += len(symbols)
                all_results["successful_downloads"] += results["summary"]["successful_downloads"]
                all_results["failed_downloads"] += results["summary"]["failed_downloads"]
                
                # Save individual category data
                category_file = self.data_dir / f"{investment_class}_{category}.json"
                with open(category_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Saved {investment_class} - {category} data to {category_file}")
            
            all_results["total_investment_classes"] += 1
        
        # Save comprehensive results
        summary_file = self.data_dir / "robust_market_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Robust market data download complete! Summary saved to {summary_file}")
        
        return all_results
    
    def get_market_statistics(self) -> Dict:
        """Calculate comprehensive market statistics"""
        
        logger.info("Calculating market statistics...")
        
        statistics = {
            "calculation_date": datetime.now().isoformat(),
            "investment_classes": {},
            "overall_statistics": {}
        }
        
        # Load all downloaded data
        for investment_class, categories in self.investment_classes.items():
            statistics["investment_classes"][investment_class] = {}
            
            for category, symbols in categories.items():
                category_file = self.data_dir / f"{investment_class}_{category}.json"
                
                if category_file.exists():
                    with open(category_file, 'r') as f:
                        data = json.load(f)
                    
                    # Calculate statistics for this category
                    category_stats = {
                        "total_symbols": len(symbols),
                        "successful_downloads": data["summary"]["successful_downloads"],
                        "success_rate": (data["summary"]["successful_downloads"] / len(symbols)) * 100,
                        "price_statistics": {},
                        "volatility_statistics": {},
                        "volume_statistics": {}
                    }
                    
                    # Calculate price and volatility statistics
                    prices = []
                    volatilities = []
                    volumes = []
                    
                    for symbol_data in data["data"].values():
                        if "summary" in symbol_data:
                            summary = symbol_data["summary"]
                            prices.append(summary.get("current_price", 0))
                            volatilities.append(summary.get("volatility", 0))
                            volumes.append(summary.get("avg_volume", 0))
                    
                    if prices:
                        category_stats["price_statistics"] = {
                            "mean": np.mean(prices),
                            "median": np.median(prices),
                            "min": np.min(prices),
                            "max": np.max(prices),
                            "std": np.std(prices)
                        }
                    
                    if volatilities:
                        category_stats["volatility_statistics"] = {
                            "mean": np.mean(volatilities),
                            "median": np.median(volatilities),
                            "min": np.min(volatilities),
                            "max": np.max(volatilities),
                            "std": np.std(volatilities)
                        }
                    
                    if volumes:
                        category_stats["volume_statistics"] = {
                            "mean": np.mean(volumes),
                            "median": np.median(volumes),
                            "min": np.min(volumes),
                            "max": np.max(volumes),
                            "std": np.std(volumes)
                        }
                    
                    statistics["investment_classes"][investment_class][category] = category_stats
        
        # Calculate overall statistics
        all_prices = []
        all_volatilities = []
        all_volumes = []
        
        for investment_class in statistics["investment_classes"].values():
            for category in investment_class.values():
                if "price_statistics" in category:
                    all_prices.extend([
                        category["price_statistics"]["mean"],
                        category["price_statistics"]["median"]
                    ])
                
                if "volatility_statistics" in category:
                    all_volatilities.extend([
                        category["volatility_statistics"]["mean"],
                        category["volatility_statistics"]["median"]
                    ])
                
                if "volume_statistics" in category:
                    all_volumes.extend([
                        category["volume_statistics"]["mean"],
                        category["volume_statistics"]["median"]
                    ])
        
        if all_prices:
            statistics["overall_statistics"]["price"] = {
                "mean": np.mean(all_prices),
                "median": np.median(all_prices),
                "std": np.std(all_prices)
            }
        
        if all_volatilities:
            statistics["overall_statistics"]["volatility"] = {
                "mean": np.mean(all_volatilities),
                "median": np.median(all_volatilities),
                "std": np.std(all_volatilities)
            }
        
        if all_volumes:
            statistics["overall_statistics"]["volume"] = {
                "mean": np.mean(all_volumes),
                "median": np.median(all_volumes),
                "std": np.std(all_volumes)
            }
        
        # Save statistics
        stats_file = self.data_dir / "robust_market_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Market statistics saved to {stats_file}")
        
        return statistics
    
    def print_market_summary(self, results: Dict):
        """Print comprehensive market data summary"""
        
        print(f"\n{'='*80}")
        print(f" ROBUST MARKET DATA DOWNLOAD SUMMARY")
        print(f"{'='*80}")
        print(f"Download Date: {results['download_date']}")
        print(f"Total Investment Classes: {results['total_investment_classes']}")
        print(f"Total Symbols: {results['total_symbols']}")
        print(f"Successful Downloads: {results['successful_downloads']}")
        print(f"Failed Downloads: {results['failed_downloads']}")
        print(f"Success Rate: {(results['successful_downloads'] / results['total_symbols'] * 100):.1f}%")
        
        print(f"\n{'='*80}")
        print(f" INVESTMENT CLASS BREAKDOWN")
        print(f"{'='*80}")
        
        for investment_class, categories in results["investment_classes"].items():
            print(f"\n📊 {investment_class.upper()}:")
            
            for category, data in categories.items():
                success_rate = (data["summary"]["successful_downloads"] / data["summary"]["total_symbols"]) * 100
                print(f"   {category}: {data['summary']['successful_downloads']}/{data['summary']['total_symbols']} ({success_rate:.1f}%)")
        
        print(f"\n{'='*80}")
        print(f" DATA SOURCES")
        print(f"{'='*80}")
        print("✅ yfinance (Primary - Most Reliable)")
        print("✅ Alpha Vantage API (Backup)")
        print("✅ Finnhub API (Fallback)")
        print("✅ All APIs: Free tier, no account required")
        
        print(f"\n{'='*80}")
        print(f" INVESTMENT CLASSES COVERED")
        print(f"{'='*80}")
        print("📈 Stocks (Large Cap, Mid Cap, Small Cap)")
        print("📊 ETFs (Equity, Bonds, Commodities)")
        print("📉 Indices (Major, International)")
        print("🪙 Commodities (Precious Metals, Energy, Agriculture)")
        print("💱 Currencies (Major, Emerging)")
        print("₿ Crypto (Major, Altcoins)")


def main():
    """Run robust market data download"""
    downloader = RobustMarketDataDownloader()
    
    # Download all market data
    results = downloader.download_all_market_data()
    
    # Calculate market statistics
    statistics = downloader.get_market_statistics()
    
    # Print summary
    downloader.print_market_summary(results)
    
    print(f"\n🎉 Robust market data download completed successfully!")
    print(f"📊 Data saved to: {downloader.data_dir}")
    print(f"📈 Statistics saved to: {downloader.data_dir}/robust_market_statistics.json")


if __name__ == "__main__":
    main() 