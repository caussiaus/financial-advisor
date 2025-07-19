"""
Enhanced Market Data Downloader

Comprehensive market data downloader with:
- Higher frequency data (daily, weekly, monthly)
- More exhaustive field coverage
- Broader range of investible securities
- Enhanced data quality and validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMarketDataDownloader:
    """Enhanced market data downloader with comprehensive coverage"""
    
    def __init__(self):
        self.data_dir = Path("data/enhanced_market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced investment classes with more exhaustive coverage
        self.investment_classes = {
            "stocks": {
                "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.A", "JNJ", "V", "PG", "UNH", "HD", "JPM", "MA"],
                "large_cap": ["AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "QCOM", "TXN", "AVGO", "ABT", "PFE", "KO", "PEP", "MRK", "TMO"],
                "mid_cap": ["SNAP", "UBER", "LYFT", "SPOT", "ZM", "SQ", "SHOP", "ROKU", "PINS", "SNAP", "TWTR", "RBLX", "PLTR", "COIN", "HOOD"],
                "small_cap": ["AMC", "GME", "BBBY", "NOK", "BB", "SNDL", "TLRY", "ACB", "CGC", "HEXO", "APHA", "CRON", "WEED", "OGI", "HEXO"],
                "growth": ["TSLA", "NVDA", "AMD", "NFLX", "CRM", "ADBE", "PYPL", "SQ", "SHOP", "ROKU", "PLTR", "COIN", "HOOD", "RBLX", "SNAP"],
                "value": ["BRK.A", "JNJ", "PG", "UNH", "HD", "JPM", "MA", "PFE", "KO", "PEP", "MRK", "TMO", "VZ", "T", "IBM"],
                "dividend": ["JNJ", "PG", "KO", "PEP", "MRK", "TMO", "VZ", "T", "IBM", "XOM", "CVX", "JPM", "BAC", "WFC", "C"],
                "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "NFLX", "CRM", "ADBE", "PYPL", "INTC", "QCOM", "TXN"],
                "healthcare": ["JNJ", "PFE", "MRK", "TMO", "ABT", "UNH", "CVS", "ANTM", "CI", "HUM", "DHR", "BDX", "ISRG", "SYK", "ZTS"],
                "financial": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "COF", "USB", "PNC", "TFC", "KEY", "HBAN"]
            },
            "etfs": {
                "equity_broad": ["SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "EFA", "EEM", "VTV", "VUG", "VBR", "VBK", "VXF", "VOT", "VO"],
                "equity_sector": ["XLK", "XLF", "XLV", "XLI", "XLE", "XLB", "XLU", "XLP", "XLY", "XLRE", "XLC", "XLB", "XLF", "XLV", "XLI"],
                "equity_international": ["EFA", "EEM", "VEA", "VWO", "VXUS", "IXUS", "ACWX", "VEU", "VSS", "VPL", "VNM", "VGT", "VHT", "VFH", "VFH"],
                "bonds_government": ["TLT", "IEF", "SHY", "BND", "AGG", "VCIT", "VCSH", "BSV", "VGIT", "VGSH", "VTEB", "MUB", "TFI", "BAB", "BNDX"],
                "bonds_corporate": ["LQD", "HYG", "VCIT", "VCSH", "BAB", "TFI", "BNDX", "IGOV", "EMB", "PCY", "LEMB", "EMLC", "EMHY", "EMB", "PCY"],
                "bonds_municipal": ["MUB", "VTEB", "TFI", "BAB", "BNDX", "IGOV", "EMB", "PCY", "LEMB", "EMLC", "EMHY", "EMB", "PCY", "MUB", "VTEB"],
                "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "XLE", "XLB", "XLI", "XLF", "XLV", "XLK", "XLP", "XLY", "XLU"],
                "real_estate": ["VNQ", "IYR", "SCHH", "RWR", "ICF", "REZ", "REM", "REET", "REGL", "REIT", "REET", "REGL", "REIT", "REET", "REGL"],
                "alternatives": ["ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "ARKX", "ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "ARKX", "ARKK", "ARKW", "ARKG"]
            },
            "indices": {
                "major_us": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^TNX", "^TYX", "^DXY", "^GC", "^CL", "^NG", "^SI", "^PL", "^PA", "^ZC"],
                "international": ["^FTSE", "^N225", "^GDAXI", "^FCHI", "^HSI", "^BSESN", "^AXJO", "^BVSP", "^MXX", "^KS11", "^TWII", "^STI", "^JKSE", "^KLSE", "^SET"],
                "emerging": ["^BSESN", "^NSEI", "^JKSE", "^KLSE", "^SET", "^TWII", "^STI", "^MXX", "^BVSP", "^KS11", "^BSESN", "^NSEI", "^JKSE", "^KLSE", "^SET"],
                "commodity": ["^GC", "^CL", "^NG", "^SI", "^PL", "^PA", "^ZC", "^ZS", "^ZW", "^KC", "^CC", "^CT", "^HO", "^RB", "^NG"],
                "currency": ["^DXY", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "USDCNY=X", "USDBRL=X", "USDINR=X", "USDMXN=X", "USDRUB=X", "USDZAR=X", "USDKRW=X", "USDSGD=X"]
            },
            "commodities": {
                "precious_metals": ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "ALI=F", "NICKEL=F", "COPPER=F", "ZINC=F", "LEAD=F", "TIN=F", "URANIUM=F", "LITHIUM=F", "COBALT=F", "RARE_EARTH=F"],
                "energy": ["CL=F", "NG=F", "HO=F", "RB=F", "BZ=F", "WTI=F", "BRENT=F", "HEATING_OIL=F", "GASOLINE=F", "PROPANE=F", "ETHANOL=F", "BIODIESEL=F", "CARBON=F", "ELECTRICITY=F", "NATURAL_GAS=F"],
                "agriculture": ["ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F", "SB=F", "CC=F", "CT=F", "KC=F", "CC=F", "CT=F", "SB=F", "CC=F", "CT=F"],
                "livestock": ["LE=F", "GF=F", "HE=F", "FC=F", "PB=F", "CATTLE=F", "HOGS=F", "PORK=F", "BEEF=F", "CHICKEN=F", "TURKEY=F", "EGGS=F", "MILK=F", "BUTTER=F", "CHEESE=F"],
                "softs": ["CC=F", "CT=F", "KC=F", "SB=F", "CC=F", "CT=F", "KC=F", "SB=F", "CC=F", "CT=F", "KC=F", "SB=F", "CC=F", "CT=F", "KC=F"]
            },
            "currencies": {
                "major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "EURCHF=X"],
                "emerging": ["USDCNY=X", "USDBRL=X", "USDINR=X", "USDMXN=X", "USDRUB=X", "USDZAR=X", "USDKRW=X", "USDSGD=X", "USDTWD=X", "USDTHB=X", "USDPHP=X", "USDIDR=X", "USDMYR=X", "USDVND=X", "USDBDT=X"],
                "exotic": ["USDBTC=X", "USDETH=X", "USDLTC=X", "USDBCH=X", "USDXRP=X", "USDDASH=X", "USDMONERO=X", "USDPOLYGON=X", "USDCARDANO=X", "USDSOLANA=X", "USDPOLKADOT=X", "USDCHAINLINK=X", "USDUNISWAP=X", "USDAVE=F", "USDSUSHI=X"],
                "commodity": ["USDAUD=X", "USDCAD=X", "USDNOK=X", "USDSEK=X", "USDDKK=X", "USDCHF=X", "USDSGD=X", "USDHKD=X", "USDSAR=X", "USDAED=X", "USDQAR=X", "USDKWD=X", "USDBHD=X", "USDOMR=X", "USDJOD=X"]
            },
            "crypto": {
                "major": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD", "XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "LINK-USD", "UNI-USD", "LTC-USD", "BCH-USD", "XLM-USD"],
                "altcoins": ["XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "LINK-USD", "UNI-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "THETA-USD"],
                "defi": ["UNI-USD", "LINK-USD", "AAVE-USD", "COMP-USD", "MKR-USD", "SUSHI-USD", "CRV-USD", "YFI-USD", "SNX-USD", "BAL-USD", "REN-USD", "ZRX-USD", "BAND-USD", "KNC-USD", "1INCH-USD"],
                "layer1": ["ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "AVAX-USD", "ATOM-USD", "ALGO-USD", "ICP-USD", "NEAR-USD", "FTM-USD", "HARMONY-USD", "CELO-USD", "COSMOS-USD", "POLKADOT-USD", "CARDANO-USD"],
                "meme": ["DOGE-USD", "SHIB-USD", "BABYDOGE-USD", "FLOKI-USD", "SAFEMOON-USD", "ELON-USD", "BONK-USD", "PEPE-USD", "WOJAK-USD", "CHAD-USD", "GIGACHAD-USD", "BASED-USD", "DEGEN-USD", "APE-USD", "MOON-USD"]
            },
            "options": {
                "spy_options": ["SPY240119C00500000", "SPY240119P00500000", "SPY240216C00500000", "SPY240216P00500000", "SPY240315C00500000"],
                "qqq_options": ["QQQ240119C00400000", "QQQ240119P00400000", "QQQ240216C00400000", "QQQ240216P00400000", "QQQ240315C00400000"],
                "tsla_options": ["TSLA240119C00250000", "TSLA240119P00250000", "TSLA240216C00250000", "TSLA240216P00250000", "TSLA240315C00250000"]
            },
            "futures": {
                "equity_futures": ["ES=F", "NQ=F", "YM=F", "RTY=F", "VX=F", "ES=F", "NQ=F", "YM=F", "RTY=F", "VX=F", "ES=F", "NQ=F", "YM=F", "RTY=F", "VX=F"],
                "commodity_futures": ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F", "LE=F", "GF=F", "HE=F", "FC=F", "PB=F"],
                "currency_futures": ["6E=F", "6B=F", "6J=F", "6S=F", "6A=F", "6C=F", "6N=F", "6M=F", "6Z=F", "6L=F", "6R=F", "6P=F", "6K=F", "6H=F", "6G=F"]
            }
        }
        
        # Enhanced data periods for different frequencies
        self.data_periods = {
            "daily": "1y",      # 1 year of daily data
            "weekly": "5y",      # 5 years of weekly data
            "monthly": "10y",    # 10 years of monthly data
            "intraday": "60d"    # 60 days of intraday data
        }
        
        # Enhanced data intervals
        self.data_intervals = {
            "daily": "1d",
            "weekly": "1wk", 
            "monthly": "1mo",
            "intraday": "1h"
        }
    
    def get_enhanced_yfinance_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict]:
        """Get enhanced data using yfinance with multiple frequencies"""
        try:
            logger.info(f"Fetching {symbol} using yfinance ({period}, {interval})...")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            
            # Get multiple frequency data
            data_frames = {}
            
            # Daily data
            df_daily = ticker.history(period="1y", interval="1d")
            if not df_daily.empty:
                data_frames["daily"] = df_daily
            
            # Weekly data
            df_weekly = ticker.history(period="5y", interval="1wk")
            if not df_weekly.empty:
                data_frames["weekly"] = df_weekly
            
            # Monthly data
            df_monthly = ticker.history(period="10y", interval="1mo")
            if not df_monthly.empty:
                data_frames["monthly"] = df_monthly
            
            # Intraday data (if available)
            try:
                df_intraday = ticker.history(period="60d", interval="1h")
                if not df_intraday.empty:
                    data_frames["intraday"] = df_intraday
            except:
                pass
            
            if not data_frames:
                return None
            
            # Calculate comprehensive metrics
            enhanced_summary = self._calculate_enhanced_metrics(data_frames, symbol)
            
            # Convert to JSON-serializable format
            json_data = {}
            for freq, df in data_frames.items():
                df_json = df.reset_index()
                df_json['Date'] = df_json['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                json_data[freq] = df_json.to_dict("records")
            
            return {
                "symbol": symbol,
                "data": json_data,
                "summary": enhanced_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced yfinance data for {symbol}: {e}")
            return None
    
    def _calculate_enhanced_metrics(self, data_frames: Dict, symbol: str) -> Dict:
        """Calculate comprehensive enhanced metrics"""
        
        summary = {
            "symbol": symbol,
            "data_quality": {},
            "price_metrics": {},
            "volatility_metrics": {},
            "volume_metrics": {},
            "technical_metrics": {},
            "risk_metrics": {},
            "correlation_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Process daily data for primary metrics
        if "daily" in data_frames:
            df = data_frames["daily"]
            
            # Basic price metrics
            summary["price_metrics"] = {
                "current_price": float(df['Close'].iloc[-1]),
                "price_change": float(df['Close'].iloc[-1] - df['Close'].iloc[0]),
                "price_change_pct": float(((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100),
                "high": float(df['High'].max()),
                "low": float(df['Low'].min()),
                "avg_price": float(df['Close'].mean()),
                "median_price": float(df['Close'].median()),
                "price_std": float(df['Close'].std()),
                "price_skew": float(df['Close'].skew()),
                "price_kurtosis": float(df['Close'].kurtosis())
            }
            
            # Volatility metrics
            returns = df['Close'].pct_change().dropna()
            summary["volatility_metrics"] = {
                "annualized_volatility": float(returns.std() * np.sqrt(252) * 100),
                "daily_volatility": float(returns.std() * 100),
                "weekly_volatility": float(returns.resample('W').std().mean() * 100),
                "monthly_volatility": float(returns.resample('M').std().mean() * 100),
                "max_drawdown": float(self._calculate_max_drawdown(df['Close'])),
                "var_95": float(np.percentile(returns, 5)),
                "var_99": float(np.percentile(returns, 1)),
                "cvar_95": float(returns[returns <= np.percentile(returns, 5)].mean()),
                "cvar_99": float(returns[returns <= np.percentile(returns, 1)].mean())
            }
            
            # Volume metrics
            summary["volume_metrics"] = {
                "avg_volume": float(df['Volume'].mean()),
                "median_volume": float(df['Volume'].median()),
                "volume_std": float(df['Volume'].std()),
                "max_volume": float(df['Volume'].max()),
                "min_volume": float(df['Volume'].min()),
                "volume_trend": float(self._calculate_volume_trend(df['Volume'])),
                "volume_price_correlation": float(df['Volume'].corr(df['Close']))
            }
            
            # Technical metrics
            summary["technical_metrics"] = {
                "rsi": float(self._calculate_rsi(df['Close'])),
                "macd": float(self._calculate_macd(df['Close'])),
                "bollinger_position": float(self._calculate_bollinger_position(df['Close'])),
                "moving_averages": {
                    "sma_20": float(df['Close'].rolling(20).mean().iloc[-1]),
                    "sma_50": float(df['Close'].rolling(50).mean().iloc[-1]),
                    "sma_200": float(df['Close'].rolling(200).mean().iloc[-1]),
                    "ema_12": float(df['Close'].ewm(span=12).mean().iloc[-1]),
                    "ema_26": float(df['Close'].ewm(span=26).mean().iloc[-1])
                }
            }
            
            # Risk metrics
            summary["risk_metrics"] = {
                "sharpe_ratio": float(self._calculate_sharpe_ratio(returns)),
                "sortino_ratio": float(self._calculate_sortino_ratio(returns)),
                "calmar_ratio": float(self._calculate_calmar_ratio(returns)),
                "information_ratio": float(self._calculate_information_ratio(returns)),
                "beta": float(self._calculate_beta(returns)),
                "alpha": float(self._calculate_alpha(returns)),
                "treynor_ratio": float(self._calculate_treynor_ratio(returns))
            }
            
            # Data quality metrics
            summary["data_quality"] = {
                "total_days": len(df),
                "start_date": df.index.min().isoformat(),
                "end_date": df.index.max().isoformat(),
                "missing_days": self._calculate_missing_days(df),
                "data_completeness": float(len(df.dropna()) / len(df) * 100),
                "outliers_detected": self._detect_outliers(df['Close']),
                "data_consistency": self._check_data_consistency(df)
            }
        
        # Process weekly and monthly data for additional metrics
        if "weekly" in data_frames:
            df_weekly = data_frames["weekly"]
            summary["weekly_metrics"] = {
                "weekly_returns_mean": float(df_weekly['Close'].pct_change().mean()),
                "weekly_returns_std": float(df_weekly['Close'].pct_change().std()),
                "weekly_volatility": float(df_weekly['Close'].pct_change().std() * np.sqrt(52) * 100)
            }
        
        if "monthly" in data_frames:
            df_monthly = data_frames["monthly"]
            summary["monthly_metrics"] = {
                "monthly_returns_mean": float(df_monthly['Close'].pct_change().mean()),
                "monthly_returns_std": float(df_monthly['Close'].pct_change().std()),
                "monthly_volatility": float(df_monthly['Close'].pct_change().std() * np.sqrt(12) * 100)
            }
        
        return summary
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())
    
    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend"""
        if len(volume) < 2:
            return 0.0
        return float((volume.iloc[-1] - volume.iloc[0]) / volume.iloc[0])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        return float(macd.iloc[-1])
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        current_price = prices.iloc[-1]
        position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        return float(position)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        return float(excess_returns.mean() / downside_std * np.sqrt(252))
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        max_dd = abs(self._calculate_max_drawdown(returns.cumsum()))
        if max_dd == 0:
            return 0.0
        return float(returns.mean() * 252 / max_dd)
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio"""
        return float(returns.mean() / returns.std() * np.sqrt(252))
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate Beta (simplified)"""
        # This would need market returns for proper calculation
        return 1.0  # Placeholder
    
    def _calculate_alpha(self, returns: pd.Series) -> float:
        """Calculate Alpha (simplified)"""
        # This would need market returns for proper calculation
        return float(returns.mean() * 252)
    
    def _calculate_treynor_ratio(self, returns: pd.Series) -> float:
        """Calculate Treynor ratio"""
        beta = self._calculate_beta(returns)
        if beta == 0:
            return 0.0
        return float(returns.mean() * 252 / beta)
    
    def _calculate_missing_days(self, df: pd.DataFrame) -> int:
        """Calculate missing trading days"""
        expected_days = len(pd.date_range(df.index.min(), df.index.max(), freq='B'))
        return expected_days - len(df)
    
    def _detect_outliers(self, prices: pd.Series) -> int:
        """Detect outliers using IQR method"""
        Q1 = prices.quantile(0.25)
        Q3 = prices.quantile(0.75)
        IQR = Q3 - Q1
        outliers = prices[(prices < (Q1 - 1.5 * IQR)) | (prices > (Q3 + 1.5 * IQR))]
        return len(outliers)
    
    def _check_data_consistency(self, df: pd.DataFrame) -> str:
        """Check data consistency"""
        if df['High'].max() < df['Low'].min():
            return "inconsistent"
        if (df['High'] < df['Low']).any():
            return "inconsistent"
        return "consistent"
    
    def download_enhanced_data(self, investment_class: str, category: str, symbols: List[str]) -> Dict:
        """Download enhanced data for a specific category"""
        
        logger.info(f"Downloading enhanced {investment_class} - {category} data...")
        
        results = {
            "investment_class": investment_class,
            "category": category,
            "symbols": symbols,
            "data": {},
            "summary": {
                "total_symbols": len(symbols),
                "successful_downloads": 0,
                "failed_downloads": 0,
                "download_date": datetime.now().isoformat(),
                "enhanced_metrics": {}
            }
        }
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Downloading {symbol} ({i+1}/{len(symbols)})...")
            
            data = self.get_enhanced_yfinance_data(symbol)
            
            if data:
                results["data"][symbol] = data
                results["summary"]["successful_downloads"] += 1
                logger.info(f"✅ Successfully downloaded {symbol}")
            else:
                results["summary"]["failed_downloads"] += 1
                logger.warning(f"❌ Failed to download {symbol}")
            
            # Rate limiting
            time.sleep(random.uniform(1.0, 2.0))
        
        return results
    
    def download_all_enhanced_data(self) -> Dict:
        """Download enhanced data for all investment classes"""
        
        logger.info("Starting enhanced market data download...")
        
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
                results = self.download_enhanced_data(investment_class, category, symbols)
                
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
        summary_file = self.data_dir / "enhanced_market_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Enhanced market data download complete! Summary saved to {summary_file}")
        
        return all_results


def main():
    """Run enhanced market data download"""
    downloader = EnhancedMarketDataDownloader()
    
    # Download all enhanced market data
    results = downloader.download_all_enhanced_data()
    
    print(f"\n🎉 Enhanced market data download completed!")
    print(f"📊 Data saved to: {downloader.data_dir}")
    print(f"📈 Enhanced metrics and comprehensive coverage!")


if __name__ == "__main__":
    main() 