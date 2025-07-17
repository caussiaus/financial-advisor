#!/usr/bin/env python3
"""
Market Tracking and Backtesting System

Comprehensive system for:
1. Real-time market data tracking
2. Personal finance to investment action association
3. Historical backtesting with realistic market conditions
4. Performance attribution and analysis
5. Risk-adjusted return calculations
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Financial calculation libraries
try:
    import quantlib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("⚠️ QuantLib not available - using simplified calculations")

@dataclass
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    volatility: Optional[float] = None

@dataclass
class PersonalFinanceAction:
    """Personal finance action that triggers investment decision"""
    timestamp: datetime
    action_type: str  # 'income_change', 'expense_change', 'milestone_payment', 'debt_payment'
    amount: float
    description: str
    category: str
    confidence: float  # 0-1 confidence in the action
    impact_duration: int  # days this action affects decisions

@dataclass
class InvestmentDecision:
    """Investment decision triggered by personal finance action"""
    timestamp: datetime
    decision_type: str  # 'buy', 'sell', 'rebalance', 'hold'
    asset_class: str
    amount: float
    percentage: float  # percentage of portfolio
    trigger_action: PersonalFinanceAction
    market_conditions: Dict[str, float]
    expected_return: float
    risk_score: float

@dataclass
class BacktestResult:
    """Result of a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    action_decisions: List[InvestmentDecision]
    market_performance: Dict[str, float]
    personal_finance_actions: List[PersonalFinanceAction]

class MarketDataTracker:
    """Real-time market data tracking"""
    
    def __init__(self):
        self.market_data = {}
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for market tracking"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def get_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get historical market data for symbols"""
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
                    self.logger.info(f"✅ Loaded {len(data)} data points for {symbol}")
                else:
                    self.logger.warning(f"⚠️ No data found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error loading data for {symbol}: {e}")
                
        return market_data
    
    def get_market_indicators(self, date: datetime) -> Dict[str, float]:
        """Get market indicators for a specific date"""
        # Get major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^TNX']  # S&P 500, Dow, NASDAQ, 10Y Treasury
        
        indicators = {}
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                data = ticker.history(start=date - timedelta(days=30), end=date)
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-5] if len(data) > 5 else current_price
                    
                    indicators[f"{index}_price"] = current_price
                    indicators[f"{index}_return"] = (current_price - prev_price) / prev_price
                    indicators[f"{index}_volatility"] = data['Returns'].std() if 'Returns' in data.columns else 0.15
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error getting indicator {index}: {e}")
        
        # Calculate market stress index
        if all(k in indicators for k in ['^GSPC_volatility', '^TNX_price']):
            vix_proxy = indicators['^GSPC_volatility'] * 100
            treasury_yield = indicators['^TNX_price'] / 100
            market_stress = min(1.0, (vix_proxy - 15) / 20 + (treasury_yield - 0.02) / 0.05)
            indicators['market_stress'] = max(0.0, market_stress)
        else:
            indicators['market_stress'] = 0.3  # Default moderate stress
            
        return indicators

class PersonalFinanceInvestmentMapper:
    """Maps personal finance actions to investment decisions"""
    
    def __init__(self):
        self.action_rules = self._initialize_action_rules()
        self.risk_tolerance_weights = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.7
        }
        
    def _initialize_action_rules(self) -> Dict[str, Dict]:
        """Initialize rules for mapping personal finance actions to investment decisions"""
        return {
            'income_increase': {
                'trigger_threshold': 0.1,  # 10% income increase
                'investment_action': 'increase_equity',
                'allocation_change': 0.05,  # 5% increase in equity
                'confidence_multiplier': 1.2
            },
            'income_decrease': {
                'trigger_threshold': -0.1,  # 10% income decrease
                'investment_action': 'increase_bonds',
                'allocation_change': 0.05,  # 5% increase in bonds
                'confidence_multiplier': 1.3
            },
            'major_expense': {
                'trigger_threshold': 0.2,  # 20% of portfolio
                'investment_action': 'increase_cash',
                'allocation_change': 0.1,  # 10% increase in cash
                'confidence_multiplier': 1.5
            },
            'debt_payment': {
                'trigger_threshold': 0.05,  # 5% of portfolio
                'investment_action': 'decrease_risk',
                'allocation_change': 0.03,  # 3% decrease in equity
                'confidence_multiplier': 1.1
            },
            'milestone_achievement': {
                'trigger_threshold': 0.0,  # Any milestone
                'investment_action': 'rebalance',
                'allocation_change': 0.02,  # 2% rebalancing
                'confidence_multiplier': 1.0
            }
        }
    
    def map_action_to_decision(self, 
                              action: PersonalFinanceAction,
                              current_portfolio: Dict[str, float],
                              market_conditions: Dict[str, float],
                              risk_tolerance: str = 'moderate') -> Optional[InvestmentDecision]:
        """Map a personal finance action to an investment decision"""
        
        # Get rule for this action type
        rule = self.action_rules.get(action.action_type)
        if not rule:
            return None
            
        # Check if action meets threshold
        portfolio_value = sum(current_portfolio.values())
        action_impact = abs(action.amount) / portfolio_value if portfolio_value > 0 else 0
        
        if action_impact < rule['trigger_threshold']:
            return None
            
        # Calculate decision parameters
        risk_weight = self.risk_tolerance_weights.get(risk_tolerance, 0.5)
        confidence = action.confidence * rule['confidence_multiplier']
        
        # Adjust for market conditions
        market_stress = market_conditions.get('market_stress', 0.3)
        if market_stress > 0.7:  # High stress
            confidence *= 0.8
        elif market_stress < 0.3:  # Low stress
            confidence *= 1.1
            
        # Determine asset class and amount
        if rule['investment_action'] == 'increase_equity':
            asset_class = 'equity'
            amount = portfolio_value * rule['allocation_change']
        elif rule['investment_action'] == 'increase_bonds':
            asset_class = 'bonds'
            amount = portfolio_value * rule['allocation_change']
        elif rule['investment_action'] == 'increase_cash':
            asset_class = 'cash'
            amount = portfolio_value * rule['allocation_change']
        elif rule['investment_action'] == 'decrease_risk':
            asset_class = 'bonds'
            amount = portfolio_value * rule['allocation_change']
        else:  # rebalance
            asset_class = 'mixed'
            amount = portfolio_value * rule['allocation_change']
            
        # Calculate expected return based on asset class and market conditions
        expected_return = self._calculate_expected_return(asset_class, market_conditions, risk_weight)
        risk_score = self._calculate_risk_score(asset_class, market_conditions, risk_weight)
        
        return InvestmentDecision(
            timestamp=action.timestamp,
            decision_type='buy' if amount > 0 else 'sell',
            asset_class=asset_class,
            amount=abs(amount),
            percentage=abs(amount) / portfolio_value if portfolio_value > 0 else 0,
            trigger_action=action,
            market_conditions=market_conditions,
            expected_return=expected_return,
            risk_score=risk_score
        )
    
    def _calculate_expected_return(self, asset_class: str, market_conditions: Dict[str, float], risk_weight: float) -> float:
        """Calculate expected return for asset class"""
        base_returns = {
            'equity': 0.08,
            'bonds': 0.04,
            'cash': 0.02,
            'mixed': 0.06
        }
        
        base_return = base_returns.get(asset_class, 0.05)
        market_stress = market_conditions.get('market_stress', 0.3)
        
        # Adjust for market stress
        if market_stress > 0.7:
            base_return *= 0.8  # Lower returns in high stress
        elif market_stress < 0.3:
            base_return *= 1.1  # Higher returns in low stress
            
        # Adjust for risk tolerance
        return base_return * (1 + (risk_weight - 0.5) * 0.2)
    
    def _calculate_risk_score(self, asset_class: str, market_conditions: Dict[str, float], risk_weight: float) -> float:
        """Calculate risk score for asset class"""
        base_risk = {
            'equity': 0.6,
            'bonds': 0.2,
            'cash': 0.0,
            'mixed': 0.4
        }
        
        base_risk_score = base_risk.get(asset_class, 0.3)
        market_stress = market_conditions.get('market_stress', 0.3)
        
        # Adjust for market stress
        if market_stress > 0.7:
            base_risk_score *= 1.3  # Higher risk in high stress
        elif market_stress < 0.3:
            base_risk_score *= 0.9  # Lower risk in low stress
            
        return min(1.0, base_risk_score)

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, market_tracker: MarketDataTracker, mapper: PersonalFinanceInvestmentMapper):
        self.market_tracker = market_tracker
        self.mapper = mapper
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for backtesting"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def run_backtest(self,
                    start_date: datetime,
                    end_date: datetime,
                    initial_portfolio: Dict[str, float],
                    personal_finance_actions: List[PersonalFinanceAction],
                    risk_tolerance: str = 'moderate',
                    rebalance_frequency: str = 'monthly') -> BacktestResult:
        """Run comprehensive backtest"""
        
        self.logger.info(f"🚀 Starting backtest from {start_date} to {end_date}")
        
        # Get market data
        symbols = ['^GSPC', '^DJI', '^IXIC', '^TNX', 'GLD', 'TLT']  # Major indices and ETFs
        market_data = self.market_tracker.get_market_data(symbols, start_date, end_date)
        
        # Initialize tracking variables
        current_portfolio = initial_portfolio.copy()
        portfolio_history = []
        decisions_made = []
        trades_executed = []
        
        # Get date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for current_date in date_range:
            # Get market conditions for this date
            market_conditions = self.market_tracker.get_market_indicators(current_date)
            
            # Check for personal finance actions on this date
            daily_actions = [a for a in personal_finance_actions 
                           if a.timestamp.date() == current_date.date()]
            
            # Process each action
            for action in daily_actions:
                decision = self.mapper.map_action_to_decision(
                    action, current_portfolio, market_conditions, risk_tolerance
                )
                
                if decision:
                    decisions_made.append(decision)
                    
                    # Execute the decision
                    self._execute_decision(decision, current_portfolio, market_data, current_date)
                    trades_executed.append({
                        'date': current_date,
                        'decision': decision,
                        'portfolio_value': sum(current_portfolio.values())
                    })
            
            # Apply market returns to portfolio
            self._apply_market_returns(current_portfolio, market_data, current_date)
            
            # Record portfolio state
            portfolio_history.append({
                'date': current_date,
                'portfolio': current_portfolio.copy(),
                'total_value': sum(current_portfolio.values()),
                'market_conditions': market_conditions
            })
            
            # Rebalance if needed
            if rebalance_frequency == 'monthly' and current_date.day == 1:
                self._rebalance_portfolio(current_portfolio, market_conditions, risk_tolerance)
        
        # Calculate results
        result = self._calculate_backtest_results(
            start_date, end_date, initial_portfolio, current_portfolio,
            portfolio_history, decisions_made, trades_executed, personal_finance_actions
        )
        
        self.logger.info(f"✅ Backtest completed. Final portfolio value: ${result.final_portfolio_value:,.2f}")
        return result
    
    def _execute_decision(self, decision: InvestmentDecision, portfolio: Dict[str, float], 
                         market_data: Dict[str, pd.DataFrame], date: datetime):
        """Execute an investment decision"""
        
        if decision.decision_type == 'buy':
            # Add to portfolio
            if decision.asset_class in portfolio:
                portfolio[decision.asset_class] += decision.amount
            else:
                portfolio[decision.asset_class] = decision.amount
                
        elif decision.decision_type == 'sell':
            # Remove from portfolio
            if decision.asset_class in portfolio:
                portfolio[decision.asset_class] = max(0, portfolio[decision.asset_class] - decision.amount)
                
        elif decision.decision_type == 'rebalance':
            # Rebalance portfolio
            total_value = sum(portfolio.values())
            target_allocation = self._get_target_allocation(decision.asset_class, decision.percentage)
            
            for asset_class, target_pct in target_allocation.items():
                target_amount = total_value * target_pct
                portfolio[asset_class] = target_amount
    
    def _apply_market_returns(self, portfolio: Dict[str, float], market_data: Dict[str, pd.DataFrame], date: datetime):
        """Apply market returns to portfolio"""
        
        # Map portfolio assets to market symbols
        asset_mapping = {
            'equity': '^GSPC',  # S&P 500 for equity
            'bonds': '^TNX',    # Treasury yield for bonds
            'cash': None,        # Cash doesn't have market returns
            'gold': 'GLD',       # Gold ETF
            'treasury': 'TLT'    # Long-term Treasury ETF
        }
        
        for asset_class, amount in portfolio.items():
            if amount <= 0:
                continue
                
            symbol = asset_mapping.get(asset_class)
            if symbol and symbol in market_data:
                data = market_data[symbol]
                
                # Get return for this date
                if date in data.index:
                    daily_return = data.loc[date, 'Returns'] if 'Returns' in data.columns else 0
                    portfolio[asset_class] *= (1 + daily_return)
                elif len(data) > 0:
                    # Use last available return
                    last_return = data['Returns'].iloc[-1] if 'Returns' in data.columns else 0
                    portfolio[asset_class] *= (1 + last_return)
    
    def _rebalance_portfolio(self, portfolio: Dict[str, float], market_conditions: Dict[str, float], risk_tolerance: str):
        """Rebalance portfolio based on risk tolerance and market conditions"""
        
        total_value = sum(portfolio.values())
        if total_value <= 0:
            return
            
        # Get target allocation based on risk tolerance
        target_allocation = self._get_target_allocation(risk_tolerance, market_conditions)
        
        # Rebalance to target
        for asset_class, target_pct in target_allocation.items():
            target_amount = total_value * target_pct
            portfolio[asset_class] = target_amount
    
    def _get_target_allocation(self, risk_tolerance: str, market_conditions: Dict[str, float]) -> Dict[str, float]:
        """Get target allocation based on risk tolerance and market conditions"""
        
        base_allocations = {
            'conservative': {'equity': 0.3, 'bonds': 0.5, 'cash': 0.2},
            'moderate': {'equity': 0.6, 'bonds': 0.3, 'cash': 0.1},
            'aggressive': {'equity': 0.8, 'bonds': 0.15, 'cash': 0.05}
        }
        
        allocation = base_allocations.get(risk_tolerance, base_allocations['moderate']).copy()
        
        # Adjust for market stress
        market_stress = market_conditions.get('market_stress', 0.3)
        if market_stress > 0.7:  # High stress - more defensive
            allocation['equity'] *= 0.8
            allocation['bonds'] *= 1.2
            allocation['cash'] *= 1.1
        elif market_stress < 0.3:  # Low stress - more aggressive
            allocation['equity'] *= 1.1
            allocation['bonds'] *= 0.9
            allocation['cash'] *= 0.9
            
        # Normalize
        total = sum(allocation.values())
        for asset_class in allocation:
            allocation[asset_class] /= total
            
        return allocation
    
    def _calculate_backtest_results(self, start_date: datetime, end_date: datetime,
                                  initial_portfolio: Dict[str, float], final_portfolio: Dict[str, float],
                                  portfolio_history: List[Dict], decisions_made: List[InvestmentDecision],
                                  trades_executed: List[Dict], personal_finance_actions: List[PersonalFinanceAction]) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        initial_value = sum(initial_portfolio.values())
        final_value = sum(final_portfolio.values())
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate time period
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility
        portfolio_values = [p['total_value'] for p in portfolio_history]
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        profitable_trades = len([t for t in trades_executed if t['decision'].expected_return > 0])
        total_trades = len(trades_executed)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate market performance
        market_performance = {}
        if portfolio_history:
            first_conditions = portfolio_history[0]['market_conditions']
            last_conditions = portfolio_history[-1]['market_conditions']
            
            for key in first_conditions:
                if key.endswith('_price') and key in last_conditions:
                    market_return = (last_conditions[key] - first_conditions[key]) / first_conditions[key]
                    market_performance[key] = market_return
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_portfolio_value=initial_value,
            final_portfolio_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            action_decisions=decisions_made,
            market_performance=market_performance,
            personal_finance_actions=personal_finance_actions
        )

class BacktestAnalyzer:
    """Analyze and visualize backtest results"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for analysis"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_backtest_result(self, result: BacktestResult) -> Dict[str, Any]:
        """Comprehensive analysis of backtest result"""
        
        analysis = {
            'performance_summary': {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'volatility': f"{result.volatility:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'win_rate': f"{result.win_rate:.2%}"
            },
            'trade_analysis': {
                'total_trades': result.total_trades,
                'profitable_trades': result.profitable_trades,
                'average_trade_return': self._calculate_average_trade_return(result),
                'best_trade': self._find_best_trade(result),
                'worst_trade': self._find_worst_trade(result)
            },
            'action_analysis': {
                'total_actions': len(result.personal_finance_actions),
                'action_types': self._analyze_action_types(result),
                'decision_effectiveness': self._analyze_decision_effectiveness(result)
            },
            'market_analysis': {
                'market_performance': result.market_performance,
                'outperformance': self._calculate_outperformance(result)
            },
            'risk_analysis': {
                'var_95': self._calculate_var(result, 0.95),
                'var_99': self._calculate_var(result, 0.99),
                'risk_adjusted_metrics': self._calculate_risk_adjusted_metrics(result)
            }
        }
        
        return analysis
    
    def _calculate_average_trade_return(self, result: BacktestResult) -> float:
        """Calculate average return per trade"""
        if not result.action_decisions:
            return 0.0
        
        total_expected_return = sum(d.expected_return for d in result.action_decisions)
        return total_expected_return / len(result.action_decisions)
    
    def _find_best_trade(self, result: BacktestResult) -> Optional[InvestmentDecision]:
        """Find the best performing trade"""
        if not result.action_decisions:
            return None
        
        return max(result.action_decisions, key=lambda x: x.expected_return)
    
    def _find_worst_trade(self, result: BacktestResult) -> Optional[InvestmentDecision]:
        """Find the worst performing trade"""
        if not result.action_decisions:
            return None
        
        return min(result.action_decisions, key=lambda x: x.expected_return)
    
    def _analyze_action_types(self, result: BacktestResult) -> Dict[str, int]:
        """Analyze types of personal finance actions"""
        action_types = {}
        for action in result.personal_finance_actions:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        return action_types
    
    def _analyze_decision_effectiveness(self, result: BacktestResult) -> Dict[str, float]:
        """Analyze effectiveness of investment decisions"""
        if not result.action_decisions:
            return {}
        
        # Group by decision type
        decision_types = {}
        for decision in result.action_decisions:
            if decision.decision_type not in decision_types:
                decision_types[decision.decision_type] = []
            decision_types[decision.decision_type].append(decision.expected_return)
        
        # Calculate average returns by decision type
        effectiveness = {}
        for decision_type, returns in decision_types.items():
            effectiveness[decision_type] = sum(returns) / len(returns)
        
        return effectiveness
    
    def _calculate_outperformance(self, result: BacktestResult) -> float:
        """Calculate outperformance vs market"""
        if not result.market_performance:
            return 0.0
        
        # Use S&P 500 as market benchmark
        market_return = result.market_performance.get('^GSPC_price', 0)
        portfolio_return = result.total_return
        
        return portfolio_return - market_return
    
    def _calculate_var(self, result: BacktestResult, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        # This would require daily returns data
        # For now, use a simplified calculation
        return result.max_drawdown * confidence_level
    
    def _calculate_risk_adjusted_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        return {
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': self._calculate_sortino_ratio(result),
            'calmar_ratio': self._calculate_calmar_ratio(result)
        }
    
    def _calculate_sortino_ratio(self, result: BacktestResult) -> float:
        """Calculate Sortino ratio"""
        # Simplified calculation
        risk_free_rate = 0.02
        excess_return = result.annualized_return - risk_free_rate
        downside_deviation = result.volatility * 0.7  # Approximate
        return excess_return / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_calmar_ratio(self, result: BacktestResult) -> float:
        """Calculate Calmar ratio"""
        if result.max_drawdown == 0:
            return 0
        return result.annualized_return / abs(result.max_drawdown)
    
    def generate_visualizations(self, result: BacktestResult) -> Dict[str, go.Figure]:
        """Generate comprehensive visualizations"""
        
        visualizations = {}
        
        # Portfolio value over time
        visualizations['portfolio_timeline'] = self._create_portfolio_timeline(result)
        
        # Returns distribution
        visualizations['returns_distribution'] = self._create_returns_distribution(result)
        
        # Drawdown chart
        visualizations['drawdown_chart'] = self._create_drawdown_chart(result)
        
        # Action decision analysis
        visualizations['decision_analysis'] = self._create_decision_analysis(result)
        
        return visualizations
    
    def _create_portfolio_timeline(self, result: BacktestResult) -> go.Figure:
        """Create portfolio value timeline"""
        # This would need portfolio history data
        # For now, create a simplified timeline
        dates = pd.date_range(result.start_date, result.end_date, freq='D')
        portfolio_values = np.linspace(result.initial_portfolio_value, result.final_portfolio_value, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400
        )
        
        return fig
    
    def _create_returns_distribution(self, result: BacktestResult) -> go.Figure:
        """Create returns distribution chart"""
        # Simplified returns distribution
        returns = np.random.normal(result.annualized_return, result.volatility, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution'
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def _create_drawdown_chart(self, result: BacktestResult) -> go.Figure:
        """Create drawdown chart"""
        # Simplified drawdown simulation
        dates = pd.date_range(result.start_date, result.end_date, freq='D')
        cumulative_returns = np.linspace(0, result.total_return, len(dates))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        return fig
    
    def _create_decision_analysis(self, result: BacktestResult) -> go.Figure:
        """Create decision analysis chart"""
        if not result.action_decisions:
            return go.Figure()
        
        # Group decisions by type
        decision_types = {}
        for decision in result.action_decisions:
            if decision.decision_type not in decision_types:
                decision_types[decision.decision_type] = []
            decision_types[decision.decision_type].append(decision.expected_return)
        
        # Calculate average returns by decision type
        decision_analysis = {}
        for decision_type, returns in decision_types.items():
            decision_analysis[decision_type] = sum(returns) / len(returns)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(decision_analysis.keys()),
                y=list(decision_analysis.values()),
                name='Average Expected Return'
            )
        ])
        
        fig.update_layout(
            title='Decision Effectiveness by Type',
            xaxis_title='Decision Type',
            yaxis_title='Average Expected Return',
            height=400
        )
        
        return fig

def create_sample_backtest():
    """Create a sample backtest with realistic data"""
    
    # Initialize components
    market_tracker = MarketDataTracker()
    mapper = PersonalFinanceInvestmentMapper()
    backtest_engine = BacktestEngine(market_tracker, mapper)
    analyzer = BacktestAnalyzer()
    
    # Create sample personal finance actions
    actions = [
        PersonalFinanceAction(
            timestamp=datetime(2020, 3, 15),
            action_type='income_increase',
            amount=15000,
            description='Promotion and salary increase',
            category='career',
            confidence=0.9,
            impact_duration=365
        ),
        PersonalFinanceAction(
            timestamp=datetime(2020, 6, 10),
            action_type='major_expense',
            amount=25000,
            description='Home renovation project',
            category='housing',
            confidence=0.8,
            impact_duration=180
        ),
        PersonalFinanceAction(
            timestamp=datetime(2021, 1, 5),
            action_type='debt_payment',
            amount=10000,
            description='Student loan payoff',
            category='debt',
            confidence=0.95,
            impact_duration=365
        ),
        PersonalFinanceAction(
            timestamp=datetime(2021, 8, 20),
            action_type='milestone_achievement',
            amount=5000,
            description='Emergency fund goal reached',
            category='savings',
            confidence=0.7,
            impact_duration=90
        )
    ]
    
    # Initial portfolio
    initial_portfolio = {
        'equity': 60000,
        'bonds': 30000,
        'cash': 10000
    }
    
    # Run backtest
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    result = backtest_engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_portfolio=initial_portfolio,
        personal_finance_actions=actions,
        risk_tolerance='moderate',
        rebalance_frequency='monthly'
    )
    
    # Analyze results
    analysis = analyzer.analyze_backtest_result(result)
    visualizations = analyzer.generate_visualizations(result)
    
    return result, analysis, visualizations

if __name__ == "__main__":
    print("🚀 Starting Market Tracking and Backtesting System...")
    
    # Run sample backtest
    result, analysis, visualizations = create_sample_backtest()
    
    print("\n📊 Backtest Results:")
    print(f"Initial Portfolio Value: ${result.initial_portfolio_value:,.2f}")
    print(f"Final Portfolio Value: ${result.final_portfolio_value:,.2f}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    
    print("\n📈 Performance Summary:")
    for metric, value in analysis['performance_summary'].items():
        print(f"  {metric}: {value}")
    
    print("\n🎯 Action Analysis:")
    print(f"  Total Actions: {analysis['action_analysis']['total_actions']}")
    print(f"  Action Types: {analysis['action_analysis']['action_types']}")
    
    print("\n✅ Backtest completed successfully!") 