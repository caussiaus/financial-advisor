"""
fsQCA Market Uncertainty Decision Analyzer

This module implements fsQCA analysis specifically for determining optimal financial
decisions and capital allocation strategies during market uncertainty.

Key Features:
- Market uncertainty surface analysis
- Capital allocation decision optimization
- Backtesting with synthesized data
- Real-time decision recommendations
- Risk-adjusted return optimization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import networkx as nx
from enum import Enum
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketUncertaintyLevel(Enum):
    """Market uncertainty levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class DecisionType(Enum):
    """Types of financial decisions"""
    CAPITAL_ALLOCATION = "capital_allocation"
    RISK_MANAGEMENT = "risk_management"
    ASSET_SELECTION = "asset_selection"
    TIMING = "timing"
    LIQUIDITY_MANAGEMENT = "liquidity_management"

@dataclass
class MarketSurface:
    """Represents a surface in financial markets"""
    surface_id: str
    coordinates: np.ndarray  # Market state coordinates
    uncertainty_level: MarketUncertaintyLevel
    volatility: float
    correlation_matrix: np.ndarray
    liquidity_metrics: Dict[str, float]
    decision_opportunities: List[Dict[str, Any]]
    metadata: Dict = field(default_factory=dict)

@dataclass
class FinancialDecision:
    """Represents a financial decision made during uncertainty"""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    market_conditions: Dict[str, float]
    uncertainty_level: MarketUncertaintyLevel
    decision_parameters: Dict[str, float]
    expected_outcome: Dict[str, float]
    confidence_score: float
    backtest_results: Optional[Dict[str, Any]] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class CapitalAllocationStrategy:
    """Represents a capital allocation strategy"""
    strategy_id: str
    uncertainty_level: MarketUncertaintyLevel
    asset_allocation: Dict[str, float]
    risk_parameters: Dict[str, float]
    timing_parameters: Dict[str, float]
    liquidity_requirements: Dict[str, float]
    success_metrics: Dict[str, float]
    backtest_performance: Dict[str, float]
    metadata: Dict = field(default_factory=dict)

@dataclass
class fsQCAMarketUncertaintyResult:
    """Result of fsQCA analysis for market uncertainty decisions"""
    solution_coverage: float
    solution_consistency: float
    necessary_conditions: Dict[str, float]
    sufficient_conditions: Dict[str, float]
    optimal_decisions: List[FinancialDecision]
    capital_allocation_strategies: List[CapitalAllocationStrategy]
    market_surfaces: List[MarketSurface]
    backtest_results: Dict[str, Any]
    decision_recommendations: Dict[str, Any]
    analysis_summary: Dict[str, Any]

class fsQCAMarketUncertaintyAnalyzer:
    """
    Analyzes financial decisions and capital allocation during market uncertainty
    using fsQCA methodology
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.uncertainty_thresholds = self.config.get('uncertainty_thresholds', {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.6,
            'extreme': 0.8
        })
        self.decision_history: List[FinancialDecision] = []
        self.market_surfaces: List[MarketSurface] = []
        self.capital_strategies: List[CapitalAllocationStrategy] = []
        self.backtest_data: Dict[str, Any] = {}
        
    def analyze_market_uncertainty_surfaces(self, market_data: Dict[str, Any]) -> List[MarketSurface]:
        """
        Analyze market surfaces to identify decision opportunities during uncertainty
        
        Args:
            market_data: Market data including prices, volumes, volatility, etc.
            
        Returns:
            List of market surfaces with decision opportunities
        """
        logger.info("🔍 Analyzing market uncertainty surfaces...")
        
        surfaces = []
        
        # Extract market features
        prices = np.array(market_data.get('prices', []))
        volumes = np.array(market_data.get('volumes', []))
        volatilities = np.array(market_data.get('volatilities', []))
        correlations = np.array(market_data.get('correlations', []))
        
        if len(prices) == 0:
            return surfaces
        
        # Create market state coordinates
        market_coordinates = np.column_stack([
            prices / np.max(prices),  # Normalized prices
            volumes / np.max(volumes),  # Normalized volumes
            volatilities,  # Raw volatility
            correlations.flatten() if correlations.size > 0 else np.zeros(len(prices))  # Correlation
        ])
        
        # Cluster market states to identify surfaces
        n_clusters = min(5, len(market_coordinates))
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(market_coordinates)
        
        # Create surfaces for each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = market_coordinates[cluster_mask]
            
            if len(cluster_coords) < 2:
                continue
            
            # Calculate surface properties
            avg_volatility = np.mean(volatilities[cluster_mask])
            avg_correlation = np.mean(correlations[cluster_mask]) if correlations.size > 0 else 0.5
            
            # Determine uncertainty level
            if avg_volatility < self.uncertainty_thresholds['low']:
                uncertainty_level = MarketUncertaintyLevel.LOW
            elif avg_volatility < self.uncertainty_thresholds['medium']:
                uncertainty_level = MarketUncertaintyLevel.MEDIUM
            elif avg_volatility < self.uncertainty_thresholds['high']:
                uncertainty_level = MarketUncertaintyLevel.HIGH
            else:
                uncertainty_level = MarketUncertaintyLevel.EXTREME
            
            # Calculate liquidity metrics
            liquidity_metrics = {
                'volume_stability': np.std(volumes[cluster_mask]) / np.mean(volumes[cluster_mask]),
                'price_efficiency': 1 - np.std(prices[cluster_mask]) / np.mean(prices[cluster_mask]),
                'market_depth': np.mean(volumes[cluster_mask]) / np.mean(prices[cluster_mask])
            }
            
            # Identify decision opportunities
            decision_opportunities = self._identify_decision_opportunities(
                cluster_coords, uncertainty_level, liquidity_metrics
            )
            
            surface = MarketSurface(
                surface_id=f"surface_{cluster_id}",
                coordinates=cluster_coords,
                uncertainty_level=uncertainty_level,
                volatility=avg_volatility,
                correlation_matrix=correlations[cluster_mask] if correlations.size > 0 else np.array([[1.0]]),
                liquidity_metrics=liquidity_metrics,
                decision_opportunities=decision_opportunities
            )
            
            surfaces.append(surface)
        
        self.market_surfaces = surfaces
        logger.info(f"✅ Identified {len(surfaces)} market surfaces")
        return surfaces
    
    def _identify_decision_opportunities(self, coordinates: np.ndarray, 
                                       uncertainty_level: MarketUncertaintyLevel,
                                       liquidity_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify decision opportunities on a market surface"""
        opportunities = []
        
        # Capital allocation opportunities
        if uncertainty_level in [MarketUncertaintyLevel.HIGH, MarketUncertaintyLevel.EXTREME]:
            opportunities.append({
                'type': 'capital_allocation',
                'action': 'increase_cash',
                'confidence': 0.8,
                'parameters': {'cash_ratio': 0.4, 'risk_reduction': 0.3}
            })
        
        # Risk management opportunities
        if liquidity_metrics['volume_stability'] > 0.5:
            opportunities.append({
                'type': 'risk_management',
                'action': 'diversify_assets',
                'confidence': 0.7,
                'parameters': {'diversification': 0.6, 'correlation_limit': 0.3}
            })
        
        # Timing opportunities
        if liquidity_metrics['price_efficiency'] < 0.5:
            opportunities.append({
                'type': 'timing',
                'action': 'wait_for_stability',
                'confidence': 0.6,
                'parameters': {'wait_period': 30, 'stability_threshold': 0.7}
            })
        
        # Asset selection opportunities
        if uncertainty_level == MarketUncertaintyLevel.EXTREME:
            opportunities.append({
                'type': 'asset_selection',
                'action': 'defensive_assets',
                'confidence': 0.9,
                'parameters': {'defensive_ratio': 0.7, 'quality_focus': 0.8}
            })
        
        return opportunities
    
    def optimize_capital_allocation(self, market_surfaces: List[MarketSurface],
                                  current_portfolio: Dict[str, float]) -> List[CapitalAllocationStrategy]:
        """
        Optimize capital allocation strategies for different uncertainty levels
        
        Args:
            market_surfaces: Identified market surfaces
            current_portfolio: Current portfolio allocation
            
        Returns:
            List of optimized capital allocation strategies
        """
        logger.info("🎯 Optimizing capital allocation strategies...")
        
        strategies = []
        
        for surface in market_surfaces:
            # Calculate optimal allocation based on uncertainty level
            if surface.uncertainty_level == MarketUncertaintyLevel.LOW:
                allocation = self._low_uncertainty_allocation(current_portfolio)
                risk_params = {'max_volatility': 0.15, 'correlation_limit': 0.4}
            elif surface.uncertainty_level == MarketUncertaintyLevel.MEDIUM:
                allocation = self._medium_uncertainty_allocation(current_portfolio)
                risk_params = {'max_volatility': 0.25, 'correlation_limit': 0.3}
            elif surface.uncertainty_level == MarketUncertaintyLevel.HIGH:
                allocation = self._high_uncertainty_allocation(current_portfolio)
                risk_params = {'max_volatility': 0.35, 'correlation_limit': 0.2}
            else:  # EXTREME
                allocation = self._extreme_uncertainty_allocation(current_portfolio)
                risk_params = {'max_volatility': 0.45, 'correlation_limit': 0.1}
            
            # Calculate timing parameters
            timing_params = self._calculate_timing_parameters(surface)
            
            # Calculate liquidity requirements
            liquidity_reqs = self._calculate_liquidity_requirements(surface)
            
            # Create strategy
            strategy = CapitalAllocationStrategy(
                strategy_id=f"strategy_{surface.surface_id}",
                uncertainty_level=surface.uncertainty_level,
                asset_allocation=allocation,
                risk_parameters=risk_params,
                timing_parameters=timing_params,
                liquidity_requirements=liquidity_reqs,
                success_metrics=self._calculate_success_metrics(surface),
                backtest_performance={},  # Will be filled by backtesting
                metadata={'surface_id': surface.surface_id}
            )
            
            strategies.append(strategy)
        
        self.capital_strategies = strategies
        logger.info(f"✅ Created {len(strategies)} capital allocation strategies")
        return strategies
    
    def _low_uncertainty_allocation(self, current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """Allocation strategy for low uncertainty"""
        return {
            'cash': 0.15,
            'bonds': 0.25,
            'stocks': 0.45,
            'real_estate': 0.10,
            'commodities': 0.05
        }
    
    def _medium_uncertainty_allocation(self, current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """Allocation strategy for medium uncertainty"""
        return {
            'cash': 0.25,
            'bonds': 0.35,
            'stocks': 0.30,
            'real_estate': 0.05,
            'commodities': 0.05
        }
    
    def _high_uncertainty_allocation(self, current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """Allocation strategy for high uncertainty"""
        return {
            'cash': 0.40,
            'bonds': 0.40,
            'stocks': 0.15,
            'real_estate': 0.03,
            'commodities': 0.02
        }
    
    def _extreme_uncertainty_allocation(self, current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """Allocation strategy for extreme uncertainty"""
        return {
            'cash': 0.60,
            'bonds': 0.30,
            'stocks': 0.05,
            'real_estate': 0.03,
            'commodities': 0.02
        }
    
    def _calculate_timing_parameters(self, surface: MarketSurface) -> Dict[str, float]:
        """Calculate timing parameters based on market surface"""
        base_timing = {
            'rebalance_frequency': 30,  # days
            'entry_threshold': 0.7,
            'exit_threshold': 0.3,
            'position_sizing': 0.1
        }
        
        # Adjust based on uncertainty level
        if surface.uncertainty_level == MarketUncertaintyLevel.HIGH:
            base_timing['rebalance_frequency'] = 15
            base_timing['position_sizing'] = 0.05
        elif surface.uncertainty_level == MarketUncertaintyLevel.EXTREME:
            base_timing['rebalance_frequency'] = 7
            base_timing['position_sizing'] = 0.02
        
        return base_timing
    
    def _calculate_liquidity_requirements(self, surface: MarketSurface) -> Dict[str, float]:
        """Calculate liquidity requirements based on market surface"""
        return {
            'minimum_cash': 0.2 if surface.uncertainty_level in [MarketUncertaintyLevel.HIGH, MarketUncertaintyLevel.EXTREME] else 0.1,
            'emergency_fund': 0.1,
            'trading_buffer': 0.05,
            'withdrawal_capacity': 0.15
        }
    
    def _calculate_success_metrics(self, surface: MarketSurface) -> Dict[str, float]:
        """Calculate success metrics for a market surface"""
        # Convert uncertainty level to numeric value
        uncertainty_value = {
            MarketUncertaintyLevel.LOW: 0.1,
            MarketUncertaintyLevel.MEDIUM: 0.3,
            MarketUncertaintyLevel.HIGH: 0.5,
            MarketUncertaintyLevel.EXTREME: 0.7
        }.get(surface.uncertainty_level, 0.3)
        
        return {
            'expected_return': 0.08 - (uncertainty_value * 0.02),
            'risk_score': surface.volatility,
            'stability_score': 1 - surface.volatility,
            'liquidity_score': surface.liquidity_metrics['market_depth']
        }
    
    def run_fsqca_analysis(self, market_data: Dict[str, Any], 
                          portfolio_data: Dict[str, float]) -> fsQCAMarketUncertaintyResult:
        """
        Run comprehensive fsQCA analysis for market uncertainty decisions
        
        Args:
            market_data: Market data for analysis
            portfolio_data: Current portfolio data
            
        Returns:
            fsQCA analysis results for market uncertainty
        """
        logger.info("🔍 Running fsQCA analysis for market uncertainty decisions...")
        
        # Analyze market surfaces
        market_surfaces = self.analyze_market_uncertainty_surfaces(market_data)
        
        # Optimize capital allocation
        capital_strategies = self.optimize_capital_allocation(market_surfaces, portfolio_data)
        
        # Generate financial decisions
        optimal_decisions = self._generate_optimal_decisions(market_surfaces, capital_strategies)
        
        # Run backtesting
        backtest_results = self._run_backtesting(market_data, capital_strategies)
        
        # Prepare fsQCA data
        fsqca_data = self._prepare_fsqca_data(optimal_decisions, capital_strategies)
        
        # Run fsQCA analysis
        solution_coverage, solution_consistency = self._calculate_fsqca_solutions(fsqca_data)
        necessary_conditions = self._find_necessary_conditions(fsqca_data)
        sufficient_conditions = self._find_sufficient_conditions(fsqca_data)
        
        # Generate decision recommendations
        decision_recommendations = self._generate_decision_recommendations(
            optimal_decisions, capital_strategies, backtest_results
        )
        
        # Create analysis summary
        analysis_summary = {
            'total_surfaces': len(market_surfaces),
            'total_strategies': len(capital_strategies),
            'total_decisions': len(optimal_decisions),
            'uncertainty_distribution': self._calculate_uncertainty_distribution(market_surfaces),
            'backtest_performance': backtest_results.get('overall_performance', {}),
            'recommendation_confidence': decision_recommendations.get('confidence', 0.0)
        }
        
        result = fsQCAMarketUncertaintyResult(
            solution_coverage=solution_coverage,
            solution_consistency=solution_consistency,
            necessary_conditions=necessary_conditions,
            sufficient_conditions=sufficient_conditions,
            optimal_decisions=optimal_decisions,
            capital_allocation_strategies=capital_strategies,
            market_surfaces=market_surfaces,
            backtest_results=backtest_results,
            decision_recommendations=decision_recommendations,
            analysis_summary=analysis_summary
        )
        
        return result
    
    def _generate_optimal_decisions(self, market_surfaces: List[MarketSurface],
                                  capital_strategies: List[CapitalAllocationStrategy]) -> List[FinancialDecision]:
        """Generate optimal financial decisions based on market surfaces and strategies"""
        decisions = []
        
        for surface in market_surfaces:
            for opportunity in surface.decision_opportunities:
                # Find corresponding strategy
                strategy = next((s for s in capital_strategies if s.uncertainty_level == surface.uncertainty_level), None)
                
                if strategy:
                    decision = FinancialDecision(
                        decision_id=f"decision_{surface.surface_id}_{opportunity['type']}",
                        timestamp=datetime.now(),
                        decision_type=DecisionType(opportunity['type']),
                        market_conditions={
                            'volatility': surface.volatility,
                            'uncertainty_level': surface.uncertainty_level.value,
                            'liquidity_score': surface.liquidity_metrics['market_depth']
                        },
                        uncertainty_level=surface.uncertainty_level,
                        decision_parameters=opportunity['parameters'],
                        expected_outcome={
                            'return': strategy.success_metrics['expected_return'],
                            'risk': strategy.success_metrics['risk_score'],
                            'stability': strategy.success_metrics['stability_score']
                        },
                        confidence_score=opportunity['confidence']
                    )
                    decisions.append(decision)
        
        return decisions
    
    def _run_backtesting(self, market_data: Dict[str, Any], 
                        capital_strategies: List[CapitalAllocationStrategy]) -> Dict[str, Any]:
        """Run backtesting on capital allocation strategies"""
        logger.info("📊 Running backtesting on capital allocation strategies...")
        
        backtest_results = {
            'strategies': {},
            'overall_performance': {},
            'risk_metrics': {},
            'timing_analysis': {}
        }
        
        # Generate synthetic market data for backtesting
        synthetic_data = self._generate_synthetic_market_data(market_data)
        
        for strategy in capital_strategies:
            # Simulate strategy performance
            performance = self._simulate_strategy_performance(strategy, synthetic_data)
            
            backtest_results['strategies'][strategy.strategy_id] = performance
            strategy.backtest_performance = performance
        
        # Calculate overall performance metrics
        all_returns = []
        all_risks = []
        
        for strategy_perf in backtest_results['strategies'].values():
            all_returns.append(strategy_perf['total_return'])
            all_risks.append(strategy_perf['volatility'])
        
        backtest_results['overall_performance'] = {
            'avg_return': np.mean(all_returns),
            'avg_risk': np.mean(all_risks),
            'best_strategy': max(backtest_results['strategies'].keys(), 
                               key=lambda k: backtest_results['strategies'][k]['sharpe_ratio'])
        }
        
        return backtest_results
    
    def _generate_synthetic_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic market data for backtesting"""
        # Use existing market data as base and add synthetic scenarios
        base_prices = np.array(market_data.get('prices', [100, 105, 110, 108, 112]))
        base_volumes = np.array(market_data.get('volumes', [1000, 1100, 1200, 1150, 1250]))
        
        # Generate synthetic scenarios
        scenarios = []
        for i in range(10):  # 10 scenarios
            # Add random walk to prices
            price_changes = np.random.normal(0, 0.02, len(base_prices))
            synthetic_prices = base_prices * (1 + price_changes)
            
            # Add volatility changes
            volatility_changes = np.random.normal(0, 0.1, len(base_prices))
            synthetic_volatilities = np.clip(np.array(market_data.get('volatilities', [0.2])) + volatility_changes, 0.05, 0.8)
            
            scenarios.append({
                'prices': synthetic_prices,
                'volumes': base_volumes * (1 + np.random.normal(0, 0.1, len(base_volumes))),
                'volatilities': synthetic_volatilities,
                'correlations': np.random.uniform(0.1, 0.9, (len(base_prices), len(base_prices)))
            })
        
        return {'scenarios': scenarios, 'base_data': market_data}
    
    def _simulate_strategy_performance(self, strategy: CapitalAllocationStrategy, 
                                     synthetic_data: Dict[str, Any]) -> Dict[str, float]:
        """Simulate performance of a capital allocation strategy"""
        returns = []
        
        for scenario in synthetic_data['scenarios']:
            # Calculate portfolio return for this scenario
            portfolio_return = 0
            for asset, allocation in strategy.asset_allocation.items():
                if asset == 'cash':
                    asset_return = 0.02  # 2% cash return
                elif asset == 'bonds':
                    asset_return = 0.04  # 4% bond return
                elif asset == 'stocks':
                    asset_return = np.random.normal(0.08, 0.15)  # Stock return with volatility
                elif asset == 'real_estate':
                    asset_return = np.random.normal(0.06, 0.10)  # Real estate return
                else:  # commodities
                    asset_return = np.random.normal(0.05, 0.20)  # Commodity return
                
                portfolio_return += allocation * asset_return
            
            returns.append(portfolio_return)
        
        # Calculate performance metrics
        total_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean([1 if r > 0 else 0 for r in returns])
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _prepare_fsqca_data(self, decisions: List[FinancialDecision],
                           strategies: List[CapitalAllocationStrategy]) -> pd.DataFrame:
        """Prepare data for fsQCA analysis"""
        fsqca_data = []
        
        # If no decisions are available, create synthetic data based on strategies
        if not decisions and strategies:
            for strategy in strategies:
                # Create synthetic decision data based on strategy
                case = {
                    'high_uncertainty': 1 if strategy.uncertainty_level in [MarketUncertaintyLevel.HIGH, MarketUncertaintyLevel.EXTREME] else 0,
                    'high_volatility': 1 if strategy.success_metrics['risk_score'] > 0.3 else 0,
                    'low_liquidity': 1 if strategy.liquidity_requirements['minimum_cash'] > 0.2 else 0,
                    'defensive_allocation': 1 if strategy.asset_allocation.get('cash', 0) > 0.3 else 0,
                    'risk_management': 1 if strategy.risk_parameters.get('max_volatility', 1) < 0.3 else 0,
                    'timing_decision': 1 if strategy.timing_parameters.get('rebalance_frequency', 30) < 20 else 0,
                    'success_achieved': 1 if strategy.success_metrics['expected_return'] > 0.06 else 0
                }
                fsqca_data.append(case)
        else:
            # Use actual decisions
            for decision in decisions:
                # Create fsQCA case
                case = {
                    'high_uncertainty': 1 if decision.uncertainty_level in [MarketUncertaintyLevel.HIGH, MarketUncertaintyLevel.EXTREME] else 0,
                    'high_volatility': 1 if decision.market_conditions['volatility'] > 0.3 else 0,
                    'low_liquidity': 1 if decision.market_conditions['liquidity_score'] < 0.5 else 0,
                    'defensive_allocation': 1 if decision.decision_type == DecisionType.CAPITAL_ALLOCATION else 0,
                    'risk_management': 1 if decision.decision_type == DecisionType.RISK_MANAGEMENT else 0,
                    'timing_decision': 1 if decision.decision_type == DecisionType.TIMING else 0,
                    'success_achieved': 1 if decision.confidence_score > 0.7 else 0
                }
                fsqca_data.append(case)
        
        # If still no data, create default case
        if not fsqca_data:
            fsqca_data.append({
                'high_uncertainty': 0,
                'high_volatility': 0,
                'low_liquidity': 0,
                'defensive_allocation': 1,
                'risk_management': 1,
                'timing_decision': 0,
                'success_achieved': 1
            })
        
        return pd.DataFrame(fsqca_data)
    
    def _calculate_fsqca_solutions(self, fsqca_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate fsQCA solution coverage and consistency"""
        # Calculate solution coverage
        outcome_cases = fsqca_data[fsqca_data['success_achieved'] == 1]
        total_cases = len(fsqca_data)
        solution_coverage = len(outcome_cases) / total_cases if total_cases > 0 else 0
        
        # Calculate solution consistency
        if len(outcome_cases) > 0:
            consistency_scores = []
            for _, case in outcome_cases.iterrows():
                # Calculate consistency based on causal conditions
                causal_conditions = case[['high_uncertainty', 'high_volatility', 'low_liquidity', 
                                       'defensive_allocation', 'risk_management', 'timing_decision']]
                consistency_score = np.mean(causal_conditions)
                consistency_scores.append(consistency_score)
            
            solution_consistency = np.mean(consistency_scores)
        else:
            solution_consistency = 0.0
        
        return solution_coverage, solution_consistency
    
    def _find_necessary_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find necessary conditions for successful decisions"""
        necessary_conditions = {}
        
        outcome_cases = fsqca_data[fsqca_data['success_achieved'] == 1]
        
        if len(outcome_cases) > 0:
            for condition in fsqca_data.columns:
                if condition != 'success_achieved':
                    condition_present = outcome_cases[condition].sum()
                    necessity_score = condition_present / len(outcome_cases)
                    necessary_conditions[condition] = necessity_score
        
        return necessary_conditions
    
    def _find_sufficient_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find sufficient conditions for successful decisions"""
        sufficient_conditions = {}
        
        for condition in fsqca_data.columns:
            if condition != 'success_achieved':
                condition_cases = fsqca_data[fsqca_data[condition] == 1]
                outcome_cases = condition_cases[condition_cases['success_achieved'] == 1]
                
                if len(condition_cases) > 0:
                    sufficiency_score = len(outcome_cases) / len(condition_cases)
                    sufficient_conditions[condition] = sufficiency_score
                else:
                    sufficient_conditions[condition] = 0.0
        
        return sufficient_conditions
    
    def _generate_decision_recommendations(self, decisions: List[FinancialDecision],
                                         strategies: List[CapitalAllocationStrategy],
                                         backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate decision recommendations based on analysis"""
        recommendations = {
            'immediate_actions': [],
            'capital_allocation': {},
            'risk_management': {},
            'timing_recommendations': {},
            'confidence': 0.0
        }
        
        # Find best performing strategy
        best_strategy_id = backtest_results['overall_performance']['best_strategy']
        best_strategy = next((s for s in strategies if s.strategy_id == best_strategy_id), None)
        
        if best_strategy:
            recommendations['capital_allocation'] = best_strategy.asset_allocation
            recommendations['risk_management'] = best_strategy.risk_parameters
            recommendations['timing_recommendations'] = best_strategy.timing_parameters
        
        # Generate immediate actions
        high_confidence_decisions = [d for d in decisions if d.confidence_score > 0.8]
        for decision in high_confidence_decisions[:3]:  # Top 3 decisions
            recommendations['immediate_actions'].append({
                'action': decision.decision_type.value,
                'parameters': decision.decision_parameters,
                'confidence': decision.confidence_score,
                'expected_outcome': decision.expected_outcome
            })
        
        # Calculate overall confidence
        if decisions:
            recommendations['confidence'] = np.mean([d.confidence_score for d in decisions])
        
        return recommendations
    
    def _calculate_uncertainty_distribution(self, surfaces: List[MarketSurface]) -> Dict[str, int]:
        """Calculate distribution of uncertainty levels across surfaces"""
        distribution = {}
        for level in MarketUncertaintyLevel:
            distribution[level.value] = len([s for s in surfaces if s.uncertainty_level == level])
        return distribution
    
    def get_real_time_recommendations(self, current_market_data: Dict[str, Any],
                                    current_portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Get real-time decision recommendations for current market conditions"""
        # Run quick analysis
        result = self.run_fsqca_analysis(current_market_data, current_portfolio)
        
        return {
            'recommendations': result.decision_recommendations,
            'confidence': result.decision_recommendations['confidence'],
            'best_strategy': result.backtest_results['overall_performance']['best_strategy'],
            'market_surfaces': len(result.market_surfaces),
            'uncertainty_level': self._determine_current_uncertainty(current_market_data)
        }
    
    def _determine_current_uncertainty(self, market_data: Dict[str, Any]) -> str:
        """Determine current market uncertainty level"""
        current_volatility = np.mean(market_data.get('volatilities', [0.2]))
        
        if current_volatility < self.uncertainty_thresholds['low']:
            return 'low'
        elif current_volatility < self.uncertainty_thresholds['medium']:
            return 'medium'
        elif current_volatility < self.uncertainty_thresholds['high']:
            return 'high'
        else:
            return 'extreme' 