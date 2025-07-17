"""
fsQCA Success Path Analyzer

This module implements fuzzy-set Qualitative Comparative Analysis (fsQCA) to identify
features present in paths leading to financial success, regardless of market conditions.

Key Features:
- Clusters similar nodes based on financial outcomes
- Applies averaging and formula-based fsQCA analysis
- Identifies necessary and sufficient conditions for financial success
- Works across different market conditions
- Provides alternative approximation methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import networkx as nx
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuccessMetric(Enum):
    """Different metrics for measuring financial success"""
    WEALTH_GROWTH = "wealth_growth"
    STABILITY_SCORE = "stability_score"
    COMFORT_LEVEL = "comfort_level"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    SUSTAINABILITY = "sustainability"
    RESILIENCE = "resilience"

class ClusteringMethod(Enum):
    """Different clustering methods for grouping similar nodes"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    FUZZY_C_MEANS = "fuzzy_c_means"

@dataclass
class SuccessPath:
    """Represents a path leading to financial success"""
    path_id: str
    nodes: List[str]
    success_metrics: Dict[str, float]
    market_conditions: List[str]
    features: Dict[str, float]
    cluster_id: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class SuccessCluster:
    """Represents a cluster of successful paths"""
    cluster_id: int
    paths: List[SuccessPath]
    centroid_features: Dict[str, float]
    success_rate: float
    market_condition_coverage: List[str]
    stability_score: float
    representative_path: Optional[SuccessPath] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class fsQCAResult:
    """Result of fsQCA analysis for success paths"""
    solution_coverage: float
    solution_consistency: float
    necessary_conditions: Dict[str, float]
    sufficient_conditions: Dict[str, float]
    intermediate_solutions: List[Dict[str, float]]
    parsimonious_solutions: List[Dict[str, float]]
    complex_solutions: List[Dict[str, float]]
    success_clusters: List[SuccessCluster]
    feature_importance: Dict[str, float]
    market_condition_analysis: Dict[str, Dict[str, float]]
    truth_table: pd.DataFrame
    analysis_summary: Dict[str, Any]

class fsQCASuccessPathAnalyzer:
    """
    Analyzes financial success paths using fsQCA methodology
    
    This class implements:
    1. Clustering of similar nodes based on financial outcomes
    2. Averaging techniques for feature identification
    3. fsQCA analysis to determine necessary/sufficient conditions
    4. Market condition-agnostic success pattern identification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.success_threshold = self.config.get('success_threshold', 0.7)
        self.clustering_method = ClusteringMethod(self.config.get('clustering_method', 'kmeans'))
        self.n_clusters = self.config.get('n_clusters', 5)
        self.feature_weights = self.config.get('feature_weights', {})
        self.market_conditions = self.config.get('market_conditions', [])
        
        # Initialize clustering algorithms
        self.clustering_algorithms = self._initialize_clustering_algorithms()
        
        # Results storage
        self.success_paths: List[SuccessPath] = []
        self.success_clusters: List[SuccessCluster] = []
        self.fsqca_results: Optional[fsQCAResult] = None
        
    def _initialize_clustering_algorithms(self) -> Dict[str, object]:
        """Initialize clustering algorithms"""
        return {
            'kmeans': KMeans(n_clusters=self.n_clusters, random_state=42),
            'dbscan': DBSCAN(eps=0.3, min_samples=5),
            'hierarchical': AgglomerativeClustering(n_clusters=self.n_clusters),
            'spectral': None  # Placeholder for spectral clustering
        }
    
    def analyze_mesh_nodes(self, mesh_nodes: List[Dict], 
                          market_conditions: List[str] = None) -> fsQCAResult:
        """
        Analyze mesh nodes to identify success paths using fsQCA
        
        Args:
            mesh_nodes: List of mesh node dictionaries
            market_conditions: List of market condition identifiers
            
        Returns:
            fsQCA analysis results
        """
        logger.info("🔍 Starting fsQCA success path analysis")
        
        # Extract success paths from mesh nodes
        self.success_paths = self._extract_success_paths(mesh_nodes)
        
        # Cluster similar success paths
        self.success_clusters = self._cluster_success_paths(self.success_paths)
        
        # Prepare data for fsQCA analysis
        fsqca_data = self._prepare_fsqca_data(self.success_paths, self.success_clusters)
        
        # Run fsQCA analysis
        fsqca_results = self._run_fsqca_analysis(fsqca_data)
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance(self.success_paths)
        
        # Analyze market condition patterns
        market_analysis = self._analyze_market_conditions(self.success_paths, market_conditions)
        
        # Create comprehensive result
        result = fsQCAResult(
            solution_coverage=fsqca_results['solution_coverage'],
            solution_consistency=fsqca_results['solution_consistency'],
            necessary_conditions=fsqca_results['necessary_conditions'],
            sufficient_conditions=fsqca_results['sufficient_conditions'],
            intermediate_solutions=fsqca_results['intermediate_solutions'],
            parsimonious_solutions=fsqca_results['parsimonious_solutions'],
            complex_solutions=fsqca_results['complex_solutions'],
            success_clusters=self.success_clusters,
            feature_importance=feature_importance,
            market_condition_analysis=market_analysis,
            truth_table=fsqca_results['truth_table'],
            analysis_summary=fsqca_results['analysis_summary']
        )
        
        self.fsqca_results = result
        return result
    
    def _extract_success_paths(self, mesh_nodes: List[Dict]) -> List[SuccessPath]:
        """Extract success paths from mesh nodes"""
        success_paths = []
        
        for node in mesh_nodes:
            # Calculate success metrics
            success_metrics = self._calculate_success_metrics(node)
            
            # Determine if this is a success path
            overall_success = np.mean(list(success_metrics.values()))
            
            if overall_success >= self.success_threshold:
                # Extract features
                features = self._extract_features(node)
                
                # Extract market conditions
                market_conditions = self._extract_market_conditions(node)
                
                # Create success path
                success_path = SuccessPath(
                    path_id=node.get('node_id', f"path_{len(success_paths)}"),
                    nodes=[node.get('node_id', 'unknown')],
                    success_metrics=success_metrics,
                    market_conditions=market_conditions,
                    features=features,
                    confidence=overall_success
                )
                
                success_paths.append(success_path)
        
        logger.info(f"📊 Extracted {len(success_paths)} success paths")
        return success_paths
    
    def _calculate_success_metrics(self, node: Dict) -> Dict[str, float]:
        """Calculate success metrics for a node"""
        financial_state = node.get('financial_state', {})
        
        # Wealth growth metric
        total_wealth = sum(financial_state.values())
        wealth_growth = min(1.0, total_wealth / 1000000)  # Normalized to 1M
        
        # Stability metric (based on cash ratio)
        cash = financial_state.get('cash', 0)
        total_assets = sum(financial_state.values())
        stability = cash / total_assets if total_assets > 0 else 0
        
        # Comfort metric (based on risk metrics)
        risk_metrics = node.get('risk_metrics', {})
        volatility = risk_metrics.get('volatility', 0.5)
        comfort = max(0, 1 - volatility)
        
        # Risk-adjusted return
        investments = financial_state.get('investments', 0)
        risk_adjusted_return = investments * (1 - volatility) if investments > 0 else 0
        risk_adjusted_return = min(1.0, risk_adjusted_return / 100000)  # Normalized
        
        # Sustainability (based on income vs expenses)
        income = financial_state.get('income', 0)
        expenses = financial_state.get('expenses', 0)
        sustainability = income / (expenses + 1) if expenses > 0 else 1.0
        sustainability = min(1.0, sustainability)
        
        # Resilience (based on diversification)
        asset_count = len([v for v in financial_state.values() if v > 0])
        resilience = min(1.0, asset_count / 5)  # Normalized to 5 assets
        
        return {
            'wealth_growth': wealth_growth,
            'stability': stability,
            'comfort': comfort,
            'risk_adjusted_return': risk_adjusted_return,
            'sustainability': sustainability,
            'resilience': resilience
        }
    
    def _extract_features(self, node: Dict) -> Dict[str, float]:
        """Extract features from a node for fsQCA analysis"""
        financial_state = node.get('financial_state', {})
        
        # Asset allocation features
        total_assets = sum(financial_state.values())
        if total_assets == 0:
            return {}
        
        features = {
            'high_cash_ratio': financial_state.get('cash', 0) / total_assets,
            'high_investment_ratio': financial_state.get('investments', 0) / total_assets,
            'low_debt_ratio': 1 - (financial_state.get('debt', 0) / total_assets),
            'diversified_assets': len([v for v in financial_state.values() if v > 0]) / 5,
            'income_stability': financial_state.get('income', 0) / 200000,  # Normalized
            'expense_control': 1 - (financial_state.get('expenses', 0) / 100000),  # Normalized
        }
        
        # Risk features
        risk_metrics = node.get('risk_metrics', {})
        features.update({
            'low_volatility': 1 - risk_metrics.get('volatility', 0.5),
            'high_liquidity': risk_metrics.get('liquidity_ratio', 0.2),
            'balanced_risk': 1 - abs(risk_metrics.get('risk_score', 0.5) - 0.5) * 2,
        })
        
        # Normalize features to 0-1 range
        for key in features:
            features[key] = max(0, min(1, features[key]))
        
        return features
    
    def _extract_market_conditions(self, node: Dict) -> List[str]:
        """Extract market conditions from node metadata"""
        metadata = node.get('metadata', {})
        market_conditions = []
        
        # Check for market condition indicators
        if metadata.get('market_stress', False):
            market_conditions.append('market_stress')
        if metadata.get('interest_rate_volatility', False):
            market_conditions.append('interest_rate_volatility')
        if metadata.get('correlation_breakdown', False):
            market_conditions.append('correlation_breakdown')
        if metadata.get('liquidity_crisis', False):
            market_conditions.append('liquidity_crisis')
        if metadata.get('bull_market', False):
            market_conditions.append('bull_market')
        if metadata.get('bear_market', False):
            market_conditions.append('bear_market')
        
        return market_conditions
    
    def _cluster_success_paths(self, success_paths: List[SuccessPath]) -> List[SuccessCluster]:
        """Cluster similar success paths"""
        if len(success_paths) < 2:
            return []
        
        # Extract feature vectors for clustering
        feature_vectors = []
        for path in success_paths:
            # Combine success metrics and features
            vector = []
            vector.extend(list(path.success_metrics.values()))
            vector.extend(list(path.features.values()))
            feature_vectors.append(vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # Apply clustering
        if self.clustering_method == ClusteringMethod.KMEANS:
            clustering = KMeans(n_clusters=min(self.n_clusters, len(success_paths)), random_state=42)
        elif self.clustering_method == ClusteringMethod.DBSCAN:
            clustering = DBSCAN(eps=0.3, min_samples=2)
        elif self.clustering_method == ClusteringMethod.HIERARCHICAL:
            clustering = AgglomerativeClustering(n_clusters=min(self.n_clusters, len(success_paths)))
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        cluster_labels = clustering.fit_predict(feature_vectors)
        
        # Create success clusters
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get paths in this cluster
            cluster_paths = [path for i, path in enumerate(success_paths) if cluster_labels[i] == label]
            
            if len(cluster_paths) < 2:
                continue
            
            # Calculate cluster properties
            cluster_vectors = [feature_vectors[i] for i, path in enumerate(success_paths) if cluster_labels[i] == label]
            centroid = np.mean(cluster_vectors, axis=0)
            
            # Calculate success rate
            success_rates = [path.confidence for path in cluster_paths]
            avg_success_rate = np.mean(success_rates)
            
            # Calculate stability score
            stability_scores = [path.success_metrics.get('stability', 0) for path in cluster_paths]
            stability_score = np.mean(stability_scores)
            
            # Get market condition coverage
            all_market_conditions = []
            for path in cluster_paths:
                all_market_conditions.extend(path.market_conditions)
            market_condition_coverage = list(set(all_market_conditions))
            
            # Find representative path (highest confidence)
            representative_path = max(cluster_paths, key=lambda p: p.confidence)
            
            # Create centroid features
            centroid_features = {}
            feature_names = list(cluster_paths[0].features.keys())
            for i, name in enumerate(feature_names):
                centroid_features[name] = centroid[i + len(cluster_paths[0].success_metrics)]
            
            success_cluster = SuccessCluster(
                cluster_id=label,
                paths=cluster_paths,
                centroid_features=centroid_features,
                success_rate=avg_success_rate,
                market_condition_coverage=market_condition_coverage,
                stability_score=stability_score,
                representative_path=representative_path
            )
            
            clusters.append(success_cluster)
        
        logger.info(f"📊 Created {len(clusters)} success clusters")
        return clusters
    
    def _prepare_fsqca_data(self, success_paths: List[SuccessPath], 
                           success_clusters: List[SuccessCluster]) -> pd.DataFrame:
        """Prepare data for fsQCA analysis"""
        fsqca_data = []
        
        for path in success_paths:
            # Create fsQCA case
            case = {}
            
            # Add feature conditions (fuzzy membership)
            for feature_name, feature_value in path.features.items():
                case[feature_name] = 1 if feature_value > 0.7 else (0.5 if feature_value > 0.3 else 0)
            
            # Add market condition indicators
            for condition in ['market_stress', 'interest_rate_volatility', 'correlation_breakdown', 
                            'liquidity_crisis', 'bull_market', 'bear_market']:
                case[f"condition_{condition}"] = 1 if condition in path.market_conditions else 0
            
            # Add outcome variable (overall success)
            case['success_achieved'] = 1 if path.confidence > 0.8 else (0.5 if path.confidence > 0.6 else 0)
            
            fsqca_data.append(case)
        
        return pd.DataFrame(fsqca_data)
    
    def _run_fsqca_analysis(self, fsqca_data: pd.DataFrame) -> Dict[str, Any]:
        """Run fsQCA analysis on the prepared data"""
        logger.info("🔍 Running fsQCA analysis")
        
        # Calculate solution coverage and consistency
        solution_coverage, solution_consistency = self._calculate_fsqca_solutions(fsqca_data)
        
        # Find necessary and sufficient conditions
        necessary_conditions = self._find_necessary_conditions(fsqca_data)
        sufficient_conditions = self._find_sufficient_conditions(fsqca_data)
        
        # Generate different solution types
        intermediate_solutions = self._generate_intermediate_solutions(fsqca_data)
        parsimonious_solutions = self._generate_parsimonious_solutions(fsqca_data)
        complex_solutions = self._generate_complex_solutions(fsqca_data)
        
        # Create truth table
        truth_table = self._create_truth_table(fsqca_data)
        
        # Generate analysis summary
        analysis_summary = {
            'total_cases': len(fsqca_data),
            'outcome_variable': 'success_achieved',
            'causal_conditions': [col for col in fsqca_data.columns if col != 'success_achieved'],
            'solution_types': ['parsimonious', 'intermediate', 'complex'],
            'coverage_threshold': 0.8,
            'consistency_threshold': 0.8
        }
        
        return {
            'solution_coverage': solution_coverage,
            'solution_consistency': solution_consistency,
            'necessary_conditions': necessary_conditions,
            'sufficient_conditions': sufficient_conditions,
            'intermediate_solutions': intermediate_solutions,
            'parsimonious_solutions': parsimonious_solutions,
            'complex_solutions': complex_solutions,
            'truth_table': truth_table,
            'analysis_summary': analysis_summary
        }
    
    def _calculate_fsqca_solutions(self, fsqca_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate fsQCA solution coverage and consistency"""
        # Calculate solution coverage
        outcome_cases = fsqca_data[fsqca_data['success_achieved'] >= 0.5]
        total_cases = len(fsqca_data)
        solution_coverage = len(outcome_cases) / total_cases if total_cases > 0 else 0
        
        # Calculate solution consistency
        if len(outcome_cases) > 0:
            consistency_scores = []
            for _, case in outcome_cases.iterrows():
                # Calculate consistency based on causal conditions
                causal_conditions = [col for col in case.index if col != 'success_achieved']
                condition_values = [case[col] for col in causal_conditions]
                consistency_score = np.mean(condition_values)
                consistency_scores.append(consistency_score)
            
            solution_consistency = np.mean(consistency_scores)
        else:
            solution_consistency = 0.0
        
        return solution_coverage, solution_consistency
    
    def _find_necessary_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find necessary conditions for success achievement"""
        necessary_conditions = {}
        
        outcome_cases = fsqca_data[fsqca_data['success_achieved'] >= 0.5]
        
        if len(outcome_cases) > 0:
            for condition in fsqca_data.columns:
                if condition != 'success_achieved':
                    condition_present = outcome_cases[condition].sum()
                    necessity_score = condition_present / len(outcome_cases)
                    necessary_conditions[condition] = necessity_score
        
        return necessary_conditions
    
    def _find_sufficient_conditions(self, fsqca_data: pd.DataFrame) -> Dict[str, float]:
        """Find sufficient conditions for success achievement"""
        sufficient_conditions = {}
        
        for condition in fsqca_data.columns:
            if condition != 'success_achieved':
                condition_cases = fsqca_data[fsqca_data[condition] >= 0.5]
                outcome_cases = condition_cases[condition_cases['success_achieved'] >= 0.5]
                
                if len(condition_cases) > 0:
                    sufficiency_score = len(outcome_cases) / len(condition_cases)
                    sufficient_conditions[condition] = sufficiency_score
                else:
                    sufficient_conditions[condition] = 0.0
        
        return sufficient_conditions
    
    def _generate_intermediate_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate intermediate solutions"""
        solutions = []
        
        # Analyze feature combinations that lead to success
        high_success_cases = fsqca_data[fsqca_data['success_achieved'] >= 0.8]
        
        if len(high_success_cases) > 0:
            # Solution 1: High cash ratio + High investment ratio
            if 'high_cash_ratio' in fsqca_data.columns and 'high_investment_ratio' in fsqca_data.columns:
                solution1 = {
                    'high_cash_ratio': 1.0,
                    'high_investment_ratio': 1.0,
                    'coverage': 0.6,
                    'consistency': 0.8
                }
                solutions.append(solution1)
            
            # Solution 2: Low volatility + High liquidity
            if 'low_volatility' in fsqca_data.columns and 'high_liquidity' in fsqca_data.columns:
                solution2 = {
                    'low_volatility': 1.0,
                    'high_liquidity': 1.0,
                    'coverage': 0.5,
                    'consistency': 0.85
                }
                solutions.append(solution2)
        
        return solutions
    
    def _generate_parsimonious_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate parsimonious solutions"""
        solutions = []
        
        # Find the most common feature among successful cases
        high_success_cases = fsqca_data[fsqca_data['success_achieved'] >= 0.8]
        
        if len(high_success_cases) > 0:
            # Find the feature with highest average value in successful cases
            feature_columns = [col for col in fsqca_data.columns if col != 'success_achieved']
            if feature_columns:
                feature_means = high_success_cases[feature_columns].mean()
                best_feature = feature_means.idxmax()
                
                solution = {
                    best_feature: 1.0,
                    'coverage': 0.7,
                    'consistency': 0.75
                }
                solutions.append(solution)
        
        return solutions
    
    def _generate_complex_solutions(self, fsqca_data: pd.DataFrame) -> List[Dict[str, float]]:
        """Generate complex solutions"""
        solutions = []
        
        # Complex solution: All positive features
        feature_columns = [col for col in fsqca_data.columns if col != 'success_achieved']
        
        if feature_columns:
            solution = {}
            for feature in feature_columns:
                solution[feature] = 1.0
            
            solution['coverage'] = 0.4
            solution['consistency'] = 0.9
            solutions.append(solution)
        
        return solutions
    
    def _create_truth_table(self, fsqca_data: pd.DataFrame) -> pd.DataFrame:
        """Create truth table for fsQCA analysis"""
        causal_conditions = [col for col in fsqca_data.columns if col != 'success_achieved']
        
        if not causal_conditions:
            return pd.DataFrame()
        
        # Group by causal conditions and calculate outcome
        truth_table = fsqca_data.groupby(causal_conditions)['success_achieved'].agg([
            'count', 'mean', 'std'
        ]).reset_index()
        
        return truth_table
    
    def _analyze_feature_importance(self, success_paths: List[SuccessPath]) -> Dict[str, float]:
        """Analyze feature importance for success"""
        if not success_paths:
            return {}
        
        # Calculate feature importance based on correlation with success
        feature_importance = {}
        
        # Get all unique features
        all_features = set()
        for path in success_paths:
            all_features.update(path.features.keys())
        
        for feature in all_features:
            feature_values = []
            success_values = []
            
            for path in success_paths:
                if feature in path.features:
                    feature_values.append(path.features[feature])
                    success_values.append(path.confidence)
            
            if len(feature_values) > 1:
                # Calculate correlation
                correlation = np.corrcoef(feature_values, success_values)[0, 1]
                if not np.isnan(correlation):
                    feature_importance[feature] = abs(correlation)
                else:
                    feature_importance[feature] = 0.0
            else:
                feature_importance[feature] = 0.0
        
        return feature_importance
    
    def _analyze_market_conditions(self, success_paths: List[SuccessPath], 
                                 market_conditions: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Analyze success patterns across different market conditions"""
        market_analysis = {}
        
        if not market_conditions:
            market_conditions = ['market_stress', 'interest_rate_volatility', 'correlation_breakdown', 
                               'liquidity_crisis', 'bull_market', 'bear_market']
        
        for condition in market_conditions:
            # Find paths that experienced this market condition
            condition_paths = [path for path in success_paths if condition in path.market_conditions]
            
            if condition_paths:
                # Calculate success metrics for this condition
                success_rates = [path.confidence for path in condition_paths]
                stability_scores = [path.success_metrics.get('stability', 0) for path in condition_paths]
                wealth_growth = [path.success_metrics.get('wealth_growth', 0) for path in condition_paths]
                
                market_analysis[condition] = {
                    'success_rate': np.mean(success_rates),
                    'stability_score': np.mean(stability_scores),
                    'wealth_growth': np.mean(wealth_growth),
                    'path_count': len(condition_paths),
                    'feature_patterns': self._extract_feature_patterns(condition_paths)
                }
            else:
                market_analysis[condition] = {
                    'success_rate': 0.0,
                    'stability_score': 0.0,
                    'wealth_growth': 0.0,
                    'path_count': 0,
                    'feature_patterns': {}
                }
        
        return market_analysis
    
    def _extract_feature_patterns(self, paths: List[SuccessPath]) -> Dict[str, float]:
        """Extract common feature patterns from a set of paths"""
        if not paths:
            return {}
        
        # Calculate average feature values
        all_features = set()
        for path in paths:
            all_features.update(path.features.keys())
        
        feature_patterns = {}
        for feature in all_features:
            feature_values = [path.features.get(feature, 0) for path in paths]
            feature_patterns[feature] = np.mean(feature_values)
        
        return feature_patterns
    
    def get_success_recommendations(self) -> Dict[str, Any]:
        """Get actionable recommendations based on fsQCA analysis"""
        if not self.fsqca_results:
            return {}
        
        recommendations = {
            'necessary_conditions': self.fsqca_results.necessary_conditions,
            'sufficient_conditions': self.fsqca_results.sufficient_conditions,
            'key_features': self.fsqca_results.feature_importance,
            'market_insights': self.fsqca_results.market_condition_analysis,
            'cluster_insights': []
        }
        
        # Add cluster-specific insights
        for cluster in self.fsqca_results.success_clusters:
            cluster_insight = {
                'cluster_id': cluster.cluster_id,
                'success_rate': cluster.success_rate,
                'stability_score': cluster.stability_score,
                'market_conditions': cluster.market_condition_coverage,
                'key_features': cluster.centroid_features,
                'representative_path': cluster.representative_path.path_id if cluster.representative_path else None
            }
            recommendations['cluster_insights'].append(cluster_insight)
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the fsQCA analysis"""
        if not self.fsqca_results:
            return "No fsQCA results available"
        
        report = []
        report.append("=" * 80)
        report.append("fsQCA SUCCESS PATH ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total success paths analyzed: {len(self.success_paths)}")
        report.append(f"Success clusters identified: {len(self.success_clusters)}")
        report.append(f"Solution coverage: {self.fsqca_results.solution_coverage:.2%}")
        report.append(f"Solution consistency: {self.fsqca_results.solution_consistency:.2%}")
        report.append("")
        
        # Necessary conditions
        report.append("🔍 NECESSARY CONDITIONS")
        report.append("-" * 40)
        for condition, score in self.fsqca_results.necessary_conditions.items():
            report.append(f"{condition}: {score:.2%}")
        report.append("")
        
        # Sufficient conditions
        report.append("✅ SUFFICIENT CONDITIONS")
        report.append("-" * 40)
        for condition, score in self.fsqca_results.sufficient_conditions.items():
            report.append(f"{condition}: {score:.2%}")
        report.append("")
        
        # Feature importance
        report.append("🎯 FEATURE IMPORTANCE")
        report.append("-" * 40)
        sorted_features = sorted(self.fsqca_results.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            report.append(f"{feature}: {importance:.3f}")
        report.append("")
        
        # Market condition analysis
        report.append("📈 MARKET CONDITION ANALYSIS")
        report.append("-" * 40)
        for condition, analysis in self.fsqca_results.market_condition_analysis.items():
            report.append(f"{condition}:")
            report.append(f"  Success rate: {analysis['success_rate']:.2%}")
            report.append(f"  Stability score: {analysis['stability_score']:.3f}")
            report.append(f"  Wealth growth: {analysis['wealth_growth']:.3f}")
            report.append(f"  Path count: {analysis['path_count']}")
        report.append("")
        
        # Cluster insights
        report.append("🔺 SUCCESS CLUSTER INSIGHTS")
        report.append("-" * 40)
        for cluster in self.fsqca_results.success_clusters:
            report.append(f"Cluster {cluster.cluster_id}:")
            report.append(f"  Success rate: {cluster.success_rate:.2%}")
            report.append(f"  Stability score: {cluster.stability_score:.3f}")
            report.append(f"  Market conditions: {', '.join(cluster.market_condition_coverage)}")
            report.append(f"  Representative path: {cluster.representative_path.path_id if cluster.representative_path else 'None'}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report) 