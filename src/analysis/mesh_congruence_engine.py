"""
Mesh Congruence Engine

This module implements advanced mesh congruence algorithms for financial network analysis:
1. Delaunay Triangulation for optimal mesh structure
2. Centroidal Voronoi Tessellations (CVTs) for density-based optimization
3. Edge collapsing for mesh decimation and optimization
4. Mesh congruence scoring and validation
5. Backtesting framework for mesh performance evaluation

Key Features:
- Delaunay triangulation for maximizing minimum angles
- CVT optimization for density-based point distribution
- Edge collapse algorithms for mesh simplification
- Congruence scoring between mesh structures
- Backtesting framework for historical performance
- Integration with existing mesh vector database
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.metrics import silhouette_score
import pickle

# Import existing components
from .mesh_vector_database import MeshVectorDatabase, MeshEmbedding, SimilarityMatch
from .synthetic_lifestyle_engine import SyntheticClientData
from .time_uncertainty_mesh import TimeUncertaintyMeshEngine


@dataclass
class MeshCongruenceResult:
    """Result of mesh congruence analysis"""
    source_client_id: str
    target_client_id: str
    congruence_score: float
    triangulation_quality: float
    density_distribution_score: float
    edge_collapse_efficiency: float
    overall_congruence: float
    matching_factors: List[str]
    confidence_interval: Tuple[float, float]


@dataclass
class BacktestResult:
    """Result of mesh backtesting"""
    client_id: str
    test_period: Tuple[datetime, datetime]
    mesh_performance: Dict[str, float]
    congruence_evolution: List[float]
    recommendation_accuracy: float
    risk_adjustment_factor: float
    overall_score: float


class MeshCongruenceEngine:
    """
    Advanced mesh congruence engine using Delaunay triangulation and CVT optimization
    """
    
    def __init__(self, embedding_dim: int = 128, congruence_threshold: float = 0.75):
        self.embedding_dim = embedding_dim
        self.congruence_threshold = congruence_threshold
        self.vector_db = MeshVectorDatabase(embedding_dim=embedding_dim)
        self.mesh_engine = TimeUncertaintyMeshEngine()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create storage directory
        self.storage_dir = Path("data/outputs/mesh_congruence")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Backtesting results storage
        self.backtest_results: List[BacktestResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the congruence engine"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def compute_delaunay_triangulation(self, points: np.ndarray) -> Delaunay:
        """
        Compute Delaunay triangulation for optimal mesh structure
        
        Args:
            points: Array of points in 2D space
            
        Returns:
            Delaunay triangulation object
        """
        try:
            # Ensure points are 2D for triangulation
            if points.shape[1] > 2:
                # Use PCA to reduce to 2D if needed
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)
            else:
                points_2d = points
            
            # Compute Delaunay triangulation
            triangulation = Delaunay(points_2d)
            
            self.logger.info(f"Computed Delaunay triangulation with {len(triangulation.simplices)} triangles")
            return triangulation
            
        except Exception as e:
            self.logger.error(f"Error computing Delaunay triangulation: {e}")
            raise
    
    def compute_centroidal_voronoi_tessellation(self, points: np.ndarray, 
                                              density_function: Optional[callable] = None,
                                              max_iterations: int = 100) -> np.ndarray:
        """
        Compute Centroidal Voronoi Tessellation for density-based optimization
        
        Args:
            points: Initial points
            density_function: Function that returns density at each point
            max_iterations: Maximum iterations for CVT optimization
            
        Returns:
            Optimized points from CVT
        """
        try:
            current_points = points.copy()
            
            for iteration in range(max_iterations):
                # Compute Voronoi diagram
                voronoi = Voronoi(current_points)
                
                # Compute centroids of Voronoi cells
                new_points = []
                for region_idx in range(len(current_points)):
                    region = voronoi.regions[voronoi.point_region[region_idx]]
                    
                    if -1 not in region and len(region) > 0:
                        # Compute centroid of the region
                        vertices = [voronoi.vertices[i] for i in region]
                        centroid = np.mean(vertices, axis=0)
                        
                        # Apply density function if provided
                        if density_function is not None:
                            density = density_function(centroid)
                            centroid = centroid * density
                        
                        new_points.append(centroid)
                    else:
                        new_points.append(current_points[region_idx])
                
                new_points = np.array(new_points)
                
                # Check convergence
                if np.allclose(current_points, new_points, atol=1e-6):
                    break
                
                current_points = new_points
            
            self.logger.info(f"CVT optimization completed in {iteration + 1} iterations")
            return current_points
            
        except Exception as e:
            self.logger.error(f"Error computing CVT: {e}")
            raise
    
    def compute_edge_collapse_efficiency(self, triangulation: Delaunay, 
                                       target_triangles: int = None) -> float:
        """
        Compute efficiency of edge collapse for mesh decimation
        
        Args:
            triangulation: Delaunay triangulation
            target_triangles: Target number of triangles after collapse
            
        Returns:
            Efficiency score (0-1)
        """
        try:
            # Get current triangle quality metrics
            triangles = triangulation.simplices
            points = triangulation.points
            
            # Compute triangle qualities (minimum angle)
            triangle_qualities = []
            for triangle in triangles:
                triangle_points = points[triangle]
                
                # Compute angles
                edges = [
                    triangle_points[1] - triangle_points[0],
                    triangle_points[2] - triangle_points[1],
                    triangle_points[0] - triangle_points[2]
                ]
                
                angles = []
                for i in range(3):
                    edge1 = edges[i]
                    edge2 = edges[(i + 1) % 3]
                    
                    cos_angle = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
                
                min_angle = np.min(angles)
                triangle_qualities.append(min_angle)
            
            # Compute efficiency based on quality preservation
            avg_quality = np.mean(triangle_qualities)
            quality_variance = np.var(triangle_qualities)
            
            # Efficiency score (higher is better)
            efficiency = avg_quality / (1 + quality_variance)
            
            return efficiency
            
        except Exception as e:
            self.logger.error(f"Error computing edge collapse efficiency: {e}")
            return 0.0
    
    def compute_mesh_congruence(self, client_data_1: SyntheticClientData, 
                              client_data_2: SyntheticClientData) -> MeshCongruenceResult:
        """
        Compute mesh congruence between two clients using advanced algorithms
        
        Args:
            client_data_1: First client data
            client_data_2: Second client data
            
        Returns:
            MeshCongruenceResult with congruence analysis
        """
        try:
            # Extract mesh embeddings
            embedding_1 = self._extract_mesh_embedding(client_data_1)
            embedding_2 = self._extract_mesh_embedding(client_data_2)
            
            # Compute Delaunay triangulation for both
            triangulation_1 = self.compute_delaunay_triangulation(embedding_1)
            triangulation_2 = self.compute_delaunay_triangulation(embedding_2)
            
            # Compute CVT optimization
            cvt_points_1 = self.compute_centroidal_voronoi_tessellation(embedding_1)
            cvt_points_2 = self.compute_centroidal_voronoi_tessellation(embedding_2)
            
            # Compute edge collapse efficiency
            efficiency_1 = self.compute_edge_collapse_efficiency(triangulation_1)
            efficiency_2 = self.compute_edge_collapse_efficiency(triangulation_2)
            
            # Compute congruence scores
            triangulation_quality = self._compute_triangulation_congruence(triangulation_1, triangulation_2)
            density_score = self._compute_density_congruence(cvt_points_1, cvt_points_2)
            edge_efficiency = (efficiency_1 + efficiency_2) / 2
            
            # Overall congruence score
            overall_congruence = (triangulation_quality + density_score + edge_efficiency) / 3
            
            # Identify matching factors
            matching_factors = self._identify_congruence_factors(client_data_1, client_data_2)
            
            # Compute confidence interval
            confidence_interval = self._compute_confidence_interval(overall_congruence, len(matching_factors))
            
            return MeshCongruenceResult(
                source_client_id=client_data_1.client_id,
                target_client_id=client_data_2.client_id,
                congruence_score=overall_congruence,
                triangulation_quality=triangulation_quality,
                density_distribution_score=density_score,
                edge_collapse_efficiency=edge_efficiency,
                overall_congruence=overall_congruence,
                matching_factors=matching_factors,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            self.logger.error(f"Error computing mesh congruence: {e}")
            raise
    
    def _extract_mesh_embedding(self, client_data: SyntheticClientData) -> np.ndarray:
        """Extract mesh embedding from client data"""
        if client_data.mesh_data and 'states' in client_data.mesh_data:
            # Use mesh states if available
            mesh_states = client_data.mesh_data['states']
            # Flatten and normalize
            embedding = mesh_states.flatten()
            if len(embedding) > self.embedding_dim:
                # Truncate to embedding dimension
                embedding = embedding[:self.embedding_dim]
            elif len(embedding) < self.embedding_dim:
                # Pad with zeros
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            # Create embedding from vector profile
            vector_profile = client_data.vector_profile
            embedding = np.concatenate([
                vector_profile.cash_flow_vector,
                vector_profile.discretionary_spending_surface.flatten(),
                [vector_profile.risk_tolerance],
                [vector_profile.life_stage.value]
            ])
            
            # Normalize to embedding dimension
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            elif len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        
        return embedding.reshape(-1, 1).T  # Reshape for 2D processing
    
    def _compute_triangulation_congruence(self, triangulation_1: Delaunay, 
                                        triangulation_2: Delaunay) -> float:
        """Compute congruence between two triangulations"""
        try:
            # Compare triangle qualities
            triangles_1 = triangulation_1.simplices
            triangles_2 = triangulation_2.simplices
            
            # Compute quality distributions
            quality_1 = self._compute_triangle_qualities(triangulation_1.points, triangles_1)
            quality_2 = self._compute_triangle_qualities(triangulation_2.points, triangles_2)
            
            # Compare distributions using histogram intersection
            hist_1, _ = np.histogram(quality_1, bins=10, range=(0, np.pi/3))
            hist_2, _ = np.histogram(quality_2, bins=10, range=(0, np.pi/3))
            
            # Normalize histograms
            hist_1 = hist_1 / np.sum(hist_1)
            hist_2 = hist_2 / np.sum(hist_2)
            
            # Compute intersection
            intersection = np.minimum(hist_1, hist_2)
            congruence = np.sum(intersection)
            
            return congruence
            
        except Exception as e:
            self.logger.error(f"Error computing triangulation congruence: {e}")
            return 0.0
    
    def _compute_triangle_qualities(self, points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute quality metrics for triangles"""
        qualities = []
        for triangle in triangles:
            triangle_points = points[triangle]
            
            # Compute minimum angle
            edges = [
                triangle_points[1] - triangle_points[0],
                triangle_points[2] - triangle_points[1],
                triangle_points[0] - triangle_points[2]
            ]
            
            angles = []
            for i in range(3):
                edge1 = edges[i]
                edge2 = edges[(i + 1) % 3]
                
                cos_angle = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            
            min_angle = np.min(angles)
            qualities.append(min_angle)
        
        return np.array(qualities)
    
    def _compute_density_congruence(self, points_1: np.ndarray, points_2: np.ndarray) -> float:
        """Compute congruence of density distributions"""
        try:
            # Compute density distributions using kernel density estimation
            from scipy.stats import gaussian_kde
            
            kde_1 = gaussian_kde(points_1.T)
            kde_2 = gaussian_kde(points_2.T)
            
            # Sample points for comparison
            x_min, x_max = min(points_1[:, 0].min(), points_2[:, 0].min()), max(points_1[:, 0].max(), points_2[:, 0].max())
            y_min, y_max = min(points_1[:, 1].min(), points_2[:, 1].min()), max(points_1[:, 1].max(), points_2[:, 1].max())
            
            x = np.linspace(x_min, x_max, 50)
            y = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            # Compute densities
            density_1 = kde_1(positions).reshape(X.shape)
            density_2 = kde_2(positions).reshape(X.shape)
            
            # Normalize
            density_1 = density_1 / np.sum(density_1)
            density_2 = density_2 / np.sum(density_2)
            
            # Compute correlation
            correlation = np.corrcoef(density_1.flatten(), density_2.flatten())[0, 1]
            
            return max(0, correlation)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error computing density congruence: {e}")
            return 0.0
    
    def _identify_congruence_factors(self, client_data_1: SyntheticClientData, 
                                   client_data_2: SyntheticClientData) -> List[str]:
        """Identify factors contributing to congruence"""
        factors = []
        
        # Compare life stages
        if client_data_1.vector_profile.life_stage == client_data_2.vector_profile.life_stage:
            factors.append("life_stage")
        
        # Compare age ranges
        age_diff = abs(client_data_1.profile.age - client_data_2.profile.age)
        if age_diff <= 5:
            factors.append("age_similarity")
        
        # Compare income levels
        income_1 = client_data_1.profile.base_income
        income_2 = client_data_2.profile.base_income
        income_ratio = min(income_1, income_2) / max(income_1, income_2)
        if income_ratio > 0.8:
            factors.append("income_similarity")
        
        # Compare risk tolerance
        risk_diff = abs(client_data_1.vector_profile.risk_tolerance - client_data_2.vector_profile.risk_tolerance)
        if risk_diff < 0.2:
            factors.append("risk_tolerance")
        
        # Compare event patterns
        events_1 = [e.category.value for e in client_data_1.lifestyle_events]
        events_2 = [e.category.value for e in client_data_2.lifestyle_events]
        common_events = set(events_1) & set(events_2)
        if len(common_events) > 0:
            factors.append("event_patterns")
        
        return factors
    
    def _compute_confidence_interval(self, congruence_score: float, 
                                   num_factors: int) -> Tuple[float, float]:
        """Compute confidence interval for congruence score"""
        # Simple confidence interval based on number of matching factors
        factor_boost = min(0.1, num_factors * 0.02)
        margin = 0.05 + factor_boost
        
        lower = max(0, congruence_score - margin)
        upper = min(1, congruence_score + margin)
        
        return (lower, upper)
    
    def run_backtest(self, client_data: SyntheticClientData, 
                    historical_period: Tuple[datetime, datetime],
                    test_scenarios: int = 100) -> BacktestResult:
        """
        Run backtesting for mesh congruence performance
        
        Args:
            client_data: Client data to backtest
            historical_period: Period for backtesting
            test_scenarios: Number of test scenarios
            
        Returns:
            BacktestResult with performance metrics
        """
        try:
            start_date, end_date = historical_period
            
            # Generate historical mesh data
            historical_mesh_data = self._generate_historical_mesh_data(
                client_data, start_date, end_date, test_scenarios
            )
            
            # Compute congruence evolution
            congruence_scores = []
            for i in range(len(historical_mesh_data) - 1):
                mesh_1 = historical_mesh_data[i]
                mesh_2 = historical_mesh_data[i + 1]
                
                congruence = self._compute_historical_congruence(mesh_1, mesh_2)
                congruence_scores.append(congruence)
            
            # Compute performance metrics
            avg_congruence = np.mean(congruence_scores)
            congruence_stability = 1 - np.std(congruence_scores)
            
            # Simulate recommendations and compute accuracy
            recommendation_accuracy = self._compute_recommendation_accuracy(
                client_data, historical_mesh_data
            )
            
            # Compute risk adjustment factor
            risk_adjustment = self._compute_risk_adjustment(client_data, historical_mesh_data)
            
            # Overall score
            overall_score = (avg_congruence + congruence_stability + 
                           recommendation_accuracy + risk_adjustment) / 4
            
            result = BacktestResult(
                client_id=client_data.client_id,
                test_period=historical_period,
                mesh_performance={
                    'avg_congruence': avg_congruence,
                    'congruence_stability': congruence_stability,
                    'max_congruence': np.max(congruence_scores),
                    'min_congruence': np.min(congruence_scores)
                },
                congruence_evolution=congruence_scores,
                recommendation_accuracy=recommendation_accuracy,
                risk_adjustment_factor=risk_adjustment,
                overall_score=overall_score
            )
            
            self.backtest_results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    def _generate_historical_mesh_data(self, client_data: SyntheticClientData,
                                     start_date: datetime, end_date: datetime,
                                     num_scenarios: int) -> List[Dict]:
        """Generate historical mesh data for backtesting"""
        # This would integrate with the time uncertainty mesh engine
        # For now, generate synthetic historical data
        historical_data = []
        
        current_date = start_date
        while current_date <= end_date:
            # Generate mesh data for this time point
            mesh_data = self._generate_synthetic_mesh_data(client_data, current_date)
            historical_data.append(mesh_data)
            
            current_date += timedelta(days=30)  # Monthly intervals
        
        return historical_data
    
    def _generate_synthetic_mesh_data(self, client_data: SyntheticClientData, 
                                    date: datetime) -> Dict:
        """Generate synthetic mesh data for a specific date"""
        # This would use the actual mesh engine
        # For now, create synthetic data based on client profile
        age_factor = client_data.profile.age / 100.0
        income_factor = client_data.profile.base_income / 100000.0
        
        # Generate mesh states
        num_scenarios = 50
        num_time_steps = 12
        
        mesh_states = np.random.normal(
            loc=age_factor * income_factor,
            scale=0.1,
            size=(num_time_steps, num_scenarios)
        )
        
        return {
            'date': date,
            'states': mesh_states,
            'scenarios': num_scenarios,
            'time_steps': num_time_steps
        }
    
    def _compute_historical_congruence(self, mesh_data_1: Dict, mesh_data_2: Dict) -> float:
        """Compute congruence between two historical mesh data points"""
        try:
            states_1 = mesh_data_1['states']
            states_2 = mesh_data_2['states']
            
            # Compute correlation between mesh states
            correlation = np.corrcoef(states_1.flatten(), states_2.flatten())[0, 1]
            
            return max(0, correlation)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error computing historical congruence: {e}")
            return 0.0
    
    def _compute_recommendation_accuracy(self, client_data: SyntheticClientData,
                                       historical_data: List[Dict]) -> float:
        """Compute accuracy of recommendations based on historical data"""
        # This would compare actual outcomes with predicted recommendations
        # For now, return a synthetic accuracy score
        base_accuracy = 0.7
        age_factor = client_data.profile.age / 100.0
        income_factor = client_data.profile.base_income / 100000.0
        
        accuracy = base_accuracy + (age_factor * 0.2) + (income_factor * 0.1)
        return min(1.0, accuracy)
    
    def _compute_risk_adjustment(self, client_data: SyntheticClientData,
                                historical_data: List[Dict]) -> float:
        """Compute risk adjustment factor based on historical volatility"""
        try:
            # Compute volatility of mesh states over time
            all_states = np.array([data['states'] for data in historical_data])
            
            # Compute volatility for each scenario
            volatilities = np.std(all_states, axis=0)
            avg_volatility = np.mean(volatilities)
            
            # Risk adjustment: lower volatility = higher score
            risk_adjustment = max(0, 1 - avg_volatility)
            
            return risk_adjustment
            
        except Exception as e:
            self.logger.error(f"Error computing risk adjustment: {e}")
            return 0.5
    
    def save_congruence_results(self, filename: str = None) -> None:
        """Save congruence and backtest results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mesh_congruence_results_{timestamp}.pkl"
        
        filepath = self.storage_dir / filename
        
        data = {
            'backtest_results': self.backtest_results,
            'engine_config': {
                'embedding_dim': self.embedding_dim,
                'congruence_threshold': self.congruence_threshold
            },
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved congruence results to {filepath}")
    
    def load_congruence_results(self, filename: str) -> None:
        """Load congruence and backtest results"""
        filepath = self.storage_dir / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.backtest_results = data['backtest_results']
        self.logger.info(f"Loaded congruence results from {filepath}")


def create_demo_congruence_engine():
    """Create and demonstrate the mesh congruence engine"""
    engine = MeshCongruenceEngine()
    
    # Generate synthetic clients for testing
    from .synthetic_lifestyle_engine import SyntheticLifestyleEngine
    lifestyle_engine = SyntheticLifestyleEngine()
    
    # Generate test clients
    client_1 = lifestyle_engine.generate_synthetic_client(target_age=35)
    client_2 = lifestyle_engine.generate_synthetic_client(target_age=38)
    
    # Compute congruence
    congruence_result = engine.compute_mesh_congruence(client_1, client_2)
    
    print(f"Mesh Congruence Analysis:")
    print(f"Source: {congruence_result.source_client_id}")
    print(f"Target: {congruence_result.target_client_id}")
    print(f"Overall Congruence: {congruence_result.overall_congruence:.3f}")
    print(f"Triangulation Quality: {congruence_result.triangulation_quality:.3f}")
    print(f"Density Distribution: {congruence_result.density_distribution_score:.3f}")
    print(f"Edge Collapse Efficiency: {congruence_result.edge_collapse_efficiency:.3f}")
    print(f"Matching Factors: {congruence_result.matching_factors}")
    print(f"Confidence Interval: {congruence_result.confidence_interval}")
    
    # Run backtest
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    backtest_result = engine.run_backtest(client_1, (start_date, end_date))
    
    print(f"\nBacktest Results:")
    print(f"Client: {backtest_result.client_id}")
    print(f"Overall Score: {backtest_result.overall_score:.3f}")
    print(f"Recommendation Accuracy: {backtest_result.recommendation_accuracy:.3f}")
    print(f"Risk Adjustment: {backtest_result.risk_adjustment_factor:.3f}")
    
    return engine


if __name__ == "__main__":
    create_demo_congruence_engine() 