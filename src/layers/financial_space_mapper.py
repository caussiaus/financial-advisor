"""
Financial Space Mapper

Responsible for:
- Clustering detection around commutators
- Topological mapping of financial state space
- Visualization of feasible vs infeasible regions
- Mesh visualization through clustering analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Protocol
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import networkx as nx
from enum import Enum


class ClusterType(Enum):
    """Types of financial state clusters"""
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    OPTIMAL = "optimal"
    HIGH_RISK = "high_risk"
    LOW_RISK = "low_risk"
    TRANSITION = "transition"


@dataclass
class FinancialState:
    """Represents a point in financial state space"""
    state_id: str
    timestamp: datetime
    coordinates: np.ndarray  # Vector representation of financial state
    asset_allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    feasibility_score: float  # 0-1, how feasible this state is
    cluster_id: Optional[int] = None
    cluster_type: Optional[ClusterType] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClusterRegion:
    """Represents a cluster region in financial space"""
    cluster_id: int
    cluster_type: ClusterType
    center: np.ndarray
    radius: float
    density: float
    states: List[FinancialState]
    boundary_points: List[np.ndarray]
    connectivity: Dict[int, float]  # Connections to other clusters
    metadata: Dict = field(default_factory=dict)


@dataclass
class FinancialSpaceMap:
    """Complete map of financial state space"""
    map_id: str
    timestamp: datetime
    clusters: List[ClusterRegion]
    connectivity_graph: nx.Graph
    feasible_regions: List[ClusterRegion]
    infeasible_regions: List[ClusterRegion]
    optimal_paths: List[List[int]]
    dimensionality: int
    coverage_score: float
    metadata: Dict = field(default_factory=dict)


class StateVectorizer(Protocol):
    """Protocol for state vectorization capabilities"""
    
    def vectorize_state(self, financial_state: Dict[str, float]) -> np.ndarray:
        """Convert financial state to vector representation"""
        ...


class ClusterDetector(Protocol):
    """Protocol for cluster detection capabilities"""
    
    def detect_clusters(self, states: List[FinancialState]) -> List[ClusterRegion]:
        """Detect clusters in financial state space"""
        ...


class FinancialSpaceMapper:
    """
    Financial Space Mapper - Clustering-based mesh visualization
    
    Responsibilities:
    - Clustering detection around commutators
    - Topological mapping of financial state space
    - Visualization of feasible vs infeasible regions
    - Mesh visualization through clustering analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.clustering_algorithms = self._initialize_clustering_algorithms()
        self.vectorization_methods = self._initialize_vectorization_methods()
        self.space_maps: Dict[str, FinancialSpaceMap] = {}
        
    def _initialize_clustering_algorithms(self) -> Dict[str, object]:
        """Initialize clustering algorithms"""
        return {
            'kmeans': KMeans(n_clusters=8, random_state=42),
            'dbscan': DBSCAN(eps=0.3, min_samples=5),
            'hierarchical': AgglomerativeClustering(n_clusters=8),
            'spectral': None  # Placeholder for spectral clustering
        }
    
    def _initialize_vectorization_methods(self) -> Dict[str, callable]:
        """Initialize state vectorization methods"""
        return {
            'asset_allocation': self._vectorize_asset_allocation,
            'risk_metrics': self._vectorize_risk_metrics,
            'combined': self._vectorize_combined_state
        }
    
    def _vectorize_asset_allocation(self, financial_state: Dict[str, float]) -> np.ndarray:
        """Vectorize based on asset allocation"""
        # Normalize asset allocation to percentages
        total = sum(financial_state.values())
        if total == 0:
            return np.zeros(len(financial_state))
        
        allocation = np.array([financial_state.get(asset, 0) / total for asset in financial_state.keys()])
        return allocation
    
    def _vectorize_risk_metrics(self, financial_state: Dict[str, float]) -> np.ndarray:
        """Vectorize based on risk metrics"""
        # Calculate risk metrics
        total_assets = sum(financial_state.values())
        cash_ratio = financial_state.get('cash', 0) / total_assets if total_assets > 0 else 0
        stock_ratio = financial_state.get('stocks', 0) / total_assets if total_assets > 0 else 0
        bond_ratio = financial_state.get('bonds', 0) / total_assets if total_assets > 0 else 0
        
        # Risk vector: [cash_ratio, stock_ratio, bond_ratio, diversification_score]
        diversification = 1 - (cash_ratio**2 + stock_ratio**2 + bond_ratio**2)  # Herfindahl index
        risk_vector = np.array([cash_ratio, stock_ratio, bond_ratio, diversification])
        
        return risk_vector
    
    def _vectorize_combined_state(self, financial_state: Dict[str, float]) -> np.ndarray:
        """Vectorize using combined asset allocation and risk metrics"""
        allocation_vector = self._vectorize_asset_allocation(financial_state)
        risk_vector = self._vectorize_risk_metrics(financial_state)
        
        # Combine vectors
        combined_vector = np.concatenate([allocation_vector, risk_vector])
        return combined_vector
    
    def generate_financial_states_from_commutators(self, commutators: List, 
                                                 initial_state: Dict[str, float],
                                                 num_samples: int = 1000) -> List[FinancialState]:
        """
        Generate financial states by applying commutators and sampling around them
        
        Args:
            commutators: List of commutator sequences
            initial_state: Initial financial state
            num_samples: Number of states to generate
            
        Returns:
            List of financial states
        """
        states = []
        
        # Generate states by applying commutators
        current_state = initial_state.copy()
        states.append(self._create_financial_state(current_state, "initial"))
        
        for i, commutator in enumerate(commutators):
            # Apply commutator
            new_state = self._apply_commutator_to_state(commutator, current_state)
            states.append(self._create_financial_state(new_state, f"commutator_{i}"))
            current_state = new_state
            
            # Generate samples around this commutator
            samples = self._sample_around_commutator(commutator, new_state, num_samples // len(commutators))
            states.extend(samples)
        
        return states
    
    def _create_financial_state(self, financial_state: Dict[str, float], 
                               state_id: str) -> FinancialState:
        """Create a FinancialState object from financial state dictionary"""
        # Vectorize state
        vector = self.vectorization_methods['combined'](financial_state)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(financial_state)
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility_score(financial_state, risk_metrics)
        
        return FinancialState(
            state_id=state_id,
            timestamp=datetime.now(),
            coordinates=vector,
            asset_allocation=financial_state.copy(),
            risk_metrics=risk_metrics,
            feasibility_score=feasibility_score
        )
    
    def _apply_commutator_to_state(self, commutator, state: Dict[str, float]) -> Dict[str, float]:
        """Apply a commutator to a financial state"""
        new_state = state.copy()
        
        # Apply move A
        new_state = self._apply_move_to_state(commutator.move_a, new_state)
        # Apply move B
        new_state = self._apply_move_to_state(commutator.move_b, new_state)
        # Apply inverse A
        new_state = self._apply_move_to_state(commutator.inverse_a, new_state)
        # Apply inverse B
        new_state = self._apply_move_to_state(commutator.inverse_b, new_state)
        
        return new_state
    
    def _apply_move_to_state(self, move, state: Dict[str, float]) -> Dict[str, float]:
        """Apply a single move to a financial state"""
        new_state = state.copy()
        
        if move.move_type.value == 'reallocate':
            if move.from_asset and move.to_asset:
                # Transfer between assets
                amount = move.amount * sum(new_state.values())
                if move.from_asset in new_state:
                    new_state[move.from_asset] = max(0, new_state[move.from_asset] - amount)
                if move.to_asset in new_state:
                    new_state[move.to_asset] = new_state.get(move.to_asset, 0) + amount
        
        return new_state
    
    def _sample_around_commutator(self, commutator, base_state: Dict[str, float], 
                                 num_samples: int) -> List[FinancialState]:
        """Generate samples around a commutator"""
        samples = []
        
        for i in range(num_samples):
            # Add noise to base state
            noisy_state = self._add_noise_to_state(base_state, noise_level=0.1)
            
            # Apply commutator with some variation
            modified_commutator = self._modify_commutator(commutator, variation=0.2)
            sample_state = self._apply_commutator_to_state(modified_commutator, noisy_state)
            
            # Create financial state
            financial_state = self._create_financial_state(
                sample_state, 
                f"sample_{commutator.sequence_id}_{i}"
            )
            samples.append(financial_state)
        
        return samples
    
    def _add_noise_to_state(self, state: Dict[str, float], noise_level: float) -> Dict[str, float]:
        """Add noise to financial state"""
        noisy_state = {}
        total = sum(state.values())
        
        for asset, value in state.items():
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level * value)
            noisy_state[asset] = max(0, value + noise)
        
        # Renormalize
        new_total = sum(noisy_state.values())
        if new_total > 0:
            for asset in noisy_state:
                noisy_state[asset] = noisy_state[asset] / new_total * total
        
        return noisy_state
    
    def _modify_commutator(self, commutator, variation: float):
        """Modify commutator with some variation"""
        # For now, return the original commutator
        # In a full implementation, this would modify the commutator parameters
        return commutator
    
    def _calculate_risk_metrics(self, financial_state: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk metrics for a financial state"""
        total_assets = sum(financial_state.values())
        
        if total_assets == 0:
            return {
                'cash_ratio': 0.0,
                'stock_ratio': 0.0,
                'bond_ratio': 0.0,
                'diversification': 0.0,
                'volatility': 0.0
            }
        
        cash_ratio = financial_state.get('cash', 0) / total_assets
        stock_ratio = financial_state.get('stocks', 0) / total_assets
        bond_ratio = financial_state.get('bonds', 0) / total_assets
        
        # Calculate diversification (1 - Herfindahl index)
        diversification = 1 - (cash_ratio**2 + stock_ratio**2 + bond_ratio**2)
        
        # Simplified volatility calculation
        volatility = stock_ratio * 0.6 + bond_ratio * 0.2 + (1 - cash_ratio) * 0.4
        
        return {
            'cash_ratio': cash_ratio,
            'stock_ratio': stock_ratio,
            'bond_ratio': bond_ratio,
            'diversification': diversification,
            'volatility': volatility
        }
    
    def _calculate_feasibility_score(self, financial_state: Dict[str, float], 
                                   risk_metrics: Dict[str, float]) -> float:
        """Calculate feasibility score (0-1) for a financial state"""
        score = 1.0
        
        # Check for negative balances
        for asset, value in financial_state.items():
            if value < 0:
                score *= 0.5
        
        # Check liquidity constraints
        cash_ratio = risk_metrics['cash_ratio']
        if cash_ratio < 0.05:  # Less than 5% cash
            score *= 0.8
        
        # Check diversification
        diversification = risk_metrics['diversification']
        if diversification < 0.3:  # Low diversification
            score *= 0.9
        
        # Check volatility
        volatility = risk_metrics['volatility']
        if volatility > 0.8:  # High volatility
            score *= 0.7
        
        return max(0.0, min(1.0, score))
    
    def detect_clusters(self, states: List[FinancialState], 
                       algorithm: str = 'kmeans') -> List[ClusterRegion]:
        """
        Detect clusters in financial state space
        
        Args:
            states: List of financial states
            algorithm: Clustering algorithm to use
            
        Returns:
            List of cluster regions
        """
        if not states:
            return []
        
        # Extract coordinates
        coordinates = np.array([state.coordinates for state in states])
        
        # Apply clustering
        if algorithm == 'kmeans':
            clustering = KMeans(n_clusters=min(8, len(states)), random_state=42)
        elif algorithm == 'dbscan':
            clustering = DBSCAN(eps=0.3, min_samples=5)
        elif algorithm == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=min(8, len(states)))
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Fit clustering
        cluster_labels = clustering.fit_predict(coordinates)
        
        # Create cluster regions
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            # Get states in this cluster
            cluster_states = [state for i, state in enumerate(states) if cluster_labels[i] == label]
            
            if len(cluster_states) < 2:
                continue
            
            # Calculate cluster properties
            cluster_coords = np.array([state.coordinates for state in cluster_states])
            center = np.mean(cluster_coords, axis=0)
            radius = np.max(cdist([center], cluster_coords))
            density = len(cluster_states) / (radius + 1e-6)
            
            # Determine cluster type
            avg_feasibility = np.mean([state.feasibility_score for state in cluster_states])
            avg_volatility = np.mean([state.risk_metrics['volatility'] for state in cluster_states])
            
            if avg_feasibility > 0.8:
                cluster_type = ClusterType.FEASIBLE
            elif avg_feasibility < 0.3:
                cluster_type = ClusterType.INFEASIBLE
            elif avg_volatility > 0.6:
                cluster_type = ClusterType.HIGH_RISK
            elif avg_volatility < 0.2:
                cluster_type = ClusterType.LOW_RISK
            else:
                cluster_type = ClusterType.TRANSITION
            
            # Calculate boundary points (simplified)
            boundary_points = self._calculate_boundary_points(cluster_coords)
            
            cluster_region = ClusterRegion(
                cluster_id=label,
                cluster_type=cluster_type,
                center=center,
                radius=radius,
                density=density,
                states=cluster_states,
                boundary_points=boundary_points,
                connectivity={}
            )
            
            clusters.append(cluster_region)
        
        return clusters
    
    def _calculate_boundary_points(self, coordinates: np.ndarray) -> List[np.ndarray]:
        """Calculate boundary points of a cluster (simplified)"""
        if len(coordinates) < 3:
            return coordinates.tolist()
        
        # Use convex hull for boundary (simplified)
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(coordinates)
            boundary_points = coordinates[hull.vertices]
            return boundary_points.tolist()
        except:
            # Fallback to extreme points
            boundary_points = []
            for i in range(coordinates.shape[1]):
                min_idx = np.argmin(coordinates[:, i])
                max_idx = np.argmax(coordinates[:, i])
                boundary_points.extend([coordinates[min_idx], coordinates[max_idx]])
            return boundary_points
    
    def create_financial_space_map(self, states: List[FinancialState], 
                                  map_id: str = None) -> FinancialSpaceMap:
        """
        Create a complete map of financial state space
        
        Args:
            states: List of financial states
            map_id: ID for the map
            
        Returns:
            Financial space map
        """
        if not map_id:
            map_id = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect clusters
        clusters = self.detect_clusters(states)
        
        # Separate clusters by type
        feasible_regions = [c for c in clusters if c.cluster_type == ClusterType.FEASIBLE]
        infeasible_regions = [c for c in clusters if c.cluster_type == ClusterType.INFEASIBLE]
        
        # Create connectivity graph
        connectivity_graph = self._create_connectivity_graph(clusters)
        
        # Find optimal paths
        optimal_paths = self._find_optimal_paths(connectivity_graph, feasible_regions)
        
        # Calculate coverage score
        coverage_score = self._calculate_coverage_score(clusters, states)
        
        # Create space map
        space_map = FinancialSpaceMap(
            map_id=map_id,
            timestamp=datetime.now(),
            clusters=clusters,
            connectivity_graph=connectivity_graph,
            feasible_regions=feasible_regions,
            infeasible_regions=infeasible_regions,
            optimal_paths=optimal_paths,
            dimensionality=states[0].coordinates.shape[0] if states else 0,
            coverage_score=coverage_score
        )
        
        # Store map
        self.space_maps[map_id] = space_map
        
        return space_map
    
    def _create_connectivity_graph(self, clusters: List[ClusterRegion]) -> nx.Graph:
        """Create connectivity graph between clusters"""
        graph = nx.Graph()
        
        for cluster in clusters:
            graph.add_node(cluster.cluster_id, cluster=cluster)
        
        # Add edges based on proximity
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                distance = np.linalg.norm(cluster1.center - cluster2.center)
                if distance < (cluster1.radius + cluster2.radius) * 1.5:
                    weight = 1.0 / (distance + 1e-6)
                    graph.add_edge(cluster1.cluster_id, cluster2.cluster_id, weight=weight)
        
        return graph
    
    def _find_optimal_paths(self, graph: nx.Graph, 
                           feasible_regions: List[ClusterRegion]) -> List[List[int]]:
        """Find optimal paths through feasible regions"""
        if len(feasible_regions) < 2:
            return []
        
        paths = []
        feasible_ids = [c.cluster_id for c in feasible_regions]
        
        # Find shortest paths between feasible regions
        for i, start_id in enumerate(feasible_ids):
            for end_id in feasible_ids[i+1:]:
                try:
                    path = nx.shortest_path(graph, start_id, end_id, weight='weight')
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def _calculate_coverage_score(self, clusters: List[ClusterRegion], 
                                states: List[FinancialState]) -> float:
        """Calculate how well the clusters cover the state space"""
        if not states:
            return 0.0
        
        # Calculate total volume covered by clusters
        total_volume = sum(cluster.radius**2 for cluster in clusters)
        
        # Normalize by number of states
        coverage = total_volume / len(states)
        
        return min(1.0, coverage)
    
    def visualize_financial_space(self, space_map: FinancialSpaceMap, 
                                output_file: str = None) -> str:
        """
        Visualize the financial space map
        
        Args:
            space_map: Financial space map to visualize
            output_file: Output file path
            
        Returns:
            HTML visualization
        """
        # Create visualization using plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Reduce dimensionality for visualization
        coordinates = np.array([state.coordinates for state in space_map.clusters[0].states])
        if coordinates.shape[1] > 2:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(coordinates)
        else:
            coords_2d = coordinates
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot clusters with different colors
        colors = {
            ClusterType.FEASIBLE: 'green',
            ClusterType.INFEASIBLE: 'red',
            ClusterType.HIGH_RISK: 'orange',
            ClusterType.LOW_RISK: 'blue',
            ClusterType.TRANSITION: 'purple'
        }
        
        for cluster in space_map.clusters:
            cluster_coords = np.array([state.coordinates for state in cluster.states])
            if cluster_coords.shape[1] > 2:
                cluster_coords_2d = pca.transform(cluster_coords)
            else:
                cluster_coords_2d = cluster_coords
            
            fig.add_trace(go.Scatter(
                x=cluster_coords_2d[:, 0],
                y=cluster_coords_2d[:, 1],
                mode='markers',
                name=f'{cluster.cluster_type.value} (Cluster {cluster.cluster_id})',
                marker=dict(
                    color=colors[cluster.cluster_type],
                    size=8,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>' +
                            'Feasibility: %{customdata[0]:.2f}<br>' +
                            'Risk: %{customdata[1]:.2f}<extra></extra>',
                text=[f'State {i}' for i in range(len(cluster.states))],
                customdata=[[state.feasibility_score, state.risk_metrics['volatility']] 
                           for state in cluster.states]
            ))
        
        # Add optimal paths
        for path in space_map.optimal_paths:
            path_coords = []
            for cluster_id in path:
                cluster = next(c for c in space_map.clusters if c.cluster_id == cluster_id)
                path_coords.append(cluster.center)
            
            path_coords = np.array(path_coords)
            if path_coords.shape[1] > 2:
                path_coords_2d = pca.transform(path_coords)
            else:
                path_coords_2d = path_coords
            
            fig.add_trace(go.Scatter(
                x=path_coords_2d[:, 0],
                y=path_coords_2d[:, 1],
                mode='lines+markers',
                name='Optimal Path',
                line=dict(color='black', width=3),
                marker=dict(size=10, color='black'),
                showlegend=True
            ))
        
        fig.update_layout(
            title="Financial State Space Map",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            width=800,
            height=600
        )
        
        # Convert to HTML
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
        
        return html
    
    def get_cluster_analysis(self, space_map: FinancialSpaceMap) -> Dict:
        """Get detailed analysis of clusters"""
        analysis = {
            'total_clusters': len(space_map.clusters),
            'feasible_clusters': len(space_map.feasible_regions),
            'infeasible_clusters': len(space_map.infeasible_regions),
            'coverage_score': space_map.coverage_score,
            'optimal_paths': len(space_map.optimal_paths),
            'cluster_details': []
        }
        
        for cluster in space_map.clusters:
            cluster_analysis = {
                'cluster_id': cluster.cluster_id,
                'cluster_type': cluster.cluster_type.value,
                'num_states': len(cluster.states),
                'avg_feasibility': np.mean([s.feasibility_score for s in cluster.states]),
                'avg_volatility': np.mean([s.risk_metrics['volatility'] for s in cluster.states]),
                'density': cluster.density,
                'radius': cluster.radius
            }
            analysis['cluster_details'].append(cluster_analysis)
        
        return analysis 