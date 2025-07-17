"""
Generates an optimized mesh structure using adaptive techniques
"""
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import platform
import math

try:
    import torch
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Use Metal for M1/M2 Macs
        METAL_AVAILABLE = torch.backends.mps.is_available()
        if METAL_AVAILABLE:
            device = torch.device("mps")
            # Set default tensor type to float32 for Metal
            torch.set_default_dtype(torch.float32)
        else:
            device = torch.device("cpu")
        CUDA_AVAILABLE = False
    else:
        # Try CUDA for other systems
        CUDA_AVAILABLE = torch.cuda.is_available()
        METAL_AVAILABLE = False
        if CUDA_AVAILABLE:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    ACCELERATION_AVAILABLE = CUDA_AVAILABLE or METAL_AVAILABLE
except ImportError:
    ACCELERATION_AVAILABLE = False
    CUDA_AVAILABLE = False
    METAL_AVAILABLE = False
    device = None

def accelerated_generate_paths(paths, initial_value, drift, volatility, dt, random_shocks):
    """Generate paths using available acceleration"""
    if not ACCELERATION_AVAILABLE:
        return paths
        
    with torch.no_grad():
        # Convert to torch tensors with float32
        paths_tensor = torch.tensor(paths, device=device, dtype=torch.float32)
        random_shocks_tensor = torch.tensor(random_shocks, device=device, dtype=torch.float32)
        
        # Convert scalar parameters to float32
        initial_value = torch.tensor(initial_value, device=device, dtype=torch.float32)
        drift = torch.tensor(drift, device=device, dtype=torch.float32)
        volatility = torch.tensor(volatility, device=device, dtype=torch.float32)
        dt = torch.tensor(dt, device=device, dtype=torch.float32)
        
        paths_tensor[:, 0] = initial_value
        for t in range(1, paths.shape[1]):
            paths_tensor[:, t] = paths_tensor[:, t-1] * torch.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * torch.sqrt(dt) * random_shocks_tensor[:, t-1]
            )
        
        return paths_tensor.cpu().numpy().astype(np.float32)

def accelerated_calculate_path_values(paths, milestone_times, milestone_values, importance_scores):
    """Calculate path values using available acceleration"""
    if not ACCELERATION_AVAILABLE:
        return np.zeros(len(paths), dtype=np.float32)
        
    with torch.no_grad():
        # Convert to torch tensors with float32
        paths_tensor = torch.tensor(paths, device=device, dtype=torch.float32)
        milestone_times_tensor = torch.tensor(milestone_times, device=device, dtype=torch.float32)
        milestone_values_tensor = torch.tensor(milestone_values, device=device, dtype=torch.float32)
        importance_scores_tensor = torch.tensor(importance_scores, device=device, dtype=torch.float32)
        
        values = torch.zeros(len(paths), device=device, dtype=torch.float32)
        
        for i in range(len(milestone_times)):
            time_idx = int(milestone_times[i])
            if time_idx < paths.shape[1]:
                diff = torch.abs(paths_tensor[:, time_idx] - milestone_values_tensor[i])
                values += importance_scores_tensor[i] / (1.0 + diff)
        
        return values.cpu().numpy().astype(np.float32)


@dataclass
class MilestoneCluster:
    """Represents a cluster of milestones that are close in time"""
    center_date: datetime
    milestones: List['FinancialMilestone']
    total_impact: float
    importance_score: float


class AdaptiveMeshGenerator:
    """
    Generates an optimized mesh structure using adaptive techniques
    """
    
    def __init__(self, initial_state: Dict[str, float], 
                 memory_manager: 'MeshMemoryManager',
                 drift: float = 0.07, 
                 volatility: float = 0.15):
        self.initial_state = {k: float(v) for k, v in initial_state.items()}  # Ensure float32
        self.memory_manager = memory_manager
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.importance_threshold = 0.01
        self.max_cluster_days = 30
        self.use_acceleration = ACCELERATION_AVAILABLE
        
        if self.use_acceleration:
            print(f"🚀 Using {'Metal' if METAL_AVAILABLE else 'CUDA' if CUDA_AVAILABLE else 'CPU'} acceleration for mesh generation")
    
    def _get_critical_dates(self, milestones: List['FinancialMilestone']) -> List[MilestoneCluster]:
        """
        Identify critical dates by clustering milestones
        """
        if not milestones:
            return []
            
        # Convert dates to numerical format for clustering
        dates = np.array([
            [m.timestamp.timestamp()] for m in milestones
        ])
        
        # Cluster milestones that are close in time
        clustering = DBSCAN(
            eps=self.max_cluster_days * 24 * 3600,  # Convert days to seconds
            min_samples=1
        ).fit(dates)
        
        # Group milestones by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(milestones[i])
        
        # Create milestone clusters
        milestone_clusters = []
        for cluster_milestones in clusters.values():
            # Calculate cluster center
            center_date = datetime.fromtimestamp(
                np.mean([m.timestamp.timestamp() for m in cluster_milestones])
            )
            
            # Calculate total financial impact
            total_impact = sum(
                m.financial_impact * m.probability 
                for m in cluster_milestones 
                if m.financial_impact
            )
            
            # Calculate importance score
            importance_score = self._calculate_cluster_importance(
                cluster_milestones, center_date
            )
            
            milestone_clusters.append(MilestoneCluster(
                center_date=center_date,
                milestones=cluster_milestones,
                total_impact=total_impact,
                importance_score=importance_score
            ))
        
        return sorted(milestone_clusters, key=lambda x: x.center_date)
    
    def _calculate_cluster_importance(self, 
                                   milestones: List['FinancialMilestone'],
                                   center_date: datetime) -> float:
        """
        Calculate importance score for a milestone cluster
        """
        total_impact = sum(m.financial_impact * m.probability for m in milestones if m.financial_impact)
        avg_probability = np.mean([m.probability for m in milestones])
        time_factor = 1.0 / (1.0 + (center_date - datetime.now()).days / 365.0)
        return total_impact * avg_probability * time_factor
    
    def _generate_paths_parallel(self, num_paths: int, num_steps: int, 
                               initial_value: float) -> np.ndarray:
        """Generate paths using available parallel processing"""
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps)).astype(np.float32)
        
        if self.use_acceleration:
            return accelerated_generate_paths(
                paths, float(initial_value), self.drift, self.volatility, 
                1.0/12.0, random_shocks
            )
        else:
            return self._generate_paths_cpu(num_paths, num_steps, initial_value)
    
    def _generate_paths_cpu(self, num_paths: int, num_steps: int, 
                          initial_value: float) -> np.ndarray:
        """CPU fallback for path generation"""
        dt = 1.0 / 12.0  # Monthly steps
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        paths[:, 0] = initial_value
        
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps)).astype(np.float32)
        
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt + 
                self.volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        return paths
    
    def _generate_path_variations(self, base_path: np.ndarray, num_variations: int = 5) -> np.ndarray:
        """Generate variations of a base path"""
        if len(base_path) == 0:
            return np.array([])
            
        num_steps = len(base_path) - 1
        variations = np.zeros((num_variations, len(base_path)), dtype=np.float32)
        
        # Generate random variations around the base path
        for i in range(num_variations):
            variation = base_path.copy()
            # Add random noise to the path while preserving the overall trend
            noise = np.random.normal(0, 0.02, size=len(base_path)).astype(np.float32)
            variation *= (1 + noise)
            variations[i] = variation
            
        return variations
    
    def _generate_important_paths(self, 
                                milestone_clusters: List['MilestoneCluster'],
                                num_paths: int = 1000) -> List[np.ndarray]:
        """
        Generate paths focusing on important milestone clusters using parallel processing
        """
        if not milestone_clusters:
            return self._generate_paths_parallel(num_paths, 60, self.initial_state['total_wealth'])
            
        time_horizon = (milestone_clusters[-1].center_date - datetime.now()).days / 365.0
        num_steps = int(time_horizon * 12)  # Monthly steps
        
        # Generate initial paths
        paths = self._generate_paths_parallel(num_paths, num_steps, self.initial_state['total_wealth'])
        
        # Calculate path values
        path_values = self._calculate_path_values_parallel(paths, milestone_clusters)
        
        # Select best paths (top 25%)
        threshold = np.percentile(path_values, 75)
        selected_indices = path_values > threshold
        selected_paths = paths[selected_indices]
        
        if len(selected_paths) == 0:
            return paths  # Return original paths if no paths meet the threshold
        
        # Generate variations for each selected path
        variations = []
        for path in selected_paths:
            path_variations = self._generate_path_variations(path)
            if len(path_variations) > 0:
                variations.append(path_variations)
        
        # Combine original paths with variations
        if variations:
            variations = np.vstack(variations)
            if len(variations) > 0:
                # Ensure dimensions match
                if variations.shape[1] == paths.shape[1]:
                    combined_paths = np.vstack([paths, variations])
                    # Limit total number of paths
                    if len(combined_paths) > num_paths:
                        indices = np.random.choice(len(combined_paths), num_paths, replace=False)
                        combined_paths = combined_paths[indices]
                    return combined_paths
        
        return paths  # Return original paths if no valid variations were generated
    
    def _generate_simple_paths(self, num_paths: int, time_horizon: float) -> List[np.ndarray]:
        """
        Generate simple GBM paths when no milestones are available
        """
        dt = 1.0 / 12.0  # Monthly steps
        num_steps = int(time_horizon / dt)
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        paths[:, 0] = self.initial_state['total_wealth']
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (num_paths, num_steps)).astype(np.float32)
        
        # Generate paths using GBM formula
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.drift - 0.5 * self.volatility**2) * dt + 
                self.volatility * np.sqrt(dt) * random_shocks[:, t-1]
            )
        
        return paths
    
    def _generate_conditional_paths(self,
                                  num_paths: int,
                                  time_horizon: float,
                                  target_value: float,
                                  target_time: float) -> List[np.ndarray]:
        """
        Generate paths conditioned to hit specific values at specific times
        """
        dt = 1.0 / 12.0  # Monthly steps
        num_steps = int(time_horizon / dt)
        
        # Initialize paths
        paths = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        paths[:, 0] = self.initial_state['total_wealth']
        
        # Generate paths using bridge sampling
        for path_idx in range(num_paths):
            # Generate bridge to target
            target_steps = int(target_time / dt)
            if target_steps > 0:
                # Calculate drift to reach target
                required_return = np.log(target_value / paths[path_idx, 0]) / target_time
                adjusted_drift = required_return + 0.5 * self.volatility**2
                
                # Generate path to target
                for t in range(1, target_steps + 1):
                    paths[path_idx, t] = paths[path_idx, t-1] * np.exp(
                        (adjusted_drift - 0.5 * self.volatility**2) * dt +
                        self.volatility * np.sqrt(dt) * np.random.normal()
                    )
            
            # Continue path after target
            if target_steps < num_steps:
                for t in range(target_steps + 1, num_steps + 1):
                    paths[path_idx, t] = paths[path_idx, t-1] * np.exp(
                        (self.drift - 0.5 * self.volatility**2) * dt +
                        self.volatility * np.sqrt(dt) * np.random.normal()
                    )
        
        return paths
    
    def _refine_high_value_paths(self, 
                                paths: List[np.ndarray],
                                milestone_clusters: List[MilestoneCluster]) -> List[np.ndarray]:
        """
        Refine paths that have high potential value
        """
        if not milestone_clusters:
            return paths
            
        refined_paths = []
        
        for path in paths:
            # Calculate path value
            path_value = self._calculate_path_value(path, milestone_clusters)
            
            if path_value > np.percentile([
                self._calculate_path_value(p, milestone_clusters) 
                for p in paths
            ], 75):
                # Generate variations around this path
                variations = self._generate_path_variations(path)
                refined_paths.extend(variations)
        
        return refined_paths
    
    def _calculate_path_values_parallel(self, paths: np.ndarray, 
                                      milestone_clusters: List['MilestoneCluster']) -> np.ndarray:
        """Calculate path values using parallel processing"""
        if not milestone_clusters:
            return np.zeros(len(paths), dtype=np.float32)
            
        # Prepare milestone data
        milestone_times = np.array([
            (c.center_date - datetime.now()).days / 30  # Convert to months
            for c in milestone_clusters
        ], dtype=np.float32)
        
        milestone_values = np.array([
            float(self.initial_state['total_wealth'] + c.total_impact)
            for c in milestone_clusters
        ], dtype=np.float32)
        
        importance_scores = np.array([
            float(c.importance_score) for c in milestone_clusters
        ], dtype=np.float32)
        
        if self.use_acceleration:
            return accelerated_calculate_path_values(
                paths, milestone_times, milestone_values, importance_scores
            )
        else:
            return np.array([
                self._calculate_path_value(path, milestone_clusters)
                for path in paths
            ], dtype=np.float32)

    def _calculate_path_value(self,
                            path: np.ndarray,
                            milestone_clusters: List[MilestoneCluster]) -> float:
        """
        Calculate the potential value of a path
        """
        value = 0.0
        dt = 1.0 / 12.0  # Monthly steps
        
        for cluster in milestone_clusters:
            cluster_time_idx = int((cluster.center_date - datetime.now()).days * dt)
            if cluster_time_idx < len(path):
                # Check if path value is close to cluster's target
                target_value = self.initial_state['total_wealth'] + cluster.total_impact
                value += (1.0 - abs(path[cluster_time_idx] - target_value) / target_value) * cluster.importance_score
        
        return value
    
    def _aggregate_similar_states(self, paths: List[np.ndarray]) -> Set[str]:
        """
        Aggregate similar financial states to reduce state space
        """
        unique_states = set()
        
        for path in paths:
            for state in path:
                # Round to reduce state space
                rounded_state = round(state, 2)
                unique_states.add(str(rounded_state))
        
        return unique_states 