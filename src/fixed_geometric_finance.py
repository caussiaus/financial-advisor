#!/usr/bin/env python3
"""
Fixed Geometric Finance System
Properly represents financial configurations as geometric patterns in high-dimensional space
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class FinancialProfile:
    """Represents a financial profile as a point in high-dimensional space"""
    client_id: str
    features: np.ndarray  # High-dimensional feature vector
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FinancialTriangle:
    """Represents a triangle of similar financial profiles"""
    triangle_id: str
    vertices: List[str]  # Client IDs
    centroid: np.ndarray
    area: float
    congruence_score: float
    financial_similarity: float
    risk_profile: str  # 'conservative', 'moderate', 'aggressive'
    life_stage: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class GeometricFinanceEngine:
    """
    Engine for representing financial configurations as geometric patterns
    """
    
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(n_features, 3))  # For visualization
        self.profiles: Dict[str, FinancialProfile] = {}
        self.triangles: List[FinancialTriangle] = []
        
    def create_financial_profile(self, client_data: Dict) -> FinancialProfile:
        """Convert client data to high-dimensional financial profile"""
        
        # Extract and normalize financial features
        features = np.array([
            client_data.get('age', 35) / 100.0,  # Normalized age
            client_data.get('income', 75000) / 200000.0,  # Normalized income
            client_data.get('risk_tolerance', 0.5),  # Already 0-1
            client_data.get('debt_to_income_ratio', 0.3),  # Already 0-1
            client_data.get('savings_rate', 0.15),  # Already 0-1
            client_data.get('investment_horizon', 20) / 50.0,  # Normalized
            client_data.get('net_worth', 500000) / 2000000.0,  # Normalized
            client_data.get('education_level', 0.5),  # Encoded
            client_data.get('family_status', 0.5),  # Encoded
            client_data.get('life_stage', 0.5),  # Encoded
        ])
        
        # Ensure we have the right number of features
        if len(features) < self.n_features:
            features = np.pad(features, (0, self.n_features - len(features)))
        elif len(features) > self.n_features:
            features = features[:self.n_features]
        
        return FinancialProfile(
            client_id=client_data.get('client_id', 'unknown'),
            features=features,
            metadata=client_data
        )
    
    def add_profile(self, profile: FinancialProfile):
        """Add a financial profile to the engine"""
        self.profiles[profile.client_id] = profile
    
    def compute_financial_similarity(self, profile1: FinancialProfile, profile2: FinancialProfile) -> float:
        """Compute similarity between two financial profiles"""
        # Use cosine similarity for high-dimensional data
        dot_product = np.dot(profile1.features, profile2.features)
        norm1 = np.linalg.norm(profile1.features)
        norm2 = np.linalg.norm(profile2.features)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def generate_financial_triangles(self) -> List[FinancialTriangle]:
        """Generate triangles of similar financial profiles"""
        if len(self.profiles) < 3:
            return []
        
        profile_ids = list(self.profiles.keys())
        triangles = []
        
        # Generate all possible triangles
        for i in range(len(profile_ids)):
            for j in range(i + 1, len(profile_ids)):
                for k in range(j + 1, len(profile_ids)):
                    
                    p1_id = profile_ids[i]
                    p2_id = profile_ids[j]
                    p3_id = profile_ids[k]
                    
                    p1 = self.profiles[p1_id]
                    p2 = self.profiles[p2_id]
                    p3 = self.profiles[p3_id]
                    
                    # Calculate similarities
                    sim_12 = self.compute_financial_similarity(p1, p2)
                    sim_13 = self.compute_financial_similarity(p1, p3)
                    sim_23 = self.compute_financial_similarity(p2, p3)
                    
                    # Average similarity as congruence score
                    congruence_score = (sim_12 + sim_13 + sim_23) / 3
                    
                    # Calculate centroid
                    centroid = (p1.features + p2.features + p3.features) / 3
                    
                    # Calculate area in feature space
                    # Use the triangle formed by the three points
                    v1 = p2.features - p1.features
                    v2 = p3.features - p1.features
                    
                    # For high-dimensional data, use the magnitude of the cross product
                    # For n-dimensional data, we can use the determinant approach
                    # or simply use the distance-based area approximation
                    if len(v1) == 2:
                        # 2D case - use cross product
                        area = np.abs(np.cross(v1, v2)) / 2.0
                    elif len(v1) == 3:
                        # 3D case - use cross product
                        area = np.linalg.norm(np.cross(v1, v2)) / 2.0
                    else:
                        # High-dimensional case - use distance-based approximation
                        # Calculate the area using the three sides of the triangle
                        side1 = np.linalg.norm(v1)
                        side2 = np.linalg.norm(v2)
                        side3 = np.linalg.norm(p3.features - p2.features)
                        
                        # Use Heron's formula for area
                        s = (side1 + side2 + side3) / 2.0
                        area = np.sqrt(s * (s - side1) * (s - side2) * (s - side3))
                    
                    # Determine risk profile based on average risk tolerance
                    avg_risk = np.mean([p1.features[2], p2.features[2], p3.features[2]])
                    if avg_risk < 0.3:
                        risk_profile = 'conservative'
                    elif avg_risk < 0.7:
                        risk_profile = 'moderate'
                    else:
                        risk_profile = 'aggressive'
                    
                    # Determine life stage
                    avg_age = np.mean([p1.features[0], p2.features[0], p3.features[0]]) * 100
                    if avg_age < 30:
                        life_stage = 'early_career'
                    elif avg_age < 50:
                        life_stage = 'mid_career'
                    elif avg_age < 65:
                        life_stage = 'pre_retirement'
                    else:
                        life_stage = 'retirement'
                    
                    triangle = FinancialTriangle(
                        triangle_id=f"triangle_{len(triangles):03d}",
                        vertices=[p1_id, p2_id, p3_id],
                        centroid=centroid,
                        area=area,
                        congruence_score=congruence_score,
                        financial_similarity=congruence_score,
                        risk_profile=risk_profile,
                        life_stage=life_stage,
                        metadata={
                            'similarities': {'1-2': sim_12, '1-3': sim_13, '2-3': sim_23},
                            'avg_age': avg_age,
                            'avg_risk': avg_risk
                        }
                    )
                    
                    triangles.append(triangle)
        
        self.triangles = triangles
        return triangles
    
    def find_similar_profiles(self, target_profile: FinancialProfile, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find profiles most similar to target"""
        similarities = []
        
        for client_id, profile in self.profiles.items():
            if client_id != target_profile.client_id:
                similarity = self.compute_financial_similarity(target_profile, profile)
                similarities.append((client_id, similarity))
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def analyze_financial_clusters(self) -> Dict:
        """Analyze the financial clusters formed by triangles"""
        if not self.triangles:
            return {}
        
        # Group triangles by risk profile and life stage
        cluster_analysis = {
            'risk_profiles': {},
            'life_stages': {},
            'high_congruence_triangles': [],
            'diverse_triangles': []
        }
        
        for triangle in self.triangles:
            # Risk profile analysis
            if triangle.risk_profile not in cluster_analysis['risk_profiles']:
                cluster_analysis['risk_profiles'][triangle.risk_profile] = []
            cluster_analysis['risk_profiles'][triangle.risk_profile].append(triangle)
            
            # Life stage analysis
            if triangle.life_stage not in cluster_analysis['life_stages']:
                cluster_analysis['life_stages'][triangle.life_stage] = []
            cluster_analysis['life_stages'][triangle.life_stage].append(triangle)
            
            # High congruence triangles (similar financial profiles)
            if triangle.congruence_score > 0.7:
                cluster_analysis['high_congruence_triangles'].append(triangle)
            
            # Diverse triangles (different strategies)
            if triangle.area > np.percentile([t.area for t in self.triangles], 75):
                cluster_analysis['diverse_triangles'].append(triangle)
        
        return cluster_analysis
    
    def visualize_financial_space(self, save_path: str = None):
        """Visualize the financial profiles in 2D space"""
        if len(self.profiles) < 2:
            print("Need at least 2 profiles for visualization")
            return
        
        # Extract features for all profiles
        features_matrix = np.array([p.features for p in self.profiles.values()])
        client_ids = list(self.profiles.keys())
        
        # Reduce to 2D for visualization
        if features_matrix.shape[1] > 2:
            features_2d = self.pca.fit_transform(features_matrix)
        else:
            features_2d = features_matrix
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot profiles
        plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                   c=range(len(client_ids)), cmap='viridis', s=100, alpha=0.7)
        
        # Add labels
        for i, client_id in enumerate(client_ids):
            plt.annotate(client_id, (features_2d[i, 0], features_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot triangles if available
        if self.triangles:
            for triangle in self.triangles[:5]:  # Show first 5 triangles
                vertices_2d = []
                for vertex_id in triangle.vertices:
                    if vertex_id in self.profiles:
                        idx = client_ids.index(vertex_id)
                        vertices_2d.append(features_2d[idx])
                
                if len(vertices_2d) == 3:
                    vertices_2d = np.array(vertices_2d)
                    plt.plot(vertices_2d[[0, 1], 0], vertices_2d[[0, 1], 1], 'r--', alpha=0.5)
                    plt.plot(vertices_2d[[1, 2], 0], vertices_2d[[1, 2], 1], 'r--', alpha=0.5)
                    plt.plot(vertices_2d[[2, 0], 0], vertices_2d[[2, 0], 1], 'r--', alpha=0.5)
        
        plt.title('Financial Profiles in Geometric Space')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Profile Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def demo_geometric_finance():
    """Demo the fixed geometric finance system"""
    print("🔺 GEOMETRIC FINANCE SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize engine
    engine = GeometricFinanceEngine(n_features=10)
    
    # Create sample financial profiles
    sample_clients = [
        {
            'client_id': 'client_1',
            'age': 35,
            'income': 85000,
            'risk_tolerance': 0.6,
            'debt_to_income_ratio': 0.25,
            'savings_rate': 0.20,
            'investment_horizon': 25,
            'net_worth': 300000,
            'education_level': 0.8,
            'family_status': 0.7,
            'life_stage': 0.4
        },
        {
            'client_id': 'client_2',
            'age': 42,
            'income': 120000,
            'risk_tolerance': 0.7,
            'debt_to_income_ratio': 0.20,
            'savings_rate': 0.25,
            'investment_horizon': 20,
            'net_worth': 500000,
            'education_level': 0.9,
            'family_status': 0.8,
            'life_stage': 0.5
        },
        {
            'client_id': 'client_3',
            'age': 28,
            'income': 65000,
            'risk_tolerance': 0.8,
            'debt_to_income_ratio': 0.35,
            'savings_rate': 0.15,
            'investment_horizon': 35,
            'net_worth': 150000,
            'education_level': 0.7,
            'family_status': 0.3,
            'life_stage': 0.2
        },
        {
            'client_id': 'client_4',
            'age': 55,
            'income': 95000,
            'risk_tolerance': 0.4,
            'debt_to_income_ratio': 0.15,
            'savings_rate': 0.30,
            'investment_horizon': 10,
            'net_worth': 800000,
            'education_level': 0.6,
            'family_status': 0.9,
            'life_stage': 0.7
        },
        {
            'client_id': 'client_5',
            'age': 38,
            'income': 110000,
            'risk_tolerance': 0.5,
            'debt_to_income_ratio': 0.22,
            'savings_rate': 0.22,
            'investment_horizon': 22,
            'net_worth': 400000,
            'education_level': 0.8,
            'family_status': 0.6,
            'life_stage': 0.4
        }
    ]
    
    print("📊 Creating financial profiles...")
    for client_data in sample_clients:
        profile = engine.create_financial_profile(client_data)
        engine.add_profile(profile)
        print(f"   ✅ Created profile for {client_data['client_id']}")
    
    print(f"\n🔺 Generating financial triangles...")
    triangles = engine.generate_financial_triangles()
    print(f"   ✅ Generated {len(triangles)} financial triangles")
    
    # Analyze results
    print("\n📊 TRIANGLE ANALYSIS")
    print("-" * 30)
    
    for i, triangle in enumerate(triangles[:3]):  # Show first 3
        print(f"Triangle {i+1}:")
        print(f"   Vertices: {triangle.vertices}")
        print(f"   Congruence Score: {triangle.congruence_score:.3f}")
        print(f"   Risk Profile: {triangle.risk_profile}")
        print(f"   Life Stage: {triangle.life_stage}")
        print(f"   Area: {triangle.area:.3f}")
        print()
    
    # Find similar profiles
    target_profile = list(engine.profiles.values())[0]
    similar_profiles = engine.find_similar_profiles(target_profile, n_similar=3)
    
    print("🔍 SIMILARITY ANALYSIS")
    print("-" * 30)
    print(f"Target: {target_profile.client_id}")
    for client_id, similarity in similar_profiles:
        print(f"   {client_id}: {similarity:.3f}")
    
    # Cluster analysis
    cluster_analysis = engine.analyze_financial_clusters()
    
    print("\n📊 CLUSTER ANALYSIS")
    print("-" * 30)
    print(f"High congruence triangles: {len(cluster_analysis['high_congruence_triangles'])}")
    print(f"Diverse triangles: {len(cluster_analysis['diverse_triangles'])}")
    
    # Visualize
    print("\n📈 Creating visualization...")
    engine.visualize_financial_space('geometric_finance_visualization.png')
    print("   ✅ Visualization saved to geometric_finance_visualization.png")
    
    return {
        'num_profiles': len(engine.profiles),
        'num_triangles': len(triangles),
        'avg_congruence': np.mean([t.congruence_score for t in triangles]) if triangles else 0.0,
        'cluster_analysis': cluster_analysis
    }

if __name__ == "__main__":
    results = demo_geometric_finance()
    print(f"\n🎉 Demo completed! Generated {results['num_triangles']} triangles with average congruence {results['avg_congruence']:.3f}") 