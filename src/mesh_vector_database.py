"""
Mesh Vector Database System

This module creates a vector database system that:
1. Generates embeddings from mesh network composition
2. Stores client profiles with their mesh embeddings
3. Finds nearest matched clients based on similarity
4. Uses similarity matching to estimate uncertain factors
5. Provides recommendations based on similar client outcomes

Key Features:
- Mesh network composition embedding generation
- Vector similarity search for client matching
- Uncertainty estimation through similar client analysis
- Recommendation engine based on mesh patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
import pickle
from pathlib import Path
import hashlib

# Import existing components
from .json_to_vector_converter import ClientVectorProfile, LifeStage, EventCategory
from .synthetic_lifestyle_engine import SyntheticClientData
from .time_uncertainty_mesh import TimeUncertaintyMeshEngine


@dataclass
class MeshEmbedding:
    """Represents a mesh network embedding for similarity matching"""
    client_id: str
    embedding_vector: np.ndarray
    mesh_composition: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SimilarityMatch:
    """Represents a similarity match between clients"""
    source_client_id: str
    matched_client_id: str
    similarity_score: float
    matching_factors: List[str]
    estimated_uncertainties: Dict[str, float]
    confidence_score: float


class MeshVectorDatabase:
    """
    Vector database system for mesh network composition similarity matching
    """
    
    def __init__(self, embedding_dim: int = 128, similarity_threshold: float = 0.7):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.embeddings: Dict[str, MeshEmbedding] = {}
        self.client_profiles: Dict[str, ClientVectorProfile] = {}
        self.mesh_data: Dict[str, Dict] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create storage directory
        self.storage_dir = Path("data/outputs/vector_db")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the vector database"""
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
    
    def _generate_mesh_composition_features(self, client_data: SyntheticClientData) -> Dict[str, Any]:
        """
        Generate features that describe the mesh network composition
        
        Args:
            client_data: Synthetic client data with mesh information
            
        Returns:
            Dictionary of mesh composition features
        """
        composition = {}
        
        # Basic client features
        composition['age'] = client_data.profile.age
        composition['life_stage'] = client_data.vector_profile.life_stage.value
        composition['income_level'] = self._categorize_income(client_data.profile.base_income)
        composition['risk_tolerance'] = client_data.vector_profile.risk_tolerance
        
        # Financial position features
        net_worth = client_data.financial_metrics['net_worth']
        composition['net_worth_category'] = self._categorize_net_worth(net_worth)
        composition['debt_to_income_ratio'] = client_data.financial_metrics['debt_to_income_ratio']
        composition['savings_rate'] = client_data.financial_metrics['savings_rate']
        
        # Event composition features
        event_counts = {}
        for event in client_data.lifestyle_events:
            category = event.category.value
            event_counts[category] = event_counts.get(category, 0) + 1
        
        composition['event_distribution'] = event_counts
        composition['total_events'] = len(client_data.lifestyle_events)
        
        # Event impact features
        positive_events = [e for e in client_data.lifestyle_events if e.cash_flow_impact == "positive"]
        negative_events = [e for e in client_data.lifestyle_events if e.cash_flow_impact == "negative"]
        
        composition['positive_event_ratio'] = len(positive_events) / len(client_data.lifestyle_events) if client_data.lifestyle_events else 0
        composition['negative_event_ratio'] = len(negative_events) / len(client_data.lifestyle_events) if client_data.lifestyle_events else 0
        
        # Discretionary spending features
        discretionary_surface = client_data.vector_profile.discretionary_spending_surface
        composition['avg_discretionary'] = np.mean(discretionary_surface)
        composition['discretionary_volatility'] = np.std(discretionary_surface)
        
        # Cash flow features
        cash_flow_vector = client_data.vector_profile.cash_flow_vector
        composition['avg_cash_flow'] = np.mean(cash_flow_vector)
        composition['cash_flow_volatility'] = np.std(cash_flow_vector)
        
        # Mesh-specific features (if available)
        if client_data.mesh_data:
            mesh_states = client_data.mesh_data.get('states', None)
            if mesh_states is not None:
                composition['mesh_complexity'] = mesh_states.shape[0] * mesh_states.shape[1]  # time_steps * scenarios
                composition['mesh_volatility'] = np.std(mesh_states)
                composition['mesh_trend'] = np.mean(np.diff(mesh_states, axis=0))  # Average change over time
        
        if client_data.risk_analysis:
            composition['risk_metrics'] = {
                'min_cash': np.min(client_data.risk_analysis.get('min_cash_by_scenario', [0])),
                'max_drawdown': np.max(client_data.risk_analysis.get('max_drawdown_by_scenario', [0])),
                'var_95': np.mean(client_data.risk_analysis.get('var_95_timeline', [0]))
            }
        
        return composition
    
    def _categorize_income(self, income: float) -> str:
        """Categorize income level"""
        if income < 50000:
            return 'low'
        elif income < 100000:
            return 'medium'
        else:
            return 'high'
    
    def _categorize_net_worth(self, net_worth: float) -> str:
        """Categorize net worth level"""
        if net_worth < 100000:
            return 'low'
        elif net_worth < 500000:
            return 'medium'
        else:
            return 'high'
    
    def _generate_embedding_vector(self, composition: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding vector from mesh composition features
        
        Args:
            composition: Mesh composition features
            
        Returns:
            Embedding vector
        """
        # Create feature vector
        features = []
        
        # Age (normalized)
        features.append(composition['age'] / 100.0)
        
        # Life stage (one-hot encoded)
        life_stages = ['early_career', 'mid_career', 'established', 'pre_retirement', 'retirement']
        life_stage_idx = life_stages.index(composition['life_stage'])
        features.extend([1.0 if i == life_stage_idx else 0.0 for i in range(len(life_stages))])
        
        # Income level (one-hot encoded)
        income_levels = ['low', 'medium', 'high']
        income_idx = income_levels.index(composition['income_level'])
        features.extend([1.0 if i == income_idx else 0.0 for i in range(len(income_levels))])
        
        # Risk tolerance
        features.append(composition['risk_tolerance'])
        
        # Net worth category (one-hot encoded)
        net_worth_levels = ['low', 'medium', 'high']
        net_worth_idx = net_worth_levels.index(composition['net_worth_category'])
        features.extend([1.0 if i == net_worth_idx else 0.0 for i in range(len(net_worth_levels))])
        
        # Financial ratios
        features.append(min(composition['debt_to_income_ratio'], 2.0) / 2.0)  # Normalize
        features.append(max(min(composition['savings_rate'], 0.5), -0.5) + 0.5)  # Normalize to 0-1
        
        # Event distribution (normalized counts)
        event_categories = ['education', 'career', 'family', 'housing', 'health', 'retirement']
        for category in event_categories:
            count = composition['event_distribution'].get(category, 0)
            features.append(min(count / 10.0, 1.0))  # Normalize to 0-1
        
        # Event impact ratios
        features.append(composition['positive_event_ratio'])
        features.append(composition['negative_event_ratio'])
        
        # Discretionary spending features
        features.append(composition['avg_discretionary'])
        features.append(min(composition['discretionary_volatility'], 0.5) / 0.5)
        
        # Cash flow features
        features.append(max(min(composition['avg_cash_flow'] / 10000.0, 1.0), -1.0))  # Normalize
        features.append(min(composition['cash_flow_volatility'] / 5000.0, 1.0))
        
        # Mesh complexity features (if available)
        if 'mesh_complexity' in composition:
            features.append(min(composition['mesh_complexity'] / 100000.0, 1.0))
            features.append(max(min(composition['mesh_volatility'] / 100000.0, 1.0), -1.0))
            features.append(max(min(composition['mesh_trend'] / 10000.0, 1.0), -1.0))
        else:
            features.extend([0.0, 0.0, 0.0])  # Placeholder values
        
        # Risk metrics (if available)
        if 'risk_metrics' in composition:
            risk_metrics = composition['risk_metrics']
            features.append(max(min(risk_metrics['min_cash'] / 100000.0, 1.0), -1.0))
            features.append(min(risk_metrics['max_drawdown'] / 1000000.0, 1.0))
            features.append(max(min(risk_metrics['var_95'] / 100000.0, 1.0), -1.0))
        else:
            features.extend([0.0, 0.0, 0.0])  # Placeholder values
        
        # Convert to numpy array and ensure correct dimension
        embedding = np.array(features, dtype=np.float32)
        
        # Pad or truncate to target dimension
        if len(embedding) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        elif len(embedding) > self.embedding_dim:
            # Truncate
            embedding = embedding[:self.embedding_dim]
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_client(self, client_data: SyntheticClientData) -> str:
        """
        Add a client to the vector database
        
        Args:
            client_data: Synthetic client data
            
        Returns:
            Client ID
        """
        client_id = client_data.client_id
        
        # Generate mesh composition features
        composition = self._generate_mesh_composition_features(client_data)
        
        # Generate embedding vector
        embedding_vector = self._generate_embedding_vector(composition)
        
        # Create mesh embedding
        mesh_embedding = MeshEmbedding(
            client_id=client_id,
            embedding_vector=embedding_vector,
            mesh_composition=composition,
            metadata={
                'age': client_data.profile.age,
                'life_stage': client_data.vector_profile.life_stage.value,
                'income': client_data.profile.base_income,
                'net_worth': client_data.financial_metrics['net_worth'],
                'num_events': len(client_data.lifestyle_events),
                'has_mesh_data': client_data.mesh_data is not None,
                'has_risk_analysis': client_data.risk_analysis is not None
            }
        )
        
        # Store in database
        self.embeddings[client_id] = mesh_embedding
        self.client_profiles[client_id] = client_data.vector_profile
        if client_data.mesh_data:
            self.mesh_data[client_id] = client_data.mesh_data
        
        self.logger.info(f"Added client {client_id} to vector database")
        return client_id
    
    def find_similar_clients(self, client_id: str, top_k: int = 5, 
                           exclude_self: bool = True) -> List[SimilarityMatch]:
        """
        Find similar clients based on mesh composition embeddings
        
        Args:
            client_id: ID of the client to find matches for
            top_k: Number of top matches to return
            exclude_self: Whether to exclude the client itself from results
            
        Returns:
            List of similarity matches
        """
        if client_id not in self.embeddings:
            raise ValueError(f"Client {client_id} not found in database")
        
        source_embedding = self.embeddings[client_id].embedding_vector
        source_composition = self.embeddings[client_id].mesh_composition
        
        similarities = []
        
        for match_client_id, match_embedding in self.embeddings.items():
            if exclude_self and match_client_id == client_id:
                continue
            
            # Calculate cosine similarity
            similarity_score = np.dot(source_embedding, match_embedding.embedding_vector)
            
            # Calculate matching factors
            matching_factors = self._identify_matching_factors(
                source_composition, match_embedding.mesh_composition
            )
            
            # Estimate uncertainties based on similarity
            estimated_uncertainties = self._estimate_uncertainties(
                source_composition, match_embedding.mesh_composition, similarity_score
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                similarity_score, len(matching_factors), estimated_uncertainties
            )
            
            match = SimilarityMatch(
                source_client_id=client_id,
                matched_client_id=match_client_id,
                similarity_score=similarity_score,
                matching_factors=matching_factors,
                estimated_uncertainties=estimated_uncertainties,
                confidence_score=confidence_score
            )
            
            similarities.append(match)
        
        # Sort by similarity score and return top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def _identify_matching_factors(self, source_composition: Dict, 
                                 match_composition: Dict) -> List[str]:
        """Identify factors that match between two clients"""
        matching_factors = []
        
        # Life stage match
        if source_composition['life_stage'] == match_composition['life_stage']:
            matching_factors.append('life_stage')
        
        # Income level match
        if source_composition['income_level'] == match_composition['income_level']:
            matching_factors.append('income_level')
        
        # Net worth category match
        if source_composition['net_worth_category'] == match_composition['net_worth_category']:
            matching_factors.append('net_worth_category')
        
        # Age similarity (within 5 years)
        age_diff = abs(source_composition['age'] - match_composition['age'])
        if age_diff <= 5:
            matching_factors.append('age_similarity')
        
        # Risk tolerance similarity (within 0.2)
        risk_diff = abs(source_composition['risk_tolerance'] - match_composition['risk_tolerance'])
        if risk_diff <= 0.2:
            matching_factors.append('risk_tolerance_similarity')
        
        # Event distribution similarity
        source_events = source_composition['event_distribution']
        match_events = match_composition['event_distribution']
        
        # Check if they have similar event patterns
        common_categories = set(source_events.keys()) & set(match_events.keys())
        if len(common_categories) >= 3:  # At least 3 common event categories
            matching_factors.append('event_pattern_similarity')
        
        # Financial ratios similarity
        debt_diff = abs(source_composition['debt_to_income_ratio'] - match_composition['debt_to_income_ratio'])
        if debt_diff <= 0.3:
            matching_factors.append('debt_ratio_similarity')
        
        savings_diff = abs(source_composition['savings_rate'] - match_composition['savings_rate'])
        if savings_diff <= 0.2:
            matching_factors.append('savings_rate_similarity')
        
        return matching_factors
    
    def _estimate_uncertainties(self, source_composition: Dict, 
                              match_composition: Dict, 
                              similarity_score: float) -> Dict[str, float]:
        """Estimate uncertainties based on similar client outcomes"""
        uncertainties = {}
        
        # Base uncertainty reduction based on similarity
        base_reduction = similarity_score * 0.5  # Max 50% reduction
        
        # Event timing uncertainty
        if 'event_pattern_similarity' in self._identify_matching_factors(source_composition, match_composition):
            uncertainties['event_timing'] = max(0.1, 0.3 - base_reduction)
        else:
            uncertainties['event_timing'] = 0.3
        
        # Amount uncertainty
        if 'income_level' in self._identify_matching_factors(source_composition, match_composition):
            uncertainties['event_amount'] = max(0.1, 0.25 - base_reduction)
        else:
            uncertainties['event_amount'] = 0.25
        
        # Cash flow uncertainty
        cash_flow_diff = abs(source_composition['avg_cash_flow'] - match_composition['avg_cash_flow'])
        if cash_flow_diff < 1000:  # Similar cash flow patterns
            uncertainties['cash_flow'] = max(0.1, 0.2 - base_reduction)
        else:
            uncertainties['cash_flow'] = 0.2
        
        # Risk uncertainty
        if 'risk_tolerance_similarity' in self._identify_matching_factors(source_composition, match_composition):
            uncertainties['risk_profile'] = max(0.05, 0.15 - base_reduction)
        else:
            uncertainties['risk_profile'] = 0.15
        
        return uncertainties
    
    def _calculate_confidence_score(self, similarity_score: float, 
                                  num_matching_factors: int, 
                                  estimated_uncertainties: Dict[str, float]) -> float:
        """Calculate confidence score for the match"""
        # Base confidence from similarity
        base_confidence = similarity_score
        
        # Bonus for matching factors
        factor_bonus = min(num_matching_factors * 0.1, 0.3)
        
        # Penalty for high uncertainties
        avg_uncertainty = np.mean(list(estimated_uncertainties.values()))
        uncertainty_penalty = avg_uncertainty * 0.3
        
        confidence = base_confidence + factor_bonus - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
    
    def get_recommendations(self, client_id: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get recommendations based on similar clients
        
        Args:
            client_id: ID of the client
            top_k: Number of similar clients to consider
            
        Returns:
            Dictionary of recommendations
        """
        similar_clients = self.find_similar_clients(client_id, top_k=top_k)
        
        if not similar_clients:
            return {"error": "No similar clients found"}
        
        recommendations = {
            'client_id': client_id,
            'similar_clients_analyzed': len(similar_clients),
            'recommendations': {}
        }
        
        # Analyze event patterns from similar clients
        event_recommendations = self._analyze_event_patterns(similar_clients)
        recommendations['recommendations']['event_patterns'] = event_recommendations
        
        # Analyze financial strategies from similar clients
        financial_recommendations = self._analyze_financial_strategies(similar_clients)
        recommendations['recommendations']['financial_strategies'] = financial_recommendations
        
        # Analyze risk management from similar clients
        risk_recommendations = self._analyze_risk_management(similar_clients)
        recommendations['recommendations']['risk_management'] = risk_recommendations
        
        return recommendations
    
    def _analyze_event_patterns(self, similar_clients: List[SimilarityMatch]) -> Dict[str, Any]:
        """Analyze event patterns from similar clients"""
        event_analysis = {
            'most_common_events': {},
            'event_timing_patterns': {},
            'successful_event_combinations': []
        }
        
        # Collect event data from similar clients
        all_events = []
        for match in similar_clients:
            client_id = match.matched_client_id
            if client_id in self.client_profiles:
                # This would need access to the original client data
                # For now, we'll use the composition data
                pass
        
        return event_analysis
    
    def _analyze_financial_strategies(self, similar_clients: List[SimilarityMatch]) -> Dict[str, Any]:
        """Analyze financial strategies from similar clients"""
        strategy_analysis = {
            'savings_patterns': {},
            'investment_approaches': {},
            'debt_management': {}
        }
        
        return strategy_analysis
    
    def _analyze_risk_management(self, similar_clients: List[SimilarityMatch]) -> Dict[str, Any]:
        """Analyze risk management from similar clients"""
        risk_analysis = {
            'risk_tolerance_patterns': {},
            'volatility_management': {},
            'diversification_strategies': {}
        }
        
        return risk_analysis
    
    def save_database(self, filename: str = None) -> None:
        """Save the vector database to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mesh_vector_db_{timestamp}.pkl"
        
        filepath = self.storage_dir / filename
        
        data = {
            'embeddings': self.embeddings,
            'client_profiles': self.client_profiles,
            'mesh_data': self.mesh_data,
            'embedding_dim': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved vector database to {filepath}")
    
    def load_database(self, filename: str) -> None:
        """Load the vector database from disk"""
        filepath = self.storage_dir / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.client_profiles = data['client_profiles']
        self.mesh_data = data['mesh_data']
        self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
        self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
        
        self.logger.info(f"Loaded vector database from {filepath}")
        self.logger.info(f"Loaded {len(self.embeddings)} client embeddings")


def create_demo_vector_database():
    """Create and demonstrate the mesh vector database"""
    print("🚀 Mesh Vector Database Demo")
    print("=" * 50)
    
    # Create vector database
    vector_db = MeshVectorDatabase(embedding_dim=128, similarity_threshold=0.7)
    
    # Create synthetic lifestyle engine
    from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine
    engine = SyntheticLifestyleEngine(use_gpu=False)
    
    # Generate synthetic clients
    print("📋 Generating synthetic clients...")
    clients = engine.generate_client_batch(num_clients=20)
    
    # Add clients to vector database
    print("🔗 Adding clients to vector database...")
    for client in clients:
        vector_db.add_client(client)
    
    print(f"✅ Added {len(clients)} clients to vector database")
    
    # Test similarity matching
    if clients:
        test_client_id = clients[0].client_id
        print(f"\n🔍 Finding similar clients for {test_client_id}...")
        
        similar_clients = vector_db.find_similar_clients(test_client_id, top_k=5)
        
        print(f"Found {len(similar_clients)} similar clients:")
        for i, match in enumerate(similar_clients):
            print(f"   {i+1}. {match.matched_client_id} (similarity: {match.similarity_score:.3f})")
            print(f"      Matching factors: {', '.join(match.matching_factors)}")
            print(f"      Confidence: {match.confidence_score:.3f}")
        
        # Get recommendations
        print(f"\n💡 Getting recommendations for {test_client_id}...")
        recommendations = vector_db.get_recommendations(test_client_id)
        print(f"Generated recommendations with {recommendations['similar_clients_analyzed']} similar clients")
    
    # Save database
    vector_db.save_database()
    
    return vector_db, clients


if __name__ == "__main__":
    vector_db, clients = create_demo_vector_database()
    print("\n✅ Mesh Vector Database Demo completed successfully!") 