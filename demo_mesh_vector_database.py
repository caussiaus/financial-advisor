#!/usr/bin/env python3
"""
Mesh Vector Database Demo

This script demonstrates the complete vector database system that:
1. Generates embeddings from mesh network composition
2. Finds nearest matched clients based on similarity
3. Uses similarity matching to estimate uncertain factors
4. Provides recommendations based on similar client outcomes
5. Shows how to use vector embeddings to map clients to nearest matches
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.mesh_vector_database import MeshVectorDatabase, SimilarityMatch
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.json_to_vector_converter import LifeStage, EventCategory


def create_comprehensive_vector_database():
    """Create a comprehensive vector database with diverse client profiles"""
    print("🚀 Creating Comprehensive Mesh Vector Database")
    print("=" * 60)
    
    # Create vector database
    vector_db = MeshVectorDatabase(embedding_dim=128, similarity_threshold=0.7)
    
    # Create synthetic lifestyle engine
    engine = SyntheticLifestyleEngine(use_gpu=False)
    
    # Generate diverse client population
    print("📋 Generating diverse client population...")
    
    # Generate clients for different life stages and ages
    clients = []
    
    # Early career clients (22-30)
    for age in [22, 25, 28, 30]:
        client = engine.generate_synthetic_client(target_age=age)
        clients.append(client)
    
    # Mid career clients (31-45)
    for age in [32, 35, 38, 42, 45]:
        client = engine.generate_synthetic_client(target_age=age)
        clients.append(client)
    
    # Established clients (46-60)
    for age in [48, 52, 55, 58]:
        client = engine.generate_synthetic_client(target_age=age)
        clients.append(client)
    
    # Pre-retirement clients (61-67)
    for age in [62, 65]:
        client = engine.generate_synthetic_client(target_age=age)
        clients.append(client)
    
    # Retirement clients (68+)
    for age in [70, 72]:
        client = engine.generate_synthetic_client(target_age=age)
        clients.append(client)
    
    print(f"✅ Generated {len(clients)} diverse clients")
    
    # Process subset with mesh engine for more detailed embeddings
    print("\n🌐 Processing clients with mesh engine...")
    mesh_clients = clients[:10]  # Process first 10 for mesh analysis
    
    for i, client in enumerate(mesh_clients):
        print(f"   Processing {client.client_id} ({i+1}/{len(mesh_clients)})...")
        client = engine.process_with_mesh_engine(
            client, 
            num_scenarios=300,  # Reduced for demo
            time_horizon_years=3
        )
    
    # Add all clients to vector database
    print("\n🔗 Adding clients to vector database...")
    for client in clients:
        vector_db.add_client(client)
    
    print(f"✅ Added {len(clients)} clients to vector database")
    
    return vector_db, clients


def analyze_similarity_matching(vector_db: MeshVectorDatabase, clients: List[SyntheticClientData]):
    """Analyze similarity matching across different client types"""
    print("\n🔍 Similarity Matching Analysis")
    print("=" * 50)
    
    # Test similarity matching for different client types
    test_cases = [
        ("Early Career", clients[0].client_id),
        ("Mid Career", clients[4].client_id),
        ("Established", clients[9].client_id),
        ("Pre-Retirement", clients[13].client_id),
        ("Retirement", clients[15].client_id)
    ]
    
    for client_type, client_id in test_cases:
        print(f"\n📊 {client_type} Client Analysis")
        print(f"   Client ID: {client_id}")
        
        # Find similar clients
        similar_clients = vector_db.find_similar_clients(client_id, top_k=5)
        
        print(f"   Found {len(similar_clients)} similar clients:")
        
        for i, match in enumerate(similar_clients):
            print(f"      {i+1}. {match.matched_client_id}")
            print(f"         Similarity Score: {match.similarity_score:.3f}")
            print(f"         Matching Factors: {', '.join(match.matching_factors)}")
            print(f"         Confidence: {match.confidence_score:.3f}")
            
            # Show uncertainty estimates
            uncertainties = match.estimated_uncertainties
            print(f"         Uncertainty Estimates:")
            for factor, uncertainty in uncertainties.items():
                print(f"            {factor}: {uncertainty:.3f}")


def demonstrate_uncertainty_estimation(vector_db: MeshVectorDatabase, clients: List[SyntheticClientData]):
    """Demonstrate how the vector database estimates uncertain factors"""
    print("\n🎯 Uncertainty Estimation Demonstration")
    print("=" * 50)
    
    # Select a test client
    test_client = clients[5]  # Mid-career client
    test_client_id = test_client.client_id
    
    print(f"📋 Test Client: {test_client_id}")
    print(f"   Age: {test_client.profile.age}")
    print(f"   Life Stage: {test_client.vector_profile.life_stage.value}")
    print(f"   Income: ${test_client.profile.base_income:,.0f}")
    print(f"   Net Worth: ${test_client.financial_metrics['net_worth']:,.0f}")
    print(f"   Risk Tolerance: {test_client.vector_profile.risk_tolerance:.3f}")
    
    # Find similar clients
    similar_clients = vector_db.find_similar_clients(test_client_id, top_k=3)
    
    print(f"\n🔍 Found {len(similar_clients)} similar clients for uncertainty estimation:")
    
    for i, match in enumerate(similar_clients):
        print(f"\n   Match {i+1}: {match.matched_client_id}")
        print(f"      Similarity: {match.similarity_score:.3f}")
        print(f"      Confidence: {match.confidence_score:.3f}")
        
        # Show how uncertainties are reduced based on similarity
        uncertainties = match.estimated_uncertainties
        print(f"      Estimated Uncertainties:")
        for factor, uncertainty in uncertainties.items():
            base_uncertainty = {
                'event_timing': 0.3,
                'event_amount': 0.25,
                'cash_flow': 0.2,
                'risk_profile': 0.15
            }.get(factor, 0.2)
            
            reduction = base_uncertainty - uncertainty
            print(f"         {factor}: {uncertainty:.3f} (reduced by {reduction:.3f})")
        
        # Show matching factors that contribute to uncertainty reduction
        print(f"      Matching Factors: {', '.join(match.matching_factors)}")


def create_similarity_visualization(vector_db: MeshVectorDatabase, clients: List[SyntheticClientData]):
    """Create visualization of client similarities"""
    print("\n📈 Creating Similarity Visualization")
    print("=" * 50)
    
    # Create similarity matrix for a subset of clients
    subset_clients = clients[:10]  # First 10 clients
    client_ids = [client.client_id for client in subset_clients]
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(client_ids), len(client_ids)))
    
    for i, client_id_1 in enumerate(client_ids):
        for j, client_id_2 in enumerate(client_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Find similarity between these two clients
                similar_clients = vector_db.find_similar_clients(client_id_1, top_k=len(client_ids))
                for match in similar_clients:
                    if match.matched_client_id == client_id_2:
                        similarity_matrix[i, j] = match.similarity_score
                        break
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Create labels with client info
    labels = []
    for client in subset_clients:
        label = f"{client.client_id}\n({client.profile.age}, {client.vector_profile.life_stage.value[:3]})"
        labels.append(label)
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Similarity Score'})
    
    plt.title('Client Similarity Matrix\n(Based on Mesh Network Composition)')
    plt.xlabel('Client ID (Age, Life Stage)')
    plt.ylabel('Client ID (Age, Life Stage)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save visualization
    plt.savefig('data/outputs/visuals/client_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Saved similarity matrix to data/outputs/visuals/client_similarity_matrix.png")


def demonstrate_recommendations(vector_db: MeshVectorDatabase, clients: List[SyntheticClientData]):
    """Demonstrate recommendation generation based on similar clients"""
    print("\n💡 Recommendation Generation Demo")
    print("=" * 50)
    
    # Test recommendations for different client types
    test_cases = [
        ("Early Career", clients[0].client_id),
        ("Mid Career", clients[5].client_id),
        ("Established", clients[10].client_id)
    ]
    
    for client_type, client_id in test_cases:
        print(f"\n📊 {client_type} Client Recommendations")
        print(f"   Client ID: {client_id}")
        
        # Get recommendations
        recommendations = vector_db.get_recommendations(client_id, top_k=3)
        
        if 'error' in recommendations:
            print(f"   Error: {recommendations['error']}")
        else:
            print(f"   Analyzed {recommendations['similar_clients_analyzed']} similar clients")
            
            # Show recommendation categories
            for category, data in recommendations['recommendations'].items():
                print(f"   {category.replace('_', ' ').title()}: {len(data)} insights")


def analyze_embedding_quality(vector_db: MeshVectorDatabase, clients: List[SyntheticClientData]):
    """Analyze the quality of embeddings and similarity matching"""
    print("\n🔬 Embedding Quality Analysis")
    print("=" * 50)
    
    # Analyze embedding distributions
    embeddings = []
    life_stages = []
    ages = []
    
    for client in clients:
        client_id = client.client_id
        if client_id in vector_db.embeddings:
            embedding = vector_db.embeddings[client_id].embedding_vector
            embeddings.append(embedding)
            life_stages.append(client.vector_profile.life_stage.value)
            ages.append(client.profile.age)
    
    embeddings = np.array(embeddings)
    
    print(f"📊 Embedding Statistics:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Mean embedding norm: {np.mean([np.linalg.norm(emb) for emb in embeddings]):.3f}")
    print(f"   Embedding std: {np.std(embeddings):.3f}")
    
    # Analyze similarity distribution
    all_similarities = []
    for i, client in enumerate(clients[:5]):  # Test first 5 clients
        similar_clients = vector_db.find_similar_clients(client.client_id, top_k=5)
        similarities = [match.similarity_score for match in similar_clients]
        all_similarities.extend(similarities)
    
    print(f"\n📈 Similarity Score Statistics:")
    print(f"   Mean similarity: {np.mean(all_similarities):.3f}")
    print(f"   Std similarity: {np.std(all_similarities):.3f}")
    print(f"   Min similarity: {np.min(all_similarities):.3f}")
    print(f"   Max similarity: {np.max(all_similarities):.3f}")
    
    # Analyze by life stage
    life_stage_similarities = {}
    for client in clients[:10]:  # Test first 10 clients
        life_stage = client.vector_profile.life_stage.value
        similar_clients = vector_db.find_similar_clients(client.client_id, top_k=3)
        similarities = [match.similarity_score for match in similar_clients]
        
        if life_stage not in life_stage_similarities:
            life_stage_similarities[life_stage] = []
        life_stage_similarities[life_stage].extend(similarities)
    
    print(f"\n📊 Similarity by Life Stage:")
    for life_stage, similarities in life_stage_similarities.items():
        print(f"   {life_stage}: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")


def main():
    """Main demo function"""
    print("🚀 Mesh Vector Database Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the vector database system for finding nearest matched")
    print("clients and estimating uncertain factors using mesh network composition embeddings.")
    print("=" * 80)
    
    # Create comprehensive vector database
    vector_db, clients = create_comprehensive_vector_database()
    
    # Analyze similarity matching
    analyze_similarity_matching(vector_db, clients)
    
    # Demonstrate uncertainty estimation
    demonstrate_uncertainty_estimation(vector_db, clients)
    
    # Create similarity visualization
    try:
        create_similarity_visualization(vector_db, clients)
    except Exception as e:
        print(f"⚠️ Could not create similarity visualization: {e}")
    
    # Demonstrate recommendations
    demonstrate_recommendations(vector_db, clients)
    
    # Analyze embedding quality
    analyze_embedding_quality(vector_db, clients)
    
    # Save database
    vector_db.save_database("comprehensive_mesh_vector_db.pkl")
    
    # Summary
    print(f"\n📊 Summary Statistics")
    print("=" * 40)
    print(f"Total clients in database: {len(vector_db.embeddings)}")
    print(f"Embedding dimension: {vector_db.embedding_dim}")
    print(f"Similarity threshold: {vector_db.similarity_threshold}")
    print(f"Life stages represented: {len(set(c.vector_profile.life_stage.value for c in clients))}")
    print(f"Age range: {min(c.profile.age for c in clients)} - {max(c.profile.age for c in clients)}")
    
    print("\n✅ Mesh Vector Database Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   - Mesh network composition embedding generation")
    print("   - Vector similarity search for client matching")
    print("   - Uncertainty estimation through similar client analysis")
    print("   - Recommendation engine based on mesh patterns")
    print("   - Quality analysis of embeddings and similarity matching")


if __name__ == "__main__":
    main() 