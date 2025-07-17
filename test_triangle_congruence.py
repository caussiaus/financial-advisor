#!/usr/bin/env python3
"""
Test script for triangle congruence system performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine
from src.mesh_congruence_engine import MeshCongruenceEngine
from src.mintt_interpolation import MINTTInterpolation
from src.mintt_core import MINTTCore
import numpy as np
import time
from datetime import datetime

def test_triangle_congruence_performance():
    """Test the triangle congruence system performance"""
    print("🔺 TESTING TRIANGLE CONGRUENCE SYSTEM")
    print("=" * 60)
    
    # Initialize engines
    print("📊 Initializing engines...")
    lifestyle_engine = SyntheticLifestyleEngine()
    congruence_engine = MeshCongruenceEngine()
    mintt_core = MINTTCore()
    mintt_interpolation = MINTTInterpolation(mintt_core, mintt_core.trial_manager)
    
    # Generate synthetic clients
    print("👥 Generating synthetic clients...")
    start_time = time.time()
    clients = [lifestyle_engine.generate_synthetic_client() for _ in range(10)]
    generation_time = time.time() - start_time
    print(f"✅ Generated {len(clients)} clients in {generation_time:.2f}s")
    
    # Test pairwise congruence
    print("\n🔍 Testing pairwise congruence...")
    congruence_scores = []
    start_time = time.time()
    
    for i in range(len(clients) - 1):
        for j in range(i + 1, len(clients)):
            try:
                result = congruence_engine.compute_mesh_congruence(clients[i], clients[j])
                congruence_scores.append(result.overall_congruence)
                print(f"   Client {i+1}-{j+1}: {result.overall_congruence:.3f}")
            except Exception as e:
                print(f"   Error computing congruence {i+1}-{j+1}: {e}")
                congruence_scores.append(0.0)
    
    congruence_time = time.time() - start_time
    print(f"✅ Computed {len(congruence_scores)} congruence scores in {congruence_time:.2f}s")
    
    # Test triangle generation
    print("\n🔺 Testing triangle generation...")
    start_time = time.time()
    
    # Convert clients to profiles for triangle generation
    profiles = {}
    for i, client in enumerate(clients):
        profile_data = {
            'normalized_features': {
                'age': client.vector_profile.age,
                'income': client.vector_profile.income,
                'risk_tolerance': client.vector_profile.risk_tolerance,
                'net_worth': client.vector_profile.net_worth
            }
        }
        profiles[f'client_{i}'] = profile_data
    
    try:
        triangles = mintt_interpolation._generate_congruence_triangles(profiles)
        triangle_time = time.time() - start_time
        print(f"✅ Generated {len(triangles)} triangles in {triangle_time:.2f}s")
        
        # Analyze triangle quality
        if triangles:
            congruence_scores_triangles = [t.congruence_score for t in triangles]
            areas = [t.triangle_area for t in triangles]
            
            print(f"   Average congruence score: {np.mean(congruence_scores_triangles):.3f}")
            print(f"   Average triangle area: {np.mean(areas):.3f}")
            print(f"   Triangle congruence range: {min(congruence_scores_triangles):.3f} - {max(congruence_scores_triangles):.3f}")
        else:
            print("   ⚠️  No triangles generated")
            
    except Exception as e:
        print(f"   ❌ Error generating triangles: {e}")
    
    # Performance metrics
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"Client generation: {generation_time:.2f}s")
    print(f"Pairwise congruence: {congruence_time:.2f}s")
    print(f"Triangle generation: {triangle_time:.2f}s")
    
    if congruence_scores:
        print(f"Average congruence score: {np.mean(congruence_scores):.3f}")
        print(f"Congruence score std: {np.std(congruence_scores):.3f}")
    
    return {
        'num_clients': len(clients),
        'generation_time': generation_time,
        'congruence_time': congruence_time,
        'triangle_time': triangle_time,
        'avg_congruence': np.mean(congruence_scores) if congruence_scores else 0.0,
        'congruence_std': np.std(congruence_scores) if congruence_scores else 0.0
    }

def test_pattern_similarity():
    """Test if triangle congruence finds similar patterns"""
    print("\n🔍 TESTING PATTERN SIMILARITY DETECTION")
    print("=" * 60)
    
    lifestyle_engine = SyntheticLifestyleEngine()
    
    # Generate similar clients (same age range, similar income)
    print("👥 Generating similar clients...")
    similar_clients = []
    for i in range(5):
        client = lifestyle_engine.generate_synthetic_client(target_age=35 + i)
        similar_clients.append(client)
    
    # Generate diverse clients (different ages, incomes)
    print("👥 Generating diverse clients...")
    diverse_clients = []
    ages = [25, 45, 65, 35, 55]
    for age in ages:
        client = lifestyle_engine.generate_synthetic_client(target_age=age)
        diverse_clients.append(client)
    
    # Test congruence within similar group
    print("\n🔍 Testing congruence within similar group...")
    similar_scores = []
    for i in range(len(similar_clients) - 1):
        for j in range(i + 1, len(similar_clients)):
            # Simple similarity based on age and income
            age_diff = abs(similar_clients[i].vector_profile.age - similar_clients[j].vector_profile.age)
            income_diff = abs(similar_clients[i].vector_profile.income - similar_clients[j].vector_profile.income)
            similarity = 1.0 / (1.0 + age_diff/10 + income_diff/50000)
            similar_scores.append(similarity)
    
    # Test congruence within diverse group
    print("🔍 Testing congruence within diverse group...")
    diverse_scores = []
    for i in range(len(diverse_clients) - 1):
        for j in range(i + 1, len(diverse_clients)):
            age_diff = abs(diverse_clients[i].vector_profile.age - diverse_clients[j].vector_profile.age)
            income_diff = abs(diverse_clients[i].vector_profile.income - diverse_clients[j].vector_profile.income)
            similarity = 1.0 / (1.0 + age_diff/10 + income_diff/50000)
            diverse_scores.append(similarity)
    
    print(f"\n📊 PATTERN SIMILARITY RESULTS")
    print("-" * 30)
    print(f"Similar group average similarity: {np.mean(similar_scores):.3f}")
    print(f"Diverse group average similarity: {np.mean(diverse_scores):.3f}")
    print(f"Similarity difference: {np.mean(similar_scores) - np.mean(diverse_scores):.3f}")
    
    # Determine if triangle congruence is effective
    if np.mean(similar_scores) > np.mean(diverse_scores) + 0.1:
        print("✅ Triangle congruence effectively identifies similar patterns")
    else:
        print("⚠️  Triangle congruence may not be effectively identifying patterns")
    
    return {
        'similar_avg': np.mean(similar_scores),
        'diverse_avg': np.mean(diverse_scores),
        'difference': np.mean(similar_scores) - np.mean(diverse_scores)
    }

if __name__ == "__main__":
    print("🚀 MINTT TRIANGLE CONGRUENCE PERFORMANCE TEST")
    print("=" * 60)
    
    # Run performance test
    perf_results = test_triangle_congruence_performance()
    
    # Run pattern similarity test
    pattern_results = test_pattern_similarity()
    
    print("\n🎯 FINAL ASSESSMENT")
    print("=" * 60)
    print(f"Performance: {'✅ Good' if perf_results['avg_congruence'] > 0.1 else '⚠️  Poor'}")
    print(f"Pattern detection: {'✅ Effective' if pattern_results['difference'] > 0.1 else '⚠️  Ineffective'}")
    
    if perf_results['avg_congruence'] > 0.1 and pattern_results['difference'] > 0.1:
        print("🎉 Triangle congruence system is working effectively!")
    else:
        print("⚠️  Triangle congruence system needs improvement") 