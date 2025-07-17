"""
Mesh Congruence System Demo

This demo showcases the complete mesh congruence system:
1. Mesh congruence algorithms (Delaunay triangulation, CVT, edge collapse)
2. Trial people data integration
3. Backtesting framework
4. Comprehensive testing
5. Performance analysis and recommendations

Key Features:
- Complete system demonstration
- Trial people data processing
- Mesh congruence analysis
- Backtesting validation
- Performance benchmarking
- Statistical validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our mesh congruence components
from src.mesh_congruence_engine import MeshCongruenceEngine, create_demo_congruence_engine
from src.mesh_backtesting_framework import MeshBacktestingFramework, create_demo_backtesting_framework
from src.comprehensive_mesh_testing import ComprehensiveMeshTesting, create_demo_comprehensive_testing
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine
from src.mesh_vector_database import MeshVectorDatabase


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def demo_mesh_congruence_algorithms():
    """Demonstrate mesh congruence algorithms"""
    print("\n" + "="*60)
    print("DEMO: Mesh Congruence Algorithms")
    print("="*60)
    
    # Create congruence engine
    engine = MeshCongruenceEngine()
    
    # Generate test data
    test_points = np.random.rand(100, 2)
    print(f"Generated {len(test_points)} test points")
    
    # Test Delaunay triangulation
    print("\n1. Testing Delaunay Triangulation...")
    triangulation = engine.compute_delaunay_triangulation(test_points)
    print(f"   - Created {len(triangulation.simplices)} triangles")
    print(f"   - Points shape: {triangulation.points.shape}")
    
    # Test CVT optimization
    print("\n2. Testing Centroidal Voronoi Tessellation...")
    cvt_points = engine.compute_centroidal_voronoi_tessellation(test_points)
    print(f"   - Optimized {len(cvt_points)} points")
    print(f"   - CVT points shape: {cvt_points.shape}")
    
    # Test edge collapse efficiency
    print("\n3. Testing Edge Collapse Efficiency...")
    efficiency = engine.compute_edge_collapse_efficiency(triangulation)
    print(f"   - Edge collapse efficiency: {efficiency:.3f}")
    
    return engine


def demo_trial_people_integration():
    """Demonstrate trial people integration"""
    print("\n" + "="*60)
    print("DEMO: Trial People Integration")
    print("="*60)
    
    # Create testing framework to load trial people
    testing = ComprehensiveMeshTesting()
    
    # Load trial people data
    print("Loading trial people data...")
    trial_people = testing.load_trial_people_data()
    print(f"Loaded {len(trial_people)} trial people")
    
    # Show sample trial person
    if trial_people:
        sample_person = trial_people[0]
        print(f"\nSample trial person: {sample_person['name']}")
        print(f"Financial profile keys: {list(sample_person['financial_profile'].keys())}")
        print(f"Goals: {sample_person['goals']}")
        print(f"Lifestyle events: {len(sample_person['lifestyle_events'])} events")
    
    # Convert to synthetic data
    print("\nConverting trial people to synthetic data...")
    synthetic_clients = testing.convert_trial_people_to_synthetic_data(trial_people)
    print(f"Converted {len(synthetic_clients)} trial people to synthetic data")
    
    # Show sample synthetic client
    if synthetic_clients:
        sample_client = synthetic_clients[0]
        print(f"\nSample synthetic client: {sample_client.client_id}")
        print(f"Age: {sample_client.profile.age}")
        print(f"Life stage: {sample_client.vector_profile.life_stage.value}")
        print(f"Risk tolerance: {sample_client.vector_profile.risk_tolerance:.2f}")
        print(f"Lifestyle events: {len(sample_client.lifestyle_events)}")
    
    return synthetic_clients


def demo_mesh_congruence_analysis(synthetic_clients):
    """Demonstrate mesh congruence analysis"""
    print("\n" + "="*60)
    print("DEMO: Mesh Congruence Analysis")
    print("="*60)
    
    if len(synthetic_clients) < 2:
        print("Need at least 2 clients for congruence analysis")
        return
    
    # Create congruence engine
    engine = MeshCongruenceEngine()
    
    # Analyze congruence between clients
    print("Analyzing mesh congruence between clients...")
    congruence_results = []
    
    for i in range(min(5, len(synthetic_clients))):
        for j in range(i + 1, min(i + 3, len(synthetic_clients))):
            client_1 = synthetic_clients[i]
            client_2 = synthetic_clients[j]
            
            print(f"\nAnalyzing congruence between {client_1.client_id} and {client_2.client_id}...")
            
            result = engine.compute_mesh_congruence(client_1, client_2)
            congruence_results.append(result)
            
            print(f"  - Overall congruence: {result.overall_congruence:.3f}")
            print(f"  - Triangulation quality: {result.triangulation_quality:.3f}")
            print(f"  - Density distribution: {result.density_distribution_score:.3f}")
            print(f"  - Edge collapse efficiency: {result.edge_collapse_efficiency:.3f}")
            print(f"  - Matching factors: {result.matching_factors}")
    
    # Summary statistics
    if congruence_results:
        congruence_scores = [r.overall_congruence for r in congruence_results]
        print(f"\nCongruence Analysis Summary:")
        print(f"  - Average congruence: {np.mean(congruence_scores):.3f}")
        print(f"  - Standard deviation: {np.std(congruence_scores):.3f}")
        print(f"  - Min congruence: {np.min(congruence_scores):.3f}")
        print(f"  - Max congruence: {np.max(congruence_scores):.3f}")
    
    return congruence_results


def demo_backtesting_framework(synthetic_clients):
    """Demonstrate backtesting framework"""
    print("\n" + "="*60)
    print("DEMO: Backtesting Framework")
    print("="*60)
    
    if not synthetic_clients:
        print("No synthetic clients available for backtesting")
        return
    
    # Create backtesting framework
    framework = MeshBacktestingFramework()
    
    # Run backtest on first client
    client = synthetic_clients[0]
    print(f"Running backtest on client: {client.client_id}")
    
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    print(f"Test period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run comprehensive backtest
    report = framework.run_comprehensive_backtest(
        client, start_date, end_date, num_scenarios=30
    )
    
    print(f"\nBacktest Results:")
    print(f"  - Scenarios tested: {len(report.scenarios)}")
    print(f"  - Total return: {report.performance_metrics.total_return:.2%}")
    print(f"  - Sharpe ratio: {report.performance_metrics.sharpe_ratio:.2f}")
    print(f"  - Max drawdown: {report.performance_metrics.max_drawdown:.2%}")
    print(f"  - Volatility: {report.performance_metrics.volatility:.2%}")
    print(f"  - Congruence stability: {report.performance_metrics.congruence_stability:.3f}")
    print(f"  - Recommendation accuracy: {report.performance_metrics.recommendation_accuracy:.2%}")
    
    # Risk analysis
    print(f"\nRisk Analysis:")
    print(f"  - VaR (95%): {report.risk_analysis['var_95']:.2%}")
    print(f"  - CVaR (95%): {report.risk_analysis['cvar_95']:.2%}")
    print(f"  - Calmar ratio: {report.risk_analysis['calmar_ratio']:.2f}")
    
    # Generate report
    report_file = framework.generate_backtest_report(report)
    print(f"\nBacktest report generated: {report_file}")
    
    return report


def demo_comprehensive_testing(synthetic_clients):
    """Demonstrate comprehensive testing"""
    print("\n" + "="*60)
    print("DEMO: Comprehensive Testing")
    print("="*60)
    
    # Create testing framework
    testing = ComprehensiveMeshTesting()
    
    # Run comprehensive tests
    print("Running comprehensive mesh testing...")
    report = testing.run_comprehensive_tests()
    
    print(f"\nComprehensive Test Results:")
    print(f"  - Overall status: {report.overall_status}")
    print(f"  - Total tests: {report.total_tests}")
    print(f"  - Passed: {report.passed_tests}")
    print(f"  - Failed: {report.failed_tests}")
    print(f"  - Warnings: {report.warning_tests}")
    print(f"  - Success rate: {report.execution_summary['success_rate']:.2%}")
    print(f"  - Total execution time: {report.execution_summary['total_execution_time']:.2f}s")
    
    print(f"\nIndividual Test Results:")
    for test_result in report.test_results:
        status_icon = "✅" if test_result.test_status == 'PASS' else "⚠️" if test_result.test_status == 'WARNING' else "❌"
        print(f"  {status_icon} {test_result.test_name}: {test_result.test_status} ({test_result.execution_time:.2f}s)")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    return report


def demo_performance_analysis(synthetic_clients):
    """Demonstrate performance analysis"""
    print("\n" + "="*60)
    print("DEMO: Performance Analysis")
    print("="*60)
    
    if len(synthetic_clients) < 2:
        print("Need at least 2 clients for performance analysis")
        return
    
    # Create congruence engine
    engine = MeshCongruenceEngine()
    
    # Performance benchmarking
    print("Running performance benchmarks...")
    
    computation_times = []
    congruence_scores = []
    
    # Test congruence computation performance
    for i in range(min(10, len(synthetic_clients))):
        for j in range(i + 1, min(i + 3, len(synthetic_clients))):
            import time
            start_time = time.time()
            
            result = engine.compute_mesh_congruence(synthetic_clients[i], synthetic_clients[j])
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            computation_times.append(computation_time)
            congruence_scores.append(result.overall_congruence)
    
    print(f"\nPerformance Analysis:")
    print(f"  - Average computation time: {np.mean(computation_times):.3f}s")
    print(f"  - Min computation time: {np.min(computation_times):.3f}s")
    print(f"  - Max computation time: {np.max(computation_times):.3f}s")
    print(f"  - Average congruence score: {np.mean(congruence_scores):.3f}")
    print(f"  - Congruence score std: {np.std(congruence_scores):.3f}")
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    print(f"  - Sample size: {len(congruence_scores)}")
    print(f"  - Correlation with computation time: {np.corrcoef(computation_times, congruence_scores)[0,1]:.3f}")
    
    return {
        'computation_times': computation_times,
        'congruence_scores': congruence_scores
    }


def demo_recommendation_system(synthetic_clients):
    """Demonstrate recommendation system"""
    print("\n" + "="*60)
    print("DEMO: Recommendation System")
    print("="*60)
    
    if not synthetic_clients:
        print("No synthetic clients available for recommendations")
        return
    
    # Create vector database
    vector_db = MeshVectorDatabase()
    
    # Add clients to database
    print("Adding clients to vector database...")
    for client in synthetic_clients[:10]:  # Add first 10 clients
        vector_db.add_client(client)
    
    print(f"Added {len(vector_db.embeddings)} clients to database")
    
    # Generate recommendations for first client
    if synthetic_clients:
        target_client = synthetic_clients[0]
        print(f"\nGenerating recommendations for {target_client.client_id}...")
        
        # Find similar clients
        similar_clients = vector_db.find_similar_clients(target_client.client_id, top_k=3)
        
        print(f"Found {len(similar_clients)} similar clients:")
        for i, match in enumerate(similar_clients, 1):
            print(f"  {i}. {match.matched_client_id} (similarity: {match.similarity_score:.3f})")
            print(f"     Matching factors: {match.matching_factors}")
            print(f"     Estimated uncertainties: {match.estimated_uncertainties}")
        
        # Get recommendations
        recommendations = vector_db.get_recommendations(target_client.client_id, top_k=3)
        
        print(f"\nRecommendations for {target_client.client_id}:")
        for category, recs in recommendations.items():
            print(f"  {category}:")
            for rec in recs:
                print(f"    - {rec}")
    
    return vector_db


def create_visualization_dashboard(synthetic_clients, congruence_results, performance_data):
    """Create visualization dashboard"""
    print("\n" + "="*60)
    print("DEMO: Visualization Dashboard")
    print("="*60)
    
    # Create dashboard directory
    dashboard_dir = Path("data/outputs/visual_timelines")
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mesh Congruence System Dashboard', fontsize=16)
    
    # 1. Congruence scores distribution
    if congruence_results:
        congruence_scores = [r.overall_congruence for r in congruence_results]
        axes[0, 0].hist(congruence_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Congruence Scores Distribution')
        axes[0, 0].set_xlabel('Congruence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(congruence_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(congruence_scores):.3f}')
        axes[0, 0].legend()
    
    # 2. Performance analysis
    if performance_data:
        axes[0, 1].scatter(performance_data['computation_times'], 
                           performance_data['congruence_scores'], alpha=0.6)
        axes[0, 1].set_title('Performance vs Congruence')
        axes[0, 1].set_xlabel('Computation Time (s)')
        axes[0, 1].set_ylabel('Congruence Score')
        
        # Add trend line
        z = np.polyfit(performance_data['computation_times'], 
                       performance_data['congruence_scores'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(performance_data['computation_times'], 
                        p(performance_data['computation_times']), "r--", alpha=0.8)
    
    # 3. Client demographics
    if synthetic_clients:
        ages = [client.profile.age for client in synthetic_clients]
        life_stages = [client.vector_profile.life_stage.value for client in synthetic_clients]
        
        # Count life stages
        stage_counts = {}
        for stage in life_stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        axes[1, 0].bar(stage_counts.keys(), stage_counts.values(), color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Client Distribution by Life Stage')
        axes[1, 0].set_xlabel('Life Stage')
        axes[1, 0].set_ylabel('Number of Clients')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Age distribution
    if synthetic_clients:
        axes[1, 1].hist(ages, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Client Age Distribution')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(ages), color='red', linestyle='--', 
                           label=f'Mean Age: {np.mean(ages):.1f}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save dashboard
    dashboard_file = dashboard_dir / "mesh_congruence_dashboard.png"
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to: {dashboard_file}")
    
    plt.show()


def main():
    """Main demo function"""
    print("Mesh Congruence System Demo")
    print("="*60)
    print("This demo showcases the complete mesh congruence system with:")
    print("- Advanced mesh congruence algorithms")
    print("- Trial people data integration")
    print("- Backtesting framework")
    print("- Comprehensive testing")
    print("- Performance analysis")
    print("- Recommendation system")
    print("="*60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Demo 1: Mesh congruence algorithms
        congruence_engine = demo_mesh_congruence_algorithms()
        
        # Demo 2: Trial people integration
        synthetic_clients = demo_trial_people_integration()
        
        # Demo 3: Mesh congruence analysis
        congruence_results = demo_mesh_congruence_analysis(synthetic_clients)
        
        # Demo 4: Backtesting framework
        backtest_report = demo_backtesting_framework(synthetic_clients)
        
        # Demo 5: Comprehensive testing
        test_report = demo_comprehensive_testing(synthetic_clients)
        
        # Demo 6: Performance analysis
        performance_data = demo_performance_analysis(synthetic_clients)
        
        # Demo 7: Recommendation system
        vector_db = demo_recommendation_system(synthetic_clients)
        
        # Demo 8: Visualization dashboard
        create_visualization_dashboard(synthetic_clients, congruence_results, performance_data)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The mesh congruence system has been demonstrated with:")
        print("✅ Advanced mesh algorithms (Delaunay, CVT, edge collapse)")
        print("✅ Trial people data integration")
        print("✅ Mesh congruence analysis")
        print("✅ Backtesting framework")
        print("✅ Comprehensive testing")
        print("✅ Performance analysis")
        print("✅ Recommendation system")
        print("✅ Visualization dashboard")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"\nDemo failed: {e}")
        raise


if __name__ == "__main__":
    main() 