#!/usr/bin/env python3
"""
Display Mesh Data and Analysis Results

This script fetches and displays mesh node information, congruence analysis,
and strategy comparison results in a readable format.
"""

import requests
import json
from datetime import datetime
import pandas as pd

# Dashboard API configuration
DASHBOARD_URL = "http://localhost:5001"
API_BASE = f"{DASHBOARD_URL}/api"

def display_clients():
    """Display all clients and their profiles"""
    print("=" * 60)
    print("📊 CLIENT PROFILES")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/clients")
    if response.status_code == 200:
        data = response.json()
        clients = data['clients']
        
        for client in clients:
            print(f"\n👤 {client['name']}")
            print(f"   Age: {client['age']}")
            print(f"   Income: ${client['income']:,.0f}")
            print(f"   Life Stage: {client['life_stage']}")
            print(f"   Risk Tolerance: {client['risk_tolerance']:.3f}")
            print(f"   Created: {client['created_at']}")
    else:
        print("❌ Failed to fetch clients")

def display_mesh_congruence():
    """Display mesh congruence analysis"""
    print("\n" + "=" * 60)
    print("🔗 MESH CONGRUENCE ANALYSIS")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/mesh/congruence")
    if response.status_code == 200:
        data = response.json()
        
        print(f"Total Clients: {data['total_clients']}")
        print(f"Total Events: {data['total_events']}")
        
        print(f"\n📋 Clients in Mesh:")
        for client in data['clients']:
            print(f"   • {client}")
        
        print(f"\n🔗 Congruence Results:")
        for result in data['congruence_results']:
            print(f"\n   {result['client_1']} ↔ {result['client_2']}")
            print(f"     Congruence: {result['congruence']:.3f}")
            print(f"     Triangulation Quality: {result['triangulation_quality']:.3f}")
            print(f"     Density Score: {result['density_score']:.3f}")
            print(f"     Edge Efficiency: {result['edge_efficiency']:.3f}")
    else:
        print("❌ Failed to fetch mesh congruence data")

def display_analytics():
    """Display dashboard analytics"""
    print("\n" + "=" * 60)
    print("📈 DASHBOARD ANALYTICS")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE}/analytics/dashboard")
    if response.status_code == 200:
        data = response.json()
        
        # Client distribution
        clients = data['clients']
        print(f"\n👥 Client Distribution:")
        print(f"   Total Clients: {clients['total']}")
        
        print(f"\n📊 By Life Stage:")
        for stage, count in clients['by_life_stage'].items():
            print(f"   • {stage}: {count}")
        
        print(f"\n💰 Income Distribution:")
        income = clients['income_distribution']
        print(f"   • Average: ${income['avg']:,.0f}")
        print(f"   • Range: ${income['min']:,.0f} - ${income['max']:,.0f}")
        
        # Performance metrics
        perf = data['performance']
        print(f"\n⚡ Performance Metrics:")
        print(f"   • Requests Processed: {perf['requests_processed']}")
        print(f"   • Average Response Time: {perf['avg_response_time']:.3f}s")
        print(f"   • Memory Usage: {perf['memory_usage_mb']:.1f} MB")
        print(f"   • Uptime: {perf['uptime_seconds']:.0f}s")
        
        # System health
        system = data['system']
        print(f"\n🏥 System Health:")
        for component, status in system['components'].items():
            print(f"   • {component}: {status}")
            
    else:
        print("❌ Failed to fetch analytics")

def display_recommendations():
    """Display recommendations for each client"""
    print("\n" + "=" * 60)
    print("💡 CLIENT RECOMMENDATIONS")
    print("=" * 60)
    
    # Get clients first
    response = requests.get(f"{API_BASE}/clients")
    if response.status_code == 200:
        clients = response.json()['clients']
        
        for client in clients:
            client_id = client['id']
            print(f"\n👤 {client['name']} ({client_id}):")
            
            # Get recommendations for this client
            rec_response = requests.get(f"{API_BASE}/recommendations/{client_id}")
            if rec_response.status_code == 200:
                recommendations = rec_response.json()
                if 'recommendations' in recommendations and recommendations['recommendations']:
                    for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                        print(f"   {i}. {rec.get('description', 'No description')}")
                        print(f"      Amount: ${rec.get('suggested_amount', 0):,.0f}")
                        print(f"      Priority: {rec.get('priority', 'Unknown')}")
                else:
                    print("   No specific recommendations available")
            else:
                print("   ❌ Failed to fetch recommendations")
    else:
        print("❌ Failed to fetch clients")

def display_strategy_comparison():
    """Display strategy comparison results"""
    print("\n" + "=" * 60)
    print("⚖️ STRATEGY COMPARISON")
    print("=" * 60)
    
    # Load the test results if available
    try:
        with open('test_results_three_way.json', 'r') as f:
            results = json.load(f)
        
        sliding_analysis = results.get('sliding_analysis', {})
        summary_stats = sliding_analysis.get('summary_stats', {})
        
        print(f"\n📊 Three-Way Strategy Comparison:")
        print(f"\n🎯 Strategy Performance:")
        print(f"   • Strategy vs Benchmark: {summary_stats.get('avg_strategy_excess', 0):.2%}")
        print(f"   • Strategy Win Rate: {summary_stats.get('strategy_win_rate', 0):.1%}")
        print(f"   • Consistent Outperformance: {summary_stats.get('consistent_strategy_outperformance', 0):.1%}")
        
        print(f"\n🛡️ Control Group Performance:")
        print(f"   • Control vs Benchmark: {summary_stats.get('avg_control_excess', 0):.2%}")
        print(f"   • Control Win Rate: {summary_stats.get('control_win_rate', 0):.1%}")
        print(f"   • Consistent Outperformance: {summary_stats.get('consistent_control_outperformance', 0):.1%}")
        
        print(f"\n⚔️ Strategy vs Control:")
        print(f"   • Strategy vs Control: {summary_stats.get('avg_strategy_vs_control_excess', 0):.2%}")
        print(f"   • Strategy Win Rate: {summary_stats.get('strategy_vs_control_win_rate', 0):.1%}")
        
        print(f"\n📈 Analysis Details:")
        print(f"   • Sliding Windows: {sliding_analysis.get('window_size', 0)} months")
        print(f"   • Total Periods: {sliding_analysis.get('total_periods', 0)} months")
        print(f"   • Windows Analyzed: {len(sliding_analysis.get('sliding_windows', []))}")
        
    except FileNotFoundError:
        print("❌ Test results file not found. Run the test script first.")
    except Exception as e:
        print(f"❌ Error loading results: {e}")

def main():
    """Display all mesh data and analysis"""
    print("🚀 MESH DATA DISPLAY")
    print("=" * 60)
    print(f"Dashboard URL: {DASHBOARD_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if dashboard is running
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/health")
        if response.status_code != 200:
            print("❌ Dashboard is not running")
            return
        print("✅ Dashboard is running and healthy")
    except Exception as e:
        print(f"❌ Cannot connect to dashboard: {e}")
        return
    
    # Display all sections
    display_clients()
    display_mesh_congruence()
    display_analytics()
    display_recommendations()
    display_strategy_comparison()
    
    print("\n" + "=" * 60)
    print("🌐 View full dashboard at: http://localhost:5001")
    print("=" * 60)

if __name__ == "__main__":
    main() 