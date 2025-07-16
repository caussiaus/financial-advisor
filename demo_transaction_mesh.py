#!/usr/bin/env python
"""
Demo Transaction Mesh Analyzer
Author: Claude 2025-07-16

Demonstrates the transaction mesh analyzer for house purchase scenario analysis.
This shows how past financial decisions constrain future options like poles in a sculpture.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

def demo_transaction_mesh_system():
    """Demonstrate the transaction mesh system with house purchase analysis"""
    
    print("🎯 TRANSACTION MESH ANALYZER DEMONSTRATION")
    print("=" * 70)
    
    print("\n🏗️ Configuration Mesh Concept:")
    print("   📊 Like a sculpture with many poles in a grid")
    print("   ❌ Each financial commitment removes poles (eliminates options)")
    print("   📈 Multi-dimensional welfare evaluation")
    print("   🔄 Real-time constraint optimization")
    
    print(f"\n🎭 Demo Scenario:")
    print(f"   📄 User uploads Case #1 IPS Individual PDF")
    print(f"   💭 User asks: 'I'm thinking of buying a house, what should I do?'")
    print(f"   🤖 System analyzes transaction history + remaining option space")
    
    try:
        from src.transaction_mesh_analyzer import TransactionMeshAnalyzer
        print("✅ Transaction mesh analyzer imported successfully")
        
        # Create a demo analysis
        demo_analysis = {
            'user_question': "I'm thinking of buying a house, what should I do with the rest of my finances?",
            'mesh_analysis': {
                'total_decision_poles': 12,
                'available_poles': 7,
                'eliminated_by_history': 5,
                'constraint_percentage': 0.42
            },
            'current_position': {
                'welfare_state': {
                    'financial_security': 0.65,
                    'stress_level': 0.45,
                    'quality_of_life': 0.60,
                    'flexibility': 0.75,
                    'growth_potential': 0.70,
                    'social_status': 0.55
                },
                'resources': {
                    'annual_income': 85000,
                    'liquid_savings': 95000,
                    'investment_portfolio': 45000,
                    'credit_score': 750,
                    'emergency_fund': 25000
                },
                'established_path': [
                    'DEMO_001_EMERGENCY_FUND',
                    'DEMO_002_GRAD_DEGREE', 
                    'DEMO_003_CAREER_ESTABLISH',
                    'DEMO_004_INVESTMENT_START',
                    'DEMO_005_CREDIT_BUILDING'
                ]
            },
            'house_purchase_analysis': {
                'scenario': {
                    'house_price': 340000,
                    'down_payment': 68000,
                    'monthly_payment': 2040,
                    'remaining_options': 6
                },
                'affordability': {
                    'down_payment_coverage': 1.40,
                    'income_ratio': 0.29,
                    'affordability_score': 0.82,
                    'recommendation': 'affordable'
                },
                'welfare_projection': {
                    'financial_security': 0.85,
                    'stress_level': 0.65,
                    'quality_of_life': 0.90,
                    'flexibility': 0.35,
                    'growth_potential': 0.60,
                    'social_status': 0.75
                }
            },
            'optimization_strategy': {
                'immediate_actions': [
                    {
                        'category': 'risk_management',
                        'priority': 'high',
                        'description': 'Rebuild emergency fund to account for homeownership risks',
                        'target_amount': 17000,
                        'rationale': 'Homeownership creates new risk categories (maintenance, repairs, market volatility)'
                    },
                    {
                        'category': 'growth_optimization',
                        'priority': 'high', 
                        'description': 'Focus on tax-advantaged growth investments',
                        'rationale': 'Reduced flexibility requires efficient growth strategies'
                    }
                ],
                'medium_term_strategy': [
                    {
                        'category': 'tax_optimization',
                        'priority': 'medium',
                        'description': 'Leverage mortgage interest deduction and homeowner tax benefits'
                    }
                ],
                'risk_mitigation': [
                    {
                        'risk': 'flexibility_risk',
                        'level': 0.9,
                        'mitigation': 'Focus on reversible investments and maintain career optionality',
                        'timeline': 'ongoing'
                    }
                ]
            },
            'configuration_mesh_visualization': {
                'nodes': [
                    {'id': 'house_purchase_scenario', 'status': 'considering', 'size': 25},
                    {'id': 'investment_growth', 'status': 'available', 'size': 15},
                    {'id': 'career_advancement', 'status': 'available', 'size': 12},
                    {'id': 'family_planning', 'status': 'available', 'size': 18},
                    {'id': 'luxury_lifestyle', 'status': 'removed', 'size': 5},
                    {'id': 'high_risk_investment', 'status': 'removed', 'size': 5}
                ],
                'edges': [
                    {'source': 'house_purchase_scenario', 'target': 'luxury_lifestyle', 'type': 'exclusion'},
                    {'source': 'investment_growth', 'target': 'career_advancement', 'type': 'dependency'}
                ],
                'legends': {
                    'green': 'Available Options',
                    'red': 'Eliminated by History', 
                    'yellow': 'Current Consideration'
                }
            },
            'similar_case_insights': {
                'similarity_score': 0.73,
                'interpretation': 'high',
                'insights': [
                    'Your profile matches 73% of successful house purchasers',
                    'Similar cases typically focus on growth investments post-purchase',
                    'Timeline optimization suggests 3-5 year planning horizon for major decisions'
                ]
            }
        }
        
        print(f"\n🔍 Analysis Components:")
        print(f"   ✓ Transaction history creates path constraints")
        print(f"   ✓ Decision poles represent available financial choices")
        print(f"   ✓ Welfare optimization across security/stress/growth/flexibility")
        print(f"   ✓ Vector similarity matching to successful cases")
        print(f"   ✓ Risk assessment with remaining option analysis")
        
        print(f"\n📊 Sample Configuration Mesh Analysis:")
        mesh = demo_analysis['mesh_analysis']
        print(f"   🎯 Total Decision Poles: {mesh['total_decision_poles']}")
        print(f"   ✅ Available Options: {mesh['available_poles']}")
        print(f"   ❌ Eliminated by History: {mesh['eliminated_by_history']}")
        print(f"   📉 Constraint Level: {mesh['constraint_percentage']:.1%}")
        
        print(f"\n🏠 House Purchase Analysis:")
        house = demo_analysis['house_purchase_analysis']['scenario']
        affordability = demo_analysis['house_purchase_analysis']['affordability']
        print(f"   💰 House Price: ${house['house_price']:,}")
        print(f"   💳 Down Payment: ${house['down_payment']:,}")
        print(f"   📅 Monthly Payment: ${house['monthly_payment']:,}")
        print(f"   ⚖️ Affordability: {affordability['recommendation']} ({affordability['affordability_score']:.1%})")
        print(f"   🔄 Remaining Options: {house['remaining_options']}")
        
        print(f"\n📈 Welfare State Projection:")
        welfare = demo_analysis['house_purchase_analysis']['welfare_projection']
        for dimension, value in welfare.items():
            color = "🟢" if value > 0.7 else "🟡" if value > 0.4 else "🔴"
            print(f"   {color} {dimension.replace('_', ' ').title()}: {value:.1%}")
        
        print(f"\n🎯 Optimization Strategy:")
        immediate = demo_analysis['optimization_strategy']['immediate_actions']
        for action in immediate:
            print(f"   • {action['category'].upper()}: {action['description']}")
        
        print(f"\n📊 Similar Case Insights:")
        insights = demo_analysis['similar_case_insights']
        print(f"   🎯 Similarity Score: {insights['similarity_score']:.1%} ({insights['interpretation']})")
        for insight in insights['insights']:
            print(f"   • {insight}")
        
        print(f"\n🎨 Configuration Mesh Visualization:")
        viz = demo_analysis['configuration_mesh_visualization']
        available_nodes = [n for n in viz['nodes'] if n['status'] == 'available']
        removed_nodes = [n for n in viz['nodes'] if n['status'] == 'removed']
        considering_nodes = [n for n in viz['nodes'] if n['status'] == 'considering']
        
        print(f"   🟢 Available Options: {[n['id'] for n in available_nodes]}")
        print(f"   🔴 Eliminated Options: {[n['id'] for n in removed_nodes]}")
        print(f"   🟡 Under Consideration: {[n['id'] for n in considering_nodes]}")
        
        # Save demo results
        with open('demo_transaction_mesh_results.json', 'w') as f:
            json.dump(demo_analysis, f, indent=2)
        
        print(f"\n✅ Demo complete! Results saved to: demo_transaction_mesh_results.json")
        print(f"🌐 Access web interface at: http://localhost:8081")
        print(f"   📄 Upload Case #1 IPS Individual.pdf")
        print(f"   🏠 Click: 'I'm thinking of buying a house, what should I do?'")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print(f"   📦 Some components not available, showing demo data only")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def demo_decision_pole_elimination():
    """Show how decision poles get eliminated by financial commitments"""
    
    print(f"\n🏗️ DECISION POLE ELIMINATION SCULPTURE DEMO")
    print("=" * 50)
    
    # Initial decision poles (sculpture with all poles)
    initial_poles = [
        "🏠 First Home Purchase",
        "🏡 Upgrade Home", 
        "📈 Aggressive Stocks",
        "🏢 Real Estate Investment",
        "🎓 Graduate Degree",
        "💼 Career Change",
        "👶 Have Children",
        "✈️ Luxury Lifestyle",
        "🏖️ Early Retirement",
        "🎯 Conservative Bonds",
        "🚗 Luxury Car",
        "🎪 High-Risk Speculation"
    ]
    
    print(f"🎨 Initial Sculpture (All Poles Available):")
    for i, pole in enumerate(initial_poles):
        print(f"   {i+1:2d}. {pole}")
    
    print(f"\n📜 Transaction History (Poles Removed):")
    
    # Transaction 1: Emergency Fund
    print(f"\n💰 Transaction 1: Built Emergency Fund ($25k)")
    print(f"   ❌ Removes: 🎪 High-Risk Speculation (no safety net needed)")
    removed_poles = ["🎪 High-Risk Speculation"]
    
    # Transaction 2: Graduate Degree
    print(f"\n🎓 Transaction 2: Completed Graduate Degree ($60k)")
    print(f"   ❌ Removes: 🎓 Graduate Degree (already completed)")
    print(f"   ❌ Removes: 💼 Career Change (need stability for loan repayment)")
    removed_poles.extend(["🎓 Graduate Degree", "💼 Career Change"])
    
    # Transaction 3: Stable Career
    print(f"\n💼 Transaction 3: Established Stable Career")
    print(f"   ❌ Removes: ✈️ Luxury Lifestyle (conserving for goals)")
    removed_poles.append("✈️ Luxury Lifestyle")
    
    # Transaction 4: Investment Portfolio
    print(f"\n📈 Transaction 4: Built Investment Portfolio ($30k)")
    print(f"   ❌ Removes: 🎯 Conservative Bonds (already investing)")
    removed_poles.append("🎯 Conservative Bonds")
    
    # Current available poles
    available_poles = [pole for pole in initial_poles if pole not in removed_poles]
    
    print(f"\n🎨 Current Sculpture State:")
    print(f"   🟢 Available Poles ({len(available_poles)}):")
    for i, pole in enumerate(available_poles):
        print(f"      {i+1}. {pole}")
    
    print(f"\n   🔴 Removed Poles ({len(removed_poles)}):")
    for i, pole in enumerate(removed_poles):
        print(f"      {i+1}. {pole}")
    
    print(f"\n🏠 Now Considering: House Purchase")
    print(f"   📊 If house purchase proceeds:")
    print(f"   ❌ Would Remove: 🚗 Luxury Car (cash flow constraint)")
    print(f"   ❌ Would Remove: 🏖️ Early Retirement (mortgage commitment)")
    print(f"   ✅ Would Keep: 🏡 Upgrade Home, 📈 Aggressive Stocks, 🏢 Real Estate Investment, 👶 Have Children")
    
    print(f"\n🎯 Configuration Analysis:")
    print(f"   📉 Constraint Level: {len(removed_poles)}/{len(initial_poles)} = {len(removed_poles)/len(initial_poles):.1%}")
    print(f"   🔄 Remaining Flexibility: {len(available_poles)}/{len(initial_poles)} = {len(available_poles)/len(initial_poles):.1%}")
    print(f"   ⚖️ House Purchase Impact: Would reduce flexibility to {(len(available_poles)-2)}/{len(initial_poles)} = {(len(available_poles)-2)/len(initial_poles):.1%}")

if __name__ == "__main__":
    demo_transaction_mesh_system()
    demo_decision_pole_elimination()
    
    print(f"\n🚀 Ready for Interactive Demo!")
    print(f"   1. Start web service: python run.py web --port 8081")
    print(f"   2. Upload Case #1 IPS Individual.pdf")
    print(f"   3. Click: 'I'm thinking of buying a house, what should I do?'")
    print(f"   4. See transaction mesh analysis in action!") 