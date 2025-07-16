#!/usr/bin/env python
# Test script to analyze optimization recommendations
# Author: ChatGPT 2025-01-16

import sys
sys.path.append('src')

from dynamic_portfolio_engine import DynamicPortfolioEngine
from life_choice_optimizer import LifeChoiceOptimizer

def test_recommendations():
    """Test and analyze optimization recommendations"""
    
    # Create portfolio engine
    client_config = {
        'income': 250000,
        'disposable_cash': 8000,
        'allowable_var': 0.15,
        'age': 42,
        'risk_profile': 3,
        'portfolio_value': 1500000,
        'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
    }
    
    portfolio_engine = DynamicPortfolioEngine(client_config)
    optimizer = LifeChoiceOptimizer(portfolio_engine)
    
    # Add sample life choices
    sample_choices = [
        ('career', 'promotion', '2023-01-15'),
        ('family', 'marriage', '2023-06-20'),
        ('lifestyle', 'buy_house', '2024-03-10'),
        ('education', 'certification', '2024-09-05'),
        ('health', 'health_improvement', '2024-12-01')
    ]
    
    for category, choice, date in sample_choices:
        optimizer.add_life_choice(category, choice, date)
    
    print("🔍 AGE-ADJUSTED OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    print(f"Client Age: {client_config['age']}")
    print(f"Portfolio Value: ${client_config['portfolio_value']:,}")
    print(f"Current Income: ${client_config['income']:,}")
    
    objectives = ['financial_growth', 'comfort_stability', 'risk_management', 'lifestyle_quality']
    
    for objective in objectives:
        result = optimizer.optimize_next_choice(objective)
        print(f"\n📊 {objective.replace('_', ' ').title()}:")
        print(f"   🏆 Best Choice: {result['best_choice']['choice']} ({result['best_choice']['category']})")
        print(f"   📈 Total Score: {result['best_choice']['total_score']:.3f}")
        print(f"   💰 Financial: {result['best_choice']['financial_score']:+.3f}")
        print(f"   😌 Comfort: {result['best_choice']['comfort_score']:+.3f}")
        print(f"   🛡️  Risk: {result['best_choice']['risk_score']:+.3f}")
        print(f"   🌟 Lifestyle: {result['best_choice']['lifestyle_score']:+.3f}")
        print(f"   📋 Impacts: {result['best_choice']['impacts']}")
        print(f"   📋 Top 3 Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"      {i}. {rec['choice']} ({rec['category']}) - Score: {rec['total_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎯 SENSE CHECK ANALYSIS:")
    print("=" * 60)
    
    # Analyze each recommendation
    for objective in objectives:
        result = optimizer.optimize_next_choice(objective)
        best = result['best_choice']
        
        print(f"\n📊 {objective.replace('_', ' ').title()}:")
        print(f"   Recommendation: {best['choice']} ({best['category']})")
        
        # Sense check logic
        if objective == 'financial_growth':
            if best['financial_score'] > 0:
                print(f"   ✅ GOOD: Positive financial impact (+{best['financial_score']:.3f})")
            else:
                print(f"   ⚠️  QUESTIONABLE: Negative financial impact ({best['financial_score']:.3f})")
                
        elif objective == 'comfort_stability':
            if best['comfort_score'] > 0:
                print(f"   ✅ GOOD: Positive comfort impact (+{best['comfort_score']:.3f})")
            else:
                print(f"   ⚠️  QUESTIONABLE: Negative comfort impact ({best['comfort_score']:.3f})")
                
        elif objective == 'risk_management':
            if best['risk_score'] > 0:
                print(f"   ✅ GOOD: Positive risk management (+{best['risk_score']:.3f})")
            else:
                print(f"   ⚠️  QUESTIONABLE: Negative risk impact ({best['risk_score']:.3f})")
                
        elif objective == 'lifestyle_quality':
            if best['lifestyle_score'] > 0:
                print(f"   ✅ GOOD: Positive lifestyle impact (+{best['lifestyle_score']:.3f})")
            else:
                print(f"   ⚠️  QUESTIONABLE: Negative lifestyle impact ({best['lifestyle_score']:.3f})")
        
        # Age appropriateness check
        if best['choice'] == 'retirement' and client_config['age'] < 55:
            print(f"   ⚠️  AGE CONCERN: Early retirement at age {client_config['age']}")
        elif best['choice'] == 'entrepreneur' and client_config['age'] > 60:
            print(f"   ⚠️  AGE CONCERN: Late entrepreneurship at age {client_config['age']}")
        elif best['choice'] == 'children' and client_config['age'] > 50:
            print(f"   ⚠️  AGE CONCERN: Late childbearing at age {client_config['age']}")

if __name__ == "__main__":
    test_recommendations() 