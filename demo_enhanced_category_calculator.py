
Enhanced Category Calculator Demonstration

This script demonstrates how the enhanced category calculator leverages computational power
to calculate detailed amounts for each category based on income percentages and actual dollar ranges.
It integrates with the existing Omega Mesh system to provide comprehensive financial analysis.import sys
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src)
sys.path.append('.')

try:
    from src.enhanced_category_calculator import EnhancedCategoryCalculator, CategoryDefinition, CategoryType, CalculationMethod
    from src.synthetic_data_generator import SyntheticFinancialDataGenerator, PersonProfile
    from src.enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone
    from src.stochastic_mesh_engine import StochasticMeshEngine
    from src.accounting_reconciliation import AccountingReconciliationEngine
    from src.financial_recommendation_engine import FinancialRecommendationEngine
except ImportError as e:
    print(f"Import error: {e}")
    print(Pleaseensure all modules are in the src/ directory")
    sys.exit(1)


class EnhancedCategoryCalculatorDemo:
    Demonstrates the enhanced category calculator's capabilities
     
    def __init__(self):
        self.category_calculator = EnhancedCategoryCalculator()
        self.synthetic_generator = SyntheticFinancialDataGenerator()
        self.pdf_processor = EnhancedPDFProcessor()
        
        # Create output directory
        os.makedirs('data/outputs/category_analysis, exist_ok=true)
        
    def run_demo(self):
lete demonstration"
        print("🚀 Enhanced Category Calculator Demonstration)
        print("=" * 60)
        
        # Step 1: Generate synthetic client profiles
        print("\n📊 Step 1: Generating synthetic client profiles...")
        profiles = self._generate_client_profiles()
        
        # Step 2: Calculate categories for each profile
        print("\n💰 Step 2: Calculating detailed category amounts...")
        all_results = {}
        
        for i, profile in enumerate(profiles):
            print(f"\nProcessing Profile {i+1}: {profile.name}")
            results = self._process_single_profile(profile)
            all_results[fprofile_{i+1}"] = results
        
        # Step 3: Generate comprehensive analysis
        print("\n📈 Step 3: Generating comprehensive analysis...")
        self._generate_comprehensive_analysis(all_results)
        
        # Step 4: Create visualizations
        print("\n🎨 Step 4ing visualizations...")
        self._create_visualizations(all_results)
        
        print("\n✅ Enhanced Category Calculator Demo Complete!)    print(Results saved to: data/outputs/category_analysis/")
    
    def _generate_client_profiles(self) -> List[PersonProfile]:
 ate diverse client profiles for testing""        profiles = []
        
        # Profile 1: Young Professional
        profiles.append(PersonProfile(
            name="Sarah Chen,
            age=28,
            occupation=Software Engineer",
            base_income=85000,
            risk_tolerance="Moderate",
            family_status="Single",
            current_assets=[object Object]checking: 1500,savings': 25000investments': 35000
            debts={'student_loans': 4500, 'credit_cards':5000      ))
        
        # Profile 2: Mid-Career Family
        profiles.append(PersonProfile(
            name=Michael Rodriguez,
            age=42,
            occupation="Marketing Director",
            base_income=120000,
            risk_tolerance="Conservative",
            family_status="Married with children",
            current_assets=[object Object]checking: 2500,savings': 5000investments':15000real_estate': 400
            debts={mortgage': 3000auto_loans': 25000      ))
        
        # Profile 3Retirement
        profiles.append(PersonProfile(
            name=Jennifer Thompson,
            age=58,
            occupation="Healthcare Administrator",
            base_income=95000,
            risk_tolerance="Conservative",
            family_status="Married",
            current_assets=[object Object]checking: 3500 savings: 10000investments': 45000retirement': 600
            debts={mortgage: 150000      ))
        
        # Profile 4: High-Income Professional
        profiles.append(PersonProfile(
            name="David Kim,
            age=35,
            occupation=Investment Banker",
            base_income=250000,
            risk_tolerance="Aggressive",
            family_status="Married",
            current_assets=[object Object]checking: 5000 savings: 10000investments':30000real_estate': 800
            debts={mortgage: 60      ))
        
        return profiles
    
    def _process_single_profile(self, profile: PersonProfile) -> Dict[str, Any]:
  s a single profile through the enhanced category calculator""        
        # Convert profile to income data
        income_data = {
        base_income': profile.base_income,
            investment_income: profile.current_assets.get('investments, 0) * 00.05return
         other_income': profile.base_income *0.1 # 10% bonus/other income
        }
        
        # Convert profile to profile data
        profile_data = [object Object]            age': profile.age,
           risk_tolerance': profile.risk_tolerance,
          family_size':2arried' in profile.family_status else 1,
       occupation': profile.occupation,
           current_assets: profile.current_assets,
          debts': profile.debts
        }
        
        # Generate synthetic milestones
        milestones = self._generate_synthetic_milestones(profile)
        
        # Calculate all categories
        calculations = self.category_calculator.calculate_all_categories(
            income_data, profile_data, milestones
        )
        
        # Generate report
        report = self.category_calculator.generate_category_report(calculations)
        
        # Export results
        output_file = f"data/outputs/category_analysis/{profile.name.lower().replace(' ', '_')}_categories.json"
        self.category_calculator.export_calculations(calculations, output_file)
        
        return {
            profile': profile,
         calculations': calculations,
            reporteport,
        output_file': output_file
        }
    
    def _generate_synthetic_milestones(self, profile: PersonProfile) -> List[Dict]:
 Generate synthetic milestones based on profile"""
        milestones = []
        
        if profile.age < 30:
            milestones.extend([
                {'category': 'education', 'description': Graduate school',financial_impact': 50000},
                {'category': 'housing', 'description': 'First home purchase',financial_impact': 200000},
            [object Object]category:emergency_fund', 'description: ency fund building',financial_impact': 2500        ])
        elif profile.age < 50:
            milestones.extend([
                {'category': 'housing', 'description': Home renovation',financial_impact': 75000},
                {'category': 'education', 'description': 'Children education',financial_impact': 100000},
                {'category: ent', 'description': 'Portfolio expansion',financial_impact': 5000        ])
        else:
            milestones.extend([
                {'category: ent', 'description': 'Retirement planning',financial_impact': 200000},
                {'category: are', 'description': 'Healthcare expenses',financial_impact': 30000},
                {'category: ent', 'description': 'Conservative portfolio',financial_impact': 1000      ])
        
        return milestones
    
    def _generate_comprehensive_analysis(self, all_results: Dict[str, Any]):
 rate comprehensive analysis across all profiles"" 
        analysis = {
      timestamp:datetime.now().isoformat(),
           total_profiles:len(all_results),
            profile_summaries:[object Object]            cross_profile_insights': {}
        }
        
        # Analyze each profile
        for profile_key, result in all_results.items():
            profile = result['profile']
            calculations = result['calculations']
            report = result['report']
            
            # Calculate key metrics
            total_income = sum(calc.percentage_of_income for calc in calculations.values())
            avg_confidence = np.mean([calc.confidence_score for calc in calculations.values()])
            
            # Find top spending categories
            expense_calculations =              calc for calc in calculations.values()
                if self.category_calculator.categories[calc.category_id].category_type == CategoryType.EXPENSE
            ]
            top_expenses = sorted(expense_calculations, key=lambda x: x.calculated_amount, reverse=True)[:3]
            
            # Find top investment categories
            investment_calculations =              calc for calc in calculations.values()
                if self.category_calculator.categories[calc.category_id].category_type in [CategoryType.INVESTMENT, CategoryType.SAVINGS]
            ]
            top_investments = sorted(investment_calculations, key=lambda x: x.calculated_amount, reverse=True)[:3]
            
            analysis[profile_summaries'][profile_key] =[object Object]
             name': profile.name,
                age': profile.age,
               income': profile.base_income,
               risk_tolerance': profile.risk_tolerance,
               total_income_allocation': total_income,
       average_confidence': avg_confidence,
                top_expenses   [object Object]                 category': calc.category_id,
                      amount: calc.calculated_amount,
                        percentage:calc.percentage_of_income
                    }
                    for calc in top_expenses
                ],
                top_investments': [
    [object Object]                 category': calc.category_id,
                      amount: calc.calculated_amount,
                        percentage:calc.percentage_of_income
                    }
                    for calc in top_investments
                ],
                recommendations': report['recommendations']
            }
        
        # Cross-profile insights
        ages = [result[profile'].age for result in all_results.values()]
        incomes = [result['profile].base_income for result in all_results.values()]
        confidence_scores = [
            np.mean([calc.confidence_score for calc in result['calculations'].values()])
            for result in all_results.values()
        ]
        
        analysis[cross_profile_insights] = [object Object]        age_range': {'min: min(ages),max: max(ages), 'avg': np.mean(ages)},
            income_range': {min:min(incomes), max:max(incomes),avg: np.mean(incomes)},
           confidence_range': {'min': min(confidence_scores), 'max': max(confidence_scores), 'avg': np.mean(confidence_scores)},
            total_categories_analyzed': sum(len(result['calculations']) for result in all_results.values())
        }
        
        # Save comprehensive analysis
        with open('data/outputs/category_analysis/comprehensive_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Comprehensive analysis saved with {len(all_results)} profiles")
    
    def _create_visualizations(self, all_results: Dict[str, Any]):
      ate visualizations for the category analysis""        
        # 1. Category Distribution by Profile
        fig1 = go.Figure()
        
        for profile_key, result in all_results.items():
            profile = result['profile']
            calculations = result['calculations']
            
            # Group by category type
            by_type = [object Object]          for calc in calculations.values():
                category = self.category_calculator.categories[calc.category_id]
                cat_type = category.category_type.value
                
                if cat_type not in by_type:
                    by_type[cat_type] =0                by_type[cat_type] += calc.calculated_amount
            
            # Add to plot
            for cat_type, amount in by_type.items():
                fig1add_trace(go.Bar(
                    name=f"{profile.name} - {cat_type}",
                    x=[cat_type],
                    y=[amount],
                    text=f${amount:,.0f}                   textposition='auto'
                ))
        
        fig1.update_layout(
            title="Category Distribution by Profile",
            xaxis_title="Category Type",
            yaxis_title="Amount ($)",
            barmode=group       )
        
        fig1_html('data/outputs/category_analysis/category_distribution.html')
        
        # 2. Income Allocation Heatmap
        categories = list(self.category_calculator.categories.keys())
        profiles = [result['profile].name for result in all_results.values()]
        
        allocation_matrix = []
        for result in all_results.values():
            calculations = result['calculations]
            row =       for category_id in categories:
                if category_id in calculations:
                    row.append(calculations[category_id].percentage_of_income * 100              else:
                    row.append(0)
            allocation_matrix.append(row)
        
        fig2 = px.imshow(
            allocation_matrix,
            x=categories,
            y=profiles,
            color_continuous_scale='RdBu',
            title=Income Allocation by Category (%)"
        )
        
        fig2_html('data/outputs/category_analysis/income_allocation_heatmap.html')
        
        # 3. Confidence Scores by Profile
        fig3 = go.Figure()
        
        for result in all_results.values():
            profile = result['profile']
            calculations = result['calculations']
            
            categories = list(calculations.keys())
            confidence_scores =calculations[cat].confidence_score for cat in categories]
            
            fig3.add_trace(go.Scatter(
                x=categories,
                y=confidence_scores,
                mode='lines+markers,              name=profile.name,
                text=[f"{score:0.2core in confidence_scores],
                textposition='top center'
            ))
        
        fig3.update_layout(
            title="Calculation Confidence Scores by Profile",
            xaxis_title="Category",
            yaxis_title="Confidence Score",
            yaxis=dict(range=0,1       )
        
        fig3_html('data/outputs/category_analysis/confidence_scores.html')
        
        print("✅ Visualizations created:)
        print("   - Category distribution by profile)
        print("   - Income allocation heatmap)
        print("   - Confidence scores by profile")


def main():
    """Main demonstration function"" try:
        demo = EnhancedCategoryCalculatorDemo()
        demo.run_demo()
        
        print(n🎯Key Features Demonstrated:)
        print("✅ Vectorized calculations using numpy arrays)
        print("✅ Multiple calculation methods (percentage, dollar range, dynamic scaling))
        print("✅ Profile-based adjustments (age, risk tolerance))
        print(✅ Milestone-based category calculations)
        print("✅ Confidence scoring for each calculation)
        print("✅ Comprehensive reporting and visualization)
        print("✅ Integration with existing Omega Mesh system")
        
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 