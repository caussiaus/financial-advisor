#!/usr/bin/env python3
"""
Refined Feature Extraction Test
Tests the enhanced feature extraction pipeline that:
1. Distinguishes between modulators (profile changers) vs features (financial events)
2. Maps features to specific accounts
3. Calculates amounts and handles taxes
4. Uses the text-to-math conversion module
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.append('src')

from src.enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone, FinancialEntity
from src.enhanced_category_calculator_fixed import EnhancedCategoryCalculator, CategoryDefinition, CategoryType, CalculationMethod
from src.stochastic_mesh_engine import StochasticMeshEngine
from src.accounting_reconciliation import AccountingReconciliationEngine

@dataclass
class RefinedFeature:
    """A refined financial feature with detailed mapping"""
    feature_id: str
    name: str
    feature_type: str  # 'modulator' or 'financial_event'
    account_mapping: Dict[str, float]  # account -> amount
    annual_amount: float
    tax_implications: Dict[str, float]  # tax_type -> amount
    confidence_score: float
    source_text: str
    timestamp: datetime

class RefinedFeatureExtractor:
    """Enhanced feature extractor that distinguishes modulators from features"""
    
    def __init__(self):
        self.pdf_processor = EnhancedPDFProcessor()
        self.category_calculator = EnhancedCategoryCalculator()
        self.accounting_engine = AccountingReconciliationEngine()
        
        # Initialize tax rates (simplified)
        self.tax_rates = {
            'federal_income': 0.22,  # 22% bracket
            'state_income': 0.05,    # 5% state
            'capital_gains': 0.15,   # 15% long-term
            'social_security': 0.062, # 6.2%
            'medicare': 0.0145       # 1.45%
        }
    
    def extract_refined_features(self, pdf_path: str) -> Tuple[List[RefinedFeature], Dict]:
        """Extract and refine features from PDF"""
        print("🔍 EXTRACTING REFINED FEATURES FROM PDF")
        print("=" * 60)
        
        # Step 1: Extract raw milestones and entities
        milestones, entities = self.pdf_processor.process_pdf(pdf_path)
        
        # Step 2: Classify as modulators vs financial events
        modulators, financial_events = self._classify_features(milestones, entities)
        
        # Step 3: Map to specific accounts and calculate amounts
        refined_features = []
        
        # Process modulators (profile changers)
        for modulator in modulators:
            refined_feature = self._process_modulator(modulator)
            if refined_feature:
                refined_features.append(refined_feature)
        
        # Process financial events
        for event in financial_events:
            refined_feature = self._process_financial_event(event)
            if refined_feature:
                refined_features.append(refined_feature)
        
        # Step 4: Calculate tax implications
        refined_features = self._calculate_tax_implications(refined_features)
        
        # Step 5: Generate summary
        summary = self._generate_extraction_summary(refined_features, modulators, financial_events)
        
        return refined_features, summary
    
    def _classify_features(self, milestones: List[FinancialMilestone], entities: List[FinancialEntity]) -> Tuple[List, List]:
        """Classify features as modulators vs financial events"""
        modulators = []
        financial_events = []
        
        # Keywords that indicate modulators (profile changers)
        modulator_keywords = [
            'promotion', 'career change', 'job loss', 'retirement', 'divorce', 'marriage',
            'child birth', 'health issue', 'disability', 'inheritance', 'lottery',
            'business start', 'business failure', 'relocation', 'education start',
            'education completion', 'military service', 'prison', 'bankruptcy'
        ]
        
        # Keywords that indicate financial events
        financial_keywords = [
            'purchase', 'sale', 'investment', 'loan', 'mortgage', 'refinance',
            'payment', 'withdrawal', 'deposit', 'transfer', 'dividend', 'interest',
            'rent', 'insurance', 'tax', 'bonus', 'commission', 'overtime'
        ]
        
        for milestone in milestones:
            description_lower = milestone.description.lower()
            
            # Check if it's a modulator
            is_modulator = any(keyword in description_lower for keyword in modulator_keywords)
            
            # Check if it's a financial event
            is_financial = any(keyword in description_lower for keyword in financial_keywords)
            
            if is_modulator:
                modulators.append(milestone)
            elif is_financial:
                financial_events.append(milestone)
            else:
                # Default to financial event if unclear
                financial_events.append(milestone)
        
        print(f"📊 CLASSIFICATION RESULTS:")
        print(f"   Modulators (profile changers): {len(modulators)}")
        print(f"   Financial events: {len(financial_events)}")
        
        return modulators, financial_events
    
    def _process_modulator(self, modulator: FinancialMilestone) -> Optional[RefinedFeature]:
        """Process a modulator (profile changer)"""
        # Modulators typically affect multiple accounts and have broader impact
        account_mapping = {}
        
        # Determine impact based on modulator type
        description_lower = modulator.description.lower()
        
        if 'promotion' in description_lower or 'career' in description_lower:
            account_mapping = {
                'salary_income': modulator.financial_impact * 0.8,  # 80% to salary
                'bonus_income': modulator.financial_impact * 0.2,   # 20% to bonus
                'retirement_savings': modulator.financial_impact * 0.15,  # 15% to retirement
                'investment_portfolio': modulator.financial_impact * 0.05   # 5% to investments
            }
        elif 'child' in description_lower or 'birth' in description_lower:
            account_mapping = {
                'childcare_expenses': modulator.financial_impact * 0.6,  # 60% to childcare
                'education_fund': modulator.financial_impact * 0.3,      # 30% to education
                'insurance_expenses': modulator.financial_impact * 0.1    # 10% to insurance
            }
        elif 'education' in description_lower:
            account_mapping = {
                'education_expenses': modulator.financial_impact * 0.8,  # 80% to education
                'student_loan_debt': modulator.financial_impact * 0.2    # 20% to debt
            }
        else:
            # Default modulator impact
            account_mapping = {
                'cash': modulator.financial_impact * 0.5,
                'savings': modulator.financial_impact * 0.3,
                'investments': modulator.financial_impact * 0.2
            }
        
        annual_amount = sum(account_mapping.values())
        
        return RefinedFeature(
            feature_id=f"modulator_{modulator.milestone_id}",
            name=modulator.description,
            feature_type='modulator',
            account_mapping=account_mapping,
            annual_amount=annual_amount,
            tax_implications={},  # Will be calculated later
            confidence_score=0.8,
            source_text=modulator.description,
            timestamp=modulator.date
        )
    
    def _process_financial_event(self, event: FinancialMilestone) -> Optional[RefinedFeature]:
        """Process a financial event"""
        # Use the category calculator to determine account mapping
        income_data = {'base_income': 100000}  # Default income
        profile_data = {'age': 35, 'risk_tolerance': 'Moderate'}
        
        # Calculate category amounts
        calculations = self.category_calculator.calculate_all_categories(
            income_data, profile_data, [{'description': event.description, 'financial_impact': event.financial_impact}]
        )
        
        # Map to specific accounts based on event type
        account_mapping = {}
        description_lower = event.description.lower()
        
        if 'education' in description_lower or 'college' in description_lower:
            account_mapping = {
                'education_expenses': event.financial_impact * 0.7,
                'education_savings': event.financial_impact * 0.3
            }
        elif 'house' in description_lower or 'mortgage' in description_lower:
            account_mapping = {
                'real_estate': event.financial_impact * 0.8,
                'mortgage_debt': event.financial_impact * 0.2
            }
        elif 'car' in description_lower or 'vehicle' in description_lower:
            account_mapping = {
                'vehicle_assets': event.financial_impact * 0.6,
                'auto_loan_debt': event.financial_impact * 0.4
            }
        elif 'investment' in description_lower or 'stock' in description_lower:
            account_mapping = {
                'investment_portfolio': event.financial_impact * 0.9,
                'investment_income': event.financial_impact * 0.1
            }
        else:
            # Default financial event mapping
            account_mapping = {
                'cash': event.financial_impact * 0.4,
                'savings': event.financial_impact * 0.3,
                'investments': event.financial_impact * 0.3
            }
        
        annual_amount = sum(account_mapping.values())
        
        return RefinedFeature(
            feature_id=f"financial_event_{event.milestone_id}",
            name=event.description,
            feature_type='financial_event',
            account_mapping=account_mapping,
            annual_amount=annual_amount,
            tax_implications={},  # Will be calculated later
            confidence_score=0.7,
            source_text=event.description,
            timestamp=event.date
        )
    
    def _calculate_tax_implications(self, features: List[RefinedFeature]) -> List[RefinedFeature]:
        """Calculate tax implications for each feature"""
        for feature in features:
            tax_implications = {}
            
            # Calculate taxes based on account types
            for account, amount in feature.account_mapping.items():
                if 'income' in account:
                    # Income is taxable
                    tax_implications['federal_income'] = amount * self.tax_rates['federal_income']
                    tax_implications['state_income'] = amount * self.tax_rates['state_income']
                    tax_implications['social_security'] = amount * self.tax_rates['social_security']
                    tax_implications['medicare'] = amount * self.tax_rates['medicare']
                
                elif 'investment' in account and 'income' in account:
                    # Investment income
                    tax_implications['capital_gains'] = amount * self.tax_rates['capital_gains']
                
                elif 'debt' in account:
                    # Debt payments may have tax deductions
                    if 'mortgage' in account:
                        tax_implications['mortgage_interest_deduction'] = amount * 0.22  # 22% deduction
                    elif 'student_loan' in account:
                        tax_implications['student_loan_interest_deduction'] = amount * 0.22
            
            feature.tax_implications = tax_implications
        
        return features
    
    def _generate_extraction_summary(self, refined_features: List[RefinedFeature], 
                                   modulators: List, financial_events: List) -> Dict:
        """Generate comprehensive summary of extraction results"""
        
        # Calculate totals
        total_modulators = len([f for f in refined_features if f.feature_type == 'modulator'])
        total_financial_events = len([f for f in refined_features if f.feature_type == 'financial_event'])
        
        total_annual_amount = sum(f.annual_amount for f in refined_features)
        total_taxes = sum(sum(f.tax_implications.values()) for f in refined_features)
        
        # Account distribution
        account_totals = {}
        for feature in refined_features:
            for account, amount in feature.account_mapping.items():
                account_totals[account] = account_totals.get(account, 0) + amount
        
        # Tax distribution
        tax_totals = {}
        for feature in refined_features:
            for tax_type, amount in feature.tax_implications.items():
                tax_totals[tax_type] = tax_totals.get(tax_type, 0) + amount
        
        return {
            'extraction_summary': {
                'total_features': len(refined_features),
                'modulators': total_modulators,
                'financial_events': total_financial_events,
                'total_annual_amount': total_annual_amount,
                'total_taxes': total_taxes,
                'net_after_taxes': total_annual_amount - total_taxes
            },
            'account_distribution': account_totals,
            'tax_distribution': tax_totals,
            'feature_details': [
                {
                    'feature_id': f.feature_id,
                    'name': f.name,
                    'type': f.feature_type,
                    'annual_amount': f.annual_amount,
                    'account_mapping': f.account_mapping,
                    'tax_implications': f.tax_implications,
                    'confidence_score': f.confidence_score
                }
                for f in refined_features
            ]
        }

def test_refined_feature_extraction():
    """Test the refined feature extraction pipeline"""
    print("🧪 TESTING REFINED FEATURE EXTRACTION")
    print("=" * 60)
    
    # Initialize the refined feature extractor
    extractor = RefinedFeatureExtractor()
    
    # Test with the Case_1_Clean.pdf
    pdf_path = 'data/inputs/uploads/Case_1_Clean.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Extract refined features
    refined_features, summary = extractor.extract_refined_features(pdf_path)
    
    # Display results
    print("\n📊 REFINED FEATURE EXTRACTION RESULTS")
    print("=" * 60)
    
    print(f"Total Features Extracted: {summary['extraction_summary']['total_features']}")
    print(f"Modulators (Profile Changers): {summary['extraction_summary']['modulators']}")
    print(f"Financial Events: {summary['extraction_summary']['financial_events']}")
    print(f"Total Annual Amount: ${summary['extraction_summary']['total_annual_amount']:,.2f}")
    print(f"Total Taxes: ${summary['extraction_summary']['total_taxes']:,.2f}")
    print(f"Net After Taxes: ${summary['extraction_summary']['net_after_taxes']:,.2f}")
    
    print("\n💰 ACCOUNT DISTRIBUTION:")
    for account, amount in summary['account_distribution'].items():
        print(f"  {account}: ${amount:,.2f}")
    
    print("\n🏛️ TAX DISTRIBUTION:")
    for tax_type, amount in summary['tax_distribution'].items():
        print(f"  {tax_type}: ${amount:,.2f}")
    
    print("\n🔍 DETAILED FEATURE ANALYSIS:")
    for feature in summary['feature_details']:
        print(f"\n  Feature: {feature['name']}")
        print(f"    Type: {feature['type']}")
        print(f"    Annual Amount: ${feature['annual_amount']:,.2f}")
        print(f"    Confidence: {feature['confidence_score']:.2f}")
        print(f"    Accounts: {feature['account_mapping']}")
        if feature['tax_implications']:
            print(f"    Taxes: {feature['tax_implications']}")
    
    # Save results
    output_file = 'refined_feature_extraction_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Test the mesh integration
    print("\n🔄 TESTING MESH INTEGRATION")
    print("=" * 60)
    
    # Initialize mesh with refined features
    initial_state = {
        'cash': 50000,
        'savings': 100000,
        'investments': 200000,
        'real_estate': 300000,
        'total_wealth': 650000
    }
    
    mesh_engine = StochasticMeshEngine(current_financial_state=initial_state)
    
    # Convert refined features to milestones for mesh
    mesh_milestones = []
    for feature in refined_features:
        mesh_milestone = FinancialMilestone(
            milestone_id=feature.feature_id,
            description=feature.name,
            date=feature.timestamp,
            event_type=feature.feature_type,
            financial_impact=feature.annual_amount,
            probability=feature.confidence_score,
            payment_flexibility=0.8
        )
        mesh_milestones.append(mesh_milestone)
    
    # Initialize mesh
    mesh_engine.initialize_mesh(mesh_milestones, time_horizon_years=10)
    
    print(f"✅ Mesh initialized with {len(mesh_milestones)} refined features")
    print(f"📈 Mesh status: {mesh_engine.get_mesh_status()}")
    
    return refined_features, summary

if __name__ == "__main__":
    test_refined_feature_extraction() 