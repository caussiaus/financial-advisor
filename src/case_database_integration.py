#!/usr/bin/env python
"""
Case Database Integration System
Author: Claude 2025-07-16

Integrates all components into a comprehensive system that uses historical case data
for income-controlled comparisons of life event timing patterns. This is the main
orchestrator that combines timeline bias, fsQCA, stress minimization, continuous
mesh optimization, and accounting reconciliation.

Key Features:
- Unified case database with comprehensive client histories
- Income-controlled timeline bias for realistic event estimation
- fsQCA analysis for finding optimal financial stability paths
- Stress minimization with accounting equation enforcement
- Continuous mesh optimization for accurate interpolation
- Real-time adaptation based on actual vs planned event timing
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import sqlite3

# Import our custom modules
from timeline_bias_engine import TimelineBiasEngine, ClientTimeline
from fuzzy_sets_optimizer import FinancialStabilityOptimizer, fsQCAResult
from financial_stress_minimizer import FinancialStressMinimizer, OptimalPath
from continuous_configuration_mesh import ContinuousConfigurationMesh
from accounting_reconciliation import AccountingReconciliationEngine, ReconciliationReport

logger = logging.getLogger(__name__)

@dataclass
class IntegratedClientCase:
    """Comprehensive client case with all analysis components"""
    client_id: str
    profile: Dict[str, Any]
    timeline_analysis: ClientTimeline
    fsqca_analysis: fsQCAResult
    optimal_path: OptimalPath
    reconciliation_reports: List[ReconciliationReport]
    mesh_optimization: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]

class CaseAnalysisOrchestrator:
    """Main orchestrator for comprehensive case analysis"""
    
    def __init__(self, case_db_path: str = "data/integrated_case_database.db"):
        self.case_db_path = Path(case_db_path)
        self.case_db_path.parent.mkdir(exist_ok=True)
        
        # Initialize all analysis engines
        self.timeline_engine = TimelineBiasEngine(str(case_db_path))
        self.fsqca_optimizer = FinancialStabilityOptimizer()
        self.stress_minimizer = FinancialStressMinimizer()
        self.accounting_engine = AccountingReconciliationEngine()
        
        # Initialize case database
        self._initialize_integrated_database()
    
    def _initialize_integrated_database(self):
        """Initialize comprehensive case database"""
        conn = sqlite3.connect(self.case_db_path)
        cursor = conn.cursor()
        
        # Create comprehensive case tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrated_cases (
                case_id TEXT PRIMARY KEY,
                client_profile TEXT,  -- JSON of client profile
                analysis_date TEXT,
                timeline_confidence REAL,
                fsqca_consistency REAL,
                stress_optimization_score REAL,
                accounting_balance_rate REAL,
                overall_quality_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS case_recommendations (
                rec_id TEXT PRIMARY KEY,
                case_id TEXT,
                recommendation_type TEXT,
                priority INTEGER,
                description TEXT,
                expected_impact REAL,
                implementation_difficulty TEXT,
                FOREIGN KEY (case_id) REFERENCES integrated_cases (case_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_results (
                result_id TEXT PRIMARY KEY,
                case_id TEXT,
                optimization_type TEXT,
                parameters TEXT,  -- JSON of parameters
                objective_values TEXT,  -- JSON of objective values
                pareto_optimal BOOLEAN,
                FOREIGN KEY (case_id) REFERENCES integrated_cases (case_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_comprehensive_case(self, client_data: Dict[str, Any], 
                                 events: List[Dict[str, Any]] = None,
                                 constraints: Dict[str, Any] = None) -> IntegratedClientCase:
        """Perform comprehensive analysis of a client case"""
        
        client_id = client_data.get('client_id', f"CLIENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.info(f"Starting comprehensive analysis for {client_id}")
        
        if events is None:
            events = []
        if constraints is None:
            constraints = self._get_default_constraints()
        
        # Extract client profile for analysis
        client_profile = self._extract_client_profile(client_data)
        
        # 1. Timeline Bias Analysis
        logger.info("Performing timeline bias analysis...")
        if events:
            timeline_analysis = self.timeline_engine.estimate_event_timeline_with_bias(
                events, client_profile
            )
        else:
            # Generate forecast if no events provided
            timeline_analysis = ClientTimeline(
                client_id=client_id,
                events=self.timeline_engine.forecast_future_events(client_profile),
                bias_adjustments=[],
                confidence_score=0.5,
                similar_cases=[]
            )
        
        # 2. fsQCA Analysis for Financial Stability
        logger.info("Performing fsQCA analysis...")
        # Prepare data for fsQCA (simulate multiple scenarios)
        fsqca_data = self._prepare_fsqca_data(client_data, timeline_analysis)
        fsqca_analysis = self.fsqca_optimizer.analyze_financial_stability_paths(fsqca_data)
        
        # 3. Financial Stress Minimization
        logger.info("Optimizing for minimal financial stress...")
        initial_state = {
            'age': client_profile.get('age', 35),
            'income': client_profile.get('income', 75000),
            'expenses': client_data.get('expenses', 50000),
            'portfolio_value': client_data.get('portfolio_value', 100000)
        }
        optimal_path = self.stress_minimizer.find_optimal_path(
            initial_state, constraints, objective='minimize_stress'
        )
        
        # 4. Continuous Mesh Optimization
        logger.info("Performing continuous mesh optimization...")
        mesh_optimization = self._perform_mesh_optimization(client_data, optimal_path)
        
        # 5. Accounting Reconciliation Analysis
        logger.info("Analyzing accounting reconciliation...")
        reconciliation_reports = self._analyze_accounting_reconciliation(
            client_data, optimal_path
        )
        
        # 6. Generate Integrated Recommendations
        logger.info("Generating integrated recommendations...")
        recommendations = self._generate_integrated_recommendations(
            timeline_analysis, fsqca_analysis, optimal_path, 
            reconciliation_reports, mesh_optimization
        )
        
        # 7. Calculate Confidence Scores
        confidence_scores = self._calculate_confidence_scores(
            timeline_analysis, fsqca_analysis, optimal_path, reconciliation_reports
        )
        
        # Create integrated case
        integrated_case = IntegratedClientCase(
            client_id=client_id,
            profile=client_profile,
            timeline_analysis=timeline_analysis,
            fsqca_analysis=fsqca_analysis,
            optimal_path=optimal_path,
            reconciliation_reports=reconciliation_reports,
            mesh_optimization=mesh_optimization,
            recommendations=recommendations,
            confidence_scores=confidence_scores
        )
        
        # Store in database
        self._store_integrated_case(integrated_case)
        
        logger.info(f"Comprehensive analysis completed for {client_id}")
        return integrated_case
    
    def _extract_client_profile(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized client profile"""
        return {
            'client_id': client_data.get('client_id'),
            'age': client_data.get('age', 35),
            'income': client_data.get('income', 75000),
            'income_level': self._classify_income_level(client_data.get('income', 75000)),
            'education': client_data.get('education', 'bachelors'),
            'family_status': client_data.get('family_status', 'single'),
            'location': client_data.get('location', 'suburban'),
            'career_stage': self._determine_career_stage(client_data.get('age', 35)),
            'risk_tolerance': client_data.get('risk_tolerance', 0.6)
        }
    
    def _classify_income_level(self, income: float) -> str:
        """Classify income into standard levels"""
        if income < 50000:
            return 'low'
        elif income < 100000:
            return 'middle'
        elif income < 250000:
            return 'high'
        else:
            return 'ultra_high'
    
    def _determine_career_stage(self, age: float) -> str:
        """Determine career stage based on age"""
        if age < 30:
            return 'early_career'
        elif age < 45:
            return 'mid_career'
        elif age < 55:
            return 'established'
        elif age < 65:
            return 'pre_retirement'
        else:
            return 'retirement'
    
    def _get_default_constraints(self) -> Dict[str, Any]:
        """Get default optimization constraints"""
        return {
            'min_cash_cushion': 3.0,
            'max_stress': 0.7,
            'min_savings_rate': 0.05,
            'max_debt_ratio': 0.4
        }
    
    def _prepare_fsqca_data(self, client_data: Dict[str, Any], 
                          timeline_analysis: ClientTimeline) -> List[Dict[str, Any]]:
        """Prepare data for fsQCA analysis"""
        # Generate multiple scenarios based on timeline events
        scenarios = []
        
        base_income = client_data.get('income', 75000)
        base_expenses = client_data.get('expenses', 50000)
        
        # Create scenarios with different configurations
        for work_intensity in [0.6, 0.8, 1.0]:
            for savings_rate in [0.1, 0.15, 0.2, 0.25]:
                for expense_ratio in [0.6, 0.7, 0.8]:
                    
                    income = base_income * (0.7 + 0.3 * work_intensity)
                    expenses = base_expenses * expense_ratio
                    savings = income * savings_rate
                    
                    # Calculate financial stress
                    stress_factors = [
                        max(0, (expenses + savings) / income - 0.8),  # Overspending
                        max(0, 0.05 - savings_rate),  # Low savings
                        work_intensity * 0.3,  # Work stress
                    ]
                    financial_stress = min(1.0, sum(stress_factors))
                    
                    scenario = {
                        'client_id': client_data.get('client_id'),
                        'income': income,
                        'income_history': [income * (1 + np.random.normal(0, 0.05)) for _ in range(3)],
                        'total_expenses': expenses,
                        'total_debt': client_data.get('total_debt', 0),
                        'emergency_fund': savings * 6,  # 6 months savings as emergency fund
                        'portfolio_return': np.random.normal(0.08, 0.1),
                        'financial_stress': financial_stress,
                        'work_intensity': work_intensity,
                        'savings_rate': savings_rate,
                        'expense_ratio': expense_ratio
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _perform_mesh_optimization(self, client_data: Dict[str, Any], 
                                 optimal_path: OptimalPath) -> Dict[str, Any]:
        """Perform continuous mesh optimization"""
        # Define optimization dimensions based on client data
        dimensions = {
            'work_intensity': (0.4, 1.0),
            'savings_rate': (0.05, 0.4),
            'expense_ratio': (0.6, 0.9),
            'risk_tolerance': (0.2, 0.8)
        }
        
        # Create mesh
        mesh = ContinuousConfigurationMesh(dimensions, base_resolution=5)
        
        # Define objective functions
        def stress_objective(coords):
            return coords['work_intensity'] * 0.3 + (1 - coords['savings_rate']) * 0.4
        
        def quality_objective(coords):
            return -(coords['savings_rate'] * 0.5 + (1 - coords['work_intensity']) * 0.3)
        
        objective_functions = {
            'stress': stress_objective,
            'negative_quality': quality_objective
        }
        
        # Generate and analyze mesh
        mesh.generate_base_mesh(objective_functions)
        mesh.adaptive_refinement(objective_functions)
        mesh.build_interpolators(list(objective_functions.keys()))
        
        # Find Pareto optimal points
        pareto_points = mesh.find_pareto_optimal_points(
            list(objective_functions.keys()), minimize=[True, True]
        )
        
        return {
            'mesh_statistics': mesh.get_mesh_statistics(),
            'pareto_points': len(pareto_points),
            'optimization_quality': 'high' if len(pareto_points) > 5 else 'medium'
        }
    
    def _analyze_accounting_reconciliation(self, client_data: Dict[str, Any], 
                                         optimal_path: OptimalPath) -> List[ReconciliationReport]:
        """Analyze accounting reconciliation for optimal path"""
        reconciliation_engine = AccountingReconciliationEngine()
        reports = []
        
        # Simulate reconciliation for first few years of optimal path
        for i, node in enumerate(optimal_path.nodes[:12]):  # First year (monthly)
            month_start = datetime.now() + timedelta(days=30*i)
            month_end = month_start + timedelta(days=30)
            
            # Create entries based on node data
            from accounting_reconciliation import AccountEntry
            from decimal import Decimal
            
            entries = [
                AccountEntry('income', 'salary', Decimal(str(node.income/12)), 
                           month_start, 'Monthly salary', 0.0, 0.0),
                AccountEntry('expense', 'living_expenses', Decimal(str(node.expenses/12)), 
                           month_start, 'Monthly expenses', 0.0, 0.1),
                AccountEntry('savings', 'monthly_savings', Decimal(str(node.savings/12)), 
                           month_start, 'Monthly savings', 0.1, -0.05)
            ]
            
            for entry in entries:
                reconciliation_engine.add_entry(entry)
            
            # Reconcile the month
            report = reconciliation_engine.reconcile_period(
                month_start, month_end, node.lifestyle_config
            )
            reports.append(report)
            
            if i >= 5:  # Limit to 6 months for demo
                break
        
        return reports
    
    def _generate_integrated_recommendations(self, timeline_analysis: ClientTimeline,
                                           fsqca_analysis: fsQCAResult,
                                           optimal_path: OptimalPath,
                                           reconciliation_reports: List[ReconciliationReport],
                                           mesh_optimization: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate integrated recommendations from all analyses"""
        recommendations = []
        
        # Timeline-based recommendations
        if timeline_analysis.confidence_score < 0.7:
            recommendations.append({
                'type': 'timeline_uncertainty',
                'priority': 'high',
                'description': f'Timeline estimates have low confidence ({timeline_analysis.confidence_score:.1%}). '
                             f'Consider providing more specific dates for life events.',
                'expected_impact': 0.3,
                'source': 'timeline_analysis'
            })
        
        # fsQCA-based recommendations
        for path in fsqca_analysis.optimal_paths[:3]:
            recommendations.append({
                'type': 'financial_stability_path',
                'priority': 'medium',
                'description': f'Follow path: {path["formula"]} for financial stability '
                             f'(consistency: {path["consistency"]:.1%})',
                'expected_impact': path['stress_reduction_potential'],
                'source': 'fsqca_analysis'
            })
        
        # Stress minimization recommendations
        if optimal_path.total_stress > 0.5:
            recommendations.append({
                'type': 'stress_reduction',
                'priority': 'high',
                'description': f'Current path has high stress ({optimal_path.total_stress:.2f}). '
                             f'Consider alternative configurations.',
                'expected_impact': 0.4,
                'source': 'stress_optimization'
            })
        
        # Accounting reconciliation recommendations
        unbalanced_periods = sum(1 for report in reconciliation_reports if not report.is_balanced)
        if unbalanced_periods > 0:
            recommendations.append({
                'type': 'accounting_balance',
                'priority': 'critical',
                'description': f'{unbalanced_periods} periods have accounting imbalances. '
                             f'Review and adjust income/expense/savings allocations.',
                'expected_impact': 0.5,
                'source': 'accounting_reconciliation'
            })
        
        # Mesh optimization recommendations
        if mesh_optimization['optimization_quality'] == 'medium':
            recommendations.append({
                'type': 'optimization_improvement',
                'priority': 'low',
                'description': 'Configuration mesh could benefit from additional refinement '
                             'for more accurate optimization.',
                'expected_impact': 0.1,
                'source': 'mesh_optimization'
            })
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
    
    def _calculate_confidence_scores(self, timeline_analysis: ClientTimeline,
                                   fsqca_analysis: fsQCAResult,
                                   optimal_path: OptimalPath,
                                   reconciliation_reports: List[ReconciliationReport]) -> Dict[str, float]:
        """Calculate confidence scores for different analysis components"""
        
        # Timeline confidence
        timeline_confidence = timeline_analysis.confidence_score
        
        # fsQCA confidence (based on consistency and sample sizes)
        fsqca_confidence = 0.0
        if fsqca_analysis.optimal_paths:
            avg_consistency = np.mean([path['consistency'] for path in fsqca_analysis.optimal_paths])
            fsqca_confidence = min(1.0, avg_consistency)
        
        # Stress optimization confidence (based on feasibility)
        stress_confidence = optimal_path.feasibility_score
        
        # Accounting confidence (based on balance rate)
        balanced_periods = sum(1 for report in reconciliation_reports if report.is_balanced)
        accounting_confidence = balanced_periods / len(reconciliation_reports) if reconciliation_reports else 0.0
        
        # Overall confidence (weighted average)
        overall_confidence = (
            timeline_confidence * 0.25 +
            fsqca_confidence * 0.25 +
            stress_confidence * 0.25 +
            accounting_confidence * 0.25
        )
        
        return {
            'timeline': timeline_confidence,
            'fsqca': fsqca_confidence,
            'stress_optimization': stress_confidence,
            'accounting': accounting_confidence,
            'overall': overall_confidence
        }
    
    def _store_integrated_case(self, case: IntegratedClientCase) -> None:
        """Store integrated case in database"""
        conn = sqlite3.connect(self.case_db_path)
        cursor = conn.cursor()
        
        # Store main case record
        cursor.execute('''
            INSERT OR REPLACE INTO integrated_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case.client_id,
            json.dumps(case.profile),
            datetime.now().isoformat(),
            case.confidence_scores['timeline'],
            case.confidence_scores['fsqca'],
            case.confidence_scores['stress_optimization'],
            case.confidence_scores['accounting'],
            case.confidence_scores['overall']
        ))
        
        # Store recommendations
        for i, rec in enumerate(case.recommendations):
            rec_id = f"{case.client_id}_REC_{i:03d}"
            cursor.execute('''
                INSERT OR REPLACE INTO case_recommendations VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                rec_id, case.client_id, rec['type'], 
                i, rec['description'], rec['expected_impact'],
                rec.get('implementation_difficulty', 'medium')
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored integrated case {case.client_id} in database")
    
    def compare_similar_cases(self, target_case: IntegratedClientCase, 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Compare with similar cases in database for insights"""
        # This would implement sophisticated case similarity matching
        # For now, return a placeholder
        return [
            {
                'case_id': 'SIMILAR_001',
                'similarity_score': 0.85,
                'key_differences': ['Higher income', 'Earlier career stage'],
                'outcome_comparison': 'Better stress management achieved'
            }
        ]
    
    def generate_comprehensive_report(self, case: IntegratedClientCase) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE FINANCIAL ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Client ID: {case.client_id}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Overall Confidence: {case.confidence_scores['overall']:.1%}")
        report.append("")
        
        # Client Profile Summary
        report.append("CLIENT PROFILE:")
        report.append(f"  Age: {case.profile['age']}")
        report.append(f"  Income: ${case.profile['income']:,.0f} ({case.profile['income_level']})")
        report.append(f"  Career Stage: {case.profile['career_stage']}")
        report.append(f"  Risk Tolerance: {case.profile['risk_tolerance']:.1%}")
        report.append("")
        
        # Timeline Analysis
        report.append("TIMELINE ANALYSIS:")
        report.append(f"  Timeline Confidence: {case.confidence_scores['timeline']:.1%}")
        report.append(f"  Events Analyzed: {len(case.timeline_analysis.events)}")
        report.append(f"  Similar Cases Used: {len(case.timeline_analysis.similar_cases)}")
        report.append("")
        
        # fsQCA Results
        report.append("FINANCIAL STABILITY PATHS (fsQCA):")
        report.append(f"  Analysis Confidence: {case.confidence_scores['fsqca']:.1%}")
        report.append(f"  Optimal Paths Found: {len(case.fsqca_analysis.optimal_paths)}")
        if case.fsqca_analysis.optimal_paths:
            top_path = case.fsqca_analysis.optimal_paths[0]
            report.append(f"  Best Path: {top_path['formula']}")
            report.append(f"  Consistency: {top_path['consistency']:.1%}")
        report.append("")
        
        # Stress Optimization
        report.append("STRESS OPTIMIZATION:")
        report.append(f"  Optimization Confidence: {case.confidence_scores['stress_optimization']:.1%}")
        report.append(f"  Total Stress Score: {case.optimal_path.total_stress:.3f}")
        report.append(f"  Average Quality of Life: {case.optimal_path.avg_quality_of_life:.1%}")
        report.append(f"  Final Portfolio Value: ${case.optimal_path.final_portfolio_value:,.0f}")
        report.append("")
        
        # Accounting Analysis
        report.append("ACCOUNTING RECONCILIATION:")
        report.append(f"  Reconciliation Confidence: {case.confidence_scores['accounting']:.1%}")
        balanced_periods = sum(1 for r in case.reconciliation_reports if r.is_balanced)
        report.append(f"  Balanced Periods: {balanced_periods}/{len(case.reconciliation_reports)}")
        if case.reconciliation_reports:
            avg_quality = np.mean([r.quality_of_life_score for r in case.reconciliation_reports])
            avg_stress = np.mean([r.stress_level for r in case.reconciliation_reports])
            report.append(f"  Average Quality of Life: {avg_quality:.1%}")
            report.append(f"  Average Stress Level: {avg_stress:.1%}")
        report.append("")
        
        # Recommendations
        if case.recommendations:
            report.append("KEY RECOMMENDATIONS:")
            for i, rec in enumerate(case.recommendations[:5], 1):
                report.append(f"  {i}. [{rec['priority'].upper()}] {rec['description']}")
                report.append(f"     Expected Impact: {rec['expected_impact']:.1%}")
            report.append("")
        
        # Mesh Optimization
        if case.mesh_optimization:
            report.append("OPTIMIZATION ANALYSIS:")
            stats = case.mesh_optimization.get('mesh_statistics', {})
            report.append(f"  Mesh Quality: {case.mesh_optimization.get('optimization_quality', 'unknown')}")
            report.append(f"  Pareto Optimal Points: {case.mesh_optimization.get('pareto_points', 0)}")
            if 'feasible_points' in stats:
                report.append(f"  Feasibility Rate: {stats.get('feasibility_rate', 0):.1%}")
        
        return "\n".join(report)

def demo_case_database_integration():
    """Demonstrate the integrated case analysis system"""
    print("🔗 CASE DATABASE INTEGRATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create sample client data
    client_data = {
        'client_id': 'DEMO_CLIENT_001',
        'age': 35,
        'income': 85000,
        'expenses': 58000,
        'portfolio_value': 150000,
        'total_debt': 25000,
        'education': 'masters',
        'family_status': 'married',
        'location': 'urban',
        'risk_tolerance': 0.6
    }
    
    # Sample life events (some without exact dates)
    events = [
        {'type': 'house_purchase', 'description': 'Planning to buy a house', 'amount': -400000},
        {'type': 'child_birth', 'description': 'Planning to have children', 'amount': -15000},
        {'type': 'career_advancement', 'description': 'Expecting promotion', 'amount': 20000},
        {'type': 'education_completion', 'description': 'Completed MBA last year', 
         'amount': -50000, 'estimated_date': '2024-06-15'}
    ]
    
    print(f"📊 Client Data Summary:")
    print(f"   • Age: {client_data['age']}")
    print(f"   • Income: ${client_data['income']:,}")
    print(f"   • Portfolio: ${client_data['portfolio_value']:,}")
    print(f"   • Events to Analyze: {len(events)}")
    
    # Initialize orchestrator
    print(f"\n🔧 Initializing Integrated Analysis System...")
    orchestrator = CaseAnalysisOrchestrator()
    
    # Perform comprehensive analysis
    print(f"\n🔍 Performing Comprehensive Analysis...")
    integrated_case = orchestrator.analyze_comprehensive_case(
        client_data=client_data,
        events=events,
        constraints={
            'min_cash_cushion': 4.0,
            'max_stress': 0.6,
            'min_savings_rate': 0.10
        }
    )
    
    # Display results summary
    print(f"\n📋 ANALYSIS RESULTS SUMMARY:")
    print(f"Case ID: {integrated_case.client_id}")
    print(f"Overall Confidence: {integrated_case.confidence_scores['overall']:.1%}")
    print(f"")
    
    print(f"Component Confidence Scores:")
    for component, score in integrated_case.confidence_scores.items():
        if component != 'overall':
            print(f"   • {component.replace('_', ' ').title()}: {score:.1%}")
    
    print(f"\nKey Metrics:")
    print(f"   • Timeline Events: {len(integrated_case.timeline_analysis.events)}")
    print(f"   • fsQCA Paths: {len(integrated_case.fsqca_analysis.optimal_paths)}")
    print(f"   • Stress Score: {integrated_case.optimal_path.total_stress:.3f}")
    print(f"   • Quality of Life: {integrated_case.optimal_path.avg_quality_of_life:.1%}")
    print(f"   • Reconciliation Periods: {len(integrated_case.reconciliation_reports)}")
    
    # Show top recommendations
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(integrated_case.recommendations[:3], 1):
        print(f"   {i}. [{rec['priority'].upper()}] {rec['description'][:80]}...")
        print(f"      Impact: {rec['expected_impact']:.1%} | Source: {rec['source']}")
    
    # Generate and display comprehensive report
    print(f"\n📄 COMPREHENSIVE REPORT:")
    print("-" * 80)
    comprehensive_report = orchestrator.generate_comprehensive_report(integrated_case)
    print(comprehensive_report)
    
    # Compare with similar cases
    print(f"\n🔍 SIMILAR CASE ANALYSIS:")
    similar_cases = orchestrator.compare_similar_cases(integrated_case)
    for case in similar_cases:
        print(f"   • {case['case_id']}: {case['similarity_score']:.1%} similar")
        print(f"     Outcome: {case['outcome_comparison']}")

if __name__ == "__main__":
    demo_case_database_integration() 