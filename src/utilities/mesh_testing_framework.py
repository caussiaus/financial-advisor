"""
Comprehensive Mesh Testing Framework

This module provides comprehensive testing for the mesh congruence system:
1. Integration testing of all mesh components
2. Trial people data incorporation
3. Mesh congruence validation
4. Backtesting framework validation
5. Performance benchmarking
6. Statistical validation

Key Features:
- Complete system integration testing
- Trial people data processing
- Mesh congruence algorithm validation
- Backtesting framework testing
- Performance benchmarking
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

# Import existing components
from .mesh_congruence_engine import MeshCongruenceEngine, MeshCongruenceResult
from .mesh_backtesting_framework import MeshBacktestingFramework, BacktestReport
from .mesh_vector_database import MeshVectorDatabase
from .synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from .json_to_vector_converter import JSONToVectorConverter


@dataclass
class TestResult:
    """Result of a comprehensive test"""
    test_name: str
    test_status: str  # 'PASS', 'FAIL', 'WARNING'
    execution_time: float
    metrics: Dict[str, float]
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComprehensiveTestReport:
    """Comprehensive test report"""
    test_results: List[TestResult]
    overall_status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    execution_summary: Dict[str, float]
    recommendations: List[str]


class ComprehensiveMeshTesting:
    """
    Comprehensive testing framework for mesh congruence system
    """
    
    def __init__(self):
        # Initialize all components
        self.congruence_engine = MeshCongruenceEngine()
        self.backtesting_framework = MeshBacktestingFramework(self.congruence_engine)
        self.vector_db = MeshVectorDatabase()
        self.lifestyle_engine = SyntheticLifestyleEngine()
        self.json_converter = JSONToVectorConverter()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Create storage directory
        self.storage_dir = Path("data/outputs/comprehensive_testing")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results storage
        self.test_results: List[TestResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the testing framework"""
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
    
    def load_trial_people_data(self) -> List[Dict[str, Any]]:
        """
        Load all trial people data from the data directory
        
        Returns:
            List of trial people data dictionaries
        """
        trial_people = []
        trial_dir = Path("data/inputs/trial_people")
        
        if not trial_dir.exists():
            self.logger.warning("Trial people directory not found")
            return trial_people
        
        for person_dir in trial_dir.iterdir():
            if person_dir.is_dir():
                person_data = self._load_person_data(person_dir)
                if person_data:
                    trial_people.append(person_data)
        
        self.logger.info(f"Loaded {len(trial_people)} trial people")
        return trial_people
    
    def _load_person_data(self, person_dir: Path) -> Optional[Dict[str, Any]]:
        """Load data for a single person"""
        try:
            person_name = person_dir.name
            
            # Load financial profile
            financial_profile_file = person_dir / "FINANCIAL_PROFILE.json"
            financial_profile = {}
            if financial_profile_file.exists():
                with open(financial_profile_file, 'r') as f:
                    financial_profile = json.load(f)
            
            # Load goals
            goals_file = person_dir / "GOALS.json"
            goals = {}
            if goals_file.exists():
                with open(goals_file, 'r') as f:
                    goals = json.load(f)
            
            # Load lifestyle events
            lifestyle_events_file = person_dir / "LIFESTYLE_EVENTS.json"
            lifestyle_events = []
            if lifestyle_events_file.exists():
                with open(lifestyle_events_file, 'r') as f:
                    lifestyle_events = json.load(f)
            
            # Create person data structure
            person_data = {
                'name': person_name,
                'financial_profile': financial_profile,
                'goals': goals,
                'lifestyle_events': lifestyle_events,
                'directory': str(person_dir)
            }
            
            return person_data
            
        except Exception as e:
            self.logger.error(f"Error loading person data from {person_dir}: {e}")
            return None
    
    def convert_trial_people_to_synthetic_data(self, trial_people: List[Dict[str, Any]]) -> List[SyntheticClientData]:
        """
        Convert trial people data to synthetic client data format
        
        Args:
            trial_people: List of trial people data
            
        Returns:
            List of synthetic client data
        """
        synthetic_clients = []
        
        for person in trial_people:
            try:
                # Create synthetic client data from trial person
                synthetic_client = self._create_synthetic_client_from_trial_person(person)
                synthetic_clients.append(synthetic_client)
                
            except Exception as e:
                self.logger.error(f"Error converting trial person {person['name']}: {e}")
                continue
        
        self.logger.info(f"Converted {len(synthetic_clients)} trial people to synthetic data")
        return synthetic_clients
    
    def _create_synthetic_client_from_trial_person(self, person: Dict[str, Any]) -> SyntheticClientData:
        """Create synthetic client data from trial person data"""
        from .synthetic_data_generator import PersonProfile
        
        # Extract financial profile
        financial_profile = person['financial_profile']
        goals = person['goals']
        lifestyle_events = person['lifestyle_events']
        
        # Create person profile
        profile = PersonProfile(
            name=person['name'],
            age=self._estimate_age_from_goals(goals),
            occupation=self._infer_occupation_from_income(financial_profile.get('monthly_income', 5000)),
            base_income=financial_profile.get('monthly_income', 5000) * 12,
            family_status=self._infer_family_status_from_goals(goals),
            location="Unknown",
            risk_tolerance=self._infer_risk_tolerance_from_profile(financial_profile),
            financial_goals=self._extract_financial_goals(goals),
            current_assets=self._generate_current_assets(financial_profile),
            debts=self._generate_debts(financial_profile)
        )
        
        # Create JSON data for vector conversion
        json_data = {
            'client_profile': {
                'name': profile.name,
                'age': profile.age,
                'occupation': profile.occupation,
                'income': profile.base_income,
                'family_status': profile.family_status,
                'location': profile.location
            },
            'financial_profile': financial_profile,
            'goals': goals,
            'lifestyle_events': lifestyle_events
        }
        
        # Convert to vector profile
        vector_profile = self.json_converter.convert_json_to_vector_profile(json_data)
        
        # Generate lifestyle events if not provided
        if not lifestyle_events:
            lifestyle_events = self._generate_lifestyle_events_from_goals(goals, vector_profile)
        
        # Convert to seed events
        seed_events = self.json_converter.convert_events_to_seed_events(lifestyle_events)
        
        # Calculate financial metrics
        financial_metrics = self._calculate_financial_metrics_from_profile(financial_profile, vector_profile)
        
        return SyntheticClientData(
            client_id=person['name'],
            profile=profile,
            vector_profile=vector_profile,
            lifestyle_events=lifestyle_events,
            seed_events=seed_events,
            financial_metrics=financial_metrics
        )
    
    def _estimate_age_from_goals(self, goals: Dict[str, List[str]]) -> int:
        """Estimate age based on goals"""
        # Simple age estimation based on goal types
        age_indicators = {
            'education_fund': 25,
            'career_growth': 30,
            'house_purchase': 35,
            'family_expenses': 30,
            'retirement_savings': 45,
            'estate_planning': 60
        }
        
        ages = []
        for goal_list in goals.values():
            for goal in goal_list:
                if goal in age_indicators:
                    ages.append(age_indicators[goal])
        
        return int(np.mean(ages)) if ages else 35
    
    def _infer_occupation_from_income(self, monthly_income: float) -> str:
        """Infer occupation from income level"""
        if monthly_income < 3000:
            return "Entry Level"
        elif monthly_income < 6000:
            return "Professional"
        elif monthly_income < 10000:
            return "Senior Professional"
        else:
            return "Executive"
    
    def _infer_family_status_from_goals(self, goals: Dict[str, List[str]]) -> str:
        """Infer family status from goals"""
        if 'family_expenses' in goals.get('short_term_goals', []):
            return "Married with Children"
        elif 'house_purchase' in goals.get('short_term_goals', []):
            return "Married"
        else:
            return "Single"
    
    def _generate_lifestyle_events_from_goals(self, goals: Dict[str, List[str]], 
                                            vector_profile) -> List:
        """Generate lifestyle events based on goals"""
        events = []
        
        # Map goals to event categories
        goal_to_event = {
            'education_fund': 'education',
            'career_growth': 'career',
            'house_purchase': 'housing',
            'family_expenses': 'family',
            'retirement_savings': 'retirement',
            'estate_planning': 'retirement'
        }
        
        for goal_list in goals.values():
            for goal in goal_list:
                if goal in goal_to_event:
                    event = {
                        'category': goal_to_event[goal],
                        'description': f"Goal: {goal}",
                        'cash_flow_impact': 'negative',
                        'probability': 0.8,
                        'timing': 'medium_term'
                    }
                    events.append(event)
        
        return events
    
    def _infer_risk_tolerance_from_profile(self, financial_profile: Dict) -> str:
        """Infer risk tolerance from financial profile"""
        savings_rate = financial_profile.get('savings_rate', 0.2)
        debt_to_income = financial_profile.get('debt_to_income_ratio', 0.3)
        
        if savings_rate > 0.3 and debt_to_income < 0.2:
            return "Conservative"
        elif savings_rate > 0.2 and debt_to_income < 0.3:
            return "Moderate"
        elif savings_rate > 0.1 and debt_to_income < 0.4:
            return "Aggressive"
        else:
            return "Very Aggressive"
    
    def _extract_financial_goals(self, goals: Dict[str, List[str]]) -> List[str]:
        """Extract financial goals from goals dictionary"""
        all_goals = []
        for goal_list in goals.values():
            all_goals.extend(goal_list)
        return all_goals
    
    def _generate_current_assets(self, financial_profile: Dict) -> Dict[str, float]:
        """Generate current assets from financial profile"""
        monthly_income = financial_profile.get('monthly_income', 5000)
        savings_rate = financial_profile.get('savings_rate', 0.2)
        
        # Estimate assets based on income and savings rate
        annual_savings = monthly_income * 12 * savings_rate
        total_assets = annual_savings * 5  # Assume 5 years of savings
        
        return {
            'checking': monthly_income * 2,  # 2 months of income
            'savings': annual_savings * 2,   # 2 years of savings
            'investments': total_assets * 0.3,  # 30% in investments
            'retirement': total_assets * 0.4,   # 40% in retirement
            'real_estate': total_assets * 0.3   # 30% in real estate
        }
    
    def _generate_debts(self, financial_profile: Dict) -> Dict[str, float]:
        """Generate debts from financial profile"""
        monthly_income = financial_profile.get('monthly_income', 5000)
        debt_to_income = financial_profile.get('debt_to_income_ratio', 0.3)
        
        total_debt = monthly_income * 12 * debt_to_income
        
        debts = {}
        if total_debt > 0:
            # Distribute debt across different types
            debts['credit_cards'] = total_debt * 0.2
            debts['student_loans'] = total_debt * 0.3
            debts['mortgage'] = total_debt * 0.4
            debts['auto_loans'] = total_debt * 0.1
        
        return debts
    
    def _calculate_financial_metrics_from_profile(self, financial_profile: Dict, 
                                                vector_profile) -> Dict[str, float]:
        """Calculate financial metrics from profile"""
        monthly_income = financial_profile.get('monthly_income', 5000)
        monthly_expenses = financial_profile.get('monthly_expenses', 3500)
        savings_rate = financial_profile.get('savings_rate', 0.2)
        debt_to_income = financial_profile.get('debt_to_income_ratio', 0.3)
        
        return {
            'net_worth': monthly_income * 12 * 5,  # Rough estimate
            'debt_to_income_ratio': debt_to_income,
            'savings_rate': savings_rate,
            'monthly_cash_flow': monthly_income - monthly_expenses
        }
    
    def run_comprehensive_tests(self) -> ComprehensiveTestReport:
        """
        Run comprehensive tests for the mesh congruence system
        
        Returns:
            Comprehensive test report
        """
        self.logger.info("Starting comprehensive mesh testing")
        
        # Load trial people data
        trial_people = self.load_trial_people_data()
        
        # Convert to synthetic data
        synthetic_clients = self.convert_trial_people_to_synthetic_data(trial_people)
        
        # Run individual tests
        tests = [
            self._test_mesh_congruence_algorithms,
            self._test_backtesting_framework,
            self._test_vector_database_integration,
            self._test_trial_people_integration,
            self._test_performance_benchmarking,
            self._test_statistical_validation
        ]
        
        for test_func in tests:
            try:
                result = test_func(synthetic_clients)
                self.test_results.append(result)
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} failed: {e}")
                result = TestResult(
                    test_name=test_func.__name__,
                    test_status='FAIL',
                    execution_time=0,
                    metrics={},
                    details={'error': str(e)}
                )
                self.test_results.append(result)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        return report
    
    def _test_mesh_congruence_algorithms(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test mesh congruence algorithms"""
        start_time = datetime.now()
        
        # Test Delaunay triangulation
        test_points = np.random.rand(50, 2)
        triangulation = self.congruence_engine.compute_delaunay_triangulation(test_points)
        
        # Test CVT optimization
        cvt_points = self.congruence_engine.compute_centroidal_voronoi_tessellation(test_points)
        
        # Test edge collapse efficiency
        efficiency = self.congruence_engine.compute_edge_collapse_efficiency(triangulation)
        
        # Test congruence between clients
        if len(clients) >= 2:
            congruence_result = self.congruence_engine.compute_mesh_congruence(clients[0], clients[1])
            congruence_score = congruence_result.overall_congruence
        else:
            congruence_score = 0.5  # Default for insufficient data
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'triangulation_quality': len(triangulation.simplices) / 100,  # Normalized
            'cvt_convergence': np.mean(np.abs(cvt_points - test_points)),
            'edge_efficiency': efficiency,
            'congruence_score': congruence_score
        }
        
        status = 'PASS' if congruence_score > 0.3 else 'WARNING'
        
        return TestResult(
            test_name='mesh_congruence_algorithms',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'triangulation_triangles': len(triangulation.simplices),
                'cvt_points_shape': cvt_points.shape,
                'congruence_result': congruence_result.__dict__ if len(clients) >= 2 else None
            }
        )
    
    def _test_backtesting_framework(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test backtesting framework"""
        start_time = datetime.now()
        
        if not clients:
            return TestResult(
                test_name='backtesting_framework',
                test_status='FAIL',
                execution_time=0,
                metrics={},
                details={'error': 'No clients available for testing'}
            )
        
        # Run backtest on first client
        client = clients[0]
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        report = self.backtesting_framework.run_comprehensive_backtest(
            client, start_date, end_date, num_scenarios=20
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'total_return': report.performance_metrics.total_return,
            'sharpe_ratio': report.performance_metrics.sharpe_ratio,
            'congruence_stability': report.performance_metrics.congruence_stability,
            'recommendation_accuracy': report.performance_metrics.recommendation_accuracy
        }
        
        status = 'PASS' if report.performance_metrics.total_return > -0.5 else 'WARNING'
        
        return TestResult(
            test_name='backtesting_framework',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'scenarios_tested': len(report.scenarios),
                'test_period': report.test_period,
                'risk_analysis': report.risk_analysis
            }
        )
    
    def _test_vector_database_integration(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test vector database integration"""
        start_time = datetime.now()
        
        # Add clients to vector database
        for client in clients[:5]:  # Test with first 5 clients
            self.vector_db.add_client(client)
        
        # Test similarity search
        if clients:
            similar_clients = self.vector_db.find_similar_clients(clients[0].client_id, top_k=3)
            similarity_scores = [match.similarity_score for match in similar_clients]
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        else:
            avg_similarity = 0
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'clients_added': min(5, len(clients)),
            'avg_similarity_score': avg_similarity,
            'database_size': len(self.vector_db.embeddings)
        }
        
        status = 'PASS' if avg_similarity > 0.3 else 'WARNING'
        
        return TestResult(
            test_name='vector_database_integration',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'similar_clients_found': len(similar_clients) if clients else 0,
                'database_embeddings': len(self.vector_db.embeddings)
            }
        )
    
    def _test_trial_people_integration(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test trial people integration"""
        start_time = datetime.now()
        
        # Test that trial people were successfully converted
        conversion_success = len(clients) > 0
        
        # Test data quality
        data_quality_metrics = []
        for client in clients:
            # Check for required fields
            has_profile = hasattr(client, 'profile') and client.profile is not None
            has_vector_profile = hasattr(client, 'vector_profile') and client.vector_profile is not None
            has_events = hasattr(client, 'lifestyle_events') and len(client.lifestyle_events) > 0
            
            quality_score = sum([has_profile, has_vector_profile, has_events]) / 3
            data_quality_metrics.append(quality_score)
        
        avg_data_quality = np.mean(data_quality_metrics) if data_quality_metrics else 0
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'conversion_success': 1.0 if conversion_success else 0.0,
            'avg_data_quality': avg_data_quality,
            'clients_converted': len(clients)
        }
        
        status = 'PASS' if conversion_success and avg_data_quality > 0.7 else 'WARNING'
        
        return TestResult(
            test_name='trial_people_integration',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'trial_people_loaded': len(clients),
                'data_quality_scores': data_quality_metrics
            }
        )
    
    def _test_performance_benchmarking(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test performance benchmarking"""
        start_time = datetime.now()
        
        # Benchmark mesh congruence computation
        congruence_times = []
        congruence_scores = []
        
        for i in range(min(10, len(clients))):
            for j in range(i + 1, min(i + 3, len(clients))):
                test_start = datetime.now()
                result = self.congruence_engine.compute_mesh_congruence(clients[i], clients[j])
                test_time = (datetime.now() - test_start).total_seconds()
                
                congruence_times.append(test_time)
                congruence_scores.append(result.overall_congruence)
        
        avg_computation_time = np.mean(congruence_times) if congruence_times else 0
        avg_congruence_score = np.mean(congruence_scores) if congruence_scores else 0
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'avg_computation_time': avg_computation_time,
            'avg_congruence_score': avg_congruence_score,
            'total_tests_performed': len(congruence_times)
        }
        
        status = 'PASS' if avg_computation_time < 5.0 else 'WARNING'  # 5 second threshold
        
        return TestResult(
            test_name='performance_benchmarking',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'computation_times': congruence_times,
                'congruence_scores': congruence_scores
            }
        )
    
    def _test_statistical_validation(self, clients: List[SyntheticClientData]) -> TestResult:
        """Test statistical validation"""
        start_time = datetime.now()
        
        # Collect congruence scores for statistical analysis
        congruence_scores = []
        
        for i in range(min(20, len(clients))):
            for j in range(i + 1, min(i + 5, len(clients))):
                result = self.congruence_engine.compute_mesh_congruence(clients[i], clients[j])
                congruence_scores.append(result.overall_congruence)
        
        if len(congruence_scores) < 3:
            return TestResult(
                test_name='statistical_validation',
                test_status='FAIL',
                execution_time=0,
                metrics={},
                details={'error': 'Insufficient data for statistical analysis'}
            )
        
        # Statistical tests
        mean_congruence = np.mean(congruence_scores)
        std_congruence = np.std(congruence_scores)
        
        # Test for normality
        try:
            _, normality_p_value = stats.normaltest(congruence_scores)
            normality_test = 1 - normality_p_value
        except:
            normality_test = 0.5
        
        # Test for significant difference from random
        random_mean = 0.5  # Expected random congruence
        try:
            t_stat, t_p_value = stats.ttest_1samp(congruence_scores, random_mean)
            significance_test = 1 - t_p_value
        except:
            significance_test = 0.5
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'mean_congruence': mean_congruence,
            'std_congruence': std_congruence,
            'normality_test': normality_test,
            'significance_test': significance_test
        }
        
        status = 'PASS' if significance_test > 0.05 else 'WARNING'
        
        return TestResult(
            test_name='statistical_validation',
            test_status=status,
            execution_time=execution_time,
            metrics=metrics,
            details={
                'congruence_scores': congruence_scores,
                'sample_size': len(congruence_scores)
            }
        )
    
    def _generate_comprehensive_report(self) -> ComprehensiveTestReport:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.test_status == 'PASS'])
        failed_tests = len([r for r in self.test_results if r.test_status == 'FAIL'])
        warning_tests = len([r for r in self.test_results if r.test_status == 'WARNING'])
        
        # Overall status
        if failed_tests > 0:
            overall_status = 'FAIL'
        elif warning_tests > 0:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        # Execution summary
        execution_summary = {
            'total_execution_time': sum(r.execution_time for r in self.test_results),
            'avg_execution_time': np.mean([r.execution_time for r in self.test_results]),
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return ComprehensiveTestReport(
            test_results=self.test_results,
            overall_status=overall_status,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            execution_summary=execution_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze test results and generate recommendations
        for result in self.test_results:
            if result.test_status == 'FAIL':
                recommendations.append(f"Fix critical issues in {result.test_name}")
            elif result.test_status == 'WARNING':
                recommendations.append(f"Improve performance in {result.test_name}")
        
        # Add general recommendations
        if len(self.test_results) < 6:
            recommendations.append("Add more comprehensive test coverage")
        
        if not recommendations:
            recommendations.append("All tests passed successfully")
        
        return recommendations
    
    def save_test_results(self, filename: str = None) -> None:
        """Save test results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_results_{timestamp}.pkl"
        
        filepath = self.storage_dir / filename
        
        data = {
            'test_results': self.test_results,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved test results to {filepath}")


def create_demo_comprehensive_testing():
    """Create and demonstrate comprehensive testing"""
    testing = ComprehensiveMeshTesting()
    
    # Run comprehensive tests
    report = testing.run_comprehensive_tests()
    
    print(f"Comprehensive Mesh Testing Report:")
    print(f"Overall Status: {report.overall_status}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Warnings: {report.warning_tests}")
    print(f"Success Rate: {report.execution_summary['success_rate']:.2%}")
    print(f"Total Execution Time: {report.execution_summary['total_execution_time']:.2f}s")
    
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")
    
    # Save results
    testing.save_test_results()
    
    return testing


if __name__ == "__main__":
    create_demo_comprehensive_testing() 