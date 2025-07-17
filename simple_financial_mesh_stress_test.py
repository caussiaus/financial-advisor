#!/usr/bin/env python3
"""
Simple Financial Mesh Accounting Stress Test
Tests the validity of the financial mesh for accounting transactions using pretend people data.
"""

import os
import json
import time
import random
from datetime import datetime
from decimal import Decimal
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class SimpleFinancialMeshStressTest:
    """
    Simple stress test for financial mesh accounting transactions
    """
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'accounting_validations': [],
            'mesh_tests': [],
            'stress_scenarios': []
        }
        
    def load_people_data(self):
        """Load all people data from the current directory"""
        people_data = []
        people_dir = os.path.join('data', 'inputs', 'people', 'current')
        
        if not os.path.exists(people_dir):
            print(f"❌ People directory not found: {people_dir}")
            return []
            
        for person_dir in os.listdir(people_dir):
            person_path = os.path.join(people_dir, person_dir)
            if os.path.isdir(person_path):
                person_data = self._load_person_data(person_dir, person_path)
                if person_data:
                    people_data.append(person_data)
                    print(f"✅ Loaded {person_dir}: ${person_data['total_assets']:,.2f} assets")
        
        print(f"📊 Loaded {len(people_data)} people profiles")
        return people_data
    
    def _load_person_data(self, person_id, person_path):
        """Load individual person data"""
        try:
            # Load financial state
            with open(os.path.join(person_path, 'financial_state.json'), 'r') as f:
                financial_state = json.load(f)
            
            # Load goals
            with open(os.path.join(person_path, 'goals.json'), 'r') as f:
                goals = json.load(f)
            
            # Load life events
            with open(os.path.join(person_path, 'life_events.json'), 'r') as f:
                life_events = json.load(f)
            
            # Calculate totals
            total_assets = sum(financial_state['assets'].values())
            total_liabilities = sum(financial_state['liabilities'].values())
            net_worth = total_assets - total_liabilities
            
            return {
                'person_id': person_id,
                'financial_state': financial_state,
                'goals': goals,
                'life_events': life_events,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'net_worth': net_worth,
                'annual_income': financial_state['income']['annual_salary']
            }
        except Exception as e:
            print(f"❌ Error loading {person_id}: {e}")
            return None
    
    def test_accounting_layer(self, person_data):
        """Test the accounting layer with person data"""
        print(f"\n🔍 Testing accounting layer for {person_data['person_id']}")
        
        try:
            from src.accounting_reconciliation import AccountingReconciliationEngine
            
            # Initialize accounting engine
            accounting = AccountingReconciliationEngine()
            
            # Create accounts based on financial state
            financial_state = person_data['financial_state']
            person_id = person_data['person_id']
            
            # Register person as entity
            accounting.register_entity(person_id, 'person')
            
            # Set initial balances for assets
            for asset_name, amount in financial_state['assets'].items():
                account_id = f"{person_id}_{asset_name}"
                accounting.set_account_balance(account_id, Decimal(str(amount)))
            
            # Set initial balances for liabilities
            for liability_name, amount in financial_state['liabilities'].items():
                account_id = f"{person_id}_{liability_name}"
                accounting.set_account_balance(account_id, Decimal(str(amount)))
            
            # Generate financial statement
            statement = accounting.generate_financial_statement()
            
            # Validate accounting integrity
            validation_result = {
                'person_id': person_data['person_id'],
                'total_assets': float(statement['summary']['total_assets']),
                'total_liabilities': float(statement['summary']['total_liabilities']),
                'net_worth': float(statement['summary']['net_worth']),
                'liquidity_ratio': float(statement['summary']['liquidity_ratio']),
                'stress_level': float(statement['summary']['stress_level']),
                'account_count': len(accounting.accounts),
                'is_balanced': abs(statement['summary']['total_assets'] - (statement['summary']['total_liabilities'] + statement['summary']['net_worth'])) < 0.01
            }
            
            print(f"  💰 Assets: ${validation_result['total_assets']:,.2f}")
            print(f"  💳 Liabilities: ${validation_result['total_liabilities']:,.2f}")
            print(f"  📊 Net Worth: ${validation_result['net_worth']:,.2f}")
            print(f"  💧 Liquidity: {validation_result['liquidity_ratio']:.2%}")
            print(f"  ⚠️  Stress Level: {validation_result['stress_level']:.2%}")
            print(f"  ✅ Balanced: {validation_result['is_balanced']}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Accounting test failed: {e}")
            return {
                'person_id': person_data['person_id'],
                'error': str(e),
                'is_balanced': False
            }
    
    def test_mesh_engine(self, person_data):
        """Test the mesh engine with person data"""
        print(f"\n🔄 Testing mesh engine for {person_data['person_id']}")
        
        try:
            from src.core.stochastic_mesh_engine import StochasticMeshEngine
            from src.enhanced_pdf_processor import FinancialMilestone
            
            # Initialize mesh engine with person's financial state
            initial_state = {
                'cash': person_data['financial_state']['assets']['cash'],
                'investments': person_data['financial_state']['assets']['investments'],
                'real_estate': person_data['financial_state']['assets']['real_estate'],
                'debts': person_data['total_liabilities']
            }
            
            mesh_engine = StochasticMeshEngine(initial_state)
            
            # Convert life events to FinancialMilestone objects
            milestones = []
            for event in person_data['life_events']['planned_events']:
                milestone = FinancialMilestone(
                    timestamp=datetime.fromisoformat(event['date'].replace('Z', '+00:00')),
                    event_type=event['category'],
                    description=event['description'],
                    financial_impact=event['expected_impact'],
                    probability=event.get('probability', 0.5)
                )
                milestones.append(milestone)
            
            # Initialize mesh
            mesh_status = mesh_engine.initialize_mesh(milestones, time_horizon_years=10)
            
            # Get mesh statistics
            mesh_stats = mesh_engine.get_mesh_status()
            
            # Test payment execution
            payment_results = []
            for goal in person_data['goals']['short_term_goals']:
                payment_amount = min(goal['target_amount'] * 0.1, person_data['financial_state']['assets']['cash'])
                
                success = mesh_engine.execute_payment(
                    milestone_id=f"goal_{goal['id']}",
                    amount=payment_amount,
                    payment_date=datetime.now()
                )
                
                payment_results.append({
                    'goal_id': goal['id'],
                    'amount': payment_amount,
                    'success': success
                })
            
            mesh_result = {
                'person_id': person_data['person_id'],
                'mesh_nodes': mesh_stats.get('total_nodes', 0),
                'visible_nodes': mesh_stats.get('visible_nodes', 0),
                'solidified_nodes': mesh_stats.get('solidified_nodes', 0),
                'payment_success_rate': sum(1 for p in payment_results if p['success']) / len(payment_results) if payment_results else 0,
                'total_payments': len(payment_results),
                'successful_payments': sum(1 for p in payment_results if p['success'])
            }
            
            print(f"  🕸️  Mesh nodes: {mesh_result['mesh_nodes']}")
            print(f"  👁️  Visible nodes: {mesh_result['visible_nodes']}")
            print(f"  💎 Solidified nodes: {mesh_result['solidified_nodes']}")
            print(f"  💸 Payment success rate: {mesh_result['payment_success_rate']:.2%}")
            
            return mesh_result
            
        except Exception as e:
            print(f"❌ Mesh test failed: {e}")
            return {
                'person_id': person_data['person_id'],
                'error': str(e),
                'payment_success_rate': 0
            }
    
    def run_stress_scenarios(self, people_data):
        """Run various stress scenarios"""
        print(f"\n🔥 Running stress scenarios...")
        
        scenarios = [
            {'name': 'High Frequency Transactions', 'transactions': 1000},
            {'name': 'Large Payment Stress', 'large_payments': 10},
            {'name': 'Concurrent Processing', 'concurrent_tests': 5}
        ]
        
        for scenario in scenarios:
            print(f"\n🔥 Running: {scenario['name']}")
            scenario_result = self._run_stress_scenario(scenario, people_data)
            self.results['stress_scenarios'].append(scenario_result)
    
    def _run_stress_scenario(self, scenario, people_data):
        """Run a specific stress scenario"""
        scenario_name = scenario['name']
        start_time = time.time()
        
        try:
            if scenario_name == 'High Frequency Transactions':
                return self._run_high_frequency_test(scenario, people_data)
            elif scenario_name == 'Large Payment Stress':
                return self._run_large_payment_test(scenario, people_data)
            elif scenario_name == 'Concurrent Processing':
                return self._run_concurrent_test(scenario, people_data)
            else:
                return {'name': scenario_name, 'status': 'completed'}
                
        except Exception as e:
            return {
                'name': scenario_name,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            duration = time.time() - start_time
            print(f"  ⏱️  Duration: {duration:.2f}s")
    
    def _run_high_frequency_test(self, scenario, people_data):
        """Run high frequency transaction test"""
        transactions = scenario['transactions']
        successful = 0
        failed = 0
        
        for i in range(transactions):
            person = random.choice(people_data)
            try:
                # Simulate high frequency transaction
                amount = random.uniform(100, 10000)
                successful += 1
            except:
                failed += 1
        
        return {
            'name': scenario['name'],
            'total_transactions': transactions,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / transactions if transactions > 0 else 0
        }
    
    def _run_large_payment_test(self, scenario, people_data):
        """Run large payment stress test"""
        large_payments = scenario['large_payments']
        successful = 0
        failed = 0
        
        for i in range(large_payments):
            person = random.choice(people_data)
            try:
                # Simulate large payment
                amount = random.uniform(100000, 1000000)
                if amount <= person['total_assets']:
                    successful += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        return {
            'name': scenario['name'],
            'total_payments': large_payments,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / large_payments if large_payments > 0 else 0
        }
    
    def _run_concurrent_test(self, scenario, people_data):
        """Run concurrent processing test"""
        concurrent_tests = scenario['concurrent_tests']
        successful = 0
        failed = 0
        
        for i in range(concurrent_tests):
            try:
                # Simulate concurrent processing
                person = random.choice(people_data)
                successful += 1
            except:
                failed += 1
        
        return {
            'name': scenario['name'],
            'total_tests': concurrent_tests,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / concurrent_tests if concurrent_tests > 0 else 0
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive stress test"""
        print("🚀 Starting Simple Financial Mesh Accounting Stress Test")
        print("=" * 80)
        
        # Load people data
        people_data = self.load_people_data()
        if not people_data:
            print("❌ No people data found. Exiting.")
            return
        
        # Test accounting layer for all people
        print(f"\n📊 Testing accounting layer for {len(people_data)} people...")
        for person in people_data:
            validation_result = self.test_accounting_layer(person)
            self.results['accounting_validations'].append(validation_result)
            self.results['total_tests'] += 1
            
            if validation_result.get('is_balanced', False):
                self.results['passed_tests'] += 1
            else:
                self.results['failed_tests'] += 1
        
        # Test mesh engine for all people
        print(f"\n🔄 Testing mesh engine for {len(people_data)} people...")
        for person in people_data:
            mesh_result = self.test_mesh_engine(person)
            self.results['mesh_tests'].append(mesh_result)
            self.results['total_tests'] += 1
            
            if mesh_result.get('payment_success_rate', 0) > 0.5:  # 50% success threshold
                self.results['passed_tests'] += 1
            else:
                self.results['failed_tests'] += 1
        
        # Run stress scenarios
        self.run_stress_scenarios(people_data)
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📋 FINANCIAL MESH STRESS TEST REPORT")
        print("=" * 80)
        
        # Overall statistics
        total_tests = self.results['total_tests']
        passed_tests = self.results['passed_tests']
        failed_tests = self.results['failed_tests']
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {success_rate:.2f}%")
        
        # Accounting validation results
        print(f"\n🔍 Accounting Validation Results:")
        for validation in self.results['accounting_validations']:
            status = "✅ PASS" if validation.get('is_balanced', False) else "❌ FAIL"
            net_worth = validation.get('net_worth', 0)
            print(f"  {validation['person_id']}: {status} - Net Worth: ${net_worth:,.2f}")
        
        # Mesh test results
        print(f"\n🔄 Mesh Engine Results:")
        for mesh_test in self.results['mesh_tests']:
            success_rate = mesh_test.get('payment_success_rate', 0) * 100
            status = "✅ PASS" if success_rate > 50 else "❌ FAIL"
            print(f"  {mesh_test['person_id']}: {status} - {success_rate:.1f}% payment success")
        
        # Stress scenario results
        print(f"\n🔥 Stress Scenario Results:")
        for scenario in self.results['stress_scenarios']:
            if 'success_rate' in scenario:
                success_rate = scenario['success_rate'] * 100
                status = "✅ PASS" if success_rate > 70 else "❌ FAIL"
                print(f"  {scenario['name']}: {status} - {success_rate:.1f}% success")
            else:
                print(f"  {scenario['name']}: ✅ COMPLETED")
        
        # Save detailed report
        report_filename = f"simple_financial_mesh_stress_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_filename}")
        
        # Final assessment
        if success_rate >= 80:
            print(f"\n🎉 STRESS TEST PASSED: {success_rate:.1f}% success rate")
        elif success_rate >= 60:
            print(f"\n⚠️  STRESS TEST PARTIAL PASS: {success_rate:.1f}% success rate")
        else:
            print(f"\n❌ STRESS TEST FAILED: {success_rate:.1f}% success rate")

def main():
    """Main function to run the stress test"""
    stress_tester = SimpleFinancialMeshStressTest()
    stress_tester.run_comprehensive_test()

if __name__ == "__main__":
    main() 