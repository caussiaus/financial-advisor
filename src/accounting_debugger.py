"""
Accounting Debugger and Sense-Checker

This module provides comprehensive debugging and validation tools for the accounting system,
ensuring that all cash flows and states make sense at any given point in time.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
from collections import defaultdict

from .accounting_reconciliation import AccountingReconciliationEngine, Account, AccountType
from .unified_cash_flow_model import UnifiedCashFlowModel, CashFlowState, CashFlowEvent

@dataclass
class AccountingValidationResult:
    """Result of accounting validation checks"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    balance_sheet_check: bool
    cash_flow_check: bool
    double_entry_check: bool
    constraint_check: bool
    stress_test_passed: bool

@dataclass
class AccountingMetrics:
    """Key accounting metrics for monitoring"""
    total_assets: Decimal
    total_liabilities: Decimal
    net_worth: Decimal
    liquidity_ratio: float
    debt_to_asset_ratio: float
    cash_flow_coverage: float
    stress_level: float
    account_balances: Dict[str, Decimal]

class AccountingDebugger:
    """
    Comprehensive debugging and validation system for accounting operations
    """
    
    def __init__(self, accounting_engine: AccountingReconciliationEngine, 
                 cash_flow_model: UnifiedCashFlowModel):
        self.accounting_engine = accounting_engine
        self.cash_flow_model = cash_flow_model
        self.validation_history = []
        self.error_log = []
        self.warning_log = []
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Validation thresholds
        self.thresholds = {
            'min_liquidity_ratio': 0.05,  # 5% minimum liquidity
            'max_debt_ratio': 0.6,        # 60% maximum debt ratio
            'min_cash_flow_coverage': 1.2, # 120% cash flow coverage
            'max_stress_level': 0.7,      # 70% maximum stress level
            'min_account_balance': 100,    # $100 minimum account balance
            'max_transaction_amount': 1000000  # $1M maximum single transaction
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debugging operations"""
        logger = logging.getLogger('accounting_debugger')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_accounting_state(self, timestamp: datetime = None) -> AccountingValidationResult:
        """Comprehensive validation of accounting state"""
        if timestamp is None:
            timestamp = datetime.now()
        
        errors = []
        warnings = []
        
        # 1. Balance Sheet Validation
        balance_sheet_valid = self._validate_balance_sheet(errors, warnings)
        
        # 2. Cash Flow Validation
        cash_flow_valid = self._validate_cash_flows(errors, warnings)
        
        # 3. Double-Entry Validation
        double_entry_valid = self._validate_double_entry_bookkeeping(errors, warnings)
        
        # 4. Constraint Validation
        constraint_valid = self._validate_constraints(errors, warnings)
        
        # 5. Stress Test
        stress_test_passed = self._run_stress_tests(errors, warnings)
        
        # Overall validation
        is_valid = (balance_sheet_valid and cash_flow_valid and 
                   double_entry_valid and constraint_valid and stress_test_passed)
        
        result = AccountingValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            balance_sheet_check=balance_sheet_valid,
            cash_flow_check=cash_flow_valid,
            double_entry_check=double_entry_valid,
            constraint_check=constraint_valid,
            stress_test_passed=stress_test_passed
        )
        
        self.validation_history.append((timestamp, result))
        
        if errors:
            self.logger.error(f"Accounting validation failed with {len(errors)} errors")
        if warnings:
            self.logger.warning(f"Accounting validation has {len(warnings)} warnings")
        
        return result
    
    def _validate_balance_sheet(self, errors: List[str], warnings: List[str]) -> bool:
        """Validate balance sheet integrity"""
        try:
            # Get all account balances
            accounts = self.accounting_engine.accounts
            total_assets = Decimal('0')
            total_liabilities = Decimal('0')
            total_equity = Decimal('0')
            total_income = Decimal('0')
            total_expenses = Decimal('0')
            
            for account_id, account in accounts.items():
                balance = account.balance
                
                if account.account_type == AccountType.ASSET:
                    total_assets += balance
                elif account.account_type == AccountType.LIABILITY:
                    total_liabilities += balance
                elif account.account_type == AccountType.EQUITY:
                    total_equity += balance
                elif account.account_type == AccountType.INCOME:
                    total_income += balance
                elif account.account_type == AccountType.EXPENSE:
                    total_expenses += balance
            
            # Check basic accounting equation: Assets = Liabilities + Equity
            calculated_equity = total_assets - total_liabilities
            equity_difference = abs(calculated_equity - total_equity)
            
            if equity_difference > Decimal('1'):  # Allow for rounding errors
                errors.append(f"Balance sheet equation violated: Assets ({total_assets}) != Liabilities ({total_liabilities}) + Equity ({total_equity})")
                return False
            
            # Check for negative asset balances
            for account_id, account in accounts.items():
                if account.account_type == AccountType.ASSET and account.balance < 0:
                    errors.append(f"Negative balance in asset account {account_id}: {account.balance}")
                    return False
            
            # Check for reasonable account balances
            for account_id, account in accounts.items():
                if abs(account.balance) > Decimal(str(self.thresholds['max_transaction_amount'])):
                    warnings.append(f"Large balance in account {account_id}: {account.balance}")
            
            return True
            
        except Exception as e:
            errors.append(f"Balance sheet validation error: {str(e)}")
            return False
    
    def _validate_cash_flows(self, errors: List[str], warnings: List[str]) -> bool:
        """Validate cash flow consistency"""
        try:
            # Get cash flow events
            cash_flow_events = self.cash_flow_model.cash_flow_events
            
            # Define cash yardstick accounts
            cash_accounts = {'cash_checking', 'cash_savings'}
            
            # Check for circular cash flows
            cash_flow_graph = defaultdict(list)
            for event in cash_flow_events:
                cash_flow_graph[event.source_account].append(event.target_account)
            
            # Detect cycles (improved: allow cash yardstick cycles)
            visited = set()
            for account in cash_flow_graph:
                if account not in visited:
                    cycle = []
                    if self._has_cycle_with_path(cash_flow_graph, account, visited, set(), cycle):
                        # Only flag as error if the cycle is not a simple cash yardstick pattern
                        if not self._is_cash_yardstick_cycle(cycle, cash_accounts):
                            errors.append(f"Circular cash flow detected involving account {account}: {cycle}")
                            return False
                        else:
                            warnings.append(f"Allowed cash yardstick cycle detected: {cycle}")
            
            # Check for reasonable cash flow amounts
            for event in cash_flow_events:
                if abs(event.amount) > self.thresholds['max_transaction_amount']:
                    warnings.append(f"Large cash flow amount in event {event.event_id}: {event.amount}")
                
                if event.amount == 0 and event.event_type != 'transfer':
                    warnings.append(f"Zero amount cash flow event: {event.event_id}")
            
            return True
            
        except Exception as e:
            errors.append(f"Cash flow validation error: {str(e)}")
            return False

    def _has_cycle_with_path(self, graph: Dict, node: str, visited: set, rec_stack: set, path: list) -> bool:
        """Check for cycles in cash flow graph and record the path"""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if self._has_cycle_with_path(graph, neighbor, visited, rec_stack, path):
                    return True
            elif neighbor in rec_stack:
                path.append(neighbor)
                return True
        
        rec_stack.remove(node)
        path.pop()
        return False

    def _is_cash_yardstick_cycle(self, cycle: list, cash_accounts: set) -> bool:
        """Return True if the cycle is a simple cash yardstick pattern"""
        # Remove duplicate at end if present
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        # Allow cycles that only involve cash accounts and direct expense/income accounts
        for acc in cycle:
            if acc not in cash_accounts and not acc.endswith('_expenses') and not acc.endswith('_income') and not acc.startswith('milestone_'):
                return False
        return True
    
    def _validate_double_entry_bookkeeping(self, errors: List[str], warnings: List[str]) -> bool:
        """Validate double-entry bookkeeping principles"""
        try:
            # Check that all transactions have equal debits and credits
            transactions = self.accounting_engine.transactions
            
            for transaction in transactions:
                total_debits = sum(entry.amount for entry in transaction.entries if entry.entry_type == 'debit')
                total_credits = sum(entry.amount for entry in transaction.entries if entry.entry_type == 'credit')
                
                if abs(total_debits - total_credits) > Decimal('0.01'):  # Allow for rounding
                    errors.append(f"Transaction {transaction.transaction_id} violates double-entry: debits={total_debits}, credits={total_credits}")
                    return False
            
            return True
            
        except Exception as e:
            errors.append(f"Double-entry validation error: {str(e)}")
            return False
    
    def _validate_constraints(self, errors: List[str], warnings: List[str]) -> bool:
        """Validate payment constraints"""
        try:
            constraints = self.accounting_engine.constraints
            
            for account_id, constraint in constraints.items():
                account = self.accounting_engine.accounts.get(account_id)
                if not account:
                    continue
                
                balance = account.balance
                
                # Check minimum balance constraint
                if constraint.min_balance and balance < constraint.min_balance:
                    errors.append(f"Account {account_id} below minimum balance: {balance} < {constraint.min_balance}")
                    return False
                
                # Check maximum single payment constraint
                if constraint.max_single_payment:
                    # This would need transaction history to fully validate
                    pass
                
                # Check daily/monthly limits (would need transaction history)
                pass
            
            return True
            
        except Exception as e:
            errors.append(f"Constraint validation error: {str(e)}")
            return False
    
    def _run_stress_tests(self, errors: List[str], warnings: List[str]) -> bool:
        """Run stress tests on the accounting system"""
        try:
            # Get current metrics
            metrics = self.calculate_accounting_metrics()
            
            # Test 1: Liquidity stress
            if metrics.liquidity_ratio < self.thresholds['min_liquidity_ratio']:
                errors.append(f"Insufficient liquidity: {metrics.liquidity_ratio:.2%} < {self.thresholds['min_liquidity_ratio']:.2%}")
                return False
            
            # Test 2: Debt ratio stress
            if metrics.debt_to_asset_ratio > self.thresholds['max_debt_ratio']:
                errors.append(f"Excessive debt ratio: {metrics.debt_to_asset_ratio:.2%} > {self.thresholds['max_debt_ratio']:.2%}")
                return False
            
            # Test 3: Cash flow coverage stress
            if metrics.cash_flow_coverage < self.thresholds['min_cash_flow_coverage']:
                warnings.append(f"Low cash flow coverage: {metrics.cash_flow_coverage:.2f} < {self.thresholds['min_cash_flow_coverage']:.2f}")
            
            # Test 4: Overall stress level
            if metrics.stress_level > self.thresholds['max_stress_level']:
                errors.append(f"High stress level: {metrics.stress_level:.2%} > {self.thresholds['max_stress_level']:.2%}")
                return False
            
            return True
            
        except Exception as e:
            errors.append(f"Stress test error: {str(e)}")
            return False
    
    def calculate_accounting_metrics(self) -> AccountingMetrics:
        """Calculate key accounting metrics"""
        accounts = self.accounting_engine.accounts
        
        # Calculate totals by account type
        total_assets = sum(acc.balance for acc in accounts.values() if acc.account_type == AccountType.ASSET)
        total_liabilities = sum(acc.balance for acc in accounts.values() if acc.account_type == AccountType.LIABILITY)
        total_income = sum(acc.balance for acc in accounts.values() if acc.account_type == AccountType.INCOME)
        total_expenses = sum(acc.balance for acc in accounts.values() if acc.account_type == AccountType.EXPENSE)
        
        # Calculate derived metrics
        net_worth = total_assets - total_liabilities
        
        # Liquidity ratio (cash / total assets)
        cash_accounts = ['cash_checking', 'cash_savings']
        cash_balance = sum(accounts.get(acc_id, Account('', '', AccountType.ASSET, Decimal('0'))).balance 
                         for acc_id in cash_accounts if accounts.get(acc_id) is not None)
        liquidity_ratio = float(cash_balance / total_assets) if total_assets > 0 else 0.0
        
        # Debt to asset ratio
        debt_to_asset_ratio = float(total_liabilities / total_assets) if total_assets > 0 else 0.0
        
        # Cash flow coverage (income / expenses)
        cash_flow_coverage = float(total_income / total_expenses) if total_expenses > 0 else float('inf')
        
        # Stress level (composite metric)
        debt_stress = min(debt_to_asset_ratio, 1.0)
        liquidity_stress = 1.0 - min(liquidity_ratio, 1.0)
        cash_flow_stress = 1.0 if cash_flow_coverage < 1.0 else 0.0
        stress_level = (debt_stress + liquidity_stress + cash_flow_stress) / 3.0
        
        # Account balances
        account_balances = {acc_id: acc.balance for acc_id, acc in accounts.items()}
        
        return AccountingMetrics(
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            net_worth=net_worth,
            liquidity_ratio=liquidity_ratio,
            debt_to_asset_ratio=debt_to_asset_ratio,
            cash_flow_coverage=cash_flow_coverage,
            stress_level=stress_level,
            account_balances=account_balances
        )
    
    def monitor_cash_flow_consistency(self, time_period_days: int = 30) -> Dict:
        """Monitor cash flow consistency over time"""
        print(f"🔍 Monitoring cash flow consistency over {time_period_days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        # Simulate cash flows
        states = self.cash_flow_model.simulate_cash_flows_over_time(
            start_date, end_date, time_step_days=1
        )
        
        # Analyze consistency
        consistency_report = {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_days': time_period_days
            },
            'cash_flow_analysis': {
                'total_states': len(states),
                'positive_cash_flow_days': len([s for s in states if s.net_cash_flow > 0]),
                'negative_cash_flow_days': len([s for s in states if s.net_cash_flow < 0]),
                'average_daily_cash_flow': float(sum(s.net_cash_flow for s in states) / len(states)),
                'cash_flow_volatility': float(np.std([float(s.net_cash_flow) for s in states]))
            },
            'account_consistency': {},
            'stress_analysis': {
                'average_stress_level': float(sum(s.stress_level for s in states) / len(states)),
                'peak_stress_level': max(s.stress_level for s in states),
                'stress_volatility': float(np.std([s.stress_level for s in states]))
            }
        }
        
        # Check account consistency
        for account_id in self.cash_flow_model.case_accounts['assets']:
            balances = [float(s.account_balances.get(account_id, 0)) for s in states]
            consistency_report['account_consistency'][account_id] = {
                'min_balance': min(balances),
                'max_balance': max(balances),
                'average_balance': np.mean(balances),
                'balance_volatility': np.std(balances),
                'negative_balance_days': len([b for b in balances if b < 0])
            }
        
        return consistency_report
    
    def generate_accounting_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive accounting report"""
        print("📊 Generating comprehensive accounting report...")
        
        # Get current validation result
        validation_result = self.validate_accounting_state()
        
        # Calculate metrics
        metrics = self.calculate_accounting_metrics()
        
        # Monitor consistency
        consistency_report = self.monitor_cash_flow_consistency()
        
        # Compile report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'overall_valid': validation_result.is_valid,
                'total_errors': len(validation_result.errors),
                'total_warnings': len(validation_result.warnings),
                'checks_passed': sum([
                    validation_result.balance_sheet_check,
                    validation_result.cash_flow_check,
                    validation_result.double_entry_check,
                    validation_result.constraint_check,
                    validation_result.stress_test_passed
                ]),
                'checks_total': 5
            },
            'accounting_metrics': {
                'total_assets': float(metrics.total_assets),
                'total_liabilities': float(metrics.total_liabilities),
                'net_worth': float(metrics.net_worth),
                'liquidity_ratio': metrics.liquidity_ratio,
                'debt_to_asset_ratio': metrics.debt_to_asset_ratio,
                'cash_flow_coverage': metrics.cash_flow_coverage,
                'stress_level': metrics.stress_level
            },
            'account_balances': {k: float(v) for k, v in metrics.account_balances.items()},
            'validation_details': {
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            },
            'consistency_report': consistency_report,
            'recommendations': self._generate_recommendations(validation_result, metrics)
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Accounting report saved to {output_file}")
        
        return report
    
    def _generate_recommendations(self, validation_result: AccountingValidationResult, 
                                metrics: AccountingMetrics) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Balance sheet recommendations
        if not validation_result.balance_sheet_check:
            recommendations.append("Review and reconcile balance sheet entries")
        
        # Cash flow recommendations
        if not validation_result.cash_flow_check:
            recommendations.append("Review cash flow event configurations")
        
        # Constraint recommendations
        if not validation_result.constraint_check:
            recommendations.append("Review payment constraints and limits")
        
        # Stress test recommendations
        if not validation_result.stress_test_passed:
            if metrics.liquidity_ratio < self.thresholds['min_liquidity_ratio']:
                recommendations.append("Increase cash reserves to improve liquidity")
            
            if metrics.debt_to_asset_ratio > self.thresholds['max_debt_ratio']:
                recommendations.append("Reduce debt levels to improve financial stability")
            
            if metrics.stress_level > self.thresholds['max_stress_level']:
                recommendations.append("Implement stress reduction strategies")
        
        # General recommendations
        if metrics.cash_flow_coverage < self.thresholds['min_cash_flow_coverage']:
            recommendations.append("Improve cash flow coverage through income increase or expense reduction")
        
        if not recommendations:
            recommendations.append("Accounting system is healthy - continue monitoring")
        
        return recommendations


def demo_accounting_debugger():
    """Demonstrate the accounting debugger"""
    print("🔍 Accounting Debugger Demo")
    print("=" * 50)
    
    # Initialize cash flow model
    initial_state = {
        'total_wealth': 764560.97,
        'cash': 764560.97 * 0.0892,
        'investments': 764560.97 * 0.9554,
        'income': 150000,
        'expenses': 60000
    }
    
    cash_flow_model = UnifiedCashFlowModel(initial_state)
    
    # Add some events
    case_events = cash_flow_model.create_case_events_from_analysis()
    for event in case_events:
        cash_flow_model.add_cash_flow_event(event)
    
    # Create debugger
    debugger = AccountingDebugger(cash_flow_model.accounting_engine, cash_flow_model)
    
    # Run validation
    print("🔍 Running accounting validation...")
    validation_result = debugger.validate_accounting_state()
    
    print(f"Validation Result: {'✅ PASSED' if validation_result.is_valid else '❌ FAILED'}")
    print(f"Errors: {len(validation_result.errors)}")
    print(f"Warnings: {len(validation_result.warnings)}")
    
    if validation_result.errors:
        print("\nErrors:")
        for error in validation_result.errors:
            print(f"  ❌ {error}")
    
    if validation_result.warnings:
        print("\nWarnings:")
        for warning in validation_result.warnings:
            print(f"  ⚠️ {warning}")
    
    # Calculate metrics
    metrics = debugger.calculate_accounting_metrics()
    print(f"\n📊 Accounting Metrics:")
    print(f"  Total Assets: ${metrics.total_assets:,.2f}")
    print(f"  Total Liabilities: ${metrics.total_liabilities:,.2f}")
    print(f"  Net Worth: ${metrics.net_worth:,.2f}")
    print(f"  Liquidity Ratio: {metrics.liquidity_ratio:.2%}")
    print(f"  Debt to Asset Ratio: {metrics.debt_to_asset_ratio:.2%}")
    print(f"  Cash Flow Coverage: {metrics.cash_flow_coverage:.2f}")
    print(f"  Stress Level: {metrics.stress_level:.2%}")
    
    # Generate report
    report = debugger.generate_accounting_report("accounting_debug_report.json")
    
    print("\n✅ Accounting debugger demo complete!")
    return debugger, validation_result, metrics


if __name__ == "__main__":
    demo_accounting_debugger() 