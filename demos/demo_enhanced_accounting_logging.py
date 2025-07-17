#!/usr/bin/env python3
"""
Demo script for Enhanced Accounting Logging

This script demonstrates the enhanced accounting logging functionality,
showing how to use flow vs balance item tracking, error logging,
reconciliation logging, and export capabilities.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decimal import Decimal
from datetime import datetime
from enhanced_accounting_logger import (
    EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory, LogItemType
)
from accounting_reconciliation import AccountingReconciliationEngine


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("=" * 60)
    print("DEMO: Basic Enhanced Accounting Logging")
    print("=" * 60)
    
    # Initialize the enhanced logger
    logger = EnhancedAccountingLogger()
    
    # Log some flow items (transactions)
    print("\n1. Logging Flow Items (Transactions):")
    print("-" * 40)
    
    logger.log_flow_item(
        category=FlowItemCategory.INCOME,
        amount=Decimal('5000.00'),
        from_account='employer',
        to_account='cash_checking',
        description='Monthly salary',
        transaction_id='TXN_001',
        reference_id='PAY_2024_01',
        metadata={'pay_period': 'monthly', 'tax_withheld': '750.00'}
    )
    
    logger.log_flow_item(
        category=FlowItemCategory.EXPENSE,
        amount=Decimal('1200.00'),
        from_account='cash_checking',
        to_account='living_expenses',
        description='Rent payment',
        transaction_id='TXN_002',
        reference_id='RENT_2024_01',
        metadata={'property_id': 'APT_123', 'due_date': '2024-01-01'}
    )
    
    logger.log_flow_item(
        category=FlowItemCategory.INVESTMENT,
        amount=Decimal('1000.00'),
        from_account='cash_checking',
        to_account='investment_account',
        description='Monthly investment contribution',
        transaction_id='TXN_003',
        metadata={'investment_type': 'index_fund', 'auto_invest': True}
    )
    
    # Log some balance items (account balances)
    print("\n2. Logging Balance Items (Account Balances):")
    print("-" * 40)
    
    logger.log_balance_item(
        category=BalanceItemCategory.CHECKING,
        account_id='cash_checking',
        balance=Decimal('2800.00'),
        previous_balance=Decimal('5000.00'),
        change_amount=Decimal('-2200.00'),
        metadata={'account_type': 'checking', 'bank': 'Chase Bank'}
    )
    
    logger.log_balance_item(
        category=BalanceItemCategory.INVESTMENT,
        account_id='investment_account',
        balance=Decimal('15000.00'),
        previous_balance=Decimal('14000.00'),
        change_amount=Decimal('1000.00'),
        metadata={'account_type': 'brokerage', 'platform': 'Vanguard'}
    )
    
    logger.log_balance_item(
        category=BalanceItemCategory.SAVINGS,
        account_id='emergency_fund',
        balance=Decimal('10000.00'),
        metadata={'account_type': 'savings', 'purpose': 'emergency_fund'}
    )
    
    # Log some errors
    print("\n3. Logging Error Items:")
    print("-" * 40)
    
    logger.log_error(
        message="Insufficient funds for transaction",
        context={
            'transaction_id': 'TXN_004',
            'requested_amount': '5000.00',
            'available_balance': '2800.00',
            'account': 'cash_checking'
        },
        error_type='insufficient_funds',
        severity='warning'
    )
    
    logger.log_error(
        message="Account not found",
        context={
            'account_id': 'nonexistent_account',
            'operation': 'balance_check'
        },
        error_type='account_not_found',
        severity='error'
    )
    
    # Log reconciliation events
    print("\n4. Logging Reconciliation Events:")
    print("-" * 40)
    
    logger.log_reconciliation(
        reconciliation_type='daily_balance_check',
        status='completed',
        details={
            'accounts_checked': 5,
            'discrepancies_found': 0,
            'total_balance': '53000.00'
        },
        metadata={'reconciliation_date': '2024-01-15'}
    )
    
    logger.log_reconciliation(
        reconciliation_type='monthly_statement_reconciliation',
        status='completed',
        details={
            'statement_period': '2024-01',
            'transactions_reconciled': 45,
            'unreconciled_items': 2
        },
        metadata={'statement_source': 'bank_statement'}
    )
    
    return logger


def demo_statistics_and_export(logger):
    """Demonstrate statistics and export functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Statistics and Export Functionality")
    print("=" * 60)
    
    # Get and display statistics
    print("\n1. Logging Statistics:")
    print("-" * 40)
    stats = logger.get_statistics()
    
    print(f"Total Flow Items: {stats['total_flow_items']}")
    print(f"Total Balance Items: {stats['total_balance_items']}")
    print(f"Total Error Items: {stats['total_error_items']}")
    print(f"Total Reconciliation Items: {stats['total_reconciliation_items']}")
    print(f"Total Flow Amount: ${stats['total_flow_amount']}")
    print(f"Total Balance Amount: ${stats['total_balance_amount']}")
    
    print("\nFlow Categories:")
    for category, count in stats['flow_categories'].items():
        print(f"  {category}: {count}")
    
    print("\nBalance Categories:")
    for category, count in stats['balance_categories'].items():
        print(f"  {category}: {count}")
    
    print("\nError Types:")
    for error_type, count in stats['error_types'].items():
        print(f"  {error_type}: {count}")
    
    # Export logs
    print("\n2. Exporting Logs:")
    print("-" * 40)
    
    # Create exports directory if it doesn't exist
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    # Export to JSON
    json_file = exports_dir / "enhanced_accounting_logs.json"
    logger.export_logs(str(json_file), format="json")
    print(f"✅ Logs exported to JSON: {json_file}")
    
    # Export to CSV
    csv_file = exports_dir / "enhanced_accounting_logs.csv"
    logger.export_logs(str(csv_file), format="csv")
    print(f"✅ Logs exported to CSV: {csv_file}")
    
    # Show recent items
    print("\n3. Recent Items:")
    print("-" * 40)
    
    print("Recent Flow Items:")
    recent_flows = logger.get_recent_items(LogItemType.FLOW, limit=3)
    for item in recent_flows:
        print(f"  {item.category.value}: ${item.amount} ({item.from_account} -> {item.to_account})")
    
    print("\nRecent Balance Items:")
    recent_balances = logger.get_recent_items(LogItemType.BALANCE, limit=3)
    for item in recent_balances:
        print(f"  {item.category.value}: {item.account_id} = ${item.balance}")
    
    print("\nRecent Error Items:")
    recent_errors = logger.get_recent_items(LogItemType.ERROR, limit=2)
    for item in recent_errors:
        print(f"  {item.error_type}: {item.message}")


def demo_accounting_integration():
    """Demonstrate integration with AccountingReconciliationEngine."""
    print("\n" + "=" * 60)
    print("DEMO: Accounting Reconciliation Engine Integration")
    print("=" * 60)
    
    # Initialize the accounting engine (which now has enhanced logging)
    accounting = AccountingReconciliationEngine()
    
    print("\n1. Setting up accounts with enhanced logging:")
    print("-" * 40)
    
    # Set account balances
    accounting.set_account_balance('cash_checking', Decimal('10000.00'))
    accounting.set_account_balance('living_expenses', Decimal('0.00'))
    accounting.set_account_balance('investment_account', Decimal('5000.00'))
    
    print("✅ Accounts initialized with enhanced logging")
    
    print("\n2. Executing transactions with enhanced logging:")
    print("-" * 40)
    
    # Execute some payments
    success, result = accounting.execute_payment(
        'cash_checking', 'living_expenses', Decimal('1500.00'), 'Rent payment'
    )
    print(f"Rent payment: {success} - {result}")
    
    success, result = accounting.execute_payment(
        'cash_checking', 'investment_account', Decimal('1000.00'), 'Investment contribution'
    )
    print(f"Investment contribution: {success} - {result}")
    
    # Try an invalid transaction
    success, result = accounting.execute_payment(
        'cash_checking', 'living_expenses', Decimal('20000.00'), 'Invalid large payment'
    )
    print(f"Invalid payment: {success} - {result}")
    
    print("\n3. Reconciliation summary with enhanced logging:")
    print("-" * 40)
    
    # Generate reconciliation summary
    summary = accounting.generate_reconciliation_summary()
    print("Reconciliation Summary:")
    for account, balance in summary.items():
        print(f"  {account}: ${balance}")
    
    print("\n4. Exporting enhanced logs from accounting engine:")
    print("-" * 40)
    
    # Export logs from the accounting engine
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    accounting_log_file = exports_dir / "accounting_engine_logs.json"
    accounting.export_enhanced_logs(str(accounting_log_file))
    print(f"✅ Accounting engine logs exported to: {accounting_log_file}")


def demo_advanced_features():
    """Demonstrate advanced logging features."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Logging Features")
    print("=" * 60)
    
    # Create a logger with file output
    log_file = "exports/advanced_logging_demo.log"
    logger = EnhancedAccountingLogger(log_file=log_file)
    
    print(f"\n1. File-based logging to: {log_file}")
    print("-" * 40)
    
    # Log complex transactions with rich metadata
    logger.log_flow_item(
        category=FlowItemCategory.INVESTMENT,
        amount=Decimal('2500.00'),
        from_account='cash_checking',
        to_account='retirement_401k',
        description='401(k) contribution',
        transaction_id='TXN_401K_001',
        reference_id='401K_2024_01',
        metadata={
            'contribution_type': 'traditional_401k',
            'employer_match': '1250.00',
            'tax_year': '2024',
            'vesting_schedule': 'immediate',
            'investment_funds': ['VTSAX', 'VTIAX', 'VBTLX']
        }
    )
    
    logger.log_flow_item(
        category=FlowItemCategory.LOAN_PAYMENT,
        amount=Decimal('800.00'),
        from_account='cash_checking',
        to_account='mortgage_loan',
        description='Mortgage payment',
        transaction_id='TXN_MORT_001',
        reference_id='MORT_2024_01',
        metadata={
            'principal': '600.00',
            'interest': '200.00',
            'escrow': '0.00',
            'loan_type': 'fixed_rate',
            'interest_rate': '3.25%',
            'remaining_balance': '180000.00'
        }
    )
    
    # Log balance items with complex scenarios
    logger.log_balance_item(
        category=BalanceItemCategory.RETIREMENT,
        account_id='retirement_401k',
        balance=Decimal('75000.00'),
        previous_balance=Decimal('72500.00'),
        change_amount=Decimal('2500.00'),
        metadata={
            'account_type': '401k',
            'employer': 'Tech Corp',
            'investment_mix': {
                'stocks': '70%',
                'bonds': '20%',
                'international': '10%'
            },
            'fees': '0.15%'
        }
    )
    
    # Log complex reconciliation
    logger.log_reconciliation(
        reconciliation_type='portfolio_rebalancing',
        status='completed',
        details={
            'rebalancing_date': '2024-01-15',
            'target_allocation': {
                'stocks': '60%',
                'bonds': '30%',
                'cash': '10%'
            },
            'current_allocation': {
                'stocks': '65%',
                'bonds': '25%',
                'cash': '10%'
            },
            'trades_executed': 3,
            'total_trade_value': '5000.00'
        },
        metadata={
            'rebalancing_frequency': 'quarterly',
            'threshold': '5%',
            'tax_considerations': 'tax_loss_harvesting_applied'
        }
    )
    
    print("✅ Advanced logging completed")
    print(f"📄 Check the log file: {log_file}")
    
    return logger


def main():
    """Run the complete enhanced accounting logging demo."""
    print("🚀 Enhanced Accounting Logging Demo")
    print("=" * 60)
    print("This demo showcases the enhanced accounting logging system")
    print("with flow vs balance item tracking, error logging, and export capabilities.")
    print()
    
    try:
        # Run all demos
        logger = demo_basic_logging()
        demo_statistics_and_export(logger)
        demo_accounting_integration()
        demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced Accounting Logging Demo Completed Successfully!")
        print("=" * 60)
        print("\n📋 Summary of what was demonstrated:")
        print("  ✅ Flow item logging (transactions, cash flows)")
        print("  ✅ Balance item logging (account balances, positions)")
        print("  ✅ Error logging with detailed context")
        print("  ✅ Reconciliation logging")
        print("  ✅ Statistics and reporting")
        print("  ✅ Export capabilities (JSON and CSV)")
        print("  ✅ Integration with AccountingReconciliationEngine")
        print("  ✅ File-based logging")
        print("  ✅ Advanced metadata support")
        
        print("\n📁 Generated files:")
        print("  - exports/enhanced_accounting_logs.json")
        print("  - exports/enhanced_accounting_logs.csv")
        print("  - exports/accounting_engine_logs.json")
        print("  - exports/advanced_logging_demo.log")
        
        print("\n🧪 To test the integration manually:")
        print("  python -c \"from src.accounting_reconciliation import AccountingReconciliationEngine; print('✅ Import successful')\"")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check that all required modules are available.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 