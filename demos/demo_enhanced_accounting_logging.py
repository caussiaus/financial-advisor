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
        error_type='insufficient_funds',
        context={
            'transaction_id': 'TXN_004',
            'requested_amount': '5000.00',
            'available_balance': '2800.00',
            'account': 'cash_checking'
        },
        severity='warning'
    )
    
    logger.log_error(
        message="Account not found",
        error_type='account_not_found',
        context={
            'account_id': 'nonexistent_account',
            'operation': 'balance_check'
        },
        severity='error'
    )
    
    # Log reconciliation events
    print("\n4. Logging Reconciliation Events:")
    print("-" * 40)
    
    logger.log_reconciliation(
        reconciliation_type='daily_balance_check',
        accounts_involved=['cash_checking', 'investment_account', 'emergency_fund'],
        total_assets=Decimal('27800.00'),
        total_liabilities=Decimal('0.00'),
        net_worth=Decimal('27800.00'),
        metadata={'reconciliation_date': '2024-01-15'}
    )
    
    logger.log_reconciliation(
        reconciliation_type='monthly_statement_reconciliation',
        accounts_involved=['cash_checking', 'investment_account'],
        total_assets=Decimal('17800.00'),
        total_liabilities=Decimal('0.00'),
        net_worth=Decimal('17800.00'),
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
    
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Flow Items: {stats['flow_items']}")
    print(f"Balance Items: {stats['balance_items']}")
    print(f"Reconciliations: {stats['reconciliations']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total Amount: ${stats['total_amount']}")
    
    print("\nCategories:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count}")
    
    # Export logs
    print("\n2. Exporting Logs:")
    print("-" * 40)
    
    export_file = "enhanced_accounting_logs.json"
    logger.export_logs(export_file)
    print(f"Logs exported to: {export_file}")
    
    # Show summary report
    print("\n3. Summary Report:")
    print("-" * 40)
    
    flow_entries = logger.get_entries_by_type(LogItemType.FLOW_ITEM)
    balance_entries = logger.get_entries_by_type(LogItemType.BALANCE_ITEM)
    error_entries = logger.get_entries_by_type(LogItemType.ERROR)
    reconciliation_entries = logger.get_entries_by_type(LogItemType.RECONCILIATION)
    
    print(f"Flow Items: {len(flow_entries)}")
    print(f"Balance Items: {len(balance_entries)}")
    print(f"Errors: {len(error_entries)}")
    print(f"Reconciliations: {len(reconciliation_entries)}")
    
    # Show recent entries
    print("\n4. Recent Entries:")
    print("-" * 40)
    
    recent_entries = logger.log_entries[-5:]  # Last 5 entries
    for entry in recent_entries:
        if hasattr(entry, 'description'):
            print(f"{entry.timestamp.strftime('%H:%M:%S')} - {entry.description}")
        else:
            print(f"{entry.timestamp.strftime('%H:%M:%S')} - {entry.message}")


def demo_integration_with_reconciliation():
    """Demonstrate integration with accounting reconciliation engine."""
    print("\n" + "=" * 60)
    print("DEMO: Integration with Accounting Reconciliation Engine")
    print("=" * 60)
    
    # Initialize the reconciliation engine
    engine = AccountingReconciliationEngine()
    
    # Add some accounts
    print("\n1. Setting up accounts:")
    print("-" * 40)
    
    # The engine should have default accounts, but let's check
    print(f"Number of accounts: {len(engine.accounts)}")
    
    # Execute some transactions
    print("\n2. Executing transactions:")
    print("-" * 40)
    
    # Set initial balances
    engine.set_account_balance("cash_checking", Decimal('10000.00'))
    engine.set_account_balance("cash_savings", Decimal('5000.00'))
    
    # Execute a payment
    success, message = engine.execute_payment(
        from_account="cash_checking",
        to_account="living_expenses",
        amount=Decimal('1200.00'),
        description="Rent payment",
        reference_id="RENT_2024_01"
    )
    
    print(f"Payment result: {success} - {message}")
    
    # Check balances
    checking_balance = engine.get_account_balance("cash_checking")
    print(f"Checking balance: ${checking_balance}")
    
    # Generate financial statement
    print("\n3. Financial Statement:")
    print("-" * 40)
    
    statement = engine.generate_financial_statement()
    print(f"Total Assets: ${statement['total_assets']}")
    print(f"Total Liabilities: ${statement['total_liabilities']}")
    print(f"Net Worth: ${statement['net_worth']}")
    
    # Export enhanced logs
    print("\n4. Exporting Enhanced Logs:")
    print("-" * 40)
    
    engine.export_enhanced_logs("reconciliation_enhanced_logs.json")
    print("Enhanced logs exported from reconciliation engine")
    
    # Get enhanced log statistics
    stats = engine.get_enhanced_log_statistics()
    print(f"Enhanced log statistics: {stats}")


def main():
    """Main demo function."""
    print("Enhanced Accounting Logging Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Basic logging
        logger = demo_basic_logging()
        
        # Demo 2: Statistics and export
        demo_statistics_and_export(logger)
        
        # Demo 3: Integration with reconciliation engine
        demo_integration_with_reconciliation()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
