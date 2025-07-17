"""
Demo: Enhanced Accounting Logging

This demo showcases the improved logging system that distinguishes between:
- Flow items: Transactions, cash flows, payments
- Balance items: Account balances, positions, net worth

The enhanced logging provides structured output with clear categorization
and detailed tracking for better financial analysis and debugging.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.accounting_reconciliation import AccountingReconciliationEngine
from src.enhanced_accounting_logger import (
    EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory
)
from decimal import Decimal
from datetime import datetime, timedelta
import json


def demo_basic_logging():
    """Demo basic enhanced logging functionality"""
    print("🔍 DEMO: Enhanced Accounting Logging")
    print("=" * 60)
    
    # Initialize accounting engine with enhanced logging
    accounting = AccountingReconciliationEngine()
    
    # Set initial balances
    print("\n💰 Setting initial balances...")
    accounting.set_account_balance("cash_checking", Decimal('50000'))
    accounting.set_account_balance("cash_savings", Decimal('100000'))
    accounting.set_account_balance("investments_stocks", Decimal('250000'))
    accounting.set_account_balance("salary", Decimal('0'))  # Income account starts at 0
    
    # Log initial balance items
    print("\n📊 Logging initial balance items...")
    for account_id, account in accounting.accounts.items():
        if account.balance > 0:
            accounting.enhanced_logger.log_balance_item(
                category=accounting._determine_balance_category(account.account_type),
                account_id=account_id,
                balance=account.balance,
                metadata={'initial_setup': True}
            )
    
    return accounting


def demo_flow_items():
    """Demo flow item logging (transactions, cash flows)"""
    print("\n💸 DEMO: Flow Items (Transactions & Cash Flows)")
    print("-" * 50)
    
    accounting = demo_basic_logging()
    
    # Simulate various types of transactions
    transactions = [
        {
            'description': 'Salary payment',
            'from_account': 'salary',
            'to_account': 'cash_checking',
            'amount': Decimal('5000'),
            'type': 'income'
        },
        {
            'description': 'Investment contribution',
            'from_account': 'cash_checking',
            'to_account': 'investments_stocks',
            'amount': Decimal('2000'),
            'type': 'investment'
        },
        {
            'description': 'Monthly rent payment',
            'from_account': 'cash_checking',
            'to_account': 'living_expenses',
            'amount': Decimal('1500'),
            'type': 'payment'
        },
        {
            'description': 'Emergency fund transfer',
            'from_account': 'cash_checking',
            'to_account': 'cash_savings',
            'amount': Decimal('1000'),
            'type': 'transfer'
        },
        {
            'description': 'Investment income',
            'from_account': 'investments_stocks',
            'to_account': 'cash_checking',
            'amount': Decimal('500'),
            'type': 'income'
        }
    ]
    
    print("🔄 Executing transactions with enhanced flow logging...")
    for i, txn in enumerate(transactions, 1):
        print(f"\nTransaction {i}: {txn['description']}")
        
        success, result = accounting.execute_payment(
            from_account=txn['from_account'],
            to_account=txn['to_account'],
            amount=txn['amount'],
            description=txn['description'],
            reference_id=f"demo_txn_{i}"
        )
        
        if success:
            print(f"✅ Transaction executed: {result}")
        else:
            print(f"❌ Transaction failed: {result}")
    
    return accounting


def demo_balance_tracking():
    """Demo balance item tracking with change detection"""
    print("\n📈 DEMO: Balance Item Tracking")
    print("-" * 50)
    
    accounting = demo_flow_items()
    
    # Log current balances after transactions
    print("\n📊 Current account balances:")
    for account_id, account in accounting.accounts.items():
        if account.balance != 0:
            print(f"  {account_id}: ${account.balance:,.2f}")
    
    # Generate reconciliation summary
    print("\n🔍 Generating reconciliation summary...")
    accounting.log_reconciliation_summary()
    
    return accounting


def demo_log_statistics():
    """Demo logging statistics and reporting"""
    print("\n📊 DEMO: Log Statistics & Reporting")
    print("-" * 50)
    
    accounting = demo_balance_tracking()
    
    # Get enhanced log statistics
    stats = accounting.get_enhanced_log_statistics()
    print("\n📈 Enhanced Logging Statistics:")
    print(f"  Total logs: {stats['total_logs']}")
    print(f"  Flow items: {stats['flow_items']}")
    print(f"  Balance items: {stats['balance_items']}")
    print(f"  Reconciliation items: {stats['reconciliation_items']}")
    print(f"  Errors: {stats['errors']}")
    
    # Generate summary report
    summary = accounting.enhanced_logger.generate_summary_report()
    print("\n📋 Summary Report:")
    print(f"  Flow Summary: {len(summary['flow_summary'])} categories")
    print(f"  Balance Summary: {len(summary['balance_summary'])} categories")
    
    # Show flow summary details
    print("\n💸 Flow Item Summary:")
    for category, data in summary['flow_summary'].items():
        print(f"  {category}: {data['count']} items, ${data['total_amount']} total")
    
    # Show balance summary details
    print("\n💰 Balance Item Summary:")
    for category, data in summary['balance_summary'].items():
        print(f"  {category}: {data['count']} items, ${data['total_balance']} total")
    
    return accounting


def demo_log_export():
    """Demo log export functionality"""
    print("\n📤 DEMO: Log Export")
    print("-" * 50)
    
    accounting = demo_log_statistics()
    
    # Export logs to different formats
    export_files = [
        "data/outputs/enhanced_logs_flow.json",
        "data/outputs/enhanced_logs_balance.json",
        "data/outputs/enhanced_logs_all.json"
    ]
    
    print("\n📁 Exporting logs to files...")
    for filepath in export_files:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        accounting.export_enhanced_logs(filepath)
        print(f"  ✅ Exported to: {filepath}")
    
    # Show sample of exported data
    print("\n📄 Sample exported log structure:")
    with open("data/outputs/enhanced_logs_all.json", 'r') as f:
        data = json.load(f)
        print(f"  Export timestamp: {data['export_timestamp']}")
        print(f"  Flow items: {len(data['logs']['flow_items'])}")
        print(f"  Balance items: {len(data['logs']['balance_items'])}")
        print(f"  Reconciliation items: {len(data['logs']['reconciliation_items'])}")
    
    return accounting


def demo_error_logging():
    """Demo error logging with context"""
    print("\n🚨 DEMO: Error Logging")
    print("-" * 50)
    
    accounting = demo_log_export()
    
    # Simulate some error scenarios
    print("\n⚠️ Testing error logging scenarios...")
    
    # Try to execute payment with insufficient funds
    print("\n1. Testing insufficient funds scenario:")
    success, result = accounting.execute_payment(
        from_account="cash_checking",
        to_account="living_expenses",
        amount=Decimal('100000'),  # More than available
        description="Large payment test",
        reference_id="error_test_1"
    )
    
    if not success:
        print(f"  ❌ Expected failure: {result}")
    
    # Try to execute payment from non-existent account
    print("\n2. Testing non-existent account scenario:")
    success, result = accounting.execute_payment(
        from_account="non_existent_account",
        to_account="living_expenses",
        amount=Decimal('1000'),
        description="Invalid account test",
        reference_id="error_test_2"
    )
    
    if not success:
        print(f"  ❌ Expected failure: {result}")
    
    # Show error statistics
    stats = accounting.get_enhanced_log_statistics()
    print(f"\n📊 Final error count: {stats['errors']}")
    
    return accounting


def demo_advanced_features():
    """Demo advanced logging features"""
    print("\n🚀 DEMO: Advanced Logging Features")
    print("-" * 50)
    
    accounting = demo_error_logging()
    
    # Demo confidence levels for flow items
    print("\n🎯 Testing confidence levels for flow items:")
    
    # High confidence transaction
    accounting.enhanced_logger.log_flow_item(
        category=FlowItemCategory.INCOME,
        amount=Decimal('5000'),
        from_account='salary',
        to_account='cash_checking',
        description='Regular salary payment',
        confidence=0.95,
        metadata={'source': 'payroll_system', 'verified': True}
    )
    
    # Low confidence transaction (estimated)
    accounting.enhanced_logger.log_flow_item(
        category=FlowItemCategory.EXPENSE,
        amount=Decimal('150'),
        from_account='cash_checking',
        to_account='living_expenses',
        description='Estimated utility payment',
        confidence=0.6,
        metadata={'source': 'estimation', 'verified': False}
    )
    
    # Demo balance tracking with change detection
    print("\n📈 Testing balance change tracking:")
    
    # Log balance with change
    accounting.enhanced_logger.log_balance_item(
        category=BalanceItemCategory.ASSET,
        account_id='cash_checking',
        balance=Decimal('45000'),
        previous_balance=Decimal('50000'),
        change_amount=Decimal('-5000'),
        metadata={'reason': 'monthly_expenses', 'period': '2024-01'}
    )
    
    # Demo reconciliation with discrepancies
    print("\n🔍 Testing reconciliation with discrepancies:")
    
    discrepancies = [
        {
            'account_id': 'cash_checking',
            'expected_balance': 45000,
            'actual_balance': 44850,
            'difference': -150,
            'reason': 'Unreconciled transaction'
        }
    ]
    
    accounting.enhanced_logger.log_reconciliation(
        reconciliation_type="monthly_reconciliation",
        accounts_involved=['cash_checking', 'cash_savings', 'investments_stocks'],
        total_assets=Decimal('395000'),
        total_liabilities=Decimal('0'),
        net_worth=Decimal('395000'),
        discrepancies=discrepancies,
        metadata={'reconciliation_method': 'manual_review', 'reviewer': 'demo_user'}
    )
    
    return accounting


def main():
    """Run the complete enhanced logging demo"""
    print("🎯 Enhanced Accounting Logging Demo")
    print("=" * 60)
    print("This demo showcases improved logging that distinguishes between:")
    print("• Flow items: Transactions, cash flows, payments")
    print("• Balance items: Account balances, positions, net worth")
    print("• Reconciliation: Balance reconciliations with discrepancy tracking")
    print("• Error logging: Detailed error context and statistics")
    print()
    
    try:
        # Run all demo sections
        accounting = demo_advanced_features()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Enhanced Accounting Logging Demo Complete!")
        print("=" * 60)
        
        final_stats = accounting.get_enhanced_log_statistics()
        print(f"📊 Final Statistics:")
        print(f"  Total logs: {final_stats['total_logs']}")
        print(f"  Flow items: {final_stats['flow_items']}")
        print(f"  Balance items: {final_stats['balance_items']}")
        print(f"  Reconciliation items: {final_stats['reconciliation_items']}")
        print(f"  Errors: {final_stats['errors']}")
        
        print(f"\n📁 Log files exported to:")
        print(f"  • data/outputs/enhanced_logs_flow.json")
        print(f"  • data/outputs/enhanced_logs_balance.json")
        print(f"  • data/outputs/enhanced_logs_all.json")
        
        print(f"\n📋 Key Improvements:")
        print(f"  • Clear distinction between flow and balance items")
        print(f"  • Structured logging with categories and metadata")
        print(f"  • Change tracking for balance items")
        print(f"  • Confidence levels for flow items")
        print(f"  • Detailed error context and statistics")
        print(f"  • Export capabilities for analysis")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 