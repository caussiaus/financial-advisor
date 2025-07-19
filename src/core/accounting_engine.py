#!/usr/bin/env python3
"""
Accounting Reconciliation Engine
Core accounting system for financial state tracking and reconciliation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FinancialTransaction:
    """Represents a financial transaction"""
    transaction_id: str
    timestamp: datetime
    amount: float
    category: str
    description: str
    transaction_type: str  # "income", "expense", "transfer", "investment"
    account: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FinancialAccount:
    """Represents a financial account"""
    account_id: str
    account_name: str
    account_type: str  # "asset", "liability", "income", "expense"
    balance: float
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

class AccountingReconciliationEngine:
    """Engine for financial accounting and reconciliation"""
    
    def __init__(self):
        self.accounts = {}
        self.transactions = []
        self.transaction_counter = 0
        self.account_counter = 0
        
        # Initialize default accounts
        self._initialize_default_accounts()
        
        logger.info("✅ Accounting Reconciliation Engine initialized")
    
    def _initialize_default_accounts(self):
        """Initialize default financial accounts"""
        default_accounts = [
            # Asset accounts
            {"name": "Cash & Checking", "type": "asset", "balance": 0.0},
            {"name": "Savings", "type": "asset", "balance": 0.0},
            {"name": "Investments - Stocks", "type": "asset", "balance": 0.0},
            {"name": "Investments - Bonds", "type": "asset", "balance": 0.0},
            {"name": "Retirement - 401k", "type": "asset", "balance": 0.0},
            {"name": "Retirement - IRA", "type": "asset", "balance": 0.0},
            {"name": "Real Estate", "type": "asset", "balance": 0.0},
            {"name": "Vehicles", "type": "asset", "balance": 0.0},
            {"name": "Other Assets", "type": "asset", "balance": 0.0},
            
            # Liability accounts
            {"name": "Credit Cards", "type": "liability", "balance": 0.0},
            {"name": "Student Loans", "type": "liability", "balance": 0.0},
            {"name": "Mortgage", "type": "liability", "balance": 0.0},
            {"name": "Auto Loans", "type": "liability", "balance": 0.0},
            {"name": "Personal Loans", "type": "liability", "balance": 0.0},
            {"name": "Other Liabilities", "type": "liability", "balance": 0.0},
            
            # Income accounts
            {"name": "Salary", "type": "income", "balance": 0.0},
            {"name": "Bonus", "type": "income", "balance": 0.0},
            {"name": "Investment Income", "type": "income", "balance": 0.0},
            {"name": "Other Income", "type": "income", "balance": 0.0},
            
            # Expense accounts
            {"name": "Housing", "type": "expense", "balance": 0.0},
            {"name": "Utilities", "type": "expense", "balance": 0.0},
            {"name": "Food", "type": "expense", "balance": 0.0},
            {"name": "Transportation", "type": "expense", "balance": 0.0},
            {"name": "Healthcare", "type": "expense", "balance": 0.0},
            {"name": "Entertainment", "type": "expense", "balance": 0.0},
            {"name": "Insurance", "type": "expense", "balance": 0.0},
            {"name": "Other Expenses", "type": "expense", "balance": 0.0}
        ]
        
        for account_info in default_accounts:
            account = FinancialAccount(
                account_id=f"account_{self.account_counter:06d}",
                account_name=account_info["name"],
                account_type=account_info["type"],
                balance=account_info["balance"]
            )
            self.accounts[account.account_id] = account
            self.account_counter += 1
    
    def add_transaction(self, amount: float, category: str, description: str, 
                       transaction_type: str, account: str, timestamp: Optional[datetime] = None) -> str:
        """Add a new financial transaction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        transaction = FinancialTransaction(
            transaction_id=f"txn_{self.transaction_counter:06d}",
            timestamp=timestamp,
            amount=amount,
            category=category,
            description=description,
            transaction_type=transaction_type,
            account=account
        )
        
        self.transactions.append(transaction)
        self.transaction_counter += 1
        
        # Update account balance
        self._update_account_balance(account, amount, transaction_type)
        
        logger.info(f"✅ Added transaction: {description} (${amount:,.2f})")
        return transaction.transaction_id
    
    def _update_account_balance(self, account: str, amount: float, transaction_type: str):
        """Update account balance based on transaction"""
        # Find account by name
        target_account = None
        for acc in self.accounts.values():
            if acc.account_name == account:
                target_account = acc
                break
        
        if target_account is None:
            logger.warning(f"⚠️ Account not found: {account}")
            return
        
        # Update balance based on transaction type
        if transaction_type == "income":
            target_account.balance += amount
        elif transaction_type == "expense":
            target_account.balance -= amount
        elif transaction_type == "transfer":
            # For transfers, we need to specify source and destination
            # This is simplified - in practice you'd have more complex logic
            target_account.balance += amount
        elif transaction_type == "investment":
            # Investment transactions can be positive (buy) or negative (sell)
            target_account.balance += amount
    
    def generate_financial_statement(self) -> Dict[str, Any]:
        """Generate comprehensive financial statement"""
        # Calculate totals by account type
        assets = {acc.account_name: {"balance": acc.balance} for acc in self.accounts.values() if acc.account_type == "asset"}
        liabilities = {acc.account_name: {"balance": acc.balance} for acc in self.accounts.values() if acc.account_type == "liability"}
        income = {acc.account_name: {"balance": acc.balance} for acc in self.accounts.values() if acc.account_type == "income"}
        expenses = {acc.account_name: {"balance": acc.balance} for acc in self.accounts.values() if acc.account_type == "expense"}
        
        # Calculate summary totals
        total_assets = sum(acc["balance"] for acc in assets.values())
        total_liabilities = sum(acc["balance"] for acc in liabilities.values())
        total_income = sum(acc["balance"] for acc in income.values())
        total_expenses = sum(acc["balance"] for acc in expenses.values())
        net_worth = total_assets - total_liabilities
        net_income = total_income - total_expenses
        
        financial_statement = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "net_worth": net_worth,
                "total_income": total_income,
                "total_expenses": total_expenses,
                "net_income": net_income
            },
            "assets": assets,
            "liabilities": liabilities,
            "income": income,
            "expenses": expenses,
            "transactions": [
                {
                    "transaction_id": txn.transaction_id,
                    "timestamp": txn.timestamp.isoformat(),
                    "amount": txn.amount,
                    "category": txn.category,
                    "description": txn.description,
                    "transaction_type": txn.transaction_type,
                    "account": txn.account
                } for txn in self.transactions[-100:]  # Last 100 transactions
            ]
        }
        
        return financial_statement
    
    def reconcile_accounts(self) -> Dict[str, Any]:
        """Reconcile all accounts and check for discrepancies"""
        reconciliation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_accounts": len(self.accounts),
            "total_transactions": len(self.transactions),
            "account_balances": {},
            "discrepancies": [],
            "reconciliation_status": "success"
        }
        
        # Check each account
        for account in self.accounts.values():
            # Calculate expected balance from transactions
            account_transactions = [txn for txn in self.transactions if txn.account == account.account_name]
            expected_balance = sum(txn.amount for txn in account_transactions if txn.transaction_type in ["income", "investment"])
            expected_balance -= sum(txn.amount for txn in account_transactions if txn.transaction_type in ["expense"])
            
            # Check for discrepancies
            if abs(account.balance - expected_balance) > 0.01:  # Allow for small rounding differences
                discrepancy = {
                    "account": account.account_name,
                    "expected_balance": expected_balance,
                    "actual_balance": account.balance,
                    "difference": account.balance - expected_balance
                }
                reconciliation_report["discrepancies"].append(discrepancy)
                reconciliation_report["reconciliation_status"] = "discrepancies_found"
            
            reconciliation_report["account_balances"][account.account_name] = {
                "balance": account.balance,
                "expected_balance": expected_balance,
                "transaction_count": len(account_transactions)
            }
        
        return reconciliation_report
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get summary of all accounts"""
        summary = {
            "total_accounts": len(self.accounts),
            "account_types": {},
            "total_balance_by_type": {},
            "recent_transactions": []
        }
        
        # Group accounts by type
        for account in self.accounts.values():
            if account.account_type not in summary["account_types"]:
                summary["account_types"][account.account_type] = []
                summary["total_balance_by_type"][account.account_type] = 0.0
            
            summary["account_types"][account.account_type].append({
                "account_id": account.account_id,
                "account_name": account.account_name,
                "balance": account.balance
            })
            
            summary["total_balance_by_type"][account.account_type] += account.balance
        
        # Add recent transactions
        recent_transactions = sorted(self.transactions, key=lambda x: x.timestamp, reverse=True)[:10]
        summary["recent_transactions"] = [
            {
                "transaction_id": txn.transaction_id,
                "timestamp": txn.timestamp.isoformat(),
                "amount": txn.amount,
                "description": txn.description,
                "category": txn.category
            } for txn in recent_transactions
        ]
        
        return summary
    
    def export_accounting_data(self, filepath: str):
        """Export accounting data to file"""
        accounting_data = {
            "accounts": {
                account_id: {
                    "account_name": account.account_name,
                    "account_type": account.account_type,
                    "balance": account.balance,
                    "currency": account.currency,
                    "metadata": account.metadata
                } for account_id, account in self.accounts.items()
            },
            "transactions": [
                {
                    "transaction_id": txn.transaction_id,
                    "timestamp": txn.timestamp.isoformat(),
                    "amount": txn.amount,
                    "category": txn.category,
                    "description": txn.description,
                    "transaction_type": txn.transaction_type,
                    "account": txn.account,
                    "metadata": txn.metadata
                } for txn in self.transactions
            ],
            "financial_statement": self.generate_financial_statement(),
            "reconciliation_report": self.reconcile_accounts()
        }
        
        with open(filepath, 'w') as f:
            json.dump(accounting_data, f, indent=2, default=str)
        
        logger.info(f"✅ Accounting data exported to {filepath}")
    
    def load_sample_data(self):
        """Load sample financial data for testing"""
        # Sample income transactions
        self.add_transaction(8000, "Salary", "Monthly salary", "income", "Salary")
        self.add_transaction(2000, "Bonus", "Annual bonus", "income", "Bonus")
        self.add_transaction(500, "Investment Income", "Dividend payment", "income", "Investment Income")
        
        # Sample expense transactions
        self.add_transaction(2500, "Housing", "Monthly rent", "expense", "Housing")
        self.add_transaction(300, "Utilities", "Electricity and water", "expense", "Utilities")
        self.add_transaction(800, "Food", "Groceries and dining", "expense", "Food")
        self.add_transaction(400, "Transportation", "Gas and public transit", "expense", "Transportation")
        self.add_transaction(200, "Healthcare", "Medical expenses", "expense", "Healthcare")
        self.add_transaction(300, "Entertainment", "Movies and activities", "expense", "Entertainment")
        
        # Sample asset transactions
        self.add_transaction(5000, "Investment", "Stock purchase", "investment", "Investments - Stocks")
        self.add_transaction(10000, "Investment", "Bond purchase", "investment", "Investments - Bonds")
        
        # Sample liability transactions
        self.add_transaction(2000, "Credit Card", "Credit card payment", "expense", "Credit Cards")
        self.add_transaction(1500, "Student Loan", "Student loan payment", "expense", "Student Loans")
        
        logger.info("✅ Sample accounting data loaded")

def main():
    """Main function for testing"""
    # Create accounting engine
    accounting_engine = AccountingReconciliationEngine()
    
    # Load sample data
    accounting_engine.load_sample_data()
    
    # Generate financial statement
    financial_statement = accounting_engine.generate_financial_statement()
    print(f"Financial Statement: {json.dumps(financial_statement['summary'], indent=2)}")
    
    # Get account summary
    account_summary = accounting_engine.get_account_summary()
    print(f"Account Summary: {json.dumps(account_summary['total_balance_by_type'], indent=2)}")
    
    # Reconcile accounts
    reconciliation_report = accounting_engine.reconcile_accounts()
    print(f"Reconciliation Status: {reconciliation_report['reconciliation_status']}")
    
    # Export data
    accounting_engine.export_accounting_data("accounting_data.json")

if __name__ == "__main__":
    main() 