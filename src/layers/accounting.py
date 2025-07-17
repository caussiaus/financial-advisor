"""
Accounting Layer

Responsible for:
- Financial state tracking and reconciliation
- Account balance management
- Transaction processing
- Financial statement generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Protocol
import numpy as np
import pandas as pd
import json
from enum import Enum


class AccountType(Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    INCOME = "income"
    EXPENSE = "expense"


class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    INVESTMENT = "investment"
    LOAN = "loan"


@dataclass
class Account:
    """Represents a financial account"""
    account_id: str
    account_type: AccountType
    name: str
    balance: float = 0.0
    currency: str = "USD"
    metadata: Dict = field(default_factory=dict)


@dataclass
class Transaction:
    """Represents a financial transaction"""
    transaction_id: str
    timestamp: datetime
    transaction_type: TransactionType
    amount: float
    from_account: Optional[str] = None
    to_account: Optional[str] = None
    description: str = ""
    category: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class FinancialState:
    """Represents a complete financial state at a point in time"""
    timestamp: datetime
    accounts: Dict[str, Account]
    transactions: List[Transaction]
    net_worth: float
    total_assets: float
    total_liabilities: float
    liquidity_ratio: float
    stress_level: float
    metadata: Dict = field(default_factory=dict)


class AccountManager(Protocol):
    """Protocol for account management capabilities"""
    
    def create_account(self, account_id: str, account_type: AccountType, name: str) -> Account:
        """Create a new account"""
        ...
    
    def update_balance(self, account_id: str, amount: float) -> bool:
        """Update account balance"""
        ...


class TransactionProcessor(Protocol):
    """Protocol for transaction processing capabilities"""
    
    def process_transaction(self, transaction: Transaction) -> bool:
        """Process a transaction"""
        ...


class AccountingLayer:
    """
    Accounting Layer - Clean API for financial state tracking
    
    Responsibilities:
    - Financial state tracking and reconciliation
    - Account balance management
    - Transaction processing
    - Financial statement generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.accounts: Dict[str, Account] = {}
        self.transactions: List[Transaction] = []
        self.financial_states: List[FinancialState] = []
        
        # Initialize default accounts
        self._initialize_default_accounts()
    
    def _initialize_default_accounts(self):
        """Initialize default financial accounts"""
        default_accounts = [
            ("cash_checking", AccountType.ASSET, "Checking Account"),
            ("cash_savings", AccountType.ASSET, "Savings Account"),
            ("investments_stocks", AccountType.ASSET, "Stock Investments"),
            ("investments_bonds", AccountType.ASSET, "Bond Investments"),
            ("investments_real_estate", AccountType.ASSET, "Real Estate"),
            ("debt_credit_cards", AccountType.LIABILITY, "Credit Card Debt"),
            ("debt_mortgage", AccountType.LIABILITY, "Mortgage"),
            ("debt_student_loans", AccountType.LIABILITY, "Student Loans"),
            ("income_salary", AccountType.INCOME, "Salary Income"),
            ("income_investment", AccountType.INCOME, "Investment Income"),
            ("expenses_living", AccountType.EXPENSE, "Living Expenses"),
            ("expenses_healthcare", AccountType.EXPENSE, "Healthcare Expenses"),
            ("expenses_education", AccountType.EXPENSE, "Education Expenses")
        ]
        
        for account_id, account_type, name in default_accounts:
            self.create_account(account_id, account_type, name)
    
    def create_account(self, account_id: str, account_type: AccountType, 
                      name: str, initial_balance: float = 0.0) -> Account:
        """
        Create a new financial account
        
        Args:
            account_id: Unique identifier for the account
            account_type: Type of account (asset, liability, income, expense)
            name: Display name for the account
            initial_balance: Initial balance for the account
            
        Returns:
            Created account object
        """
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")
        
        account = Account(
            account_id=account_id,
            account_type=account_type,
            name=name,
            balance=initial_balance
        )
        
        self.accounts[account_id] = account
        return account
    
    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID"""
        return self.accounts.get(account_id)
    
    def update_account_balance(self, account_id: str, amount: float) -> bool:
        """
        Update account balance
        
        Args:
            account_id: ID of the account to update
            amount: Amount to add (positive) or subtract (negative)
            
        Returns:
            Success status
        """
        if account_id not in self.accounts:
            return False
        
        self.accounts[account_id].balance += amount
        return True
    
    def process_transaction(self, transaction: Transaction) -> bool:
        """
        Process a financial transaction
        
        Args:
            transaction: Transaction to process
            
        Returns:
            Success status
        """
        # Validate transaction
        if not self._validate_transaction(transaction):
            return False
        
        # Update account balances
        if transaction.from_account:
            if not self.update_account_balance(transaction.from_account, -transaction.amount):
                return False
        
        if transaction.to_account:
            if not self.update_account_balance(transaction.to_account, transaction.amount):
                return False
        
        # Record transaction
        self.transactions.append(transaction)
        
        # Update financial state
        self._update_financial_state(transaction.timestamp)
        
        return True
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction before processing"""
        # Check if accounts exist
        if transaction.from_account and transaction.from_account not in self.accounts:
            return False
        
        if transaction.to_account and transaction.to_account not in self.accounts:
            return False
        
        # Check if sufficient funds for withdrawal
        if (transaction.transaction_type == TransactionType.WITHDRAWAL and 
            transaction.from_account):
            account = self.accounts[transaction.from_account]
            if account.balance < transaction.amount:
                return False
        
        return True
    
    def _update_financial_state(self, timestamp: datetime):
        """Update financial state after transaction"""
        # Calculate current financial metrics
        total_assets = sum(
            account.balance for account in self.accounts.values() 
            if account.account_type == AccountType.ASSET
        )
        
        total_liabilities = sum(
            account.balance for account in self.accounts.values() 
            if account.account_type == AccountType.LIABILITY
        )
        
        net_worth = total_assets - total_liabilities
        
        # Calculate liquidity ratio (cash / total assets)
        cash_accounts = [
            account for account in self.accounts.values() 
            if account.account_type == AccountType.ASSET and "cash" in account.account_id.lower()
        ]
        total_cash = sum(account.balance for account in cash_accounts)
        liquidity_ratio = total_cash / total_assets if total_assets > 0 else 0
        
        # Calculate stress level (simplified)
        stress_level = self._calculate_stress_level(total_assets, total_liabilities, liquidity_ratio)
        
        # Create financial state
        financial_state = FinancialState(
            timestamp=timestamp,
            accounts=self.accounts.copy(),
            transactions=self.transactions.copy(),
            net_worth=net_worth,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            liquidity_ratio=liquidity_ratio,
            stress_level=stress_level
        )
        
        self.financial_states.append(financial_state)
    
    def _calculate_stress_level(self, total_assets: float, total_liabilities: float, 
                              liquidity_ratio: float) -> float:
        """Calculate financial stress level (0-1)"""
        stress = 0.0
        
        # Debt-to-asset ratio stress
        if total_assets > 0:
            debt_ratio = total_liabilities / total_assets
            stress += min(1.0, debt_ratio * 2)  # Scale debt ratio
        
        # Liquidity stress
        if liquidity_ratio < 0.1:  # Less than 10% cash
            stress += 0.3
        elif liquidity_ratio < 0.2:  # Less than 20% cash
            stress += 0.1
        
        # Net worth stress
        if total_assets < 10000:  # Low assets
            stress += 0.2
        
        return min(1.0, stress)
    
    def get_current_financial_state(self) -> Optional[FinancialState]:
        """Get the most recent financial state"""
        if not self.financial_states:
            return None
        
        return self.financial_states[-1]
    
    def generate_financial_statement(self, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Generate comprehensive financial statement
        
        Args:
            as_of_date: Date for the statement (defaults to current)
            
        Returns:
            Financial statement dictionary
        """
        if not as_of_date:
            as_of_date = datetime.now()
        
        # Get financial state closest to the date
        target_state = None
        for state in reversed(self.financial_states):
            if state.timestamp <= as_of_date:
                target_state = state
                break
        
        if not target_state:
            # Create current state if none exists
            self._update_financial_state(as_of_date)
            target_state = self.financial_states[-1]
        
        # Organize accounts by type
        assets = {k: v for k, v in target_state.accounts.items() 
                 if v.account_type == AccountType.ASSET}
        liabilities = {k: v for k, v in target_state.accounts.items() 
                      if v.account_type == AccountType.LIABILITY}
        income = {k: v for k, v in target_state.accounts.items() 
                 if v.account_type == AccountType.INCOME}
        expenses = {k: v for k, v in target_state.accounts.items() 
                   if v.account_type == AccountType.EXPENSE}
        
        return {
            'as_of_date': target_state.timestamp.isoformat(),
            'summary': {
                'total_assets': target_state.total_assets,
                'total_liabilities': target_state.total_liabilities,
                'net_worth': target_state.net_worth,
                'liquidity_ratio': target_state.liquidity_ratio,
                'stress_level': target_state.stress_level
            },
            'assets': {k: {'balance': v.balance, 'name': v.name} 
                      for k, v in assets.items()},
            'liabilities': {k: {'balance': v.balance, 'name': v.name} 
                           for k, v in liabilities.items()},
            'income': {k: {'balance': v.balance, 'name': v.name} 
                      for k, v in income.items()},
            'expenses': {k: {'balance': v.balance, 'name': v.name} 
                        for k, v in expenses.items()},
            'transactions': [
                {
                    'id': t.transaction_id,
                    'timestamp': t.timestamp.isoformat(),
                    'type': t.transaction_type.value,
                    'amount': t.amount,
                    'from_account': t.from_account,
                    'to_account': t.to_account,
                    'description': t.description,
                    'category': t.category
                }
                for t in target_state.transactions
            ]
        }
    
    def reconcile_accounts(self) -> Dict:
        """
        Reconcile account balances and identify discrepancies
        
        Returns:
            Reconciliation report
        """
        reconciliation_report = {
            'timestamp': datetime.now().isoformat(),
            'accounts_reconciled': len(self.accounts),
            'discrepancies': [],
            'total_transactions': len(self.transactions)
        }
        
        # Check for negative balances in asset accounts
        for account_id, account in self.accounts.items():
            if account.account_type == AccountType.ASSET and account.balance < 0:
                reconciliation_report['discrepancies'].append({
                    'account_id': account_id,
                    'account_name': account.name,
                    'issue': 'negative_balance',
                    'balance': account.balance
                })
        
        # Check for positive balances in liability accounts (unusual)
        for account_id, account in self.accounts.items():
            if account.account_type == AccountType.LIABILITY and account.balance > 0:
                reconciliation_report['discrepancies'].append({
                    'account_id': account_id,
                    'account_name': account.name,
                    'issue': 'positive_liability_balance',
                    'balance': account.balance
                })
        
        return reconciliation_report
    
    def get_account_history(self, account_id: str, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict]:
        """
        Get transaction history for a specific account
        
        Args:
            account_id: ID of the account
            start_date: Start date for history
            end_date: End date for history
            
        Returns:
            List of transactions for the account
        """
        if account_id not in self.accounts:
            return []
        
        account_transactions = []
        
        for transaction in self.transactions:
            # Check if transaction involves this account
            if (transaction.from_account == account_id or 
                transaction.to_account == account_id):
                
                # Apply date filters
                if start_date and transaction.timestamp < start_date:
                    continue
                if end_date and transaction.timestamp > end_date:
                    continue
                
                account_transactions.append({
                    'transaction_id': transaction.transaction_id,
                    'timestamp': transaction.timestamp.isoformat(),
                    'type': transaction.transaction_type.value,
                    'amount': transaction.amount,
                    'description': transaction.description,
                    'category': transaction.category,
                    'direction': 'out' if transaction.from_account == account_id else 'in'
                })
        
        return sorted(account_transactions, key=lambda x: x['timestamp'])
    
    def export_accounting_data(self, filepath: str):
        """Export accounting data to file"""
        data = {
            'accounts': {k: v.__dict__ for k, v in self.accounts.items()},
            'transactions': [t.__dict__ for t in self.transactions],
            'financial_states': [s.__dict__ for s in self.financial_states]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, default=str, indent=2)
    
    def import_accounting_data(self, filepath: str):
        """Import accounting data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Import accounts
        for account_id, account_data in data['accounts'].items():
            account = Account(**account_data)
            self.accounts[account_id] = account
        
        # Import transactions
        for transaction_data in data['transactions']:
            transaction = Transaction(**transaction_data)
            self.transactions.append(transaction)
        
        # Import financial states
        for state_data in data['financial_states']:
            financial_state = FinancialState(**state_data)
            self.financial_states.append(financial_state) 