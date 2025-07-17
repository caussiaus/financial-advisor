import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import json
from enum import Enum
import logging
from src.enhanced_accounting_logger import (
    EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory, LogItemType
)

# Enhanced logging imports
from src.enhanced_accounting_logger import (
    EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory, LogItemType
)


class AccountType(Enum):
    ASSET = "asset"
    LIABILITY = "liability" 
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"


class TransactionType(Enum):
    PAYMENT = "payment"
    INCOME = "income"
    INVESTMENT = "investment"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    MILESTONE_PAYMENT = "milestone_payment"


@dataclass
class Account:
    """Represents a financial account with balance tracking"""
    account_id: str
    name: str
    account_type: AccountType
    balance: Decimal
    currency: str = "USD"
    is_active: bool = True
    parent_account: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Transaction:
    """Represents a financial transaction with double-entry bookkeeping"""
    transaction_id: str
    timestamp: datetime
    transaction_type: TransactionType
    amount: Decimal
    debit_account: str
    credit_account: str
    description: str
    reference_id: Optional[str] = None  # Links to milestone or payment
    is_pending: bool = False
    created_by: str = "system"
    metadata: Dict = field(default_factory=dict)


@dataclass
class PaymentConstraint:
    """Defines constraints for payments"""
    account_id: str
    min_balance: Decimal
    max_single_payment: Optional[Decimal] = None
    daily_limit: Optional[Decimal] = None
    monthly_limit: Optional[Decimal] = None
    requires_approval: bool = False
    blocked_dates: List[datetime] = field(default_factory=list)


class AccountingReconciliationEngine:
    """
    Comprehensive accounting system that ensures all payments respect balance constraints
    and maintains accurate double-entry bookkeeping for the Omega mesh
    """
    
    def __init__(self):
        self.accounts = {}  # account_id -> Account
        self.transactions = []  # List of all transactions
        self.constraints = {}  # account_id -> PaymentConstraint
        self.pending_transactions = []  # Transactions awaiting approval
        self.logger = self._setup_logging()
        # Initialize enhanced logging
        self.enhanced_logger = EnhancedAccountingLogger()
        # Initialize default accounts
        self._initialize_default_accounts()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for accounting operations"""
        logger = logging.getLogger('accounting_reconciliation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _determine_flow_category(self, transaction: Transaction) -> FlowItemCategory:
        """
        Determine the flow category for a transaction based on its type and accounts.
        
        Args:
            transaction: The transaction to categorize
            
        Returns:
            FlowItemCategory for the transaction
        """
        # Map transaction types to flow categories
        type_to_category = {
            TransactionType.PAYMENT: FlowItemCategory.EXPENSE,
            TransactionType.INCOME: FlowItemCategory.INCOME,
            TransactionType.INVESTMENT: FlowItemCategory.INVESTMENT,
            TransactionType.WITHDRAWAL: FlowItemCategory.EXPENSE,
            TransactionType.TRANSFER: FlowItemCategory.TRANSFER,
            TransactionType.MILESTONE_PAYMENT: FlowItemCategory.EXPENSE
        }
        
        # Get base category from transaction type
        base_category = type_to_category.get(transaction.transaction_type, FlowItemCategory.OTHER)
        
        # Refine based on account types
        debit_account = self.accounts.get(transaction.debit_account)
        credit_account = self.accounts.get(transaction.credit_account)
        
        if debit_account and credit_account:
            # Specific refinements based on account types
            if (debit_account.account_type == AccountType.ASSET and 
                credit_account.account_type == AccountType.EXPENSE):
                return FlowItemCategory.EXPENSE
            elif (debit_account.account_type == AccountType.INCOME and 
                  credit_account.account_type == AccountType.ASSET):
                return FlowItemCategory.INCOME
            elif (debit_account.account_type == AccountType.ASSET and 
                  credit_account.account_type == AccountType.ASSET):
                return FlowItemCategory.TRANSFER
            elif (debit_account.account_type == AccountType.ASSET and 
                  credit_account.account_type == AccountType.LIABILITY):
                return FlowItemCategory.LOAN_PAYMENT
        
        return base_category
    
    def _determine_balance_category(self, account_type: AccountType) -> BalanceItemCategory:
        """
        Determine the balance category for an account based on its type.
        
        Args:
            account_type: The account type to categorize
            
        Returns:
            BalanceItemCategory for the account
        """
        # Map account types to balance categories
        type_to_category = {
            AccountType.ASSET: BalanceItemCategory.CASH,
            AccountType.LIABILITY: BalanceItemCategory.LOAN,
            AccountType.EQUITY: BalanceItemCategory.OTHER,
            AccountType.INCOME: BalanceItemCategory.OTHER,
            AccountType.EXPENSE: BalanceItemCategory.OTHER
        }
        
        return type_to_category.get(account_type, BalanceItemCategory.OTHER)
    
    def _initialize_default_accounts(self):
        """Initialize standard chart of accounts"""
        default_accounts = [
            # Assets
            Account("cash_checking", "Checking Account", AccountType.ASSET, Decimal('0')),
            Account("cash_savings", "Savings Account", AccountType.ASSET, Decimal('0')),
            Account("investments_stocks", "Stock Investments", AccountType.ASSET, Decimal('0')),
            Account("investments_bonds", "Bond Investments", AccountType.ASSET, Decimal('0')),
            Account("investments_retirement", "Retirement Accounts", AccountType.ASSET, Decimal('0')),
            Account("real_estate", "Real Estate", AccountType.ASSET, Decimal('0')),
            
            # Liabilities
            Account("mortgage", "Mortgage Loan", AccountType.LIABILITY, Decimal('0')),
            Account("student_loans", "Student Loans", AccountType.LIABILITY, Decimal('0')),
            Account("credit_cards", "Credit Cards", AccountType.LIABILITY, Decimal('0')),
            Account("auto_loans", "Auto Loans", AccountType.LIABILITY, Decimal('0')),
            
            # Equity
            Account("net_worth", "Net Worth", AccountType.EQUITY, Decimal('0')),
            
            # Income
            Account("salary", "Salary Income", AccountType.INCOME, Decimal('0')),
            Account("investment_income", "Investment Income", AccountType.INCOME, Decimal('0')),
            Account("other_income", "Other Income", AccountType.INCOME, Decimal('0')),
            
            # Expenses
            Account("milestone_payments", "Milestone Payments", AccountType.EXPENSE, Decimal('0')),
            Account("living_expenses", "Living Expenses", AccountType.EXPENSE, Decimal('0')),
            Account("investment_expenses", "Investment Expenses", AccountType.EXPENSE, Decimal('0'))
        ]
        
        for account in default_accounts:
            self.accounts[account.account_id] = account
            
        # Set up default constraints
        self._setup_default_constraints()
    
    def _setup_default_constraints(self):
        """Setup default payment constraints for accounts"""
        # Checking account constraints
        self.constraints["cash_checking"] = PaymentConstraint(
            account_id="cash_checking",
            min_balance=Decimal('1000'),  # Keep minimum $1000
            max_single_payment=Decimal('50000'),  # Max $50k single payment
            daily_limit=Decimal('100000'),  # $100k daily limit
            monthly_limit=Decimal('500000')  # $500k monthly limit
        )
        
        # Savings account constraints
        self.constraints["cash_savings"] = PaymentConstraint(
            account_id="cash_savings",
            min_balance=Decimal('5000'),  # Keep minimum $5000
            max_single_payment=Decimal('100000'),
            daily_limit=Decimal('50000'),
            monthly_limit=Decimal('200000')
        )
        
        # Investment account constraints
        self.constraints["investments_stocks"] = PaymentConstraint(
            account_id="investments_stocks",
            min_balance=Decimal('0'),
            max_single_payment=None,  # No limit on investment liquidation
            requires_approval=True  # Require approval for large liquidations
        )
    
    def add_account(self, account: Account) -> bool:
        """Add a new account to the system"""
        if account.account_id in self.accounts:
            self.logger.warning(f"Account {account.account_id} already exists")
            return False
        
        self.accounts[account.account_id] = account
        self.logger.info(f"Added account: {account.name} ({account.account_id})")
        return True
    
    def set_constraint(self, constraint: PaymentConstraint):
        """Set payment constraints for an account"""
        if constraint.account_id not in self.accounts:
            raise ValueError(f"Account {constraint.account_id} does not exist")
        
        self.constraints[constraint.account_id] = constraint
        self.logger.info(f"Set constraints for account {constraint.account_id}")
    
    def validate_payment(self, from_account: str, amount: Decimal, 
                        transaction_date: datetime = None) -> Tuple[bool, str]:
        """
        Validate if a payment can be made while respecting all constraints
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        if transaction_date is None:
            transaction_date = datetime.now()
        
        # Check if account exists
        if from_account not in self.accounts:
            return False, f"Account {from_account} does not exist"
        
        account = self.accounts[from_account]
        constraint = self.constraints.get(from_account)
        
        # Check if account is active
        if not account.is_active:
            return False, f"Account {from_account} is inactive"
        
        # Check sufficient balance
        if account.balance < amount:
            return False, f"Insufficient balance: ${account.balance} available, ${amount} requested"
        
        # Check minimum balance constraint
        if constraint and constraint.min_balance:
            remaining_balance = account.balance - amount
            if remaining_balance < constraint.min_balance:
                return False, f"Payment would violate minimum balance requirement of ${constraint.min_balance}"
        
        # Check single payment limit
        if constraint and constraint.max_single_payment:
            if amount > constraint.max_single_payment:
                return False, f"Payment exceeds single payment limit of ${constraint.max_single_payment}"
        
        # Check daily limit
        if constraint and constraint.daily_limit:
            daily_total = self._get_daily_payment_total(from_account, transaction_date)
            if daily_total + amount > constraint.daily_limit:
                return False, f"Payment would exceed daily limit of ${constraint.daily_limit}"
        
        # Check monthly limit
        if constraint and constraint.monthly_limit:
            monthly_total = self._get_monthly_payment_total(from_account, transaction_date)
            if monthly_total + amount > constraint.monthly_limit:
                return False, f"Payment would exceed monthly limit of ${constraint.monthly_limit}"
        
        # Check blocked dates
        if constraint and constraint.blocked_dates:
            if transaction_date.date() in [d.date() for d in constraint.blocked_dates]:
                return False, f"Payments are blocked on {transaction_date.date()}"
        
        return True, "Payment validation passed"
    
    def execute_payment(self, from_account: str, to_account: str, amount: Decimal,
                       description: str, reference_id: str = None,
                       transaction_date: datetime = None, 
                       force_approval: bool = False) -> Tuple[bool, str]:
        """
        Execute a payment with full accounting reconciliation
        
        Returns:
            (success, transaction_id_or_error_message)
        """
        if transaction_date is None:
            transaction_date = datetime.now()
        
        # Convert amount to Decimal for precision
        amount = Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        # Validate the payment
        is_valid, validation_message = self.validate_payment(from_account, amount, transaction_date)
        if not is_valid and not force_approval:
            self.logger.warning(f"Payment validation failed: {validation_message}")
            return False, validation_message
        
        # Check if approval is required
        constraint = self.constraints.get(from_account)
        requires_approval = constraint and constraint.requires_approval and not force_approval
        
        # Create transaction
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.transactions)}"
        transaction = Transaction(
            transaction_id=transaction_id,
            timestamp=transaction_date,
            transaction_type=TransactionType.MILESTONE_PAYMENT,
            amount=amount,
            debit_account=from_account,
            credit_account=to_account,
            description=description,
            reference_id=reference_id,
            is_pending=requires_approval,
            metadata={
                'validation_bypassed': not is_valid,
                'validation_message': validation_message
            }
        )
        
        if requires_approval:
            self.pending_transactions.append(transaction)
            self.logger.info(f"Transaction {transaction_id} added to pending approval queue")
            return True, f"Transaction {transaction_id} pending approval"
        else:
            # Execute immediately
            return self._execute_transaction(transaction)
    
    def _execute_transaction(self, transaction: Transaction) -> Tuple[bool, str]:
        """Execute a validated transaction and update account balances"""
        try:
            # Store previous balances for logging
            previous_debit_balance = self.accounts[transaction.debit_account].balance
            previous_credit_balance = self.accounts[transaction.credit_account].balance
            
            # Update balances based on double-entry bookkeeping
            debit_account = self.accounts[transaction.debit_account]
            credit_account = self.accounts[transaction.credit_account]
            
            # For asset accounts: debit increases, credit decreases
            # For liability/equity/income accounts: credit increases, debit decreases
            # For expense accounts: debit increases, credit decreases
            
            if debit_account.account_type in [AccountType.ASSET, AccountType.EXPENSE]:
                debit_account.balance -= transaction.amount  # Money going out
            else:
                debit_account.balance += transaction.amount
            
            if credit_account.account_type in [AccountType.ASSET, AccountType.EXPENSE]:
                credit_account.balance += transaction.amount  # Money coming in
            else:
                credit_account.balance -= transaction.amount
            
            # Update timestamps
            debit_account.last_updated = transaction.timestamp
            credit_account.last_updated = transaction.timestamp
            
            # Record transaction
            transaction.is_pending = False
            self.transactions.append(transaction)
            
            # Enhanced logging for flow items
            flow_category = self._determine_flow_category(transaction)
            self.enhanced_logger.log_flow_item(
                category=flow_category,
                amount=transaction.amount,
                from_account=transaction.debit_account,
                to_account=transaction.credit_account,
                description=transaction.description,
                transaction_id=transaction.transaction_id,
                reference_id=transaction.reference_id,
                metadata={
                    'transaction_type': transaction.transaction_type.value,
                    'created_by': transaction.created_by
                }
            )
            
            # Enhanced logging for balance changes
            self.enhanced_logger.log_balance_item(
                category=self._determine_balance_category(debit_account.account_type),
                account_id=transaction.debit_account,
                balance=debit_account.balance,
                previous_balance=previous_debit_balance,
                change_amount=debit_account.balance - previous_debit_balance,
                metadata={'transaction_id': transaction.transaction_id}
            )
            
            self.enhanced_logger.log_balance_item(
                category=self._determine_balance_category(credit_account.account_type),
                account_id=transaction.credit_account,
                balance=credit_account.balance,
                previous_balance=previous_credit_balance,
                change_amount=credit_account.balance - previous_credit_balance,
                metadata={'transaction_id': transaction.transaction_id}
            )
            
            self.logger.info(f"Transaction {transaction.transaction_id} executed successfully")
            
            # Update net worth
            self._update_net_worth()
            
            return True, transaction.transaction_id
            
        except Exception as e:
            # Enhanced error logging
            self.enhanced_logger.log_error(f"Transaction execution failed: {str(e)}", {
                'transaction_id': transaction.transaction_id,
                'debit_account': transaction.debit_account,
                'credit_account': transaction.credit_account,
                'amount': str(transaction.amount)
            })
            self.logger.error(f"Error executing transaction {transaction.transaction_id}: {str(e)}")
            self.enhanced_logger.log_error(f"Transaction execution failed: {str(e)}", {
                'transaction_id': transaction.transaction_id,
                'debit_account': transaction.debit_account,
                'credit_account': transaction.credit_account,
                'amount': str(transaction.amount)
            })
            return False, f"Transaction failed: {str(e)}"
    
    def approve_pending_transaction(self, transaction_id: str) -> Tuple[bool, str]:
        """Approve a pending transaction"""
        for i, transaction in enumerate(self.pending_transactions):
            if transaction.transaction_id == transaction_id:
                # Remove from pending and execute
                pending_txn = self.pending_transactions.pop(i)
                return self._execute_transaction(pending_txn)
        
        return False, f"Pending transaction {transaction_id} not found"
    
    def reject_pending_transaction(self, transaction_id: str, reason: str) -> bool:
        """Reject a pending transaction"""
        for i, transaction in enumerate(self.pending_transactions):
            if transaction.transaction_id == transaction_id:
                rejected_txn = self.pending_transactions.pop(i)
                rejected_txn.metadata['rejection_reason'] = reason
                self.logger.info(f"Transaction {transaction_id} rejected: {reason}")
                return True
        
        return False
    
    def _get_daily_payment_total(self, account_id: str, date: datetime) -> Decimal:
        """Get total payments from an account on a specific date"""
        daily_total = Decimal('0')
        target_date = date.date()
        
        for transaction in self.transactions:
            if (transaction.debit_account == account_id and 
                transaction.timestamp.date() == target_date and
                not transaction.is_pending):
                daily_total += transaction.amount
        
        return daily_total
    
    def _get_monthly_payment_total(self, account_id: str, date: datetime) -> Decimal:
        """Get total payments from an account in a specific month"""
        monthly_total = Decimal('0')
        target_month = date.replace(day=1).date()
        
        for transaction in self.transactions:
            if (transaction.debit_account == account_id and 
                transaction.timestamp.replace(day=1).date() == target_month and
                not transaction.is_pending):
                monthly_total += transaction.amount
        
        return monthly_total
    
    def _update_net_worth(self):
        """Update net worth calculation"""
        total_assets = Decimal('0')
        total_liabilities = Decimal('0')
        
        for account in self.accounts.values():
            if account.account_type == AccountType.ASSET:
                total_assets += account.balance
            elif account.account_type == AccountType.LIABILITY:
                total_liabilities += account.balance
        
        net_worth = total_assets - total_liabilities
        self.accounts["net_worth"].balance = net_worth
        self.accounts["net_worth"].last_updated = datetime.now()
    
    def get_account_balance(self, account_id: str) -> Optional[Decimal]:
        """Get current balance of an account"""
        if account_id in self.accounts:
            return self.accounts[account_id].balance
        return None
    
    def set_account_balance(self, account_id: str, balance: Decimal):
        """Set account balance (for initialization)"""
        if account_id in self.accounts:
            self.accounts[account_id].balance = Decimal(str(balance))
            self.accounts[account_id].last_updated = datetime.now()
            self._update_net_worth()
    
    def register_entity(self, entity_state: Dict):
        """Register a financial entity with the accounting system."""
        entity_name = entity_state.get('entity_name', 'Unknown')
        entity_type = entity_state.get('entity_type', 'person')
        balances = entity_state.get('balances', {})

        # Create entity-specific accounts
        for account_name, balance in balances.items():
            account_id = f"{entity_name.lower().replace(' ', '_')}_{account_name}"

            # Determine account type based on the balance name
            if account_name in ['salary', 'income', 'investment_income']:
                account_type = AccountType.INCOME
            elif account_name in ['savings', 'investments', 'education_fund', 'housing', 'investment']:
                account_type = AccountType.ASSET
            elif account_name in ['allowance', 'expenses']:
                account_type = AccountType.EXPENSE
            else:
                account_type = AccountType.ASSET  # Default to asset

            # Create the account
            account = Account(
                account_id=account_id,
                name=f"{entity_name} - {account_name.replace('_', ' ').title()}",
                account_type=account_type,
                balance=Decimal(str(balance))
            )

            self.accounts[account_id] = account
            self.logger.info(f"Registered entity account: {account_id} with balance ${balance}")
    
    def get_payment_capacity(self, account_id: str) -> Dict[str, Decimal]:
        """
        Get the maximum payment capacity for an account considering all constraints
        """
        if account_id not in self.accounts:
            return {}
        
        account = self.accounts[account_id]
        constraint = self.constraints.get(account_id)
        
        # Start with current balance
        max_payment = account.balance
        
        # Apply minimum balance constraint
        if constraint and constraint.min_balance:
            max_payment = max(Decimal('0'), account.balance - constraint.min_balance)
        
        # Apply single payment limit
        if constraint and constraint.max_single_payment:
            max_payment = min(max_payment, constraint.max_single_payment)
        
        # Apply daily limit
        today_payments = self._get_daily_payment_total(account_id, datetime.now())
        if constraint and constraint.daily_limit:
            remaining_daily = constraint.daily_limit - today_payments
            max_payment = min(max_payment, remaining_daily)
        
        # Apply monthly limit
        month_payments = self._get_monthly_payment_total(account_id, datetime.now())
        if constraint and constraint.monthly_limit:
            remaining_monthly = constraint.monthly_limit - month_payments
            max_payment = min(max_payment, remaining_monthly)
        
        return {
            'max_single_payment': max(Decimal('0'), max_payment),
            'current_balance': account.balance,
            'minimum_balance_required': constraint.min_balance if constraint else Decimal('0'),
            'daily_limit_remaining': constraint.daily_limit - today_payments if constraint and constraint.daily_limit else None,
            'monthly_limit_remaining': constraint.monthly_limit - month_payments if constraint and constraint.monthly_limit else None
        }
    
    def generate_financial_statement(self, as_of_date: datetime = None) -> Dict:
        """Generate a comprehensive financial statement"""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        assets = {}
        liabilities = {}
        equity = {}
        income = {}
        expenses = {}
        
        for account_id, account in self.accounts.items():
            account_data = {
                'name': account.name,
                'balance': float(account.balance),
                'last_updated': account.last_updated.isoformat()
            }
            
            if account.account_type == AccountType.ASSET:
                assets[account_id] = account_data
            elif account.account_type == AccountType.LIABILITY:
                liabilities[account_id] = account_data
            elif account.account_type == AccountType.EQUITY:
                equity[account_id] = account_data
            elif account.account_type == AccountType.INCOME:
                income[account_id] = account_data
            elif account.account_type == AccountType.EXPENSE:
                expenses[account_id] = account_data
        
        total_assets = sum(float(self.accounts[aid].balance) for aid in assets.keys())
        total_liabilities = sum(float(self.accounts[lid].balance) for lid in liabilities.keys())
        net_worth = total_assets - total_liabilities
        
        return {
            'as_of_date': as_of_date.isoformat(),
            'assets': assets,
            'liabilities': liabilities,
            'equity': equity,
            'income': income,
            'expenses': expenses,
            'summary': {
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'net_worth': net_worth
            },
            'pending_transactions': len(self.pending_transactions),
            'total_transactions': len(self.transactions)
        }
    
    def get_transaction_history(self, account_id: str = None, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> List[Dict]:
        """Get transaction history with optional filtering"""
        filtered_transactions = self.transactions
        
        if account_id:
            filtered_transactions = [
                txn for txn in filtered_transactions 
                if txn.debit_account == account_id or txn.credit_account == account_id
            ]
        
        if start_date:
            filtered_transactions = [
                txn for txn in filtered_transactions 
                if txn.timestamp >= start_date
            ]
        
        if end_date:
            filtered_transactions = [
                txn for txn in filtered_transactions 
                if txn.timestamp <= end_date
            ]
        
        return [
            {
                'transaction_id': txn.transaction_id,
                'timestamp': txn.timestamp.isoformat(),
                'type': txn.transaction_type.value,
                'amount': float(txn.amount),
                'debit_account': txn.debit_account,
                'credit_account': txn.credit_account,
                'description': txn.description,
                'reference_id': txn.reference_id,
                'is_pending': txn.is_pending
            }
            for txn in filtered_transactions
        ]
    
    def export_accounting_data(self, filepath: str):
        """Export all accounting data to JSON"""
        export_data = {
            'accounts': {
                aid: {
                    'name': acc.name,
                    'type': acc.account_type.value,
                    'balance': float(acc.balance),
                    'currency': acc.currency,
                    'is_active': acc.is_active,
                    'created_date': acc.created_date.isoformat(),
                    'last_updated': acc.last_updated.isoformat()
                }
                for aid, acc in self.accounts.items()
            },
            'transactions': self.get_transaction_history(),
            'constraints': {
                aid: {
                    'min_balance': float(constraint.min_balance),
                    'max_single_payment': float(constraint.max_single_payment) if constraint.max_single_payment else None,
                    'daily_limit': float(constraint.daily_limit) if constraint.daily_limit else None,
                    'monthly_limit': float(constraint.monthly_limit) if constraint.monthly_limit else None,
                    'requires_approval': constraint.requires_approval
                }
                for aid, constraint in self.constraints.items()
            },
            'pending_transactions': [
                {
                    'transaction_id': txn.transaction_id,
                    'timestamp': txn.timestamp.isoformat(),
                    'amount': float(txn.amount),
                    'debit_account': txn.debit_account,
                    'credit_account': txn.credit_account,
                    'description': txn.description
                }
                for txn in self.pending_transactions
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Accounting data exported to {filepath}")
    
    def log_reconciliation_summary(self, reconciliation_type: str = "daily"):
        """
        Log a reconciliation summary with enhanced logging.
        
        Args:
            reconciliation_type: Type of reconciliation (daily, monthly, etc.)
        """
        try:
            # Generate reconciliation data
            total_assets = Decimal('0')
            total_liabilities = Decimal('0')
            total_income = Decimal('0')
            total_expenses = Decimal('0')
            
            for account in self.accounts.values():
                if account.account_type == AccountType.ASSET:
                    total_assets += account.balance
                elif account.account_type == AccountType.LIABILITY:
                    total_liabilities += account.balance
                elif account.account_type == AccountType.INCOME:
                    total_income += account.balance
                elif account.account_type == AccountType.EXPENSE:
                    total_expenses += account.balance
            
            net_worth = total_assets - total_liabilities
            
            # Log reconciliation with enhanced logger
            self.enhanced_logger.log_reconciliation(
                reconciliation_type=reconciliation_type,
                status='completed',
                details={
                    'total_assets': str(total_assets),
                    'total_liabilities': str(total_liabilities),
                    'total_income': str(total_income),
                    'total_expenses': str(total_expenses),
                    'net_worth': str(net_worth),
                    'accounts_reconciled': len(self.accounts),
                    'transactions_processed': len(self.transactions)
                },
                metadata={
                    'reconciliation_date': datetime.now().isoformat(),
                    'system_version': 'enhanced_accounting_v1.0'
                }
            )
            
            self.logger.info(f"{reconciliation_type.capitalize()} reconciliation completed successfully")
            
        except Exception as e:
            self.enhanced_logger.log_error(f"Reconciliation summary failed: {str(e)}", {
                'reconciliation_type': reconciliation_type,
                'error_type': 'reconciliation_failure'
            })
            self.logger.error(f"Error during reconciliation summary: {str(e)}")
    
    def export_enhanced_logs(self, file_path: str):
        """
        Export enhanced logs from the accounting engine.
        
        Args:
            file_path: Path to export the enhanced logs
        """
        try:
            self.enhanced_logger.export_logs(file_path, format="json")
            self.logger.info(f"Enhanced logs exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export enhanced logs: {str(e)}")
    
    def generate_reconciliation_summary(self) -> Dict[str, Decimal]:
        """
        Generate a reconciliation summary of all account balances.
        
        Returns:
            Dictionary mapping account IDs to balances
        """
        summary = {}
        for account_id, account in self.accounts.items():
            summary[account_id] = account.balance
        
        # Log the reconciliation summary
        self.log_reconciliation_summary("summary")
        
        return summary


if __name__ == "__main__":
    # Example usage
    accounting = AccountingReconciliationEngine()
    
    # Initialize some balances
    accounting.set_account_balance("cash_checking", Decimal('50000'))
    accounting.set_account_balance("cash_savings", Decimal('100000'))
    accounting.set_account_balance("investments_stocks", Decimal('250000'))
    
    print("Accounting system initialized!")
    print(f"Financial statement: {accounting.generate_financial_statement()}")
    
    # Test payment validation
    is_valid, message = accounting.validate_payment("cash_checking", Decimal('5000'))
    print(f"Payment validation: {is_valid} - {message}")
    
    # Test payment capacity
    capacity = accounting.get_payment_capacity("cash_checking")
    print(f"Payment capacity: {capacity}")
    
    # Execute a payment
    success, result = accounting.execute_payment(
        "cash_checking", "milestone_payments", Decimal('5000'),
        "Payment for education milestone", "milestone_education_2024"
    )
    print(f"Payment execution: {success} - {result}")