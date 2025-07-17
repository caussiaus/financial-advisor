"""
Enhanced Accounting Logger

This module provides structured logging for financial accounting operations,
distinguishing between flow items (transactions, cash flows) and balance items
(account balances, positions). It supports detailed logging with metadata,
export capabilities, and statistics reporting.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path


class LogItemType(Enum):
    """Types of accounting items for logging"""
    FLOW_ITEM = "flow_item"      # Transactions, cash flows, payments
    BALANCE_ITEM = "balance_item" # Account balances, positions, net worth
    RECONCILIATION = "reconciliation"  # Balance reconciliations
    VALIDATION = "validation"     # Payment validations
    ERROR = "error"              # Errors and exceptions


class FlowItemCategory(Enum):
    """Categories for flow items (transactions, cash flows, payments)"""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    INVESTMENT = "investment"
    LOAN_PAYMENT = "loan_payment"
    INTEREST = "interest"
    DIVIDEND = "dividend"
    TAX = "tax"
    INSURANCE = "insurance"
    UTILITY = "utility"
    ENTERTAINMENT = "entertainment"
    FOOD = "food"
    TRANSPORTATION = "transportation"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    LOAN = "loan"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"
    OTHER = "other"


class BalanceItemCategory(Enum):
    """Categories for balance items (account balances, positions)"""
    CASH = "cash"
    CHECKING = "checking"
    SAVINGS = "savings"
    INVESTMENT = "investment"
    RETIREMENT = "retirement"
    CREDIT = "credit"
    LOAN = "loan"
    MORTGAGE = "mortgage"
    INSURANCE = "insurance"
    REAL_ESTATE = "real_estate"
    VEHICLE = "vehicle"
    BUSINESS = "business"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    NET_WORTH = "net_worth"
    LIQUIDITY = "liquidity"
    POSITION = "position"
    OTHER = "other"


@dataclass
class FlowItemLog:
    """Structured log entry for flow items (transactions, cash flows)"""
    timestamp: datetime
    category: FlowItemCategory
    amount: Decimal
    item_type: LogItemType = LogItemType.FLOW_ITEM
    from_account: Optional[str] = None
    to_account: Optional[str] = None
    description: str = ""
    transaction_id: Optional[str] = None
    reference_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'item_type': self.item_type.value,
            'category': self.category.value,
            'amount': str(self.amount),
            'from_account': self.from_account,
            'to_account': self.to_account,
            'description': self.description,
            'transaction_id': self.transaction_id,
            'reference_id': self.reference_id,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class BalanceItemLog:
    """Structured log entry for balance items (account balances, positions)"""
    timestamp: datetime
    category: BalanceItemCategory
    account_id: str
    balance: Decimal
    item_type: LogItemType = LogItemType.BALANCE_ITEM
    previous_balance: Optional[Decimal] = None
    change_amount: Optional[Decimal] = None
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'item_type': self.item_type.value,
            'category': self.category.value,
            'account_id': self.account_id,
            'balance': str(self.balance),
            'previous_balance': str(self.previous_balance) if self.previous_balance else None,
            'change_amount': str(self.change_amount) if self.change_amount else None,
            'currency': self.currency,
            'metadata': self.metadata
        }


@dataclass
class ReconciliationLog:
    """Structured log entry for reconciliation events"""
    timestamp: datetime
    reconciliation_type: str
    accounts_involved: List[str]
    total_assets: Decimal
    total_liabilities: Decimal
    net_worth: Decimal
    item_type: LogItemType = LogItemType.RECONCILIATION
    status: str = "completed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'item_type': self.item_type.value,
            'reconciliation_type': self.reconciliation_type,
            'accounts_involved': self.accounts_involved,
            'total_assets': str(self.total_assets),
            'total_liabilities': str(self.total_liabilities),
            'net_worth': str(self.net_worth),
            'status': self.status,
            'metadata': self.metadata
        }


@dataclass
class ErrorLog:
    """Structured log entry for errors and exceptions"""
    timestamp: datetime
    message: str
    error_type: str
    context: Dict[str, Any]
    item_type: LogItemType = LogItemType.ERROR
    severity: str = "error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'item_type': self.item_type.value,
            'message': self.message,
            'error_type': self.error_type,
            'context': self.context,
            'severity': self.severity
        }


class EnhancedAccountingLogger:
    """Enhanced logging system for financial accounting operations"""
    
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Configure file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.log_entries: List[Union[FlowItemLog, BalanceItemLog, ReconciliationLog, ErrorLog]] = []
        self.stats = {
            'flow_items': 0,
            'balance_items': 0,
            'reconciliations': 0,
            'errors': 0,
            'total_amount': Decimal('0'),
            'categories': {}
        }
    
    def log_flow_item(self, category: FlowItemCategory, amount: Decimal, 
                     from_account: Optional[str] = None, to_account: Optional[str] = None,
                     description: str = "", transaction_id: Optional[str] = None,
                     reference_id: Optional[str] = None, confidence: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> FlowItemLog:
        """Log a flow item (transaction, cash flow, payment)"""
        log_entry = FlowItemLog(
            timestamp=datetime.now(),
            category=category,
            amount=amount,
            from_account=from_account,
            to_account=to_account,
            description=description,
            transaction_id=transaction_id,
            reference_id=reference_id,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.log_entries.append(log_entry)
        self.stats['flow_items'] += 1
        self.stats['total_amount'] += amount
        
        category_name = category.value
        if category_name not in self.stats['categories']:
            self.stats['categories'][category_name] = 0
        self.stats['categories'][category_name] += 1
        
        self.logger.info(f"Flow item logged: {category.value} - {amount} "
                        f"({from_account} -> {to_account}) - {description}")
        
        return log_entry
    
    def log_balance_item(self, category: BalanceItemCategory, account_id: str, 
                        balance: Decimal, previous_balance: Optional[Decimal] = None,
                        change_amount: Optional[Decimal] = None, currency: str = "USD",
                        metadata: Optional[Dict[str, Any]] = None) -> BalanceItemLog:
        """Log a balance item (account balance, position)"""
        log_entry = BalanceItemLog(
            timestamp=datetime.now(),
            category=category,
            account_id=account_id,
            balance=balance,
            previous_balance=previous_balance,
            change_amount=change_amount,
            currency=currency,
            metadata=metadata or {}
        )
        
        self.log_entries.append(log_entry)
        self.stats['balance_items'] += 1
        
        self.logger.info(f"Balance item logged: {category.value} - {account_id} - {balance} {currency}")
        
        return log_entry
    
    def log_reconciliation(self, reconciliation_type: str, accounts_involved: List[str],
                         total_assets: Decimal, total_liabilities: Decimal, net_worth: Decimal,
                         status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> ReconciliationLog:
        """Log a reconciliation event"""
        log_entry = ReconciliationLog(
            timestamp=datetime.now(),
            reconciliation_type=reconciliation_type,
            accounts_involved=accounts_involved,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            net_worth=net_worth,
            status=status,
            metadata=metadata or {}
        )
        
        self.log_entries.append(log_entry)
        self.stats['reconciliations'] += 1
        
        self.logger.info(f"Reconciliation logged: {reconciliation_type} - "
                        f"Assets: {total_assets}, Liabilities: {total_liabilities}, "
                        f"Net Worth: {net_worth}")
        
        return log_entry
    
    def log_error(self, message: str, error_type: str, context: Dict[str, Any],
                 severity: str = "error") -> ErrorLog:
        """Log an error or exception"""
        log_entry = ErrorLog(
            timestamp=datetime.now(),
            message=message,
            error_type=error_type,
            context=context,
            severity=severity
        )
        
        self.log_entries.append(log_entry)
        self.stats['errors'] += 1
        
        self.logger.error(f"Error logged: {error_type} - {message}")
        
        return log_entry
    
    def export_logs(self, file_path: str, format: str = "json") -> None:
        """Export logs to file"""
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump([entry.to_dict() for entry in self.log_entries], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Logs exported to {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'total_entries': len(self.log_entries),
            'flow_items': self.stats['flow_items'],
            'balance_items': self.stats['balance_items'],
            'reconciliations': self.stats['reconciliations'],
            'errors': self.stats['errors'],
            'total_amount': str(self.stats['total_amount']),
            'categories': self.stats['categories']
        }
    
    def clear_logs(self) -> None:
        """Clear all logged entries"""
        self.log_entries.clear()
        self.stats = {
            'flow_items': 0,
            'balance_items': 0,
            'reconciliations': 0,
            'errors': 0,
            'total_amount': Decimal('0'),
            'categories': {}
        }
        self.logger.info("Logs cleared")
    
    def get_entries_by_type(self, item_type: LogItemType) -> List[Union[FlowItemLog, BalanceItemLog, ReconciliationLog, ErrorLog]]:
        """Get all entries of a specific type"""
        return [entry for entry in self.log_entries if entry.item_type == item_type]
    
    def get_entries_by_category(self, category: Union[FlowItemCategory, BalanceItemCategory]) -> List[Union[FlowItemLog, BalanceItemLog]]:
        """Get all entries of a specific category"""
        return [entry for entry in self.log_entries 
                if hasattr(entry, 'category') and entry.category == category]
    
    def get_entries_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Union[FlowItemLog, BalanceItemLog, ReconciliationLog, ErrorLog]]:
        """Get all entries within a date range"""
        return [entry for entry in self.log_entries 
                if start_date <= entry.timestamp <= end_date]
