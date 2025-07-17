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
from dataclasses import dataclass, asdict
from pathlib import Path


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
    OTHER = "other"


class LogItemType(Enum):
    """Types of log items"""
    FLOW = "flow"
    BALANCE = "balance"
    ERROR = "error"
    RECONCILIATION = "reconciliation"


@dataclass
class FlowItem:
    """Represents a flow item (transaction, cash flow, payment)"""
    category: FlowItemCategory
    amount: Decimal
    from_account: str
    to_account: str
    description: str
    transaction_id: str
    reference_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BalanceItem:
    """Represents a balance item (account balance, position)"""
    category: BalanceItemCategory
    account_id: str
    balance: Decimal
    previous_balance: Optional[Decimal] = None
    change_amount: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ErrorItem:
    """Represents an error log item"""
    message: str
    error_type: str
    context: Dict[str, Any]
    timestamp: Optional[datetime] = None
    severity: str = "error"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReconciliationItem:
    """Represents a reconciliation log item"""
    reconciliation_type: str
    status: str
    details: Dict[str, Any]
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class EnhancedAccountingLogger:
    """
    Enhanced accounting logger with structured logging for flow vs balance items.
    
    This logger provides:
    - Structured logging for flow items (transactions, cash flows)
    - Structured logging for balance items (account balances, positions)
    - Error logging with detailed context
    - Reconciliation logging
    - Export capabilities
    - Statistics reporting
    """
    
    def __init__(self, log_level: int = logging.INFO, log_file: Optional[str] = None):
        """
        Initialize the enhanced accounting logger.
        
        Args:
            log_level: Logging level (default: INFO)
            log_file: Optional log file path
        """
        self.flow_items: List[FlowItem] = []
        self.balance_items: List[BalanceItem] = []
        self.error_items: List[ErrorItem] = []
        self.reconciliation_items: List[ReconciliationItem] = []
        
        # Setup standard logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_flow_item(self, category: FlowItemCategory, amount: Decimal, 
                     from_account: str, to_account: str, description: str,
                     transaction_id: str, reference_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a flow item (transaction, cash flow, payment).
        
        Args:
            category: Flow item category
            amount: Transaction amount
            from_account: Source account
            to_account: Destination account
            description: Transaction description
            transaction_id: Unique transaction ID
            reference_id: Optional reference ID
            metadata: Optional metadata dictionary
        """
        flow_item = FlowItem(
            category=category,
            amount=amount,
            from_account=from_account,
            to_account=to_account,
            description=description,
            transaction_id=transaction_id,
            reference_id=reference_id,
            metadata=metadata or {}
        )
        
        self.flow_items.append(flow_item)
        
        # Log to standard logger
        self.logger.info(
            f"Flow item logged: {category.value} - {amount} "
            f"({from_account} -> {to_account}) - {description}"
        )
    
    def log_balance_item(self, category: BalanceItemCategory, account_id: str,
                        balance: Decimal, previous_balance: Optional[Decimal] = None,
                        change_amount: Optional[Decimal] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a balance item (account balance, position).
        
        Args:
            category: Balance item category
            account_id: Account identifier
            balance: Current balance
            previous_balance: Previous balance (optional)
            change_amount: Change amount (optional)
            metadata: Optional metadata dictionary
        """
        balance_item = BalanceItem(
            category=category,
            account_id=account_id,
            balance=balance,
            previous_balance=previous_balance,
            change_amount=change_amount,
            metadata=metadata or {}
        )
        
        self.balance_items.append(balance_item)
        
        # Log to standard logger
        change_info = ""
        if change_amount is not None:
            change_info = f" (change: {change_amount})"
        
        self.logger.info(
            f"Balance item logged: {category.value} - {account_id} = {balance}{change_info}"
        )
    
    def log_error(self, message: str, context: Dict[str, Any], 
                  error_type: str = "general", severity: str = "error") -> None:
        """
        Log an error with detailed context.
        
        Args:
            message: Error message
            context: Error context dictionary
            error_type: Type of error
            severity: Error severity level
        """
        error_item = ErrorItem(
            message=message,
            error_type=error_type,
            context=context,
            severity=severity
        )
        
        self.error_items.append(error_item)
        
        # Log to standard logger
        self.logger.error(f"Error logged: {message} - {error_type} - {context}")
    
    def log_reconciliation(self, reconciliation_type: str, status: str,
                          details: Dict[str, Any], 
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a reconciliation event.
        
        Args:
            reconciliation_type: Type of reconciliation
            status: Reconciliation status
            details: Reconciliation details
            metadata: Optional metadata dictionary
        """
        reconciliation_item = ReconciliationItem(
            reconciliation_type=reconciliation_type,
            status=status,
            details=details,
            metadata=metadata or {}
        )
        
        self.reconciliation_items.append(reconciliation_item)
        
        # Log to standard logger
        self.logger.info(
            f"Reconciliation logged: {reconciliation_type} - {status} - {details}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged items.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            "total_flow_items": len(self.flow_items),
            "total_balance_items": len(self.balance_items),
            "total_error_items": len(self.error_items),
            "total_reconciliation_items": len(self.reconciliation_items),
            "flow_categories": {},
            "balance_categories": {},
            "error_types": {},
            "total_flow_amount": Decimal('0'),
            "total_balance_amount": Decimal('0')
        }
        
        # Flow item statistics
        for item in self.flow_items:
            category = item.category.value
            stats["flow_categories"][category] = stats["flow_categories"].get(category, 0) + 1
            stats["total_flow_amount"] += item.amount
        
        # Balance item statistics
        for item in self.balance_items:
            category = item.category.value
            stats["balance_categories"][category] = stats["balance_categories"].get(category, 0) + 1
            stats["total_balance_amount"] += item.balance
        
        # Error statistics
        for item in self.error_items:
            error_type = item.error_type
            stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
        
        return stats
    
    def export_logs(self, file_path: str, format: str = "json") -> None:
        """
        Export logs to a file.
        
        Args:
            file_path: Output file path
            format: Export format ("json" or "csv")
        """
        if format.lower() == "json":
            self._export_json(file_path)
        elif format.lower() == "csv":
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, file_path: str) -> None:
        """Export logs to JSON format."""
        export_data = {
            "flow_items": [asdict(item) for item in self.flow_items],
            "balance_items": [asdict(item) for item in self.balance_items],
            "error_items": [asdict(item) for item in self.error_items],
            "reconciliation_items": [asdict(item) for item in self.reconciliation_items],
            "statistics": self.get_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Convert Decimal objects to strings for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            else:
                return obj
        
        export_data = convert_decimals(export_data)
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Logs exported to JSON: {file_path}")
    
    def _export_csv(self, file_path: str) -> None:
        """Export logs to CSV format."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write flow items
            writer.writerow(["Type", "Category", "Amount", "From Account", "To Account", 
                           "Description", "Transaction ID", "Reference ID", "Timestamp"])
            for item in self.flow_items:
                writer.writerow([
                    "flow", item.category.value, str(item.amount), item.from_account,
                    item.to_account, item.description, item.transaction_id,
                    item.reference_id or "", item.timestamp.isoformat()
                ])
            
            # Write balance items
            writer.writerow([])  # Empty row for separation
            writer.writerow(["Type", "Category", "Account ID", "Balance", "Previous Balance",
                           "Change Amount", "Timestamp"])
            for item in self.balance_items:
                writer.writerow([
                    "balance", item.category.value, item.account_id, str(item.balance),
                    str(item.previous_balance) if item.previous_balance else "",
                    str(item.change_amount) if item.change_amount else "",
                    item.timestamp.isoformat()
                ])
        
        self.logger.info(f"Logs exported to CSV: {file_path}")
    
    def clear_logs(self) -> None:
        """Clear all logged items."""
        self.flow_items.clear()
        self.balance_items.clear()
        self.error_items.clear()
        self.reconciliation_items.clear()
        self.logger.info("All logs cleared")
    
    def get_recent_items(self, item_type: LogItemType, limit: int = 10) -> List[Any]:
        """
        Get recent items of a specific type.
        
        Args:
            item_type: Type of items to retrieve
            limit: Maximum number of items to return
            
        Returns:
            List of recent items
        """
        if item_type == LogItemType.FLOW:
            return self.flow_items[-limit:]
        elif item_type == LogItemType.BALANCE:
            return self.balance_items[-limit:]
        elif item_type == LogItemType.ERROR:
            return self.error_items[-limit:]
        elif item_type == LogItemType.RECONCILIATION:
            return self.reconciliation_items[-limit:]
        else:
            raise ValueError(f"Unknown item type: {item_type}") 