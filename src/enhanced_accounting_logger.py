"""
Enhanced Accounting Logger

<<<<<<< HEAD
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
=======
Provides detailed logging for flow items vs balance items with clear categorization
and structured output for better financial tracking and debugging.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import os
from pathlib import Path


class LogItemType(Enum):
    """Types of accounting items for logging"""
    FLOW_ITEM = "flow_item"      # Transactions, cash flows, payments
    BALANCE_ITEM = "balance_item" # Account balances, positions, net worth
    RECONCILIATION = "reconciliation"  # Balance reconciliations
    VALIDATION = "validation"     # Payment validations
    ERROR = "error"              # Errors and exceptions


class FlowItemCategory(Enum):
    """Categories for flow items"""
>>>>>>> c8b4580f8e727ce4557c1196fdf5337c7741ca85
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    INVESTMENT = "investment"
<<<<<<< HEAD
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
=======
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    LOAN = "loan"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"


class BalanceItemCategory(Enum):
    """Categories for balance items"""
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"
    NET_WORTH = "net_worth"
    LIQUIDITY = "liquidity"
    POSITION = "position"


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
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
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
            'discrepancies': self.discrepancies,
            'metadata': self.metadata
        }
>>>>>>> c8b4580f8e727ce4557c1196fdf5337c7741ca85


class EnhancedAccountingLogger:
    """
<<<<<<< HEAD
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
=======
    Enhanced accounting logger that provides detailed logging for flow items vs balance items
    """
    
    def __init__(self, log_directory: str = "logs", log_level: str = "INFO"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Log storage
        self.flow_items: List[FlowItemLog] = []
        self.balance_items: List[BalanceItemLog] = []
        self.reconciliation_items: List[ReconciliationLog] = []
        
        # Statistics
        self.stats = {
            'flow_items_logged': 0,
            'balance_items_logged': 0,
            'reconciliation_items_logged': 0,
            'errors_logged': 0
        }
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('enhanced_accounting_logger')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            self.log_directory / "accounting_detailed.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_flow_item(self, 
                     category: FlowItemCategory,
                     amount: Union[float, Decimal],
                     from_account: Optional[str] = None,
                     to_account: Optional[str] = None,
                     description: str = "",
                     transaction_id: Optional[str] = None,
                     reference_id: Optional[str] = None,
                     confidence: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> FlowItemLog:
        """
        Log a flow item (transaction, cash flow, payment)
        
        Args:
            category: Type of flow item
            amount: Amount of the flow
            from_account: Source account
            to_account: Destination account
            description: Description of the flow
            transaction_id: Transaction identifier
            reference_id: Reference identifier
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Created flow item log entry
        """
        # Convert amount to Decimal
        if isinstance(amount, float):
            amount = Decimal(str(amount))
        
        flow_item = FlowItemLog(
            timestamp=datetime.now(),
>>>>>>> c8b4580f8e727ce4557c1196fdf5337c7741ca85
            category=category,
            amount=amount,
            from_account=from_account,
            to_account=to_account,
            description=description,
            transaction_id=transaction_id,
            reference_id=reference_id,
<<<<<<< HEAD
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
=======
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Store log entry
        self.flow_items.append(flow_item)
        self.stats['flow_items_logged'] += 1
        
        # Log to console with structured format
        self.logger.info(
            f"FLOW_ITEM | {category.value.upper()} | "
            f"${amount:,.2f} | {from_account or 'N/A'} → {to_account or 'N/A'} | "
            f"{description}"
        )
        
        return flow_item
    
    def log_balance_item(self,
                        category: BalanceItemCategory,
                        account_id: str,
                        balance: Union[float, Decimal],
                        previous_balance: Optional[Union[float, Decimal]] = None,
                        change_amount: Optional[Union[float, Decimal]] = None,
                        currency: str = "USD",
                        metadata: Optional[Dict[str, Any]] = None) -> BalanceItemLog:
        """
        Log a balance item (account balance, position)
        
        Args:
            category: Type of balance item
            account_id: Account identifier
            balance: Current balance
            previous_balance: Previous balance (for change tracking)
            change_amount: Amount of change
            currency: Currency code
            metadata: Additional metadata
            
        Returns:
            Created balance item log entry
        """
        # Convert amounts to Decimal
        if isinstance(balance, float):
            balance = Decimal(str(balance))
        if previous_balance is not None and isinstance(previous_balance, float):
            previous_balance = Decimal(str(previous_balance))
        if change_amount is not None and isinstance(change_amount, float):
            change_amount = Decimal(str(change_amount))
        
        balance_item = BalanceItemLog(
            timestamp=datetime.now(),
>>>>>>> c8b4580f8e727ce4557c1196fdf5337c7741ca85
            category=category,
            account_id=account_id,
            balance=balance,
            previous_balance=previous_balance,
            change_amount=change_amount,
<<<<<<< HEAD
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
=======
            currency=currency,
            metadata=metadata or {}
        )
        
        # Store log entry
        self.balance_items.append(balance_item)
        self.stats['balance_items_logged'] += 1
        
        # Log to console with structured format
        change_str = f" ({change_amount:+,.2f})" if change_amount else ""
        self.logger.info(
            f"BALANCE_ITEM | {category.value.upper()} | "
            f"{account_id} | ${balance:,.2f}{change_str} | {currency}"
        )
        
        return balance_item
    
    def log_reconciliation(self,
                          reconciliation_type: str,
                          accounts_involved: List[str],
                          total_assets: Union[float, Decimal],
                          total_liabilities: Union[float, Decimal],
                          net_worth: Union[float, Decimal],
                          discrepancies: Optional[List[Dict[str, Any]]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> ReconciliationLog:
        """
        Log a reconciliation event
        
        Args:
            reconciliation_type: Type of reconciliation
            accounts_involved: List of accounts involved
            total_assets: Total assets
            total_liabilities: Total liabilities
            net_worth: Net worth
            discrepancies: List of discrepancies found
            metadata: Additional metadata
            
        Returns:
            Created reconciliation log entry
        """
        # Convert amounts to Decimal
        if isinstance(total_assets, float):
            total_assets = Decimal(str(total_assets))
        if isinstance(total_liabilities, float):
            total_liabilities = Decimal(str(total_liabilities))
        if isinstance(net_worth, float):
            net_worth = Decimal(str(net_worth))
        
        reconciliation_item = ReconciliationLog(
            timestamp=datetime.now(),
            reconciliation_type=reconciliation_type,
            accounts_involved=accounts_involved,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            net_worth=net_worth,
            discrepancies=discrepancies or [],
            metadata=metadata or {}
        )
        
        # Store log entry
        self.reconciliation_items.append(reconciliation_item)
        self.stats['reconciliation_items_logged'] += 1
        
        # Log to console with structured format
        discrepancy_count = len(reconciliation_item.discrepancies)
        self.logger.info(
            f"RECONCILIATION | {reconciliation_type.upper()} | "
            f"Assets: ${total_assets:,.2f} | Liabilities: ${total_liabilities:,.2f} | "
            f"Net Worth: ${net_worth:,.2f} | Discrepancies: {discrepancy_count}"
        )
        
        return reconciliation_item
    
    def log_error(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Log an error with context"""
        self.stats['errors_logged'] += 1
        context_str = f" | Context: {json.dumps(context)}" if context else ""
        self.logger.error(f"ACCOUNTING_ERROR | {error_message}{context_str}")
    
    def export_logs(self, filepath: str, log_type: str = "all"):
        """
        Export logs to JSON file
        
        Args:
            filepath: Path to export file
            log_type: Type of logs to export ("flow", "balance", "reconciliation", "all")
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'logs': {}
        }
        
        if log_type in ["flow", "all"]:
            export_data['logs']['flow_items'] = [item.to_dict() for item in self.flow_items]
        
        if log_type in ["balance", "all"]:
            export_data['logs']['balance_items'] = [item.to_dict() for item in self.balance_items]
        
        if log_type in ["reconciliation", "all"]:
            export_data['logs']['reconciliation_items'] = [item.to_dict() for item in self.reconciliation_items]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Logs exported to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'total_logs': len(self.flow_items) + len(self.balance_items) + len(self.reconciliation_items),
            'flow_items': len(self.flow_items),
            'balance_items': len(self.balance_items),
            'reconciliation_items': len(self.reconciliation_items),
            'errors': self.stats['errors_logged'],
            'statistics': self.stats
        }
    
    def clear_logs(self):
        """Clear all stored logs"""
        self.flow_items.clear()
        self.balance_items.clear()
        self.reconciliation_items.clear()
        self.stats = {
            'flow_items_logged': 0,
            'balance_items_logged': 0,
            'reconciliation_items_logged': 0,
            'errors_logged': 0
        }
        self.logger.info("All logs cleared")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of logged items"""
        # Flow item summary
        flow_summary = {}
        for item in self.flow_items:
            category = item.category.value
            if category not in flow_summary:
                flow_summary[category] = {
                    'count': 0,
                    'total_amount': Decimal('0'),
                    'avg_amount': Decimal('0')
                }
            flow_summary[category]['count'] += 1
            flow_summary[category]['total_amount'] += item.amount
        
        # Calculate averages
        for category in flow_summary:
            if flow_summary[category]['count'] > 0:
                flow_summary[category]['avg_amount'] = (
                    flow_summary[category]['total_amount'] / flow_summary[category]['count']
                )
        
        # Balance item summary
        balance_summary = {}
        for item in self.balance_items:
            category = item.category.value
            if category not in balance_summary:
                balance_summary[category] = {
                    'count': 0,
                    'total_balance': Decimal('0'),
                    'avg_balance': Decimal('0')
                }
            balance_summary[category]['count'] += 1
            balance_summary[category]['total_balance'] += item.balance
        
        # Calculate averages
        for category in balance_summary:
            if balance_summary[category]['count'] > 0:
                balance_summary[category]['avg_balance'] = (
                    balance_summary[category]['total_balance'] / balance_summary[category]['count']
                )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'flow_summary': {k: {**v, 'total_amount': str(v['total_amount']), 'avg_amount': str(v['avg_amount'])} 
                           for k, v in flow_summary.items()},
            'balance_summary': {k: {**v, 'total_balance': str(v['total_balance']), 'avg_balance': str(v['avg_balance'])} 
                              for k, v in balance_summary.items()}
        } 
>>>>>>> c8b4580f8e727ce4557c1196fdf5337c7741ca85
