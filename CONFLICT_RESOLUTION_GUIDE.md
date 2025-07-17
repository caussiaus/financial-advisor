# Conflict Resolution Guide for Enhanced Accounting Logging

## Overview
This guide helps resolve conflicts when pulling the enhanced accounting logging changes (commit `16d29f7`) on other servers.

## Files Modified/Added
- `src/enhanced_accounting_logger.py` (NEW)
- `src/accounting_reconciliation.py` (MODIFIED)
- `demos/demo_enhanced_accounting_logging.py` (NEW)

## Conflict Resolution Strategy

### 1. Safe Pull Strategy
```bash
# Before pulling, backup current state
git stash push -m "backup_before_enhanced_logging_pull"

# Fetch latest changes
git fetch origin

# Check what conflicts might occur
git diff origin/pdf-extraction-enhancement --name-only

# Pull with conflict resolution
git pull origin pdf-extraction-enhancement
```

### 2. Manual Conflict Resolution

#### If `src/accounting_reconciliation.py` has conflicts:

**Local changes to preserve:**
- Any custom account configurations
- Any custom payment constraints
- Any business logic specific to your environment

**Remote changes to accept:**
- Enhanced logging integration
- New helper methods (`_determine_flow_category`, `_determine_balance_category`)
- New logging methods (`log_reconciliation_summary`, `export_enhanced_logs`)

**Resolution steps:**
1. Look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
2. In the `__init__` method, ensure both local and remote changes are preserved:
   ```python
   def __init__(self):
       # Your local customizations
       self.accounts = {}
       self.transactions = []
       self.constraints = {}
       self.pending_transactions = []
       self.logger = self._setup_logging()
       
       # NEW: Enhanced logging integration
       self.enhanced_logger = EnhancedAccountingLogger()
       
       # Your local customizations
       self._initialize_default_accounts()
   ```

3. In the `_execute_transaction` method, preserve both local logic and enhanced logging:
   ```python
   def _execute_transaction(self, transaction: Transaction) -> Tuple[bool, str]:
       try:
           # Store previous balances for logging
           previous_debit_balance = self.accounts[transaction.debit_account].balance
           previous_credit_balance = self.accounts[transaction.credit_account].balance
           
           # YOUR LOCAL TRANSACTION LOGIC HERE
           # ... existing transaction processing ...
           
           # NEW: Enhanced logging for flow items
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
           
           # NEW: Enhanced logging for balance changes
           self.enhanced_logger.log_balance_item(
               category=self._determine_balance_category(debit_account.account_type),
               account_id=transaction.debit_account,
               balance=debit_account.balance,
               previous_balance=previous_debit_balance,
               change_amount=debit_account.balance - previous_debit_balance,
               metadata={'transaction_id': transaction.transaction_id}
           )
           
           # YOUR LOCAL SUCCESS LOGIC HERE
           return True, transaction.transaction_id
           
       except Exception as e:
           # NEW: Enhanced error logging
           self.enhanced_logger.log_error(f"Transaction execution failed: {str(e)}", {
               'transaction_id': transaction.transaction_id,
               'debit_account': transaction.debit_account,
               'credit_account': transaction.credit_account,
               'amount': str(transaction.amount)
           })
           return False, f"Transaction failed: {str(e)}"
   ```

#### If `src/enhanced_accounting_logger.py` conflicts:
This is a new file, so conflicts are unlikely. If it exists locally:
1. Backup your local version: `cp src/enhanced_accounting_logger.py src/enhanced_accounting_logger.py.backup`
2. Accept the remote version: `git checkout origin/pdf-extraction-enhancement -- src/enhanced_accounting_logger.py`
3. Manually merge any customizations from your backup

#### If `demos/demo_enhanced_accounting_logging.py` conflicts:
This is a new demo file. If you have a local version:
1. Compare the files: `diff demos/demo_enhanced_accounting_logging.py origin/pdf-extraction-enhancement:demos/demo_enhanced_accounting_logging.py`
2. Decide which version to keep or merge manually

### 3. Import Path Conflicts

If you get import errors after merging:

**Check these import statements in `src/accounting_reconciliation.py`:**
```python
# Should be:
from .enhanced_accounting_logger import (
    EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory, LogItemType
)

# And ensure this import exists:
from typing import List, Dict, Tuple, Optional, Any
```

### 4. Testing After Resolution

After resolving conflicts, test the integration:

```bash
# Test the enhanced logging
python demos/demo_enhanced_accounting_logging.py

# Check for import errors
python -c "from src.accounting_reconciliation import AccountingReconciliationEngine; print('✅ Import successful')"

# Run a quick test
python -c "
from src.accounting_reconciliation import AccountingReconciliationEngine
from decimal import Decimal
accounting = AccountingReconciliationEngine()
accounting.set_account_balance('cash_checking', Decimal('10000'))
success, result = accounting.execute_payment('cash_checking', 'living_expenses', Decimal('1000'), 'Test payment')
print(f'Test payment: {success} - {result}')
"
```

### 5. Rollback Strategy

If conflicts become too complex:

```bash
# Abort the merge
git merge --abort

# Restore your backup
git stash pop

# Try a different approach: cherry-pick specific commits
git cherry-pick 16d29f7 --no-commit
# Then manually resolve conflicts in the working directory
```

### 6. Verification Checklist

After resolving conflicts, verify:

- [ ] `src/enhanced_accounting_logger.py` exists and imports correctly
- [ ] `src/accounting_reconciliation.py` has the enhanced logging integration
- [ ] `demos/demo_enhanced_accounting_logging.py` runs without errors
- [ ] No import errors when importing `AccountingReconciliationEngine`
- [ ] Enhanced logging works: transactions are logged with flow/balance categories
- [ ] Export functionality works: logs can be exported to JSON files

### 7. Common Issues and Solutions

**Issue:** `ModuleNotFoundError: No module named 'enhanced_accounting_logger'`
**Solution:** Check import path in `src/accounting_reconciliation.py` - should be `from .enhanced_accounting_logger import`

**Issue:** `TypeError: non-default argument follows default argument`
**Solution:** The dataclass field order has been fixed in the remote version. Accept the remote version.

**Issue:** `NameError: name 'Any' is not defined`
**Solution:** Ensure `from typing import List, Dict, Tuple, Optional, Any` is in the imports.

## Emergency Rollback

If everything goes wrong:

```bash
# Hard reset to before the problematic merge
git reset --hard HEAD~1

# Or reset to a known good state
git reset --hard <known_good_commit_hash>

# Restore your stashed changes
git stash pop
```

## Support

If you encounter issues not covered in this guide:
1. Check the git log for the exact commit: `git log --oneline -5`
2. Compare with remote: `git diff origin/pdf-extraction-enhancement`
3. Create a backup branch: `git checkout -b backup_before_enhanced_logging`
4. Document the specific error and share for assistance 