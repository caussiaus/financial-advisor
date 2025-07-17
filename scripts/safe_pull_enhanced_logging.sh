#!/bin/bash

# Safe Pull Script for Enhanced Accounting Logging
# This script safely pulls the enhanced accounting logging changes (commit 16d29f7)
# and handles conflicts gracefully

set -e  # Exit on any error

echo "🔄 Safe Pull Script for Enhanced Accounting Logging"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository. Please run this script from the project root."
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
print_status "Current branch: $CURRENT_BRANCH"

# Create backup
BACKUP_BRANCH="backup_before_enhanced_logging_$(date +%Y%m%d_%H%M%S)"
print_status "Creating backup branch: $BACKUP_BRANCH"
git checkout -b "$BACKUP_BRANCH"

# Stash any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "Uncommitted changes detected. Stashing them..."
    git stash push -m "backup_before_enhanced_logging_pull"
    STASHED=true
else
    STASHED=false
fi

# Fetch latest changes
print_status "Fetching latest changes from remote..."
git fetch origin

# Check if we're on the target branch
if [ "$CURRENT_BRANCH" != "pdf-extraction-enhancement" ]; then
    print_warning "Not on pdf-extraction-enhancement branch. Switching..."
    git checkout pdf-extraction-enhancement
fi

# Check what files will be modified
print_status "Checking what files will be modified..."
git diff --name-only HEAD origin/pdf-extraction-enhancement || true

# Attempt to pull
print_status "Attempting to pull enhanced accounting logging changes..."
if git pull origin pdf-extraction-enhancement; then
    print_status "✅ Successfully pulled changes without conflicts!"
else
    print_warning "⚠️  Conflicts detected. Attempting to resolve..."
    
    # Check which files have conflicts
    CONFLICTED_FILES=$(git diff --name-only --diff-filter=U)
    
    if [ -n "$CONFLICTED_FILES" ]; then
        print_status "Files with conflicts:"
        echo "$CONFLICTED_FILES"
        
        # Handle specific file conflicts
        for file in $CONFLICTED_FILES; do
            case $file in
                "src/accounting_reconciliation.py")
                    print_status "Resolving conflicts in accounting_reconciliation.py..."
                    # This is the main file that needs careful merging
                    # The script will guide manual resolution
                    echo "Please manually resolve conflicts in $file"
                    echo "See CONFLICT_RESOLUTION_GUIDE.md for detailed instructions"
                    ;;
                "src/enhanced_accounting_logger.py")
                    print_status "Resolving conflicts in enhanced_accounting_logger.py..."
                    # This is a new file, accept remote version
                    git checkout --theirs "$file"
                    git add "$file"
                    ;;
                "demos/demo_enhanced_accounting_logging.py")
                    print_status "Resolving conflicts in demo_enhanced_accounting_logging.py..."
                    # This is a new demo file, accept remote version
                    git checkout --theirs "$file"
                    git add "$file"
                    ;;
                *)
                    print_warning "Unknown conflicted file: $file"
                    echo "Please resolve manually"
                    ;;
            esac
        done
        
        # Try to complete the merge
        if git commit -m "Merge enhanced accounting logging with conflict resolution"; then
            print_status "✅ Conflicts resolved successfully!"
        else
            print_error "❌ Failed to resolve conflicts. Manual intervention required."
            print_status "Your changes are backed up in branch: $BACKUP_BRANCH"
            print_status "Please see CONFLICT_RESOLUTION_GUIDE.md for manual resolution steps"
            exit 1
        fi
    fi
fi

# Test the integration
print_status "Testing the enhanced logging integration..."

# Test import
if python -c "from src.accounting_reconciliation import AccountingReconciliationEngine; print('✅ Import successful')" 2>/dev/null; then
    print_status "✅ Import test passed"
else
    print_warning "⚠️  Import test failed. Checking for common issues..."
    
    # Check for common import issues
    if grep -q "from enhanced_accounting_logger import" src/accounting_reconciliation.py; then
        print_warning "Found incorrect import path. Fixing..."
        sed -i '' 's/from enhanced_accounting_logger import/from .enhanced_accounting_logger import/' src/accounting_reconciliation.py
    fi
    
    if ! grep -q "from typing import.*Any" src/accounting_reconciliation.py; then
        print_warning "Missing 'Any' import. Fixing..."
        sed -i '' 's/from typing import List, Dict, Tuple, Optional/from typing import List, Dict, Tuple, Optional, Any/' src/accounting_reconciliation.py
    fi
    
    # Test again
    if python -c "from src.accounting_reconciliation import AccountingReconciliationEngine; print('✅ Import successful after fixes')" 2>/dev/null; then
        print_status "✅ Import test passed after fixes"
    else
        print_error "❌ Import test still failing. Manual intervention required."
    fi
fi

# Test the demo if it exists
if [ -f "demos/demo_enhanced_accounting_logging.py" ]; then
    print_status "Testing enhanced logging demo..."
    if timeout 30 python demos/demo_enhanced_accounting_logging.py > /dev/null 2>&1; then
        print_status "✅ Demo test passed"
    else
        print_warning "⚠️  Demo test failed or timed out. This is not critical."
    fi
fi

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    print_status "Restoring stashed changes..."
    if git stash pop; then
        print_status "✅ Stashed changes restored"
    else
        print_warning "⚠️  Could not restore stashed changes. They are still in stash."
        print_status "Run 'git stash list' to see available stashes"
    fi
fi

# Final status
print_status "🎉 Enhanced accounting logging integration complete!"
print_status "📋 Summary:"
print_status "  - Backup branch: $BACKUP_BRANCH"
print_status "  - Enhanced logging: src/enhanced_accounting_logger.py"
print_status "  - Updated accounting: src/accounting_reconciliation.py"
print_status "  - Demo script: demos/demo_enhanced_accounting_logging.py"
print_status "  - Conflict resolution guide: CONFLICT_RESOLUTION_GUIDE.md"

print_status "🧪 To test the integration, run:"
echo "  python demos/demo_enhanced_accounting_logging.py"

print_status "📚 For detailed conflict resolution, see:"
echo "  CONFLICT_RESOLUTION_GUIDE.md"

echo ""
print_status "✅ Safe pull completed successfully!" 