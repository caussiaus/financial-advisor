#!/usr/bin/env python
"""
Accounting Reconciliation System
Author: Claude 2025-07-16

Ensures that at each node in the financial path, income, expenses, and savings
reconcile exactly like an accounting equation: Income = Expenses + Savings.

Handles different quality of life variations where you could be:
- Working hard to save for goals (charity, education, RRSP, child TFSA)
- Living frugally but working less (high savings, low expenses, low income)
- Balanced approach with moderate work/spending/saving

Key Features:
- Strict accounting equation enforcement: ΣIncome = ΣExpenses + ΣSavings
- Multi-account reconciliation (checking, savings, investment, RRSP, TFSA)
- Cash flow timing and period matching
- Quality of life impact assessment for each reconciliation scenario
- Automatic adjustment mechanisms when equation doesn't balance
- Goal-based allocation reconciliation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AccountEntry:
    """Individual accounting entry"""
    account_type: str  # 'income', 'expense', 'savings', 'investment'
    category: str  # Specific category (salary, rent, rrsp, etc.)
    amount: Decimal
    date: datetime
    description: str
    quality_impact: float  # Impact on quality of life (-1 to 1)
    stress_impact: float   # Impact on stress level (0 to 1)
    goal_allocation: Dict[str, Decimal] = field(default_factory=dict)

@dataclass
class AccountBalance:
    """Balance state for a specific account"""
    account_name: str
    opening_balance: Decimal
    closing_balance: Decimal
    period_inflows: Decimal
    period_outflows: Decimal
    net_change: Decimal
    reconciliation_difference: Decimal

@dataclass
class ReconciliationReport:
    """Report showing reconciliation status"""
    period_start: datetime
    period_end: datetime
    total_income: Decimal
    total_expenses: Decimal
    total_savings: Decimal
    accounting_difference: Decimal
    is_balanced: bool
    account_balances: Dict[str, AccountBalance]
    quality_of_life_score: float
    stress_level: float
    goal_achievement_rates: Dict[str, float]
    recommendations: List[str]

class AccountingReconciliationEngine:
    """Main engine for accounting reconciliation and balancing"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize reconciliation engine
        
        Args:
            tolerance: Maximum acceptable difference in accounting equation (dollars)
        """
        self.tolerance = Decimal(str(tolerance))
        self.accounts = {
            'checking': Decimal('0'),
            'savings': Decimal('0'),
            'investment': Decimal('0'),
            'rrsp': Decimal('0'),
            'tfsa': Decimal('0'),
            'child_tfsa': Decimal('0'),
            'emergency_fund': Decimal('0')
        }
        self.entries = []
        self.reconciliation_history = []
        
    def add_entry(self, entry: AccountEntry) -> None:
        """Add an accounting entry"""
        self.entries.append(entry)
        logger.debug(f"Added entry: {entry.category} ${entry.amount}")
    
    def reconcile_period(self, start_date: datetime, end_date: datetime,
                        lifestyle_config: Dict[str, Any]) -> ReconciliationReport:
        """Reconcile accounts for a specific period"""
        
        logger.info(f"Reconciling period {start_date.date()} to {end_date.date()}")
        
        # Filter entries for this period
        period_entries = [
            entry for entry in self.entries
            if start_date <= entry.date <= end_date
        ]
        
        # Calculate totals by type
        income_entries = [e for e in period_entries if e.account_type == 'income']
        expense_entries = [e for e in period_entries if e.account_type == 'expense']
        savings_entries = [e for e in period_entries if e.account_type == 'savings']
        investment_entries = [e for e in period_entries if e.account_type == 'investment']
        
        total_income = sum(entry.amount for entry in income_entries)
        total_expenses = sum(entry.amount for entry in expense_entries)
        total_savings = sum(entry.amount for entry in savings_entries)
        total_investments = sum(entry.amount for entry in investment_entries)
        
        # Total outflows (expenses + savings + investments)
        total_outflows = total_expenses + total_savings + total_investments
        
        # Calculate accounting difference
        accounting_difference = total_income - total_outflows
        is_balanced = abs(accounting_difference) <= self.tolerance
        
        # Update account balances
        account_balances = self._update_account_balances(period_entries)
        
        # Calculate quality of life and stress
        quality_score = self._calculate_quality_of_life(period_entries, lifestyle_config)
        stress_level = self._calculate_stress_level(period_entries, lifestyle_config, is_balanced)
        
        # Calculate goal achievement rates
        goal_rates = self._calculate_goal_achievement(period_entries, lifestyle_config)
        
        # Generate recommendations if not balanced
        recommendations = []
        if not is_balanced:
            recommendations = self._generate_balancing_recommendations(
                accounting_difference, lifestyle_config, period_entries
            )
        
        report = ReconciliationReport(
            period_start=start_date,
            period_end=end_date,
            total_income=total_income,
            total_expenses=total_expenses,
            total_savings=total_savings + total_investments,
            accounting_difference=accounting_difference,
            is_balanced=is_balanced,
            account_balances=account_balances,
            quality_of_life_score=quality_score,
            stress_level=stress_level,
            goal_achievement_rates=goal_rates,
            recommendations=recommendations
        )
        
        self.reconciliation_history.append(report)
        return report
    
    def _update_account_balances(self, entries: List[AccountEntry]) -> Dict[str, AccountBalance]:
        """Update account balances based on entries"""
        account_changes = {account: Decimal('0') for account in self.accounts}
        
        # Process each entry
        for entry in entries:
            if entry.account_type == 'income':
                # Income goes to checking by default
                account_changes['checking'] += entry.amount
                
            elif entry.account_type == 'expense':
                # Expenses come from checking
                account_changes['checking'] -= entry.amount
                
            elif entry.account_type == 'savings':
                # Determine target account based on category
                if 'rrsp' in entry.category.lower():
                    account_changes['checking'] -= entry.amount
                    account_changes['rrsp'] += entry.amount
                elif 'tfsa' in entry.category.lower():
                    if 'child' in entry.category.lower():
                        account_changes['checking'] -= entry.amount
                        account_changes['child_tfsa'] += entry.amount
                    else:
                        account_changes['checking'] -= entry.amount
                        account_changes['tfsa'] += entry.amount
                elif 'emergency' in entry.category.lower():
                    account_changes['checking'] -= entry.amount
                    account_changes['emergency_fund'] += entry.amount
                else:
                    # Regular savings
                    account_changes['checking'] -= entry.amount
                    account_changes['savings'] += entry.amount
                    
            elif entry.account_type == 'investment':
                # Investments come from checking
                account_changes['checking'] -= entry.amount
                account_changes['investment'] += entry.amount
        
        # Create balance reports
        balances = {}
        for account_name, change in account_changes.items():
            opening_balance = self.accounts[account_name]
            closing_balance = opening_balance + change
            
            inflows = max(Decimal('0'), change)
            outflows = max(Decimal('0'), -change)
            
            balance = AccountBalance(
                account_name=account_name,
                opening_balance=opening_balance,
                closing_balance=closing_balance,
                period_inflows=inflows,
                period_outflows=outflows,
                net_change=change,
                reconciliation_difference=Decimal('0')  # Would calculate if needed
            )
            balances[account_name] = balance
            
            # Update stored balance
            self.accounts[account_name] = closing_balance
        
        return balances
    
    def _calculate_quality_of_life(self, entries: List[AccountEntry], 
                                 lifestyle_config: Dict[str, Any]) -> float:
        """Calculate quality of life score based on entries and lifestyle"""
        quality_score = 0.5  # Base quality
        
        # Weight quality impacts from entries
        total_amount = sum(abs(entry.amount) for entry in entries)
        if total_amount > 0:
            weighted_quality = sum(
                entry.quality_impact * float(abs(entry.amount)) / float(total_amount)
                for entry in entries
            )
            quality_score += weighted_quality * 0.3
        
        # Lifestyle configuration impacts
        work_intensity = lifestyle_config.get('work_intensity', 0.7)
        spending_level = lifestyle_config.get('spending_level', 0.8)
        savings_rate = lifestyle_config.get('savings_rate', 0.15)
        
        # Work-life balance
        if work_intensity < 0.7:
            quality_score += 0.15  # Better work-life balance
        else:
            quality_score -= (work_intensity - 0.7) * 0.2  # Overwork penalty
        
        # Spending comfort
        quality_score += (spending_level - 0.6) * 0.25
        
        # Financial security from savings
        quality_score += min(0.2, savings_rate * 0.5)
        
        # Goal achievement quality boost
        charity_entries = [e for e in entries if 'charity' in e.category.lower()]
        education_entries = [e for e in entries if 'education' in e.category.lower()]
        
        if charity_entries:
            charity_amount = sum(float(e.amount) for e in charity_entries)
            quality_score += min(0.1, charity_amount / 10000 * 0.1)  # Fulfillment from giving
        
        if education_entries:
            quality_score += 0.05  # Growth from learning
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_stress_level(self, entries: List[AccountEntry], 
                              lifestyle_config: Dict[str, Any], 
                              is_balanced: bool) -> float:
        """Calculate financial stress level"""
        stress = 0.0
        
        # Accounting imbalance stress
        if not is_balanced:
            stress += 0.3  # Major stress from financial inconsistency
        
        # Weight stress impacts from entries
        total_amount = sum(abs(entry.amount) for entry in entries)
        if total_amount > 0:
            weighted_stress = sum(
                entry.stress_impact * float(abs(entry.amount)) / float(total_amount)
                for entry in entries
            )
            stress += weighted_stress * 0.2
        
        # Lifestyle configuration stress
        work_intensity = lifestyle_config.get('work_intensity', 0.7)
        savings_rate = lifestyle_config.get('savings_rate', 0.15)
        
        # Work intensity stress
        stress += work_intensity * 0.25
        
        # High savings pressure stress
        if savings_rate > 0.25:
            stress += (savings_rate - 0.25) * 0.4
        
        # Cash flow adequacy stress
        income_total = float(sum(e.amount for e in entries if e.account_type == 'income'))
        expense_total = float(sum(e.amount for e in entries if e.account_type == 'expense'))
        
        if income_total > 0:
            expense_ratio = expense_total / income_total
            if expense_ratio > 0.8:
                stress += (expense_ratio - 0.8) * 0.5  # High expense ratio stress
        
        # Goal pressure stress
        goal_entries = [e for e in entries if e.goal_allocation]
        if goal_entries:
            total_goal_allocation = sum(
                sum(float(amount) for amount in entry.goal_allocation.values())
                for entry in goal_entries
            )
            if income_total > 0 and total_goal_allocation / income_total > 0.3:
                stress += 0.15  # Too many competing goals
        
        return min(1.0, stress)
    
    def _calculate_goal_achievement(self, entries: List[AccountEntry], 
                                  lifestyle_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate achievement rates for different financial goals"""
        goal_targets = lifestyle_config.get('goal_targets', {
            'charity': 5000,
            'education': 15000,
            'rrsp': 10000,
            'child_tfsa': 3000,
            'emergency_fund': 15000
        })
        
        goal_actual = {}
        
        # Sum up actual allocations by goal
        for entry in entries:
            for goal, amount in entry.goal_allocation.items():
                if goal not in goal_actual:
                    goal_actual[goal] = Decimal('0')
                goal_actual[goal] += amount
        
        # Calculate achievement rates
        achievement_rates = {}
        for goal, target in goal_targets.items():
            actual = float(goal_actual.get(goal, Decimal('0')))
            achievement_rates[goal] = min(1.0, actual / target) if target > 0 else 1.0
        
        return achievement_rates
    
    def _generate_balancing_recommendations(self, difference: Decimal, 
                                         lifestyle_config: Dict[str, Any],
                                         entries: List[AccountEntry]) -> List[str]:
        """Generate recommendations to balance the accounting equation"""
        recommendations = []
        
        if difference > 0:
            # Surplus: Income > Expenses + Savings
            recommendations.append(f"Surplus of ${difference:.2f} - consider increasing savings or investments")
            
            savings_rate = lifestyle_config.get('savings_rate', 0.15)
            if savings_rate < 0.2:
                recommendations.append("Increase savings rate to build financial security")
            
            charity_allocation = lifestyle_config.get('charity_allocation', 0.0)
            if charity_allocation < 0.05:
                recommendations.append("Consider increasing charitable giving for tax benefits and fulfillment")
                
        else:
            # Deficit: Income < Expenses + Savings
            deficit = abs(difference)
            recommendations.append(f"Deficit of ${deficit:.2f} - immediate action required")
            
            work_intensity = lifestyle_config.get('work_intensity', 0.7)
            spending_level = lifestyle_config.get('spending_level', 0.8)
            
            # Income solutions
            if work_intensity < 0.9:
                income_boost = deficit * (1 - work_intensity) * 0.5
                recommendations.append(f"Increase work intensity to boost income by ~${income_boost:.0f}")
            
            # Expense solutions
            if spending_level > 0.7:
                expense_cut = deficit * spending_level * 0.3
                recommendations.append(f"Reduce discretionary spending by ~${expense_cut:.0f}")
            
            # Savings solutions
            savings_entries = [e for e in entries if e.account_type == 'savings']
            if savings_entries:
                total_savings = sum(float(e.amount) for e in savings_entries)
                if total_savings > float(deficit):
                    recommendations.append(f"Temporarily reduce savings by ${deficit:.0f}")
            
            # Emergency recommendations
            if deficit > 1000:
                recommendations.append("Consider using emergency fund if necessary")
                recommendations.append("Review and prioritize essential vs. non-essential expenses")
        
        return recommendations
    
    def auto_balance_period(self, start_date: datetime, end_date: datetime,
                          lifestyle_config: Dict[str, Any], 
                          balance_method: str = 'adjust_savings') -> ReconciliationReport:
        """Automatically balance the accounting equation for a period"""
        
        # First, get the current reconciliation
        initial_report = self.reconcile_period(start_date, end_date, lifestyle_config)
        
        if initial_report.is_balanced:
            logger.info("Period already balanced")
            return initial_report
        
        logger.info(f"Auto-balancing deficit of ${initial_report.accounting_difference:.2f}")
        
        # Apply balancing method
        if balance_method == 'adjust_savings':
            self._balance_by_adjusting_savings(initial_report.accounting_difference)
        elif balance_method == 'adjust_expenses':
            self._balance_by_adjusting_expenses(initial_report.accounting_difference)
        elif balance_method == 'adjust_income':
            self._balance_by_adjusting_income(initial_report.accounting_difference)
        elif balance_method == 'proportional':
            self._balance_proportionally(initial_report.accounting_difference)
        
        # Re-reconcile after adjustments
        balanced_report = self.reconcile_period(start_date, end_date, lifestyle_config)
        return balanced_report
    
    def _balance_by_adjusting_savings(self, difference: Decimal) -> None:
        """Balance by adjusting savings amounts"""
        if difference < 0:  # Deficit
            # Reduce savings to balance
            deficit = abs(difference)
            savings_entries = [e for e in self.entries if e.account_type == 'savings']
            
            # Reduce proportionally
            total_savings = sum(e.amount for e in savings_entries)
            if total_savings > 0:
                for entry in savings_entries:
                    reduction_ratio = min(1.0, float(deficit / total_savings))
                    reduction = entry.amount * Decimal(str(reduction_ratio))
                    entry.amount -= reduction
                    deficit -= reduction
                    
                    logger.debug(f"Reduced {entry.category} savings by ${reduction:.2f}")
        else:
            # Surplus - increase savings
            surplus = difference
            # Add to general savings
            savings_entry = AccountEntry(
                account_type='savings',
                category='surplus_allocation',
                amount=surplus,
                date=datetime.now(),
                description='Auto-allocated surplus to savings',
                quality_impact=0.1,
                stress_impact=-0.05
            )
            self.entries.append(savings_entry)
    
    def _balance_by_adjusting_expenses(self, difference: Decimal) -> None:
        """Balance by adjusting expense amounts"""
        if difference < 0:  # Deficit
            deficit = abs(difference)
            discretionary_categories = ['entertainment', 'dining', 'shopping', 'travel']
            
            # Find discretionary expenses to reduce
            discretionary_entries = [
                e for e in self.entries 
                if e.account_type == 'expense' and 
                any(cat in e.category.lower() for cat in discretionary_categories)
            ]
            
            total_discretionary = sum(e.amount for e in discretionary_entries)
            if total_discretionary > 0:
                for entry in discretionary_entries:
                    reduction_ratio = min(0.5, float(deficit / total_discretionary))  # Max 50% reduction
                    reduction = entry.amount * Decimal(str(reduction_ratio))
                    entry.amount -= reduction
                    deficit -= reduction
                    
                    # Adjust quality impact for reduced spending
                    entry.quality_impact -= 0.1
                    entry.stress_impact += 0.05
                    
                    logger.debug(f"Reduced {entry.category} expense by ${reduction:.2f}")
    
    def _balance_by_adjusting_income(self, difference: Decimal) -> None:
        """Balance by adjusting income (work intensity)"""
        if difference < 0:  # Deficit
            deficit = abs(difference)
            
            # Add overtime or bonus income
            income_entry = AccountEntry(
                account_type='income',
                category='overtime_adjustment',
                amount=deficit,
                date=datetime.now(),
                description='Additional income to balance equation',
                quality_impact=-0.05,  # Working more reduces quality
                stress_impact=0.1      # But increases stress
            )
            self.entries.append(income_entry)
            logger.debug(f"Added ${deficit:.2f} additional income")
    
    def _balance_proportionally(self, difference: Decimal) -> None:
        """Balance by proportional adjustments across categories"""
        if abs(difference) < self.tolerance:
            return
        
        if difference < 0:  # Deficit
            deficit = abs(difference)
            
            # 60% from savings reduction, 30% from expense reduction, 10% from income increase
            savings_adjustment = deficit * Decimal('0.6')
            expense_adjustment = deficit * Decimal('0.3')
            income_adjustment = deficit * Decimal('0.1')
            
            self._balance_by_adjusting_savings(-savings_adjustment)
            self._balance_by_adjusting_expenses(-expense_adjustment)
            self._balance_by_adjusting_income(income_adjustment)
        else:
            # Surplus - allocate to savings
            self._balance_by_adjusting_savings(difference)
    
    def generate_detailed_reconciliation_report(self, start_date: datetime, 
                                              end_date: datetime) -> str:
        """Generate a detailed reconciliation report"""
        report = self.reconcile_period(start_date, end_date, {})
        
        output = []
        output.append("="*60)
        output.append("DETAILED ACCOUNTING RECONCILIATION REPORT")
        output.append("="*60)
        output.append(f"Period: {start_date.date()} to {end_date.date()}")
        output.append(f"Status: {'✓ BALANCED' if report.is_balanced else '✗ UNBALANCED'}")
        output.append("")
        
        output.append("ACCOUNTING EQUATION:")
        output.append(f"Total Income:    ${report.total_income:>12,.2f}")
        output.append(f"Total Expenses:  ${report.total_expenses:>12,.2f}")
        output.append(f"Total Savings:   ${report.total_savings:>12,.2f}")
        output.append(f"{'':15} {'-'*15}")
        output.append(f"Difference:      ${report.accounting_difference:>12,.2f}")
        output.append("")
        
        output.append("ACCOUNT BALANCES:")
        for account_name, balance in report.account_balances.items():
            if balance.net_change != 0:
                output.append(f"{account_name.title():15} ${balance.opening_balance:>10,.2f} → ${balance.closing_balance:>10,.2f}")
        output.append("")
        
        output.append("QUALITY METRICS:")
        output.append(f"Quality of Life: {report.quality_of_life_score:>12.1%}")
        output.append(f"Stress Level:    {report.stress_level:>12.1%}")
        output.append("")
        
        if report.goal_achievement_rates:
            output.append("GOAL ACHIEVEMENT:")
            for goal, rate in report.goal_achievement_rates.items():
                output.append(f"{goal.title():15} {rate:>12.1%}")
            output.append("")
        
        if report.recommendations:
            output.append("RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                output.append(f"{i:2d}. {rec}")
        
        return "\n".join(output)

def demo_accounting_reconciliation():
    """Demonstrate accounting reconciliation system"""
    print("⚖️ ACCOUNTING RECONCILIATION SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Initialize reconciliation engine
    engine = AccountingReconciliationEngine(tolerance=0.01)
    
    # Define lifestyle configuration
    lifestyle_config = {
        'work_intensity': 0.8,     # 80% work effort
        'spending_level': 0.75,    # 75% of comfortable spending
        'savings_rate': 0.20,      # 20% savings rate
        'charity_allocation': 0.05, # 5% for charity
        'goal_targets': {
            'charity': 4000,
            'education': 0,  # No education this period
            'rrsp': 12000,
            'child_tfsa': 2500,
            'emergency_fund': 10000
        }
    }
    
    print(f"📊 Lifestyle Configuration:")
    for key, value in lifestyle_config.items():
        if key != 'goal_targets':
            if isinstance(value, float):
                print(f"   • {key.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Create sample entries for one month
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    # Income entries
    income_entries = [
        AccountEntry('income', 'salary', Decimal('7000'), datetime(2025, 1, 15), 
                    'Monthly salary', 0.1, 0.1),
        AccountEntry('income', 'bonus', Decimal('1000'), datetime(2025, 1, 31), 
                    'Performance bonus', 0.2, 0.0)
    ]
    
    # Expense entries
    expense_entries = [
        AccountEntry('expense', 'rent', Decimal('2000'), datetime(2025, 1, 1), 
                    'Monthly rent', -0.1, 0.1),
        AccountEntry('expense', 'groceries', Decimal('800'), datetime(2025, 1, 15), 
                    'Food expenses', 0.1, 0.0),
        AccountEntry('expense', 'utilities', Decimal('300'), datetime(2025, 1, 31), 
                    'Electricity, water, internet', -0.05, 0.05),
        AccountEntry('expense', 'entertainment', Decimal('400'), datetime(2025, 1, 20), 
                    'Dining and activities', 0.3, -0.1),
        AccountEntry('expense', 'transportation', Decimal('500'), datetime(2025, 1, 10), 
                    'Car payment and gas', -0.1, 0.1)
    ]
    
    # Savings entries
    savings_entries = [
        AccountEntry('savings', 'rrsp_contribution', Decimal('1000'), datetime(2025, 1, 31), 
                    'Monthly RRSP contribution', 0.1, -0.05, {'rrsp': Decimal('1000')}),
        AccountEntry('savings', 'child_tfsa', Decimal('200'), datetime(2025, 1, 31), 
                    'Child education savings', 0.15, -0.05, {'child_tfsa': Decimal('200')}),
        AccountEntry('savings', 'emergency_fund', Decimal('500'), datetime(2025, 1, 31), 
                    'Emergency fund contribution', 0.05, -0.1, {'emergency_fund': Decimal('500')}),
        AccountEntry('savings', 'charity_donation', Decimal('400'), datetime(2025, 1, 31), 
                    'Monthly charitable giving', 0.2, 0.0, {'charity': Decimal('400')})
    ]
    
    # Add all entries
    all_entries = income_entries + expense_entries + savings_entries
    for entry in all_entries:
        engine.add_entry(entry)
    
    print(f"\n📝 Added {len(all_entries)} accounting entries for January 2025")
    
    # Show initial totals
    total_income = sum(e.amount for e in income_entries)
    total_expenses = sum(e.amount for e in expense_entries)
    total_savings = sum(e.amount for e in savings_entries)
    
    print(f"\n💰 Initial Totals:")
    print(f"   Income:   ${total_income:>8,.2f}")
    print(f"   Expenses: ${total_expenses:>8,.2f}")
    print(f"   Savings:  ${total_savings:>8,.2f}")
    print(f"   Balance:  ${total_income - total_expenses - total_savings:>8,.2f}")
    
    # Perform reconciliation
    print(f"\n⚖️ RECONCILIATION RESULTS:")
    report = engine.reconcile_period(start_date, end_date, lifestyle_config)
    
    print(f"Status: {'✓ BALANCED' if report.is_balanced else '✗ UNBALANCED'}")
    print(f"Accounting Difference: ${report.accounting_difference:.2f}")
    print(f"Quality of Life Score: {report.quality_of_life_score:.1%}")
    print(f"Stress Level: {report.stress_level:.1%}")
    
    # Show account balances
    print(f"\n🏦 ACCOUNT BALANCES:")
    for account_name, balance in report.account_balances.items():
        if balance.net_change != 0:
            print(f"   {account_name.title():15} ${balance.closing_balance:>8,.2f} "
                  f"({'+' if balance.net_change >= 0 else ''}{balance.net_change:.2f})")
    
    # Show goal achievement
    print(f"\n🎯 GOAL ACHIEVEMENT:")
    for goal, rate in report.goal_achievement_rates.items():
        status = "✓" if rate >= 1.0 else "○" if rate >= 0.8 else "✗"
        print(f"   {status} {goal.title():15} {rate:>6.1%}")
    
    # Show recommendations if any
    if report.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Demonstrate auto-balancing if needed
    if not report.is_balanced:
        print(f"\n🔧 AUTO-BALANCING DEMONSTRATION:")
        balanced_report = engine.auto_balance_period(start_date, end_date, lifestyle_config)
        print(f"After auto-balance: {'✓ BALANCED' if balanced_report.is_balanced else '✗ STILL UNBALANCED'}")
        print(f"New difference: ${balanced_report.accounting_difference:.2f}")
    
    # Generate detailed report
    print(f"\n📋 DETAILED RECONCILIATION REPORT:")
    detailed_report = engine.generate_detailed_reconciliation_report(start_date, end_date)
    print(detailed_report)

if __name__ == "__main__":
    demo_accounting_reconciliation() 