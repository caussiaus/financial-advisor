"""
Unified Cash Flow Model with Time Uncertainty Integration

This module creates a comprehensive cash flow model that:
1. Incorporates time uncertainty mesh for event timing/amount uncertainty
2. Integrates with existing accounting system using case-specific account names
3. Provides real-time cash flow tracking and state monitoring
4. Ensures accounting consistency at any point in time
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal
import json
import logging

from .core.time_uncertainty_mesh import TimeUncertaintyMeshEngine, SeedEvent, EventVector
from .accounting_reconciliation import AccountingReconciliationEngine, Account, AccountType, PaymentConstraint
from .vectorized_accounting import VectorizedAccountingEngine, AccountingState

@dataclass
class CashFlowEvent:
    """Represents a cash flow event with accounting integration"""
    event_id: str
    description: str
    estimated_date: str
    amount: float
    source_account: str  # Account to debit from
    target_account: str  # Account to credit to
    event_type: str  # 'income', 'expense', 'transfer', 'investment'
    timing_volatility: float = 0.2
    amount_volatility: float = 0.15
    drift_rate: float = 0.03
    probability: float = 0.8
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    
    def to_seed_event(self) -> SeedEvent:
        """Convert to SeedEvent for time uncertainty mesh"""
        return SeedEvent(
            event_id=self.event_id,
            description=self.description,
            estimated_date=self.estimated_date,
            amount=self.amount,
            timing_volatility=self.timing_volatility,
            amount_volatility=self.amount_volatility,
            drift_rate=self.drift_rate,
            probability=self.probability,
            category=self.category,
            dependencies=self.dependencies
        )

@dataclass
class CashFlowState:
    """Represents the complete cash flow state at a point in time"""
    timestamp: datetime
    account_balances: Dict[str, Decimal]
    cash_flow_events: List[CashFlowEvent]
    net_cash_flow: Decimal
    total_assets: Decimal
    total_liabilities: Decimal
    net_worth: Decimal
    liquidity_ratio: float
    stress_level: float

class UnifiedCashFlowModel:
    """
    Unified cash flow model that integrates time uncertainty with accounting
    """
    
    def __init__(self, initial_state: Dict[str, float]):
        self.initial_state = initial_state
        self.time_uncertainty_engine = TimeUncertaintyMeshEngine(use_gpu=True)
        self.accounting_engine = AccountingReconciliationEngine()
        self.vectorized_accounting = VectorizedAccountingEngine()
        
        # Case-specific account configuration based on Case_1 analysis
        self.case_accounts = {
            'assets': {
                'cash_checking': 'Checking Account',
                'cash_savings': 'Savings Account', 
                'investments_stocks': 'Stock Investments',
                'investments_bonds': 'Bond Investments',
                'investments_retirement': 'Retirement Accounts',
                'real_estate': 'Real Estate'
            },
            'liabilities': {
                'mortgage': 'Mortgage Loan',
                'student_loans': 'Student Loans',
                'credit_cards': 'Credit Cards',
                'auto_loans': 'Auto Loans'
            },
            'income': {
                'salary': 'Salary Income',
                'investment_income': 'Investment Income',
                'other_income': 'Other Income'
            },
            'expenses': {
                'milestone_payments': 'Milestone Payments',
                'living_expenses': 'Living Expenses',
                'investment_expenses': 'Investment Expenses',
                'education_expenses': 'Education Expenses'
            }
        }
        
        # Initialize accounting system
        self._initialize_accounting_system()
        
        # Cash flow tracking
        self.cash_flow_events = []
        self.cash_flow_history = []
        self.current_state = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cash flow operations"""
        logger = logging.getLogger('unified_cash_flow_model')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_accounting_system(self):
        """Initialize accounting system with case-specific accounts"""
        print("🏦 Initializing accounting system with case-specific accounts...")
        
        # Set initial balances based on case data
        total_wealth = self.initial_state.get('total_wealth', 764560.97)  # From case analysis
        
        # Initialize accounts with case-specific balances
        initial_balances = {
            'cash_checking': total_wealth * 0.0446,  # 4.46% cash allocation
            'cash_savings': total_wealth * 0.0446,   # Additional savings
            'investments_stocks': total_wealth * 0.4435,  # 44.35% equity allocation
            'investments_bonds': total_wealth * 0.5119,   # 51.19% bonds allocation
            'investments_retirement': total_wealth * 0.2,  # 20% retirement
            'real_estate': total_wealth * 0.3,  # 30% real estate
            'mortgage': total_wealth * 0.4,  # 40% mortgage
            'student_loans': 0,  # No student loans initially
            'credit_cards': 0,   # No credit card debt initially
            'auto_loans': 0,     # No auto loans initially
            'salary': 150000,    # Annual salary
            'investment_income': total_wealth * 0.04,  # 4% investment income
            'other_income': 0,
            'milestone_payments': 0,
            'living_expenses': 60000,  # Annual living expenses
            'investment_expenses': 0,
            'education_expenses': 0
        }
        
        # Set account balances in accounting engine
        for account_id, balance in initial_balances.items():
            self.accounting_engine.set_account_balance(account_id, Decimal(str(balance)))
        
        print(f"✅ Accounting system initialized with ${total_wealth:,.2f} total wealth")
    
    def add_cash_flow_event(self, event: CashFlowEvent):
        """Add a cash flow event to the model"""
        self.cash_flow_events.append(event)
        self.logger.info(f"Added cash flow event: {event.event_id} - {event.description}")
    
    def create_case_events_from_analysis(self) -> List[CashFlowEvent]:
        """Create cash flow events based on Case_1 analysis"""
        events = []
        
        # Family planning event (2020)
        events.append(CashFlowEvent(
            event_id="first_child_born",
            description="First child born - increased need for financial stability",
            estimated_date="2020-03-15",
            amount=-25000,
            source_account="cash_savings",
            target_account="milestone_payments",
            event_type="expense",
            timing_volatility=0.3,
            amount_volatility=0.2,
            drift_rate=0.05,
            probability=0.9,
            category="family_planning"
        ))
        
        # Career advancement event (2021)
        events.append(CashFlowEvent(
            event_id="promotion_senior_role",
            description="Promotion to senior role - increased income and bonus",
            estimated_date="2021-01-15",
            amount=30000,
            source_account="salary",
            target_account="cash_checking",
            event_type="income",
            timing_volatility=0.2,
            amount_volatility=0.15,
            drift_rate=0.03,
            probability=0.95,
            category="career_advancement"
        ))
        
        # Portfolio rebalancing event (2022)
        events.append(CashFlowEvent(
            event_id="covid_market_recovery",
            description="COVID market recovery - increased risk tolerance",
            estimated_date="2022-06-01",
            amount=0,  # Neutral impact
            source_account="investments_bonds",
            target_account="investments_stocks",
            event_type="transfer",
            timing_volatility=0.4,
            amount_volatility=0.1,
            drift_rate=0.02,
            probability=0.85,
            category="portfolio_rebalancing"
        ))
        
        # Education planning event (2023)
        events.append(CashFlowEvent(
            event_id="education_fund_start",
            description="Starting education fund - considering Johns Hopkins vs McGill",
            estimated_date="2023-09-01",
            amount=-15000,
            source_account="cash_savings",
            target_account="education_expenses",
            event_type="expense",
            timing_volatility=0.2,
            amount_volatility=0.25,
            drift_rate=0.06,
            probability=0.9,
            category="education_planning"
        ))
        
        # Education decision event (2024)
        events.append(CashFlowEvent(
            event_id="chose_mcgill_over_johns_hopkins",
            description="Chose McGill over Johns Hopkins - cost concerns",
            estimated_date="2024-02-15",
            amount=75000,  # Positive impact (savings from choosing cheaper option)
            source_account="education_expenses",
            target_account="cash_savings",
            event_type="income",
            timing_volatility=0.3,
            amount_volatility=0.2,
            drift_rate=0.04,
            probability=1.0,
            category="education_decision"
        ))
        
        # Work arrangement change (2025)
        events.append(CashFlowEvent(
            event_id="switched_to_part_time",
            description="Switched to part-time work for better work-life balance",
            estimated_date="2025-01-01",
            amount=-40000,
            source_account="salary",
            target_account="living_expenses",
            event_type="expense",
            timing_volatility=0.5,
            amount_volatility=0.3,
            drift_rate=-0.02,
            probability=0.95,
            category="work_arrangement"
        ))
        
        return events
    
    def initialize_time_uncertainty_mesh(self, num_scenarios: int = 10000, 
                                       time_horizon_years: float = 10) -> Tuple[Dict, Dict]:
        """Initialize time uncertainty mesh with cash flow events"""
        print("🎯 Initializing time uncertainty mesh with cash flow events...")
        
        # Convert cash flow events to seed events
        seed_events = [event.to_seed_event() for event in self.cash_flow_events]
        
        # Initialize mesh
        mesh_data, risk_analysis = self.time_uncertainty_engine.initialize_mesh_with_time_uncertainty(
            seed_events,
            num_scenarios=num_scenarios,
            time_horizon_years=time_horizon_years
        )
        
        print(f"✅ Time uncertainty mesh initialized with {len(seed_events)} events")
        return mesh_data, risk_analysis
    
    def calculate_cash_flow_state(self, timestamp: datetime) -> CashFlowState:
        """Calculate cash flow state at a specific timestamp"""
        
        # Get current account balances
        account_balances = {}
        for account_id in self.case_accounts['assets']:
            balance = self.accounting_engine.get_account_balance(account_id)
            account_balances[account_id] = balance if balance is not None else Decimal('0')
        
        for account_id in self.case_accounts['liabilities']:
            balance = self.accounting_engine.get_account_balance(account_id)
            account_balances[account_id] = balance if balance is not None else Decimal('0')
        
        for account_id in self.case_accounts['income']:
            balance = self.accounting_engine.get_account_balance(account_id)
            account_balances[account_id] = balance if balance is not None else Decimal('0')
        
        for account_id in self.case_accounts['expenses']:
            balance = self.accounting_engine.get_account_balance(account_id)
            account_balances[account_id] = balance if balance is not None else Decimal('0')
        
        # Calculate totals
        total_assets = sum(account_balances.get(acc, Decimal('0')) 
                          for acc in self.case_accounts['assets'].keys())
        total_liabilities = sum(account_balances.get(acc, Decimal('0')) 
                               for acc in self.case_accounts['liabilities'].keys())
        net_worth = total_assets - total_liabilities
        
        # Calculate net cash flow (income - expenses)
        total_income = sum(account_balances.get(acc, Decimal('0')) 
                          for acc in self.case_accounts['income'].keys())
        total_expenses = sum(account_balances.get(acc, Decimal('0')) 
                            for acc in self.case_accounts['expenses'].keys())
        net_cash_flow = total_income - total_expenses
        
        # Calculate liquidity ratio
        cash_assets = sum(account_balances.get(acc, Decimal('0')) 
                         for acc in ['cash_checking', 'cash_savings'])
        liquidity_ratio = float(cash_assets / total_assets) if total_assets > 0 else 0.0
        
        # Calculate stress level based on cash flow and debt ratios
        debt_ratio = float(total_liabilities / total_assets) if total_assets > 0 else 0.0
        cash_flow_stress = 1.0 if net_cash_flow < 0 else 0.0
        debt_stress = min(debt_ratio, 1.0)
        stress_level = (cash_flow_stress + debt_stress) / 2.0
        
        return CashFlowState(
            timestamp=timestamp,
            account_balances=account_balances,
            cash_flow_events=self.cash_flow_events,
            net_cash_flow=net_cash_flow,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            net_worth=net_worth,
            liquidity_ratio=liquidity_ratio,
            stress_level=stress_level
        )
    
    def process_cash_flow_event(self, event: CashFlowEvent, timestamp: datetime) -> bool:
        """Process a cash flow event and update accounting state"""
        try:
            # Check if event should occur at this timestamp
            event_date = pd.to_datetime(event.estimated_date)
            if abs((event_date - timestamp).days) > 30:  # Event not due yet
                return False
            
            # Check if we have sufficient funds
            source_balance = self.accounting_engine.get_account_balance(event.source_account)
            if event.amount < 0 and abs(Decimal(str(event.amount))) > source_balance:
                self.logger.warning(f"Insufficient funds in {event.source_account} for event {event.event_id}")
                return False
            
            # Process the transaction
            if event.event_type == 'income':
                # Credit target account, debit source account
                self.accounting_engine.set_account_balance(
                    event.target_account,
                    self.accounting_engine.get_account_balance(event.target_account) + Decimal(str(event.amount))
                )
                self.accounting_engine.set_account_balance(
                    event.source_account,
                    self.accounting_engine.get_account_balance(event.source_account) - Decimal(str(event.amount))
                )
            elif event.event_type == 'expense':
                # Debit target account, credit source account
                self.accounting_engine.set_account_balance(
                    event.target_account,
                    self.accounting_engine.get_account_balance(event.target_account) + Decimal(str(abs(event.amount)))
                )
                self.accounting_engine.set_account_balance(
                    event.source_account,
                    self.accounting_engine.get_account_balance(event.source_account) - Decimal(str(abs(event.amount)))
                )
            elif event.event_type == 'transfer':
                # Transfer between accounts
                transfer_amount = abs(event.amount) if event.amount != 0 else 10000  # Default transfer amount
                self.accounting_engine.set_account_balance(
                    event.target_account,
                    self.accounting_engine.get_account_balance(event.target_account) + Decimal(str(transfer_amount))
                )
                self.accounting_engine.set_account_balance(
                    event.source_account,
                    self.accounting_engine.get_account_balance(event.source_account) - Decimal(str(transfer_amount))
                )
            
            self.logger.info(f"Processed cash flow event: {event.event_id} - ${event.amount:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing cash flow event {event.event_id}: {str(e)}")
            return False
    
    def simulate_cash_flows_over_time(self, start_date: datetime, end_date: datetime, 
                                    time_step_days: int = 30) -> List[CashFlowState]:
        """Simulate cash flows over time and track states"""
        print(f"📊 Simulating cash flows from {start_date} to {end_date}")
        
        states = []
        current_date = start_date
        
        while current_date <= end_date:
            # Process any due cash flow events
            for event in self.cash_flow_events:
                self.process_cash_flow_event(event, current_date)
            
            # Calculate current state
            state = self.calculate_cash_flow_state(current_date)
            states.append(state)
            
            # Apply monthly growth/inflation
            self._apply_monthly_adjustments(current_date)
            
            current_date += timedelta(days=time_step_days)
        
        self.cash_flow_history = states
        print(f"✅ Simulated {len(states)} cash flow states")
        return states
    
    def _apply_monthly_adjustments(self, timestamp: datetime):
        """Apply monthly adjustments to accounts (growth, inflation, etc.)"""
        # Investment growth
        for account_id in ['investments_stocks', 'investments_bonds', 'investments_retirement']:
            current_balance = self.accounting_engine.get_account_balance(account_id)
            if current_balance is None:
                continue  # Skip if account doesn't exist
                
            if account_id == 'investments_stocks':
                growth_rate = 0.08 / 12  # 8% annual growth
            elif account_id == 'investments_bonds':
                growth_rate = 0.04 / 12  # 4% annual growth
            else:  # retirement
                growth_rate = 0.06 / 12  # 6% annual growth
            
            new_balance = current_balance * (1 + Decimal(str(growth_rate)))
            self.accounting_engine.set_account_balance(account_id, new_balance)
        
        # Income growth (2% annual)
        for account_id in ['salary', 'investment_income']:
            current_balance = self.accounting_engine.get_account_balance(account_id)
            if current_balance is None:
                continue  # Skip if account doesn't exist
                
            growth_rate = 0.02 / 12  # 2% annual growth
            new_balance = current_balance * (1 + Decimal(str(growth_rate)))
            self.accounting_engine.set_account_balance(account_id, new_balance)
        
        # Expense inflation (3% annual)
        for account_id in ['living_expenses', 'education_expenses']:
            current_balance = self.accounting_engine.get_account_balance(account_id)
            if current_balance is None:
                continue  # Skip if account doesn't exist
                
            inflation_rate = 0.03 / 12  # 3% annual inflation
            new_balance = current_balance * (1 + Decimal(str(inflation_rate)))
            self.accounting_engine.set_account_balance(account_id, new_balance)

        # Interest on negative cash balances (borrowing yardstick)
        interest_rate = Decimal('0.06') / Decimal('12')  # 6% annual, monthly
        for cash_account in ['cash_checking', 'cash_savings']:
            cash_balance = self.accounting_engine.get_account_balance(cash_account)
            if cash_balance is not None and cash_balance < 0:
                interest = abs(cash_balance) * interest_rate
                # Add to interest_expense (create if needed)
                if self.accounting_engine.get_account_balance('interest_expense') is None:
                    self.accounting_engine.set_account_balance('interest_expense', Decimal('0'))
                current_interest_expense = self.accounting_engine.get_account_balance('interest_expense')
                self.accounting_engine.set_account_balance('interest_expense', current_interest_expense + interest)
                # Optionally, reduce cash further to reflect interest payment
                self.accounting_engine.set_account_balance(cash_account, cash_balance - interest)
    
    def get_cash_flow_summary(self) -> Dict:
        """Get comprehensive cash flow summary"""
        if not self.cash_flow_history:
            return {}
        
        latest_state = self.cash_flow_history[-1]
        initial_state = self.cash_flow_history[0]
        
        return {
            'period': {
                'start_date': initial_state.timestamp.isoformat(),
                'end_date': latest_state.timestamp.isoformat(),
                'duration_days': (latest_state.timestamp - initial_state.timestamp).days
            },
            'financial_summary': {
                'initial_net_worth': float(initial_state.net_worth),
                'final_net_worth': float(latest_state.net_worth),
                'net_worth_change': float(latest_state.net_worth - initial_state.net_worth),
                'net_worth_growth_rate': float((float(latest_state.net_worth) / float(initial_state.net_worth)) ** (365 / (latest_state.timestamp - initial_state.timestamp).days) - 1) if initial_state.net_worth > 0 else 0.0,
                'final_liquidity_ratio': latest_state.liquidity_ratio,
                'final_stress_level': latest_state.stress_level
            },
            'cash_flow_analysis': {
                'total_events_processed': len(self.cash_flow_events),
                'positive_events': len([e for e in self.cash_flow_events if e.amount > 0]),
                'negative_events': len([e for e in self.cash_flow_events if e.amount < 0]),
                'total_cash_flow_impact': sum(e.amount for e in self.cash_flow_events),
                'average_monthly_cash_flow': float(latest_state.net_cash_flow)
            },
            'account_summary': {
                'total_assets': float(latest_state.total_assets),
                'total_liabilities': float(latest_state.total_liabilities),
                'debt_to_asset_ratio': float(latest_state.total_liabilities / latest_state.total_assets) if latest_state.total_assets > 0 else 0.0
            }
        }
    
    def export_cash_flow_data(self, filepath: str):
        """Export cash flow data to JSON"""
        export_data = {
            'model_configuration': {
                'initial_state': self.initial_state,
                'case_accounts': self.case_accounts,
                'total_events': len(self.cash_flow_events)
            },
            'cash_flow_events': [
                {
                    'event_id': event.event_id,
                    'description': event.description,
                    'estimated_date': event.estimated_date,
                    'amount': event.amount,
                    'source_account': event.source_account,
                    'target_account': event.target_account,
                    'event_type': event.event_type,
                    'category': event.category,
                    'probability': event.probability
                }
                for event in self.cash_flow_events
            ],
            'cash_flow_history': [
                {
                    'timestamp': state.timestamp.isoformat(),
                    'net_cash_flow': float(state.net_cash_flow),
                    'total_assets': float(state.total_assets),
                    'total_liabilities': float(state.total_liabilities),
                    'net_worth': float(state.net_worth),
                    'liquidity_ratio': state.liquidity_ratio,
                    'stress_level': state.stress_level,
                    'account_balances': {k: float(v) for k, v in state.account_balances.items()}
                }
                for state in self.cash_flow_history
            ],
            'summary': self.get_cash_flow_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Cash flow data exported to {filepath}")


def demo_unified_cash_flow_model():
    """Demonstrate the unified cash flow model"""
    print("🎯 Unified Cash Flow Model Demo")
    print("=" * 50)
    
    # Initialize with case-specific initial state
    initial_state = {
        'total_wealth': 764560.97,  # From case analysis
        'cash': 764560.97 * 0.0892,  # 8.92% cash allocation
        'investments': 764560.97 * 0.9554,  # 95.54% investments
        'income': 150000,  # Annual salary
        'expenses': 60000   # Annual expenses
    }
    
    # Create unified model
    model = UnifiedCashFlowModel(initial_state)
    
    # Add case-specific events
    case_events = model.create_case_events_from_analysis()
    for event in case_events:
        model.add_cash_flow_event(event)
    
    print(f"📋 Added {len(case_events)} case-specific cash flow events")
    
    # Initialize time uncertainty mesh
    mesh_data, risk_analysis = model.initialize_time_uncertainty_mesh(
        num_scenarios=5000,  # Reduced for demo
        time_horizon_years=5
    )
    
    # Simulate cash flows over time
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    states = model.simulate_cash_flows_over_time(start_date, end_date)
    
    # Get summary
    summary = model.get_cash_flow_summary()
    print("\n📊 Cash Flow Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Export data
    model.export_cash_flow_data("unified_cash_flow_data.json")
    
    print("\n✅ Unified cash flow model demo complete!")
    return model, mesh_data, risk_analysis


if __name__ == "__main__":
    demo_unified_cash_flow_model() 