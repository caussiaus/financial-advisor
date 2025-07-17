"""
Vectorized accounting engine for efficient batch processing of financial states.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from decimal import Decimal
import pandas as pd


@dataclass
class AccountingState:
    """Represents the state of all accounts at a point in time"""
    cash: Decimal
    investments: Dict[str, Decimal]
    debts: Dict[str, Decimal]
    timestamp: np.datetime64


class VectorizedAccountingEngine:
    """
    Efficient accounting engine using vectorized operations
    """
    
    def __init__(self):
        self.operation_cache = {}
        self.common_operations = self._initialize_common_operations()
        self.account_types = {
            'assets': ['cash', 'savings', 'investments', 'retirement', 'real_estate'],
            'liabilities': ['mortgage', 'loans', 'credit_cards'],
            'income': ['salary', 'investment_income', 'other_income'],
            'expenses': ['housing', 'utilities', 'insurance', 'discretionary']
        }
        
    def _initialize_common_operations(self) -> Dict[str, np.ndarray]:
        """
        Initialize lookup tables for common operations
        """
        operations = {
            'mortgage_payment': self._create_mortgage_table(),
            'investment_growth': self._create_investment_table(),
            'debt_paydown': self._create_debt_table()
        }
        return operations
    
    def _create_mortgage_table(self, 
                             rates: List[float] = np.linspace(0.03, 0.08, 51),
                             terms: List[int] = [15, 30],
                             amounts: List[float] = np.linspace(100000, 1000000, 91)) -> np.ndarray:
        """
        Create lookup table for mortgage calculations
        """
        table = np.zeros((len(rates), len(terms), len(amounts)))
        
        for i, rate in enumerate(rates):
            for j, term in enumerate(terms):
                for k, amount in enumerate(amounts):
                    # Monthly payment calculation
                    monthly_rate = rate / 12
                    num_payments = term * 12
                    table[i, j, k] = amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        return table
    
    def _create_investment_table(self,
                               returns: List[float] = np.linspace(-0.20, 0.20, 41),
                               horizons: List[int] = range(1, 31),
                               allocations: List[float] = np.linspace(0, 1, 11)) -> np.ndarray:
        """
        Create lookup table for investment returns
        """
        table = np.zeros((len(returns), len(horizons), len(allocations)))
        
        for i, ret in enumerate(returns):
            for j, horizon in enumerate(horizons):
                for k, allocation in enumerate(allocations):
                    # Compound return calculation
                    table[i, j, k] = (1 + ret) ** horizon * allocation
        
        return table
    
    def _create_debt_table(self,
                          rates: List[float] = np.linspace(0.05, 0.25, 21),
                          terms: List[int] = range(1, 11),
                          amounts: List[float] = np.linspace(1000, 50000, 50)) -> np.ndarray:
        """
        Create lookup table for debt calculations
        """
        table = np.zeros((len(rates), len(terms), len(amounts)))
        
        for i, rate in enumerate(rates):
            for j, term in enumerate(terms):
                for k, amount in enumerate(amounts):
                    # Monthly payment calculation
                    monthly_rate = rate / 12
                    num_payments = term * 12
                    table[i, j, k] = amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        return table
    
    def batch_process_states(self, 
                           states: np.ndarray,
                           operations: List[Dict],
                           time_horizon: float) -> np.ndarray:
        """
        Process multiple financial states in parallel
        """
        # Convert states to matrix form
        state_matrix = self._states_to_matrix(states)
        
        # Apply common operations using lookup tables
        processed_states = self._apply_common_operations(state_matrix, operations)
        
        # Process unique cases
        final_states = self._process_unique_cases(processed_states, operations)
        
        return final_states
    
    def _states_to_matrix(self, states: List[AccountingState]) -> np.ndarray:
        """
        Convert list of states to matrix form for vectorized operations
        """
        num_states = len(states)
        num_accounts = len(self.account_types['assets']) + len(self.account_types['liabilities'])
        
        matrix = np.zeros((num_states, num_accounts))
        
        for i, state in enumerate(states):
            # Assets
            for j, asset in enumerate(self.account_types['assets']):
                if asset == 'cash':
                    matrix[i, j] = float(state.cash)
                elif asset in state.investments:
                    matrix[i, j] = float(state.investments[asset])
            
            # Liabilities
            liability_offset = len(self.account_types['assets'])
            for j, liability in enumerate(self.account_types['liabilities']):
                if liability in state.debts:
                    matrix[i, liability_offset + j] = float(state.debts[liability])
        
        return matrix
    
    def _apply_common_operations(self,
                               state_matrix: np.ndarray,
                               operations: List[Dict]) -> np.ndarray:
        """
        Apply common financial operations using lookup tables
        """
        result_matrix = state_matrix.copy()
        
        for operation in operations:
            op_type = operation['type']
            if op_type in self.common_operations:
                lookup_table = self.common_operations[op_type]
                
                # Find closest indices in lookup table
                indices = self._find_lookup_indices(operation, lookup_table.shape)
                
                # Apply operation using lookup table
                result_matrix = self._apply_lookup_operation(
                    result_matrix, lookup_table[indices], operation
                )
        
        return result_matrix
    
    def _find_lookup_indices(self, 
                           operation: Dict,
                           table_shape: Tuple) -> Tuple[int, ...]:
        """
        Find closest indices in lookup table for given operation parameters
        """
        indices = []
        
        if operation['type'] == 'mortgage_payment':
            # Find rate index
            rate_idx = np.searchsorted(
                np.linspace(0.03, 0.08, table_shape[0]),
                operation['rate']
            )
            
            # Find term index
            term_idx = np.searchsorted([15, 30], operation['term'])
            
            # Find amount index
            amount_idx = np.searchsorted(
                np.linspace(100000, 1000000, table_shape[2]),
                operation['amount']
            )
            
            indices = [rate_idx, term_idx, amount_idx]
            
        elif operation['type'] == 'investment_growth':
            # Similar logic for investment operations
            return_idx = np.searchsorted(
                np.linspace(-0.20, 0.20, table_shape[0]),
                operation['return']
            )
            
            horizon_idx = np.searchsorted(
                range(1, 31),
                operation['horizon']
            )
            
            allocation_idx = np.searchsorted(
                np.linspace(0, 1, table_shape[2]),
                operation['allocation']
            )
            
            indices = [return_idx, horizon_idx, allocation_idx]
        
        return tuple(indices)
    
    def _apply_lookup_operation(self,
                              state_matrix: np.ndarray,
                              lookup_values: np.ndarray,
                              operation: Dict) -> np.ndarray:
        """
        Apply operation using looked up values
        """
        result = state_matrix.copy()
        
        if operation['type'] == 'mortgage_payment':
            # Apply mortgage payment to relevant accounts
            liability_idx = len(self.account_types['assets'])
            mortgage_idx = self.account_types['liabilities'].index('mortgage')
            cash_idx = self.account_types['assets'].index('cash')
            
            # Update mortgage balance and cash
            result[:, liability_idx + mortgage_idx] -= lookup_values
            result[:, cash_idx] -= lookup_values
            
        elif operation['type'] == 'investment_growth':
            # Apply investment growth
            investment_idx = self.account_types['assets'].index('investments')
            result[:, investment_idx] *= (1 + lookup_values)
        
        return result
    
    def _process_unique_cases(self,
                            state_matrix: np.ndarray,
                            operations: List[Dict]) -> np.ndarray:
        """
        Process operations that don't fit common patterns
        """
        result = state_matrix.copy()
        
        for operation in operations:
            if operation['type'] not in self.common_operations:
                # Handle custom operation
                if operation['type'] == 'custom_transfer':
                    from_idx = self._get_account_index(operation['from_account'])
                    to_idx = self._get_account_index(operation['to_account'])
                    amount = operation['amount']
                    
                    # Perform transfer
                    result[:, from_idx] -= amount
                    result[:, to_idx] += amount
        
        return result
    
    def _get_account_index(self, account_name: str) -> int:
        """
        Get matrix index for account name
        """
        for category, accounts in self.account_types.items():
            if account_name in accounts:
                offset = 0
                if category == 'liabilities':
                    offset = len(self.account_types['assets'])
                return offset + accounts.index(account_name)
        raise ValueError(f"Unknown account: {account_name}")
    
    def matrix_to_states(self, 
                        matrix: np.ndarray,
                        timestamps: List[np.datetime64]) -> List[AccountingState]:
        """
        Convert matrix back to AccountingState objects
        """
        states = []
        
        for i in range(matrix.shape[0]):
            # Extract assets
            cash = Decimal(str(matrix[i, 0]))
            investments = {}
            for j, asset in enumerate(self.account_types['assets'][1:], 1):
                if matrix[i, j] != 0:
                    investments[asset] = Decimal(str(matrix[i, j]))
            
            # Extract liabilities
            debts = {}
            liability_offset = len(self.account_types['assets'])
            for j, liability in enumerate(self.account_types['liabilities']):
                if matrix[i, liability_offset + j] != 0:
                    debts[liability] = Decimal(str(matrix[i, liability_offset + j]))
            
            states.append(AccountingState(
                cash=cash,
                investments=investments,
                debts=debts,
                timestamp=timestamps[i]
            ))
        
        return states 