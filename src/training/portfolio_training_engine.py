"""
Portfolio Training Engine with Stochastic Gradient Descent

This module implements a training process that uses stochastic gradient descent
to optimize portfolio composition changes across multiple mesh engines.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import logging
from decimal import Decimal
import random

# Import existing mesh engines
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine
from src.integration.mesh_engine_layer import MeshEngineLayer
from src.accounting_reconciliation import AccountingReconciliationEngine
from src.enhanced_accounting_logger import EnhancedAccountingLogger, FlowItemCategory, BalanceItemCategory


@dataclass
class PortfolioWeights:
    """Represents portfolio allocation weights"""
    cash: float = 0.2
    bonds: float = 0.3
    stocks: float = 0.4
    real_estate: float = 0.1
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cash': self.cash,
            'bonds': self.bonds,
            'stocks': self.stocks,
            'real_estate': self.real_estate
        }
    
    def from_dict(self, weights: Dict[str, float]):
        self.cash = weights.get('cash', 0.2)
        self.bonds = weights.get('bonds', 0.3)
        self.stocks = weights.get('stocks', 0.4)
        self.real_estate = weights.get('real_estate', 0.1)
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = self.cash + self.bonds + self.stocks + self.real_estate
        if total > 0:
            self.cash /= total
            self.bonds /= total
            self.stocks /= total
            self.real_estate /= total


@dataclass
class TrainingScenario:
    """Represents a training scenario for portfolio optimization"""
    scenario_id: str
    initial_wealth: float
    time_horizon_years: float
    risk_tolerance: float  # 0.0 to 1.0
    age: int
    income_growth_rate: float
    market_volatility: float
    target_return: float
    constraints: Dict[str, Any] = field(default_factory=dict)


class PortfolioTrainingEngine:
    """
    Training engine that uses stochastic gradient descent to optimize
    portfolio composition changes across multiple mesh engines.
    """
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.logger = self._setup_logging()
        self.enhanced_logger = EnhancedAccountingLogger()
        
        # Training history
        self.training_history = {
            'losses': [],
            'portfolio_weights': [],
            'returns': [],
            'volatilities': []
        }
        
        # Initialize mesh engines
        self.stochastic_mesh = None
        self.time_uncertainty_mesh = None
        self.mesh_engine_layer = None
        self.accounting_engine = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training operations"""
        logger = logging.getLogger('portfolio_training_engine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_mesh_engines(self, initial_financial_state: Dict[str, float]):
        """Initialize all mesh engines for training"""
        self.logger.info("🚀 Initializing mesh engines for portfolio training...")
        
        # Initialize stochastic mesh engine
        self.stochastic_mesh = StochasticMeshEngine(initial_financial_state)
        
        # Initialize time uncertainty mesh engine
        self.time_uncertainty_mesh = TimeUncertaintyMeshEngine(use_gpu=True)
        
        # Initialize mesh engine layer
        self.mesh_engine_layer = MeshEngineLayer()
        
        # Initialize accounting engine
        self.accounting_engine = AccountingReconciliationEngine()
        
        self.logger.info("✅ All mesh engines initialized successfully")
    
    def generate_training_scenarios(self, num_scenarios: int = 100) -> List[TrainingScenario]:
        """Generate diverse training scenarios"""
        self.logger.info(f"📊 Generating {num_scenarios} training scenarios...")
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate random scenario parameters
            scenario = TrainingScenario(
                scenario_id=f"scenario_{i:04d}",
                initial_wealth=random.uniform(100000, 2000000),
                time_horizon_years=random.uniform(5, 20),
                risk_tolerance=random.uniform(0.1, 0.9),
                age=random.randint(25, 65),
                income_growth_rate=random.uniform(0.02, 0.08),
                market_volatility=random.uniform(0.15, 0.35),
                target_return=random.uniform(0.05, 0.12),
                constraints={
                    'max_stock_allocation': random.uniform(0.6, 0.8),
                    'min_cash_allocation': random.uniform(0.05, 0.15),
                    'max_real_estate_allocation': random.uniform(0.2, 0.4)
                }
            )
            scenarios.append(scenario)
        
        self.logger.info(f"✅ Generated {len(scenarios)} training scenarios")
        return scenarios
    
    def calculate_portfolio_loss(self, weights: PortfolioWeights, scenario: TrainingScenario) -> float:
        """
        Calculate loss function for portfolio optimization
        
        Loss components:
        1. Expected return deviation from target
        2. Risk penalty based on volatility
        3. Constraint violation penalty
        """
        # Normalize weights
        weights.normalize()
        
        # Calculate expected return (simplified model)
        expected_return = (
            weights.cash * 0.02 +  # Cash return
            weights.bonds * 0.04 +  # Bond return
            weights.stocks * 0.08 +  # Stock return
            weights.real_estate * 0.06  # Real estate return
        )
        
        # Calculate portfolio volatility (simplified)
        portfolio_volatility = (
            weights.cash * 0.01 +  # Cash volatility
            weights.bonds * 0.05 +  # Bond volatility
            weights.stocks * scenario.market_volatility +  # Stock volatility
            weights.real_estate * 0.15  # Real estate volatility
        )
        
        # Calculate loss components
        return_deviation = (expected_return - scenario.target_return) ** 2
        risk_penalty = portfolio_volatility ** 2 * (1 - scenario.risk_tolerance)
        
        # Constraint violations
        constraint_penalty = 0.0
        if weights.stocks > scenario.constraints.get('max_stock_allocation', 0.8):
            constraint_penalty += (weights.stocks - scenario.constraints['max_stock_allocation']) ** 2
        if weights.cash < scenario.constraints.get('min_cash_allocation', 0.05):
            constraint_penalty += (scenario.constraints['min_cash_allocation'] - weights.cash) ** 2
        if weights.real_estate > scenario.constraints.get('max_real_estate_allocation', 0.3):
            constraint_penalty += (weights.real_estate - scenario.constraints['max_real_estate_allocation']) ** 2
        
        total_loss = return_deviation + risk_penalty + constraint_penalty
        
        # Log portfolio metrics
        self.enhanced_logger.log_balance_item(
            category=BalanceItemCategory.INVESTMENT,
            account_id=f"portfolio_{scenario.scenario_id}",
            balance=Decimal(str(scenario.initial_wealth)),
            metadata={
                'expected_return': expected_return,
                'portfolio_volatility': portfolio_volatility,
                'weights': weights.to_dict(),
                'target_return': scenario.target_return,
                'risk_tolerance': scenario.risk_tolerance
            }
        )
        
        return total_loss
    
    def calculate_gradients(self, weights: PortfolioWeights, scenario: TrainingScenario) -> Dict[str, float]:
        """Calculate gradients for stochastic gradient descent"""
        epsilon = 1e-6
        gradients = {}
        
        # Calculate gradients for each weight component
        for weight_name in ['cash', 'bonds', 'stocks', 'real_estate']:
            # Create perturbed weights
            weights_plus = PortfolioWeights()
            weights_plus.from_dict(weights.to_dict())
            weights_minus = PortfolioWeights()
            weights_minus.from_dict(weights.to_dict())
            
            # Perturb the specific weight
            setattr(weights_plus, weight_name, getattr(weights_plus, weight_name) + epsilon)
            setattr(weights_minus, weight_name, getattr(weights_minus, weight_name) - epsilon)
            
            # Calculate finite difference gradient
            loss_plus = self.calculate_portfolio_loss(weights_plus, scenario)
            loss_minus = self.calculate_portfolio_loss(weights_minus, scenario)
            
            gradient = (loss_plus - loss_minus) / (2 * epsilon)
            gradients[weight_name] = gradient
        
        return gradients
    
    def update_portfolio_weights(self, weights: PortfolioWeights, gradients: Dict[str, float]):
        """Update portfolio weights using stochastic gradient descent"""
        weights.cash -= self.learning_rate * gradients.get('cash', 0.0)
        weights.bonds -= self.learning_rate * gradients.get('bonds', 0.0)
        weights.stocks -= self.learning_rate * gradients.get('stocks', 0.0)
        weights.real_estate -= self.learning_rate * gradients.get('real_estate', 0.0)
        
        # Ensure weights are non-negative
        weights.cash = max(0.0, weights.cash)
        weights.bonds = max(0.0, weights.bonds)
        weights.stocks = max(0.0, weights.stocks)
        weights.real_estate = max(0.0, weights.real_estate)
        
        # Normalize weights
        weights.normalize()
    
    def train_portfolio_optimization(self, scenarios: List[TrainingScenario], 
                                   num_epochs: int = 100) -> Dict[str, Any]:
        """
        Train portfolio optimization using stochastic gradient descent
        
        Args:
            scenarios: List of training scenarios
            num_epochs: Number of training epochs
            
        Returns:
            Training results and optimized portfolio weights
        """
        self.logger.info(f"🎯 Starting portfolio optimization training...")
        self.logger.info(f"   - {len(scenarios)} scenarios")
        self.logger.info(f"   - {num_epochs} epochs")
        self.logger.info(f"   - Learning rate: {self.learning_rate}")
        
        # Initialize optimal weights
        optimal_weights = PortfolioWeights()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_weights = []
            
            # Shuffle scenarios for stochastic training
            random.shuffle(scenarios)
            
            # Process scenarios in batches
            for i in range(0, len(scenarios), self.batch_size):
                batch_scenarios = scenarios[i:i + self.batch_size]
                batch_losses = []
                batch_gradients = []
                
                for scenario in batch_scenarios:
                    # Calculate loss and gradients
                    loss = self.calculate_portfolio_loss(optimal_weights, scenario)
                    gradients = self.calculate_gradients(optimal_weights, scenario)
                    
                    batch_losses.append(loss)
                    batch_gradients.append(gradients)
                
                # Average gradients across batch
                avg_gradients = {}
                for key in ['cash', 'bonds', 'stocks', 'real_estate']:
                    avg_gradients[key] = np.mean([g[key] for g in batch_gradients])
                
                # Update weights
                self.update_portfolio_weights(optimal_weights, avg_gradients)
                
                # Store batch metrics
                epoch_losses.extend(batch_losses)
                epoch_weights.append(optimal_weights.to_dict())
            
            # Calculate epoch metrics
            avg_epoch_loss = np.mean(epoch_losses)
            avg_epoch_return = self._calculate_portfolio_return(optimal_weights)
            avg_epoch_volatility = self._calculate_portfolio_volatility(optimal_weights)
            
            # Store training history
            self.training_history['losses'].append(avg_epoch_loss)
            self.training_history['portfolio_weights'].append(optimal_weights.to_dict())
            self.training_history['returns'].append(avg_epoch_return)
            self.training_history['volatilities'].append(avg_epoch_volatility)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                               f"Loss={avg_epoch_loss:.6f}, "
                               f"Return={avg_epoch_return:.4f}, "
                               f"Vol={avg_epoch_volatility:.4f}")
                
                # Log to enhanced logger
                self.enhanced_logger.log_flow_item(
                    category=FlowItemCategory.INVESTMENT,
                    amount=Decimal(str(avg_epoch_return)),
                    description=f"Training epoch {epoch + 1} portfolio return",
                    transaction_id=f"training_epoch_{epoch + 1}",
                    metadata={
                        'epoch': epoch + 1,
                        'loss': avg_epoch_loss,
                        'volatility': avg_epoch_volatility,
                        'weights': optimal_weights.to_dict()
                    }
                )
        
        # Final training results
        training_results = {
            'optimal_weights': optimal_weights.to_dict(),
            'final_loss': self.training_history['losses'][-1],
            'final_return': self.training_history['returns'][-1],
            'final_volatility': self.training_history['volatilities'][-1],
            'training_history': self.training_history,
            'num_epochs': num_epochs,
            'num_scenarios': len(scenarios)
        }
        
        self.logger.info("✅ Portfolio optimization training completed")
        self.logger.info(f"   - Final loss: {training_results['final_loss']:.6f}")
        self.logger.info(f"   - Final return: {training_results['final_return']:.4f}")
        self.logger.info(f"   - Final volatility: {training_results['final_volatility']:.4f}")
        
        return training_results
    
    def _calculate_portfolio_return(self, weights: PortfolioWeights) -> float:
        """Calculate expected portfolio return"""
        return (
            weights.cash * 0.02 +
            weights.bonds * 0.04 +
            weights.stocks * 0.08 +
            weights.real_estate * 0.06
        )
    
    def _calculate_portfolio_volatility(self, weights: PortfolioWeights) -> float:
        """Calculate expected portfolio volatility"""
        return (
            weights.cash * 0.01 +
            weights.bonds * 0.05 +
            weights.stocks * 0.20 +
            weights.real_estate * 0.15
        )
    
    def run_mesh_integration_test(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test the trained portfolio weights with mesh engines"""
        self.logger.info("🔄 Running mesh integration test with trained portfolio...")
        
        # Create test scenario
        test_scenario = TrainingScenario(
            scenario_id="test_scenario",
            initial_wealth=1000000,
            time_horizon_years=10,
            risk_tolerance=0.5,
            age=35,
            income_growth_rate=0.05,
            market_volatility=0.25,
            target_return=0.08
        )
        
        # Apply trained weights
        trained_weights = PortfolioWeights()
        trained_weights.from_dict(training_results['optimal_weights'])
        
        # Test with stochastic mesh engine
        if self.stochastic_mesh:
            initial_state = {
                'cash': test_scenario.initial_wealth * trained_weights.cash,
                'bonds': test_scenario.initial_wealth * trained_weights.bonds,
                'stocks': test_scenario.initial_wealth * trained_weights.stocks,
                'real_estate': test_scenario.initial_wealth * trained_weights.real_estate
            }
            
            # Initialize mesh with test scenario
            milestones = self._create_test_milestones(test_scenario)
            mesh_status = self.stochastic_mesh.initialize_mesh(milestones, test_scenario.time_horizon_years)
            
            # Get payment options and analyze
            payment_options = self.stochastic_mesh.get_payment_options()
            
            mesh_test_results = {
                'mesh_status': mesh_status,
                'payment_options_count': len(payment_options),
                'initial_state': initial_state,
                'applied_weights': trained_weights.to_dict()
            }
        else:
            mesh_test_results = {'error': 'Stochastic mesh not initialized'}
        
        # Test with accounting engine
        if self.accounting_engine:
            # Register test account
            self.accounting_engine.register_entity("test_portfolio", "portfolio")
            
            # Set initial balances
            for asset_type, amount in initial_state.items():
                account_id = f"test_portfolio_{asset_type}"
                self.accounting_engine.set_account_balance(account_id, Decimal(str(amount)))
            
            # Generate financial statement
            financial_statement = self.accounting_engine.generate_financial_statement()
            
            accounting_test_results = {
                'financial_statement': financial_statement,
                'total_assets': float(financial_statement['total_assets']),
                'net_worth': float(financial_statement['net_worth'])
            }
        else:
            accounting_test_results = {'error': 'Accounting engine not initialized'}
        
        integration_results = {
            'mesh_test': mesh_test_results,
            'accounting_test': accounting_test_results,
            'trained_weights': trained_weights.to_dict(),
            'test_scenario': {
                'initial_wealth': test_scenario.initial_wealth,
                'time_horizon_years': test_scenario.time_horizon_years,
                'risk_tolerance': test_scenario.risk_tolerance
            }
        }
        
        self.logger.info("✅ Mesh integration test completed")
        return integration_results
    
    def _create_test_milestones(self, scenario: TrainingScenario) -> List:
        """Create test milestones for mesh integration"""
        milestones = []
        current_time = datetime.now()
        
        # Create milestones over the time horizon
        for year in range(int(scenario.time_horizon_years)):
            milestone = {
                'timestamp': current_time + timedelta(days=365 * year),
                'event_type': 'portfolio_rebalancing',
                'description': f'Annual portfolio rebalancing year {year + 1}',
                'financial_impact': scenario.initial_wealth * 0.05,  # 5% annual adjustment
                'probability': 1.0
            }
            milestones.append(milestone)
        
        return milestones
    
    def export_training_results(self, training_results: Dict[str, Any], 
                              filepath: str = "portfolio_training_results.json"):
        """Export training results to file"""
        # Convert numpy arrays to lists for JSON serialization
        export_data = training_results.copy()
        
        # Convert training history
        for key in ['losses', 'returns', 'volatilities']:
            if key in export_data['training_history']:
                export_data['training_history'][key] = [float(x) for x in export_data['training_history'][key]]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Training results exported to {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics from enhanced logger"""
        return self.enhanced_logger.get_statistics() 