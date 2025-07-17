"""
Omega Mesh Integration Engine

This module orchestrates the end-to-end pipeline for adaptive, scenario-based financial planning:
- NLP-based milestone/entity extraction from narrative PDFs
- Stochastic mesh (omega) generation for all possible financial futures
- Adaptive database and similarity matching for event clustering
- Ultra-flexible payment execution and scheduling
- Real-time accounting and constraint validation
- Monthly recommendations and configuration matrices
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from .enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone, FinancialEntity
from .stochastic_mesh_engine import StochasticMeshEngine
from .accounting_reconciliation import AccountingReconciliationEngine
from .adaptive_mesh_generator import AdaptiveMeshGenerator
from .financial_recommendation_engine import FinancialRecommendationEngine
from .mesh_memory_manager import MeshMemoryManager
from .enhanced_client_profile import TextToMathematicalConverter, ClientProfile, VectorDBManager

class OmegaMeshIntegration:
    """
    Orchestrates the Omega Mesh financial planning pipeline.
    """
    def __init__(self, initial_financial_state: Dict[str, float]):
        # NLP-based PDF processor
        self.pdf_processor = EnhancedPDFProcessor()
        # Stochastic mesh engine (omega mesh)
        self.mesh_engine = StochasticMeshEngine(current_financial_state=initial_financial_state)
        # Memory manager for efficient mesh storage
        self.memory_manager = MeshMemoryManager(max_nodes=10000)
        # Adaptive mesh generator for similarity matching and clustering
        self.adaptive_mesh = AdaptiveMeshGenerator(initial_state=initial_financial_state, memory_manager=self.memory_manager)
        # Double-entry accounting engine
        self.accounting_engine = AccountingReconciliationEngine()
        # Recommendation engine
        self.recommendation_engine = FinancialRecommendationEngine(self.mesh_engine, self.accounting_engine)
        # Enhanced client profile system
        self.client_profile_converter = TextToMathematicalConverter()
        self.vector_db = VectorDBManager()

        self.milestones: List[FinancialMilestone] = []
        self.entities: List[FinancialEntity] = []
        self.payment_history: List[Dict] = []
        self.system_status = {
            'initialized': True,
            'mesh_active': False,
            'milestones_loaded': False,
            'entities_loaded': False,
            'last_update': datetime.now()
        }
        self._initialize_accounting_state(initial_financial_state)
        print("🌟 Omega Mesh Integration System Initialized!")
        print("Ready to process PDFs and create stochastic financial mesh.")

    def _initialize_accounting_state(self, initial_state: Dict[str, float]):
        """Initialize accounting engine with initial financial state."""
        print(f"💰 Accounting initialized with ${initial_state.get('total_wealth', 0):,.2f} total wealth")
        # Set initial balances for key accounts
        if 'total_wealth' in initial_state:
            total_wealth = initial_state['total_wealth']
            from decimal import Decimal
            self.accounting_engine.set_account_balance('cash_checking', Decimal(str(total_wealth * 0.1)))
            self.accounting_engine.set_account_balance('cash_savings', Decimal(str(total_wealth * 0.2)))
            self.accounting_engine.set_account_balance('investments_stocks', Decimal(str(total_wealth * 0.3)))
            self.accounting_engine.set_account_balance('investments_bonds', Decimal(str(total_wealth * 0.2)))
            self.accounting_engine.set_account_balance('investments_retirement', Decimal(str(total_wealth * 0.2)))

    def process_ips_document(self, pdf_path: str) -> Tuple[List[FinancialMilestone], List[FinancialEntity]]:
        """
        Process IPS document to extract milestones and entities, then initialize the mesh.
        """
        print(f"📄 Processing IPS document: {pdf_path}")
        try:
            # Extract basic milestones and entities
            self.milestones, self.entities = self.pdf_processor.process_pdf(pdf_path)
            
            # Extract comprehensive client profile from text content
            text_content = self._extract_text_content(pdf_path)
            self.client_profile = self.client_profile_converter.extract_client_profile(
                text_content, self.entities, self.milestones
            )
            
            # Add client profile to vector DB for similarity matching
            self.vector_db.add_profile(self.client_profile)
            
            # Get financial backing recommendations from similar profiles
            financial_backing = self.vector_db.get_financial_backing(self.client_profile)
            
            print(f"🎯 Extracted {len(self.milestones)} financial milestones.")
            print(f"👥 Extracted {len(self.entities)} financial entities.")
            print(f"👤 Client Archetype: {self.client_profile.archetype.value}")
            print(f"💰 Portfolio Value: ${self.client_profile.portfolio_value:,0.2f}")
            print(f"📊 Risk Tolerance: {self.client_profile.risk_tolerance.value}")
            
            # Initialize entity accounting with enhanced profile
            self._initialize_entity_accounting_with_profile(self.client_profile)
            
            # Initialize mesh with shock modeling
            self._initialize_mesh_with_shocks(self.client_profile)
            
            self.system_status['milestones_loaded'] = True
            self.system_status['entities_loaded'] = True
            self.system_status['mesh_active'] = True
            self.system_status['last_update'] = datetime.now()
            self.system_status['client_profile'] = {
                'archetype': self.client_profile.archetype.value,
                'risk_tolerance': self.client_profile.risk_tolerance.value,
                'portfolio_value': self.client_profile.portfolio_value,
                'financial_backing': financial_backing
            }
            
            return self.milestones, self.entities
        except Exception as e:
            print(f"Error processing document: {e}")
            import traceback
            print(traceback.format_exc())
            return [], []

    def _extract_text_content(self, pdf_path: str) -> str:
        """Extract text content from PDF for client profile analysis."""
        try:
            import pdfplumber
            text_content = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"
            return text_content
        except Exception as e:
            print(f"Error extracting text content: {e}")
            return ""

    def _initialize_entity_accounting_with_profile(self, client_profile: ClientProfile):
        """Initialize accounting framework with enhanced client profile."""
        print("🏦 Initializing entity-based accounting framework with enhanced profile...")
        
        # Apply mathematical implications from client profile
        cash_flow_multipliers = client_profile.cash_flow_multipliers
        risk_adjustment_factors = client_profile.risk_adjustment_factors
        
        for entity in self.entities:
            entity_state = {
                'entity_name': entity.name,
                'entity_type': entity.entity_type,
                'balances': entity.initial_balances.copy(),
                'transactions': [],
                'milestones': [m for m in self.milestones if m.entity == entity.name]
            }
            
            # Apply cash flow multipliers to balances
            for balance_type, amount in entity_state['balances'].items():
                if balance_type in cash_flow_multipliers:
                    entity_state['balances'][balance_type] = amount * cash_flow_multipliers[balance_type]
            
            self.accounting_engine.register_entity(entity_state)
            print(f"📊 Registered {entity.name} with enhanced balances: {entity_state['balances']}")

    def _initialize_mesh_with_shocks(self, client_profile: ClientProfile):
        """Initialize mesh with shock modeling from client profile."""
        print("⚡ Initializing mesh with shock modeling...")
        
        # Add shock milestones to the mesh
        shock_milestones = []
        for shock in client_profile.potential_shocks:
            shock_milestone = FinancialMilestone(
                timestamp=datetime.now() + timedelta(days=365),  # Default to 1 year
                event_type='shock',
                description=shock['description'],
                financial_impact=shock['financial_impact'],
                probability=shock['probability'],
                entity='system',
                metadata={'shock_type': shock['type'], 'timing': shock['timing']}
            )
            shock_milestones.append(shock_milestone)
        
        # Combine regular milestones with shock milestones
        all_milestones = self.milestones + shock_milestones
        
        # Initialize mesh with enhanced milestone set
        self.mesh_engine.initialize_mesh(all_milestones)
        
        print(f"✅ Initialized mesh with {len(all_milestones)} milestones ({len(shock_milestones)} shocks)")

    def _initialize_entity_accounting(self):
        """Initialize accounting framework for all entities."""
        print("🏦 Initializing entity-based accounting framework...")
        for entity in self.entities:
            entity_state = {
                'entity_name': entity.name,
                'entity_type': entity.entity_type,
                'balances': entity.initial_balances.copy(),
                'transactions': [],
                'milestones': [m for m in self.milestones if m.entity == entity.name]
            }
            self.accounting_engine.register_entity(entity_state)
            print(f"  📊 Registered {entity.name} with balances: {entity.initial_balances}")

    def _create_sample_milestones(self) -> List[FinancialMilestone]:
        """Create sample milestones for demonstration purposes."""
        now = datetime.now()
        return [
            FinancialMilestone(timestamp=now, event_type='education', description='Start college', financial_impact=80000, probability=0.9, entity='Child'),
            FinancialMilestone(timestamp=now, event_type='housing', description='Buy house', financial_impact=300000, probability=0.7, entity='Horatio'),
            FinancialMilestone(timestamp=now, event_type='investment', description='Portfolio rebalancing', financial_impact=50000, probability=0.8, entity='Horatio')
        ]

    def _create_sample_entities(self) -> List[FinancialEntity]:
        """Create sample entities for demonstration purposes."""
        return [
            FinancialEntity(name='Horatio', entity_type='person', initial_balances={'salary': 100000, 'savings': 20000}, metadata={'age': 45, 'occupation': 'Professional'}),
            FinancialEntity(name='Child', entity_type='child', initial_balances={'education_fund': 0}, metadata={'age': 15, 'education_level': 'school'})
        ]

    def demonstrate_flexible_payment(self, milestone_idx: int = 0, amount: Optional[float] = None, payment_date: Optional[datetime] = None) -> Dict:
        """
        Demonstrate ultra-flexible payment execution with entity tracking.
        """
        print("💳 Demonstrating ultra-flexible payment execution...")
        if not self.milestones:
            print("No milestones available for payment demonstration.")
            return {}
        milestone = self.milestones[milestone_idx]
        entity_name = milestone.entity or 'Unknown'
        payment_amount = amount if amount is not None else (milestone.financial_impact or 10000)
        payment_date = payment_date or datetime.now()
        print(f"Processing payment for: {milestone.event_type} | Entity: {entity_name} | Amount: ${payment_amount:,.2f}")
        payment_result = self.mesh_engine.execute_payment(
            milestone_id=f"{entity_name}_{milestone.event_type}",
            amount=payment_amount,
            payment_date=payment_date
        )
        if payment_result:
            self._update_entity_balances(entity_name, milestone.event_type, payment_amount)
        return {
            'milestone': milestone.event_type,
            'entity': entity_name,
            'amount': payment_amount,
            'payment_successful': payment_result,
            'timestamp': payment_date.isoformat()
        }

    def _update_entity_balances(self, entity_name: str, event_type: str, amount: float):
        """Update entity balances after payment."""
        for entity in self.entities:
            if entity.name == entity_name:
                if event_type in entity.initial_balances:
                    entity.initial_balances[event_type] += amount
                else:
                    entity.initial_balances[event_type] = amount
                print(f"Updated {entity_name} {event_type} balance: ${entity.initial_balances[event_type]:,.2f}")
                break

    def run_adaptive_mesh_similarity(self):
        """
        Run adaptive mesh generator to cluster similar events and states for optimization.
        """
        print("🔄 Running adaptive mesh similarity matching...")
        if not self.milestones:
            print("No milestones loaded for similarity matching.")
            return set()
        mesh_paths = self.mesh_engine.get_all_paths()
        milestone_clusters = self.adaptive_mesh.cluster_milestones(self.milestones)
        unique_states = self.adaptive_mesh._aggregate_similar_states(mesh_paths)
        print(f"Aggregated {len(unique_states)} unique financial states after similarity matching.")
        return unique_states

    def generate_monthly_recommendations(self, months_ahead: int = 12) -> List[Dict]:
        """
        Generate monthly recommendations for purchases, investments, and reallocations.
        """
        print("📅 Generating monthly recommendations...")
        if not self.milestones:
            print("No milestones loaded for recommendations.")
            return []
        profile_data = {'base_income': 100000, 'risk_tolerance': 'moderate'}  # Example; should be dynamic
        recommendations = self.recommendation_engine.generate_monthly_recommendations(
            self.milestones, profile_data, months_ahead=months_ahead
        )
        print(f"Generated {len(recommendations)} monthly recommendations.")
        return recommendations

    def create_configuration_matrix(self, scenarios: int = 3) -> Dict:
        """
        Create configuration matrices showing possible financial paths and allocations over time.
        """
        print("🧮 Creating configuration matrix...")
        if not self.milestones:
            print("No milestones loaded for configuration matrix.")
            return {}
        profile_data = {'base_income': 100000, 'risk_tolerance': 'moderate'}  # Example; should be dynamic
        config_matrix = self.recommendation_engine.create_configuration_matrix(
            'user', self.milestones, profile_data, scenarios=scenarios
        )
        print("Configuration matrix created.")
        return config_matrix

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            **self.system_status,
            'milestones_count': len(self.milestones),
            'entities_count': len(self.entities),
            'mesh_nodes_count': len(self.mesh_engine.nodes) if hasattr(self.mesh_engine, 'nodes') else 0
        } 

    def get_client_profile_summary(self) -> Dict:
        """A summary of the client profile and financial implications."""
        if not hasattr(self, 'client_profile'):
            return {}
        
        return {
            'archetype': self.client_profile.archetype.value,
            'risk_tolerance': self.client_profile.risk_tolerance.value,
            'life_stage': self.client_profile.life_stage.value,
            'portfolio_value': self.client_profile.portfolio_value,
            'net_worth': self.client_profile.net_worth,
            'base_income': self.client_profile.base_income,
            'family_size': self.client_profile.family_size,
            'dependents': self.client_profile.dependents,
            'potential_shocks': len(self.client_profile.potential_shocks),
            'cash_flow_multipliers': self.client_profile.cash_flow_multipliers,
            'risk_adjustment_factors': self.client_profile.risk_adjustment_factors
        }

    def get_financial_backing_recommendations(self) -> Dict:
        """Financial backing recommendations from vector DB."""
        if not hasattr(self, 'client_profile'):
            return {}
        
        return self.vector_db.get_financial_backing(self.client_profile) 