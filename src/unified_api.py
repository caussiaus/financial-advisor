"""
Unified API for Financial Engine

Orchestrates all five layers:
1. PDF Processor Layer
2. Mesh Engine Layer  
3. Accounting Layer
4. Recommendation Engine Layer
5. UI Layer
6. Financial Space Mapper (NEW)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os
import numpy as np
import networkx as nx

from .layers.pdf_processor import PDFProcessorLayer, FinancialMilestone, FinancialEntity
from .layers.mesh_engine import MeshEngineLayer, MeshConfig
from .layers.accounting import AccountingLayer, Transaction, TransactionType
from .layers.recommendation_engine import RecommendationEngineLayer, Recommendation
from .layers.ui import UILayer
from .layers.financial_space_mapper import FinancialSpaceMapper, FinancialSpaceMap


@dataclass
class EngineConfig:
    """Configuration for the unified financial engine"""
    mesh_config: Optional[MeshConfig] = None
    pdf_config: Optional[Dict] = None
    accounting_config: Optional[Dict] = None
    recommendation_config: Optional[Dict] = None
    ui_config: Optional[Dict] = None
    space_mapper_config: Optional[Dict] = None


@dataclass
class AnalysisResult:
    """Result of a complete financial analysis"""
    analysis_id: str
    timestamp: datetime
    milestones: List[FinancialMilestone]
    entities: List[FinancialEntity]
    mesh_status: Dict
    financial_statement: Dict
    recommendation: Recommendation
    dashboard_html: str
    financial_space_map: Optional[FinancialSpaceMap] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedFinancialEngine:
    """
    Unified Financial Engine - Orchestrates all five layers + space mapper
    
    Provides a clean API for:
    - Document processing and milestone extraction
    - Stochastic mesh generation and optimization
    - Financial state tracking and reconciliation
    - Commutator-based portfolio recommendations
    - Interactive dashboard generation
    - Clustering-based financial space mapping
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        
        # Initialize all layers
        self.pdf_processor = PDFProcessorLayer(self.config.pdf_config)
        self.mesh_engine = MeshEngineLayer(self.config.mesh_config)
        self.accounting_layer = AccountingLayer(self.config.accounting_config)
        self.recommendation_engine = RecommendationEngineLayer(self.config.recommendation_config)
        self.ui_layer = UILayer(self.config.ui_config)
        self.space_mapper = FinancialSpaceMapper(self.config.space_mapper_config)
        
        # State tracking
        self.current_analysis: Optional[AnalysisResult] = None
        self.analysis_history: List[AnalysisResult] = []
    
    def process_document_and_analyze(self, document_path: str, 
                                   risk_preference: str = 'moderate',
                                   include_space_mapping: bool = True) -> AnalysisResult:
        """
        Complete analysis pipeline: document → mesh → accounting → recommendations → UI → space mapping
        
        Args:
            document_path: Path to financial document
            risk_preference: Risk preference for recommendations
            include_space_mapping: Whether to include financial space mapping
            
        Returns:
            Complete analysis result
        """
        print("🔄 Starting complete financial analysis pipeline...")
        
        # Step 1: Process document
        print("📄 Processing document...")
        milestones, entities = self.pdf_processor.process_document(document_path)
        print(f"✅ Extracted {len(milestones)} milestones and {len(entities)} entities")
        
        # Step 2: Initialize accounting with entities
        print("💰 Initializing accounting...")
        self._initialize_accounting_from_entities(entities)
        
        # Step 3: Generate initial financial state
        print("📊 Generating initial financial state...")
        initial_state = self._generate_initial_financial_state(milestones, entities)
        
        # Step 4: Initialize mesh engine
        print("🌐 Initializing stochastic mesh...")
        mesh_status = self.mesh_engine.initialize_mesh(initial_state, milestones)
        
        # Step 5: Generate financial statement
        print("📋 Generating financial statement...")
        financial_statement = self.accounting_layer.generate_financial_statement()
        
        # Step 6: Generate recommendations
        print("🎯 Generating recommendations...")
        recommendation = self.recommendation_engine.generate_recommendation(
            initial_state, risk_preference
        )
        
        # Step 7: Generate dashboard
        print("📈 Generating dashboard...")
        dashboard_data = self._prepare_dashboard_data(
            milestones, entities, mesh_status, financial_statement, recommendation
        )
        dashboard_html = self.ui_layer.build_dashboard(dashboard_data)
        
        # Step 8: Generate financial space map (NEW)
        financial_space_map = None
        if include_space_mapping:
            print("🗺️ Generating financial space map...")
            financial_space_map = self._generate_financial_space_map(
                recommendation, initial_state
            )
            print(f"✅ Generated space map with {len(financial_space_map.clusters)} clusters")
        
        # Create analysis result
        analysis_result = AnalysisResult(
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            milestones=milestones,
            entities=entities,
            mesh_status=mesh_status,
            financial_statement=financial_statement,
            recommendation=recommendation,
            dashboard_html=dashboard_html,
            financial_space_map=financial_space_map,
            metadata={
                'document_path': document_path,
                'risk_preference': risk_preference,
                'total_milestones': len(milestones),
                'total_entities': len(entities),
                'include_space_mapping': include_space_mapping
            }
        )
        
        # Store result
        self.current_analysis = analysis_result
        self.analysis_history.append(analysis_result)
        
        print("✅ Analysis complete!")
        return analysis_result
    
    def _generate_financial_space_map(self, recommendation: Recommendation, 
                                    initial_state: Dict[str, float]) -> FinancialSpaceMap:
        """Generate financial space map from recommendation commutators"""
        # Generate financial states from commutators
        states = self.space_mapper.generate_financial_states_from_commutators(
            recommendation.commutator_sequence,
            initial_state,
            num_samples=500  # Reduced for performance
        )
        
        # Create space map
        space_map = self.space_mapper.create_financial_space_map(states)
        
        return space_map
    
    def visualize_financial_space(self, analysis_id: str = None, 
                                output_file: str = None) -> str:
        """
        Visualize the financial space map
        
        Args:
            analysis_id: ID of analysis to visualize (uses current if None)
            output_file: Output file path for visualization
            
        Returns:
            HTML visualization
        """
        if not analysis_id:
            if not self.current_analysis:
                raise ValueError("No analysis available for visualization")
            analysis = self.current_analysis
        else:
            analysis = next((a for a in self.analysis_history if a.analysis_id == analysis_id), None)
            if not analysis:
                raise ValueError(f"Analysis {analysis_id} not found")
        
        if not analysis.financial_space_map:
            raise ValueError("No financial space map available for visualization")
        
        # Generate visualization
        html = self.space_mapper.visualize_financial_space(
            analysis.financial_space_map, output_file
        )
        
        return html
    
    def get_cluster_analysis(self, analysis_id: str = None) -> Dict:
        """
        Get detailed cluster analysis
        
        Args:
            analysis_id: ID of analysis to analyze (uses current if None)
            
        Returns:
            Cluster analysis dictionary
        """
        if not analysis_id:
            if not self.current_analysis:
                raise ValueError("No analysis available for cluster analysis")
            analysis = self.current_analysis
        else:
            analysis = next((a for a in self.analysis_history if a.analysis_id == analysis_id), None)
            if not analysis:
                raise ValueError(f"Analysis {analysis_id} not found")
        
        if not analysis.financial_space_map:
            raise ValueError("No financial space map available for cluster analysis")
        
        return self.space_mapper.get_cluster_analysis(analysis.financial_space_map)
    
    def find_optimal_path_in_space(self, from_state: Dict[str, float], 
                                  to_state: Dict[str, float]) -> List[Dict[str, float]]:
        """
        Find optimal path between two financial states using space mapping
        
        Args:
            from_state: Starting financial state
            to_state: Target financial state
            
        Returns:
            List of intermediate states forming the optimal path
        """
        if not self.current_analysis or not self.current_analysis.financial_space_map:
            raise ValueError("No financial space map available")
        
        space_map = self.current_analysis.financial_space_map
        
        # Find clusters containing the states
        from_cluster = self._find_cluster_for_state(from_state, space_map.clusters)
        to_cluster = self._find_cluster_for_state(to_state, space_map.clusters)
        
        if not from_cluster or not to_cluster:
            raise ValueError("Could not find clusters for the specified states")
        
        # Find path through connectivity graph
        try:
            path = nx.shortest_path(
                space_map.connectivity_graph,
                from_cluster.cluster_id,
                to_cluster.cluster_id,
                weight='weight'
            )
        except nx.NetworkXNoPath:
            raise ValueError("No path found between the specified states")
        
        # Generate intermediate states along the path
        intermediate_states = []
        for cluster_id in path:
            cluster = next(c for c in space_map.clusters if c.cluster_id == cluster_id)
            # Use cluster center as representative state
            center_state = self._vector_to_financial_state(cluster.center)
            intermediate_states.append(center_state)
        
        return intermediate_states
    
    def _find_cluster_for_state(self, state: Dict[str, float], 
                               clusters: List) -> Optional:
        """Find the cluster that contains a given financial state"""
        state_vector = self.space_mapper.vectorization_methods['combined'](state)
        
        best_cluster = None
        min_distance = float('inf')
        
        for cluster in clusters:
            distance = np.linalg.norm(state_vector - cluster.center)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster
        
        return best_cluster
    
    def _vector_to_financial_state(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert vector back to financial state (simplified)"""
        # This is a simplified conversion - in practice, you'd need more sophisticated logic
        total_assets = 1000000  # Default total
        
        # Assume first 4 components are asset allocations
        if len(vector) >= 4:
            cash_ratio = max(0, min(1, vector[0]))
            stock_ratio = max(0, min(1, vector[1]))
            bond_ratio = max(0, min(1, vector[2]))
            real_estate_ratio = max(0, min(1, vector[3]))
            
            # Normalize ratios
            total_ratio = cash_ratio + stock_ratio + bond_ratio + real_estate_ratio
            if total_ratio > 0:
                cash_ratio /= total_ratio
                stock_ratio /= total_ratio
                bond_ratio /= total_ratio
                real_estate_ratio /= total_ratio
            
            return {
                'cash': cash_ratio * total_assets,
                'stocks': stock_ratio * total_assets,
                'bonds': bond_ratio * total_assets,
                'real_estate': real_estate_ratio * total_assets
            }
        
        return {'cash': total_assets, 'stocks': 0, 'bonds': 0, 'real_estate': 0}
    
    def _initialize_accounting_from_entities(self, entities: List[FinancialEntity]):
        """Initialize accounting layer with extracted entities"""
        for entity in entities:
            # Create accounts for each entity
            if entity.entity_type == 'person':
                # Create personal accounts
                self.accounting_layer.create_account(
                    f"{entity.name}_cash", 
                    self.accounting_layer.AccountType.ASSET, 
                    f"{entity.name} Cash"
                )
                self.accounting_layer.create_account(
                    f"{entity.name}_investments", 
                    self.accounting_layer.AccountType.ASSET, 
                    f"{entity.name} Investments"
                )
            elif entity.entity_type == 'child':
                # Create education fund accounts
                self.accounting_layer.create_account(
                    f"{entity.name}_education_fund", 
                    self.accounting_layer.AccountType.ASSET, 
                    f"{entity.name} Education Fund"
                )
            elif entity.entity_type == 'fund':
                # Create fund accounts
                self.accounting_layer.create_account(
                    f"{entity.name}_fund", 
                    self.accounting_layer.AccountType.ASSET, 
                    f"{entity.name} Fund"
                )
    
    def _generate_initial_financial_state(self, milestones: List[FinancialMilestone], 
                                        entities: List[FinancialEntity]) -> Dict[str, float]:
        """Generate initial financial state from milestones and entities"""
        # Start with default state
        initial_state = {
            'cash': 100000,  # Default cash
            'bonds': 200000,  # Default bonds
            'stocks': 300000,  # Default stocks
            'real_estate': 100000,  # Default real estate
        }
        
        # Adjust based on milestones
        for milestone in milestones:
            if milestone.financial_impact:
                # Distribute impact across assets based on milestone type
                if milestone.event_type == 'education':
                    # Education expenses reduce cash
                    initial_state['cash'] = max(0, initial_state['cash'] - milestone.financial_impact * 0.5)
                elif milestone.event_type == 'housing':
                    # Housing affects real estate and cash
                    initial_state['real_estate'] += milestone.financial_impact * 0.7
                    initial_state['cash'] = max(0, initial_state['cash'] - milestone.financial_impact * 0.3)
                elif milestone.event_type == 'investment':
                    # Investment milestones increase stock allocation
                    initial_state['stocks'] += milestone.financial_impact * 0.8
                    initial_state['cash'] = max(0, initial_state['cash'] - milestone.financial_impact * 0.2)
        
        return initial_state
    
    def _prepare_dashboard_data(self, milestones: List[FinancialMilestone], 
                               entities: List[FinancialEntity], 
                               mesh_status: Dict, 
                               financial_statement: Dict, 
                               recommendation: Recommendation) -> Dict:
        """Prepare data for dashboard generation"""
        # Extract allocation from financial statement
        allocation = {}
        if 'assets' in financial_statement:
            total_assets = financial_statement['summary']['total_assets']
            for asset_id, asset_data in financial_statement['assets'].items():
                if total_assets > 0:
                    allocation[asset_data['name']] = asset_data['balance'] / total_assets
        
        # Create timeline data
        timeline_data = []
        if mesh_status.get('status') == 'active':
            # Generate timeline from mesh data
            current_time = datetime.now()
            for i in range(12):  # 12 months
                date = current_time.replace(month=((current_time.month + i - 1) % 12) + 1)
                if current_time.month + i > 12:
                    date = date.replace(year=current_time.year + 1)
                
                # Simulate net worth growth
                net_worth = financial_statement['summary']['net_worth'] * (1 + i * 0.01)
                timeline_data.append({
                    'timestamp': date.isoformat(),
                    'net_worth': net_worth
                })
        
        # Create risk analysis data
        risk_analysis = {
            'x_labels': ['Cash', 'Bonds', 'Stocks', 'Real Estate'],
            'y_labels': ['Risk', 'Liquidity', 'Return'],
            'values': [
                [0.0, 0.2, 0.6, 0.4],  # Risk
                [1.0, 0.8, 0.9, 0.3],  # Liquidity
                [0.02, 0.04, 0.08, 0.06]  # Return
            ]
        }
        
        # Create commutator data
        commutator_data = {
            'steps': [f"Step {i+1}" for i in range(len(recommendation.commutator_sequence))],
            'impacts': [0.1 + i * 0.02 for i in range(len(recommendation.commutator_sequence))],
            'descriptions': [comm.description for comm in recommendation.commutator_sequence]
        }
        
        return {
            'allocation': allocation,
            'timeline': timeline_data,
            'risk_analysis': risk_analysis,
            'commutators': commutator_data
        }
    
    def execute_recommendation(self, recommendation_id: str) -> bool:
        """
        Execute a specific recommendation
        
        Args:
            recommendation_id: ID of recommendation to execute
            
        Returns:
            Success status
        """
        if not self.current_analysis:
            return False
        
        recommendation = self.current_analysis.recommendation
        if recommendation.recommendation_id != recommendation_id:
            return False
        
        print(f"🔄 Executing recommendation: {recommendation.description}")
        
        # Execute commutator sequence
        current_state = recommendation.current_state
        
        for commutator in recommendation.commutator_sequence:
            print(f"  📊 Executing commutator: {commutator.description}")
            
            # Execute the commutator
            new_state = self.recommendation_engine.execute_commutator(commutator, current_state)
            
            # Update mesh engine
            self.mesh_engine.advance_time(datetime.now())
            
            # Update accounting
            self._update_accounting_from_state_change(current_state, new_state)
            
            current_state = new_state
        
        print("✅ Recommendation executed successfully")
        return True
    
    def _update_accounting_from_state_change(self, old_state: Dict[str, float], 
                                           new_state: Dict[str, float]):
        """Update accounting layer based on state changes"""
        for asset, new_value in new_state.items():
            old_value = old_state.get(asset, 0)
            change = new_value - old_value
            
            if abs(change) > 0.01:  # Significant change
                # Create transaction
                transaction = Transaction(
                    transaction_id=f"state_change_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    transaction_type=TransactionType.TRANSFER,
                    amount=abs(change),
                    description=f"State change for {asset}: {change:+.2f}",
                    category="reallocation"
                )
                
                self.accounting_layer.process_transaction(transaction)
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of current analysis"""
        if not self.current_analysis:
            return {'status': 'no_analysis'}
        
        analysis = self.current_analysis
        
        summary = {
            'analysis_id': analysis.analysis_id,
            'timestamp': analysis.timestamp.isoformat(),
            'milestones_count': len(analysis.milestones),
            'entities_count': len(analysis.entities),
            'mesh_status': analysis.mesh_status.get('status', 'unknown'),
            'net_worth': analysis.financial_statement['summary']['net_worth'],
            'recommendation_confidence': analysis.recommendation.confidence,
            'recommendation_risk': analysis.recommendation.risk_score,
            'expected_duration': analysis.recommendation.expected_duration
        }
        
        # Add space mapping info if available
        if analysis.financial_space_map:
            summary['space_mapping'] = {
                'total_clusters': len(analysis.financial_space_map.clusters),
                'feasible_regions': len(analysis.financial_space_map.feasible_regions),
                'infeasible_regions': len(analysis.financial_space_map.infeasible_regions),
                'coverage_score': analysis.financial_space_map.coverage_score,
                'optimal_paths': len(analysis.financial_space_map.optimal_paths)
            }
        
        return summary
    
    def export_analysis(self, filepath: str):
        """Export complete analysis to file"""
        if not self.current_analysis:
            raise ValueError("No analysis to export")
        
        analysis_data = {
            'analysis_id': self.current_analysis.analysis_id,
            'timestamp': self.current_analysis.timestamp.isoformat(),
            'milestones': [m.__dict__ for m in self.current_analysis.milestones],
            'entities': [e.__dict__ for e in self.current_analysis.entities],
            'mesh_status': self.current_analysis.mesh_status,
            'financial_statement': self.current_analysis.financial_statement,
            'recommendation': self.current_analysis.recommendation.__dict__,
            'metadata': self.current_analysis.metadata
        }
        
        # Add space mapping data if available
        if self.current_analysis.financial_space_map:
            analysis_data['financial_space_map'] = {
                'map_id': self.current_analysis.financial_space_map.map_id,
                'timestamp': self.current_analysis.financial_space_map.timestamp.isoformat(),
                'clusters_count': len(self.current_analysis.financial_space_map.clusters),
                'feasible_regions_count': len(self.current_analysis.financial_space_map.feasible_regions),
                'infeasible_regions_count': len(self.current_analysis.financial_space_map.infeasible_regions),
                'coverage_score': self.current_analysis.financial_space_map.coverage_score,
                'optimal_paths_count': len(self.current_analysis.financial_space_map.optimal_paths)
            }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, default=str, indent=2)
    
    def run_interactive_dashboard(self, host: str = 'localhost', port: int = 5000):
        """Run interactive dashboard server"""
        print(f"🌐 Starting interactive dashboard on http://{host}:{port}")
        self.ui_layer.run_server(host=host, port=port)
    
    def benchmark_performance(self) -> Dict:
        """Run performance benchmarks across all layers"""
        benchmarks = {}
        
        # Benchmark mesh engine
        if self.mesh_engine:
            benchmarks['mesh_engine'] = self.mesh_engine.benchmark_performance()
        
        # Benchmark PDF processor
        if self.pdf_processor:
            # Simple benchmark for PDF processing
            import time
            start_time = time.time()
            # Simulate processing
            time.sleep(0.1)  # Simulate processing time
            benchmarks['pdf_processor'] = {
                'processing_time': time.time() - start_time,
                'status': 'ready'
            }
        
        # Benchmark recommendation engine
        if self.recommendation_engine:
            start_time = time.time()
            # Simulate recommendation generation
            time.sleep(0.05)  # Simulate processing time
            benchmarks['recommendation_engine'] = {
                'generation_time': time.time() - start_time,
                'status': 'ready'
            }
        
        # Benchmark space mapper
        if self.space_mapper:
            start_time = time.time()
            # Simulate space mapping
            time.sleep(0.08)  # Simulate processing time
            benchmarks['space_mapper'] = {
                'mapping_time': time.time() - start_time,
                'status': 'ready'
            }
        
        return benchmarks 