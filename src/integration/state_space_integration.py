"""
Enhanced Mesh Integration

This module integrates the enhanced mesh system with the existing unified cash flow model
to provide path-dependent analysis and state-space mesh capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import logging
from pathlib import Path

from ..core.state_space_mesh_engine import EnhancedMeshEngine, EnhancedMeshNode, CashFlowSeries, FinancialState
from ..unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent
from ..core.time_uncertainty_mesh import TimeUncertaintyMeshEngine

class EnhancedMeshIntegration:
    """
    Integration layer that connects enhanced mesh with existing systems
    
    This provides:
    - Path-dependent analysis using full cash flow series
    - State-space mesh visualization
    - Integration with existing time uncertainty mesh
    - Real-time cash flow tracking with historical context
    """
    
    def __init__(self, initial_state: Dict[str, float]):
        self.initial_state = initial_state
        self.enhanced_mesh = EnhancedMeshEngine(initial_state)
        self.unified_model = UnifiedCashFlowModel(initial_state)
        self.time_uncertainty_engine = TimeUncertaintyMeshEngine(use_gpu=True)
        
        # Integration state
        self.current_scenario_id = 0
        self.scenario_nodes = {}  # Map scenario_id to node_id
        self.path_analyses = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('enhanced_mesh_integration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_system(self) -> Dict[str, Any]:
        """Initialize the integrated system"""
        print("🚀 Initializing Enhanced Mesh Integration System")
        print("=" * 60)
        
        # Initialize enhanced mesh
        initial_node_id = self.enhanced_mesh.create_initial_node()
        print(f"✅ Enhanced mesh initialized with node: {initial_node_id}")
        
        # Initialize unified cash flow model
        case_events = self.unified_model.create_case_events_from_analysis()
        for event in case_events:
            self.unified_model.add_cash_flow_event(event)
        print(f"✅ Unified cash flow model initialized with {len(case_events)} events")
        
        # Initialize time uncertainty mesh
        mesh_data, risk_analysis = self.unified_model.initialize_time_uncertainty_mesh(
            num_scenarios=1000,
            time_horizon_years=5
        )
        print(f"✅ Time uncertainty mesh initialized with {len(mesh_data['time_steps'])} time steps")
        
        # Create scenario mappings
        self._create_scenario_mappings(mesh_data)
        
        return {
            'enhanced_mesh_nodes': len(self.enhanced_mesh.nodes),
            'unified_events': len(self.unified_model.cash_flow_events),
            'time_mesh_scenarios': len(mesh_data['scenario_weights']),
            'risk_analysis': risk_analysis
        }
    
    def _create_scenario_mappings(self, mesh_data: Dict):
        """Create mappings between time uncertainty scenarios and enhanced mesh nodes"""
        print("🔗 Creating scenario mappings...")
        
        time_steps = mesh_data['time_steps']
        states = mesh_data['states']
        scenario_weights = mesh_data['scenario_weights']
        
        for scenario_id in range(states.shape[1]):
            # Create a path for this scenario
            scenario_path = []
            
            for t, timestamp in enumerate(time_steps):
                # Get financial state for this time/scenario
                financial_state = states[t, scenario_id, :]
                
                # Create cash flow event for this transition
                if t > 0:
                    # Calculate cash flow from previous state
                    prev_state = states[t-1, scenario_id, :]
                    cash_flow = financial_state[0] - prev_state[0]  # Cash change
                    
                    if abs(cash_flow) > 1.0:  # Only create event if significant
                        event_id = f"scenario_{scenario_id}_time_{t}"
                        category = "investment_growth" if cash_flow > 0 else "expense"
                        source = "investments" if cash_flow > 0 else "cash"
                        target = "cash" if cash_flow > 0 else "expenses"
                        
                        # Add to enhanced mesh
                        node_id = self.enhanced_mesh.add_cash_flow_event(
                            event_id, cash_flow, category, source, target, timestamp.to_pydatetime()
                        )
                        scenario_path.append(node_id)
            
            # Store scenario mapping
            self.scenario_nodes[scenario_id] = scenario_path
        
        print(f"✅ Created {len(self.scenario_nodes)} scenario mappings")
    
    def add_real_cash_flow_event(self, event: CashFlowEvent) -> str:
        """Add a real cash flow event to both systems"""
        # Add to unified model
        self.unified_model.add_cash_flow_event(event)
        
        # Add to enhanced mesh
        timestamp = datetime.fromisoformat(event.estimated_date)
        node_id = self.enhanced_mesh.add_cash_flow_event(
            event.event_id,
            event.amount,
            event.event_type,
            event.source_account,
            event.target_account,
            timestamp
        )
        
        self.logger.info(f"Added real cash flow event: {event.event_id} -> Node {node_id}")
        return node_id
    
    def analyze_path_dependencies(self, target_node_id: str = None) -> Dict[str, Any]:
        """Analyze path dependencies for a specific node"""
        if target_node_id is None:
            target_node_id = self.enhanced_mesh.current_position
        
        if target_node_id not in self.enhanced_mesh.nodes:
            return {}
        
        node = self.enhanced_mesh.nodes[target_node_id]
        
        # Get path analysis
        path_analysis = self.enhanced_mesh.get_path_analysis(target_node_id)
        
        # Find similar nodes (path clustering)
        similar_nodes = self.enhanced_mesh.find_similar_nodes(
            target_node_id, method='combined', threshold=0.6
        )
        
        # Analyze cash flow patterns
        cf_series = node.cash_flow_series
        pattern_analysis = {
            'total_events': len(cf_series.amounts),
            'net_flow': cf_series.get_net_flow(),
            'volatility': cf_series.get_volatility(),
            'trend': cf_series.get_trend(),
            'cumulative_flow': cf_series.get_cumulative_flow().tolist(),
            'event_distribution': self._analyze_event_distribution(cf_series),
            'timing_patterns': self._analyze_timing_patterns(cf_series)
        }
        
        # Analyze financial state evolution
        state_evolution = self._analyze_state_evolution(target_node_id)
        
        # Path risk analysis
        risk_analysis = {
            'current_risk_score': node.risk_score,
            'utility_score': node.utility_score,
            'similar_paths_risk': self._analyze_similar_paths_risk(similar_nodes),
            'path_volatility': pattern_analysis['volatility'],
            'liquidity_stress': self._calculate_liquidity_stress(node)
        }
        
        return {
            'node_id': target_node_id,
            'path_analysis': path_analysis,
            'pattern_analysis': pattern_analysis,
            'state_evolution': state_evolution,
            'risk_analysis': risk_analysis,
            'similar_nodes': similar_nodes[:10],  # Top 10 similar nodes
            'path_summary': node.get_path_summary()
        }
    
    def _analyze_event_distribution(self, cf_series: CashFlowSeries) -> Dict[str, Any]:
        """Analyze distribution of cash flow events"""
        if not cf_series.amounts:
            return {}
        
        # Categorize events
        positive_events = [amt for amt in cf_series.amounts if amt > 0]
        negative_events = [amt for amt in cf_series.amounts if amt < 0]
        
        # Category distribution
        category_counts = {}
        for category in cf_series.categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'positive_count': len(positive_events),
            'negative_count': len(negative_events),
            'positive_total': sum(positive_events),
            'negative_total': sum(negative_events),
            'category_distribution': category_counts,
            'largest_positive': max(positive_events) if positive_events else 0,
            'largest_negative': min(negative_events) if negative_events else 0
        }
    
    def _analyze_timing_patterns(self, cf_series: CashFlowSeries) -> Dict[str, Any]:
        """Analyze timing patterns in cash flows"""
        if len(cf_series.timestamps) < 2:
            return {}
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(cf_series.timestamps)):
            interval = (cf_series.timestamps[i] - cf_series.timestamps[i-1]).days
            intervals.append(interval)
        
        # Monthly patterns
        monthly_totals = {}
        for i, timestamp in enumerate(cf_series.timestamps):
            month_key = timestamp.strftime('%Y-%m')
            monthly_totals[month_key] = monthly_totals.get(month_key, 0) + cf_series.amounts[i]
        
        return {
            'average_interval_days': np.mean(intervals) if intervals else 0,
            'interval_volatility': np.std(intervals) if intervals else 0,
            'monthly_patterns': monthly_totals,
            'total_duration_days': (cf_series.timestamps[-1] - cf_series.timestamps[0]).days if len(cf_series.timestamps) > 1 else 0
        }
    
    def _analyze_state_evolution(self, node_id: str) -> Dict[str, Any]:
        """Analyze how financial state evolved to reach this node"""
        node = self.enhanced_mesh.nodes[node_id]
        
        # Get path to this node
        path_nodes = self._get_path_to_node(node_id)
        
        if not path_nodes:
            return {}
        
        # Analyze state changes along path
        state_changes = []
        for i in range(1, len(path_nodes)):
            prev_node = path_nodes[i-1]
            curr_node = path_nodes[i]
            
            net_worth_change = curr_node.financial_state.net_worth - prev_node.financial_state.net_worth
            liquidity_change = curr_node.financial_state.liquidity_ratio - prev_node.financial_state.liquidity_ratio
            
            state_changes.append({
                'node_id': curr_node.node_id,
                'net_worth_change': net_worth_change,
                'liquidity_change': liquidity_change,
                'risk_score_change': curr_node.risk_score - prev_node.risk_score,
                'utility_score_change': curr_node.utility_score - prev_node.utility_score
            })
        
        return {
            'total_state_changes': len(state_changes),
            'net_worth_evolution': [change['net_worth_change'] for change in state_changes],
            'liquidity_evolution': [change['liquidity_change'] for change in state_changes],
            'risk_evolution': [change['risk_score_change'] for change in state_changes],
            'utility_evolution': [change['utility_score_change'] for change in state_changes],
            'cumulative_net_worth_change': sum(change['net_worth_change'] for change in state_changes)
        }
    
    def _get_path_to_node(self, node_id: str) -> List[EnhancedMeshNode]:
        """Get the complete path to a node"""
        if node_id not in self.enhanced_mesh.nodes:
            return []
        
        node = self.enhanced_mesh.nodes[node_id]
        path = [node]
        
        # Traverse up the path
        current_node = node
        while current_node.parent_nodes:
            parent_id = current_node.parent_nodes[0]  # Take first parent
            if parent_id in self.enhanced_mesh.nodes:
                parent_node = self.enhanced_mesh.nodes[parent_id]
                path.insert(0, parent_node)
                current_node = parent_node
            else:
                break
        
        return path
    
    def _analyze_similar_paths_risk(self, similar_nodes: List[Tuple[str, float]]) -> Dict[str, float]:
        """Analyze risk patterns across similar paths"""
        if not similar_nodes:
            return {}
        
        risk_scores = []
        utility_scores = []
        
        for node_id, similarity in similar_nodes:
            if node_id in self.enhanced_mesh.nodes:
                node = self.enhanced_mesh.nodes[node_id]
                risk_scores.append(node.risk_score)
                utility_scores.append(node.utility_score)
        
        return {
            'average_risk': np.mean(risk_scores) if risk_scores else 0,
            'risk_volatility': np.std(risk_scores) if risk_scores else 0,
            'average_utility': np.mean(utility_scores) if utility_scores else 0,
            'utility_volatility': np.std(utility_scores) if utility_scores else 0
        }
    
    def _calculate_liquidity_stress(self, node: EnhancedMeshNode) -> float:
        """Calculate liquidity stress score"""
        liquidity_ratio = node.financial_state.liquidity_ratio
        cash_flows = node.cash_flow_series.amounts
        
        # Base stress from liquidity ratio
        base_stress = max(0, 1 - liquidity_ratio)
        
        # Additional stress from recent negative cash flows
        recent_negative_flows = sum(1 for amt in cash_flows[-5:] if amt < 0)  # Last 5 flows
        flow_stress = recent_negative_flows / 5.0
        
        return (base_stress + flow_stress) / 2.0
    
    def generate_path_visualization_data(self, node_id: str = None) -> Dict[str, Any]:
        """Generate data for path visualization"""
        if node_id is None:
            node_id = self.enhanced_mesh.current_position
        
        if node_id not in self.enhanced_mesh.nodes:
            return {}
        
        node = self.enhanced_mesh.nodes[node_id]
        path_nodes = self._get_path_to_node(node_id)
        
        # Create visualization data
        viz_data = {
            'path_nodes': [],
            'cash_flow_series': [],
            'risk_evolution': [],
            'utility_evolution': []
        }
        
        for i, path_node in enumerate(path_nodes):
            # Node data
            viz_data['path_nodes'].append({
                'node_id': path_node.node_id,
                'timestamp': path_node.timestamp.isoformat(),
                'net_worth': path_node.financial_state.net_worth,
                'risk_score': path_node.risk_score,
                'utility_score': path_node.utility_score,
                'step': i
            })
            
            # Cash flow series
            cf_series = path_node.cash_flow_series
            viz_data['cash_flow_series'].append({
                'node_id': path_node.node_id,
                'timestamps': [ts.isoformat() for ts in cf_series.timestamps],
                'amounts': cf_series.amounts,
                'cumulative': cf_series.get_cumulative_flow().tolist()
            })
            
            # Risk and utility evolution
            viz_data['risk_evolution'].append({
                'step': i,
                'risk_score': path_node.risk_score,
                'timestamp': path_node.timestamp.isoformat()
            })
            
            viz_data['utility_evolution'].append({
                'step': i,
                'utility_score': path_node.utility_score,
                'timestamp': path_node.timestamp.isoformat()
            })
        
        return viz_data
    
    def export_integration_data(self, filepath: str):
        """Export integration data for analysis"""
        integration_data = {
            'enhanced_mesh': {
                'initial_state': self.enhanced_mesh.initial_state,
                'current_position': self.enhanced_mesh.current_position,
                'nodes': {node_id: node.to_dict() for node_id, node in self.enhanced_mesh.nodes.items()}
            },
            'scenario_mappings': self.scenario_nodes,
            'path_analyses': self.path_analyses,
            'system_stats': {
                'total_nodes': len(self.enhanced_mesh.nodes),
                'total_scenarios': len(self.scenario_nodes),
                'current_node_id': self.enhanced_mesh.current_position
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(integration_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported integration data to {filepath}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        enhanced_stats = self.enhanced_mesh.get_mesh_statistics()
        
        return {
            'enhanced_mesh_stats': enhanced_stats,
            'scenario_count': len(self.scenario_nodes),
            'current_node': self.enhanced_mesh.current_position,
            'total_cash_flow_events': sum(len(node.cash_flow_series.amounts) for node in self.enhanced_mesh.nodes.values()),
            'average_path_length': enhanced_stats.get('average_series_length', 0),
            'risk_distribution': enhanced_stats.get('risk_score_stats', {}),
            'utility_distribution': enhanced_stats.get('utility_score_stats', {})
        }

def demo_enhanced_mesh_integration():
    """Demonstrate the enhanced mesh integration system"""
    print("🚀 Enhanced Mesh Integration Demo")
    print("=" * 60)
    
    # Initialize system
    initial_state = {
        'cash': 100000,
        'investments': 500000,
        'income': 150000,
        'expenses': 60000
    }
    
    integration = EnhancedMeshIntegration(initial_state)
    
    # Initialize system
    init_result = integration.initialize_system()
    print(f"✅ System initialized: {init_result}")
    
    # Add some real cash flow events
    events = [
        CashFlowEvent(
            event_id="salary_payment_1",
            description="Monthly salary payment",
            estimated_date="2024-01-15",
            amount=12500,
            source_account="income",
            target_account="cash",
            event_type="income"
        ),
        CashFlowEvent(
            event_id="mortgage_payment_1",
            description="Monthly mortgage payment",
            estimated_date="2024-01-15",
            amount=-3000,
            source_account="cash",
            target_account="expenses",
            event_type="expense"
        ),
        CashFlowEvent(
            event_id="investment_dividend_1",
            description="Investment dividend",
            estimated_date="2024-02-01",
            amount=2000,
            source_account="investments",
            target_account="cash",
            event_type="income"
        )
    ]
    
    for event in events:
        node_id = integration.add_real_cash_flow_event(event)
        print(f"📊 Added event {event.event_id} -> Node {node_id}")
    
    # Analyze path dependencies
    current_node_id = integration.enhanced_mesh.current_position
    path_analysis = integration.analyze_path_dependencies(current_node_id)
    
    print(f"\n🔍 Path Analysis for Node {current_node_id}:")
    print(f"Total events: {path_analysis['pattern_analysis']['total_events']}")
    print(f"Net flow: ${path_analysis['pattern_analysis']['net_flow']:,.2f}")
    print(f"Risk score: {path_analysis['risk_analysis']['current_risk_score']:.3f}")
    print(f"Utility score: {path_analysis['risk_analysis']['utility_score']:.3f}")
    
    # Get system summary
    summary = integration.get_system_summary()
    print(f"\n📈 System Summary:")
    print(f"Total nodes: {summary['enhanced_mesh_stats']['total_nodes']}")
    print(f"Total scenarios: {summary['scenario_count']}")
    print(f"Average path length: {summary['average_path_length']:.1f}")
    
    # Generate visualization data
    viz_data = integration.generate_path_visualization_data()
    print(f"\n📊 Visualization data generated for {len(viz_data['path_nodes'])} path nodes")
    
    print("\n✅ Enhanced mesh integration demo completed!")

if __name__ == "__main__":
    demo_enhanced_mesh_integration() 