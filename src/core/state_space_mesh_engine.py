"""
Enhanced Mesh Node with Full Cash Flow Series Tracking

This module implements the state-space mesh approach where:
- Each node represents a financial state with complete cash flow history
- Nodes store full series of cash flows (not just snapshots)
- The mesh represents all possible financial evolutions
- Path-dependent analysis is supported through cash flow series comparison
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from decimal import Decimal
import json
import logging
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class CashFlowSeries:
    """Represents a complete series of cash flows up to a point in time"""
    timestamps: List[datetime]
    amounts: List[float]
    categories: List[str]
    sources: List[str]
    targets: List[str]
    event_ids: List[str]
    
    def __post_init__(self):
        """Validate cash flow series data"""
        if not (len(self.timestamps) == len(self.amounts) == len(self.categories) == 
                len(self.sources) == len(self.targets) == len(self.event_ids)):
            raise ValueError("All cash flow series arrays must have the same length")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for analysis"""
        return np.array([
            [ts.timestamp() for ts in self.timestamps],
            self.amounts,
            [hash(cat) % 1000 for cat in self.categories],  # Numeric encoding
            [hash(src) % 1000 for src in self.sources],
            [hash(tgt) % 1000 for tgt in self.targets],
            [hash(eid) % 1000 for eid in self.event_ids]
        ]).T
    
    def get_cumulative_flow(self) -> np.ndarray:
        """Get cumulative cash flow over time"""
        return np.cumsum(self.amounts)
    
    def get_net_flow(self) -> float:
        """Get net cash flow"""
        return sum(self.amounts)
    
    def get_volatility(self) -> float:
        """Calculate cash flow volatility"""
        if len(self.amounts) < 2:
            return 0.0
        return np.std(self.amounts)
    
    def get_trend(self) -> float:
        """Calculate cash flow trend (positive = increasing, negative = decreasing)"""
        if len(self.amounts) < 2:
            return 0.0
        return np.polyfit(range(len(self.amounts)), self.amounts, 1)[0]
    
    def similarity_to(self, other: 'CashFlowSeries', method: str = 'dtw') -> float:
        """Calculate similarity to another cash flow series"""
        if method == 'dtw':
            return self._dtw_similarity(other)
        elif method == 'wasserstein':
            return self._wasserstein_similarity(other)
        elif method == 'euclidean':
            return self._euclidean_similarity(other)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _dtw_similarity(self, other: 'CashFlowSeries') -> float:
        """Dynamic Time Warping similarity"""
        # Simple DTW implementation
        s1 = np.array(self.amounts)
        s2 = np.array(other.amounts)
        
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                             dtw_matrix[i, j-1], 
                                             dtw_matrix[i-1, j-1])
        
        return 1.0 / (1.0 + dtw_matrix[n, m])  # Convert to similarity
    
    def _wasserstein_similarity(self, other: 'CashFlowSeries') -> float:
        """Wasserstein distance similarity"""
        try:
            distance = wasserstein_distance(self.amounts, other.amounts)
            return 1.0 / (1.0 + distance)
        except:
            return 0.0
    
    def _euclidean_similarity(self, other: 'CashFlowSeries') -> float:
        """Euclidean distance similarity"""
        s1 = np.array(self.amounts)
        s2 = np.array(other.amounts)
        
        # Pad shorter series with zeros
        max_len = max(len(s1), len(s2))
        s1_padded = np.pad(s1, (0, max_len - len(s1)), 'constant')
        s2_padded = np.pad(s2, (0, max_len - len(s2)), 'constant')
        
        distance = np.linalg.norm(s1_padded - s2_padded)
        return 1.0 / (1.0 + distance)

@dataclass
class FinancialState:
    """Represents a complete financial state at a point in time"""
    timestamp: datetime
    account_balances: Dict[str, float]
    total_assets: float
    total_liabilities: float
    net_worth: float
    liquidity_ratio: float
    debt_to_income_ratio: float
    investment_allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for analysis"""
        features = [
            self.total_assets,
            self.total_liabilities,
            self.net_worth,
            self.liquidity_ratio,
            self.debt_to_income_ratio
        ]
        
        # Add account balances
        for account, balance in self.account_balances.items():
            features.append(balance)
        
        # Add investment allocation
        for asset, allocation in self.investment_allocation.items():
            features.append(allocation)
        
        # Add risk metrics
        for metric, value in self.risk_metrics.items():
            features.append(value)
        
        return np.array(features)

@dataclass
class EnhancedMeshNode:
    """
    Enhanced mesh node that stores complete cash flow history and financial state
    
    This represents a point in the financial state-space with:
    - Complete cash flow series up to this point
    - Current financial state
    - Path information (how this state was reached)
    - Future possibilities (child nodes)
    """
    node_id: str
    timestamp: datetime
    financial_state: FinancialState
    cash_flow_series: CashFlowSeries
    probability: float
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    transition_events: List[str] = field(default_factory=list)
    path_metadata: Dict[str, Any] = field(default_factory=dict)
    visibility_radius: float = 1.0
    is_solidified: bool = False
    risk_score: float = 0.0
    utility_score: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics after initialization"""
        self._calculate_risk_score()
        self._calculate_utility_score()
    
    def _calculate_risk_score(self):
        """Calculate risk score based on financial state and cash flow history"""
        # Liquidity risk
        liquidity_risk = max(0, 1 - self.financial_state.liquidity_ratio)
        
        # Debt risk
        debt_risk = min(1, self.financial_state.debt_to_income_ratio / 0.4)  # 40% DTI threshold
        
        # Cash flow volatility risk
        flow_volatility = self.cash_flow_series.get_volatility()
        volatility_risk = min(1, flow_volatility / 10000)  # Normalize to 0-1
        
        # Net worth trend risk
        net_worth_trend = self.cash_flow_series.get_trend()
        trend_risk = max(0, -net_worth_trend / 10000)  # Negative trend = higher risk
        
        # Combined risk score (weighted average)
        self.risk_score = (0.3 * liquidity_risk + 
                          0.3 * debt_risk + 
                          0.2 * volatility_risk + 
                          0.2 * trend_risk)
    
    def _calculate_utility_score(self):
        """Calculate utility score based on financial health and growth"""
        # Growth utility
        growth_utility = max(0, self.cash_flow_series.get_trend() / 10000)
        
        # Wealth utility
        wealth_utility = min(1, self.financial_state.net_worth / 1000000)  # Normalize to 1M
        
        # Stability utility (inverse of risk)
        stability_utility = 1 - self.risk_score
        
        # Combined utility score
        self.utility_score = (0.4 * growth_utility + 
                             0.3 * wealth_utility + 
                             0.3 * stability_utility)
    
    def similarity_to(self, other: 'EnhancedMeshNode', method: str = 'combined') -> float:
        """Calculate similarity to another node"""
        if method == 'cash_flow':
            return self.cash_flow_series.similarity_to(other.cash_flow_series)
        elif method == 'financial_state':
            return self._financial_state_similarity(other)
        elif method == 'combined':
            flow_sim = self.cash_flow_series.similarity_to(other.cash_flow_series)
            state_sim = self._financial_state_similarity(other)
            return 0.6 * flow_sim + 0.4 * state_sim
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _financial_state_similarity(self, other: 'EnhancedMeshNode') -> float:
        """Calculate financial state similarity"""
        vec1 = self.financial_state.to_vector()
        vec2 = other.financial_state.to_vector()
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return max(0, similarity)  # Ensure non-negative
    
    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of the path to this node"""
        return {
            'node_id': self.node_id,
            'timestamp': self.timestamp.isoformat(),
            'net_worth': self.financial_state.net_worth,
            'total_cash_flows': len(self.cash_flow_series.amounts),
            'net_cash_flow': self.cash_flow_series.get_net_flow(),
            'risk_score': self.risk_score,
            'utility_score': self.utility_score,
            'parent_count': len(self.parent_nodes),
            'child_count': len(self.child_nodes),
            'is_solidified': self.is_solidified
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'timestamp': self.timestamp.isoformat(),
            'financial_state': {
                'account_balances': self.financial_state.account_balances,
                'total_assets': self.financial_state.total_assets,
                'total_liabilities': self.financial_state.total_liabilities,
                'net_worth': self.financial_state.net_worth,
                'liquidity_ratio': self.financial_state.liquidity_ratio,
                'debt_to_income_ratio': self.financial_state.debt_to_income_ratio,
                'investment_allocation': self.financial_state.investment_allocation,
                'risk_metrics': self.financial_state.risk_metrics
            },
            'cash_flow_series': {
                'timestamps': [ts.isoformat() for ts in self.cash_flow_series.timestamps],
                'amounts': self.cash_flow_series.amounts,
                'categories': self.cash_flow_series.categories,
                'sources': self.cash_flow_series.sources,
                'targets': self.cash_flow_series.targets,
                'event_ids': self.cash_flow_series.event_ids
            },
            'probability': self.probability,
            'parent_nodes': self.parent_nodes,
            'child_nodes': self.child_nodes,
            'transition_events': self.transition_events,
            'path_metadata': self.path_metadata,
            'visibility_radius': self.visibility_radius,
            'is_solidified': self.is_solidified,
            'risk_score': self.risk_score,
            'utility_score': self.utility_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedMeshNode':
        """Create node from dictionary"""
        # Reconstruct cash flow series
        cf_data = data['cash_flow_series']
        cash_flow_series = CashFlowSeries(
            timestamps=[datetime.fromisoformat(ts) for ts in cf_data['timestamps']],
            amounts=cf_data['amounts'],
            categories=cf_data['categories'],
            sources=cf_data['sources'],
            targets=cf_data['targets'],
            event_ids=cf_data['event_ids']
        )
        
        # Reconstruct financial state
        fs_data = data['financial_state']
        financial_state = FinancialState(
            timestamp=datetime.fromisoformat(data['timestamp']),
            account_balances=fs_data['account_balances'],
            total_assets=fs_data['total_assets'],
            total_liabilities=fs_data['total_liabilities'],
            net_worth=fs_data['net_worth'],
            liquidity_ratio=fs_data['liquidity_ratio'],
            debt_to_income_ratio=fs_data['debt_to_income_ratio'],
            investment_allocation=fs_data['investment_allocation'],
            risk_metrics=fs_data['risk_metrics']
        )
        
        return cls(
            node_id=data['node_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            financial_state=financial_state,
            cash_flow_series=cash_flow_series,
            probability=data['probability'],
            parent_nodes=data['parent_nodes'],
            child_nodes=data['child_nodes'],
            transition_events=data['transition_events'],
            path_metadata=data['path_metadata'],
            visibility_radius=data['visibility_radius'],
            is_solidified=data['is_solidified'],
            risk_score=data['risk_score'],
            utility_score=data['utility_score']
        )

class EnhancedMeshEngine:
    """
    Enhanced mesh engine that operates on nodes with full cash flow series
    
    This implements the state-space mesh approach where:
    - Each node represents a complete financial history
    - The mesh encodes all possible financial evolutions
    - Path-dependent analysis is supported
    - Similarity is calculated over cash flow series
    """
    
    def __init__(self, initial_state: Dict[str, float]):
        self.initial_state = initial_state
        self.nodes: Dict[str, EnhancedMeshNode] = {}
        self.current_position: Optional[str] = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('enhanced_mesh_engine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_initial_node(self, timestamp: datetime = None) -> str:
        """Create the initial node with starting financial state"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create initial cash flow series (empty)
        initial_cash_flow = CashFlowSeries(
            timestamps=[],
            amounts=[],
            categories=[],
            sources=[],
            targets=[],
            event_ids=[]
        )
        
        # Create initial financial state
        initial_financial_state = FinancialState(
            timestamp=timestamp,
            account_balances=self.initial_state,
            total_assets=sum(v for k, v in self.initial_state.items() if 'debt' not in k.lower()),
            total_liabilities=sum(v for k, v in self.initial_state.items() if 'debt' in k.lower()),
            net_worth=sum(v for k, v in self.initial_state.items() if 'debt' not in k.lower()) - 
                     sum(v for k, v in self.initial_state.items() if 'debt' in k.lower()),
            liquidity_ratio=0.2,  # Default
            debt_to_income_ratio=0.0,  # Default
            investment_allocation={'stocks': 0.6, 'bonds': 0.4},  # Default
            risk_metrics={'volatility': 0.0, 'var_95': 0.0, 'max_drawdown': 0.0}
        )
        
        # Create initial node
        initial_node = EnhancedMeshNode(
            node_id="initial_0",
            timestamp=timestamp,
            financial_state=initial_financial_state,
            cash_flow_series=initial_cash_flow,
            probability=1.0
        )
        
        self.nodes[initial_node.node_id] = initial_node
        self.current_position = initial_node.node_id
        
        self.logger.info(f"Created initial node: {initial_node.node_id}")
        return initial_node.node_id
    
    def add_cash_flow_event(self, event_id: str, amount: float, category: str, 
                           source: str, target: str, timestamp: datetime = None) -> str:
        """Add a cash flow event and create a new node"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if not self.current_position:
            raise ValueError("No current position. Call create_initial_node first.")
        
        current_node = self.nodes[self.current_position]
        
        # Create new cash flow series with the event
        new_timestamps = current_node.cash_flow_series.timestamps + [timestamp]
        new_amounts = current_node.cash_flow_series.amounts + [amount]
        new_categories = current_node.cash_flow_series.categories + [category]
        new_sources = current_node.cash_flow_series.sources + [source]
        new_targets = current_node.cash_flow_series.targets + [target]
        new_event_ids = current_node.cash_flow_series.event_ids + [event_id]
        
        new_cash_flow_series = CashFlowSeries(
            timestamps=new_timestamps,
            amounts=new_amounts,
            categories=new_categories,
            sources=new_sources,
            targets=new_targets,
            event_ids=new_event_ids
        )
        
        # Update financial state
        new_account_balances = current_node.financial_state.account_balances.copy()
        if source in new_account_balances:
            new_account_balances[source] = max(0, new_account_balances[source] - amount)
        if target in new_account_balances:
            new_account_balances[target] = new_account_balances[target] + amount
        
        # Recalculate financial metrics
        total_assets = sum(v for k, v in new_account_balances.items() if 'debt' not in k.lower())
        total_liabilities = sum(v for k, v in new_account_balances.items() if 'debt' in k.lower())
        net_worth = total_assets - total_liabilities
        
        new_financial_state = FinancialState(
            timestamp=timestamp,
            account_balances=new_account_balances,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            net_worth=net_worth,
            liquidity_ratio=min(1.0, new_account_balances.get('cash', 0) / max(1, total_liabilities)),
            debt_to_income_ratio=total_liabilities / max(1, new_account_balances.get('income', 1)),
            investment_allocation=current_node.financial_state.investment_allocation,
            risk_metrics=current_node.financial_state.risk_metrics
        )
        
        # Create new node
        new_node_id = f"node_{len(self.nodes)}"
        new_node = EnhancedMeshNode(
            node_id=new_node_id,
            timestamp=timestamp,
            financial_state=new_financial_state,
            cash_flow_series=new_cash_flow_series,
            probability=current_node.probability,  # Inherit probability
            parent_nodes=[self.current_position],
            transition_events=[event_id]
        )
        
        # Add to mesh
        self.nodes[new_node_id] = new_node
        
        # Update parent-child relationships
        current_node.child_nodes.append(new_node_id)
        
        # Update current position
        self.current_position = new_node_id
        
        self.logger.info(f"Added cash flow event {event_id}: ${amount} from {source} to {target}")
        return new_node_id
    
    def find_similar_nodes(self, target_node_id: str, method: str = 'combined', 
                          threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find nodes similar to the target node"""
        if target_node_id not in self.nodes:
            return []
        
        target_node = self.nodes[target_node_id]
        similarities = []
        
        for node_id, node in self.nodes.items():
            if node_id == target_node_id:
                continue
            
            similarity = target_node.similarity_to(node, method)
            if similarity >= threshold:
                similarities.append((node_id, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def get_path_analysis(self, node_id: str) -> Dict[str, Any]:
        """Analyze the path to a specific node"""
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        
        # Get path summary
        path_summary = node.get_path_summary()
        
        # Analyze cash flow patterns
        cf_series = node.cash_flow_series
        flow_analysis = {
            'total_events': len(cf_series.amounts),
            'net_flow': cf_series.get_net_flow(),
            'volatility': cf_series.get_volatility(),
            'trend': cf_series.get_trend(),
            'positive_flows': sum(1 for amt in cf_series.amounts if amt > 0),
            'negative_flows': sum(1 for amt in cf_series.amounts if amt < 0),
            'largest_inflow': max(cf_series.amounts) if cf_series.amounts else 0,
            'largest_outflow': min(cf_series.amounts) if cf_series.amounts else 0
        }
        
        # Analyze financial state evolution
        state_analysis = {
            'net_worth_change': node.financial_state.net_worth - self.initial_state.get('total_wealth', 0),
            'liquidity_ratio': node.financial_state.liquidity_ratio,
            'debt_ratio': node.financial_state.debt_to_income_ratio,
            'risk_score': node.risk_score,
            'utility_score': node.utility_score
        }
        
        return {
            'path_summary': path_summary,
            'cash_flow_analysis': flow_analysis,
            'state_analysis': state_analysis
        }
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get overall mesh statistics"""
        if not self.nodes:
            return {}
        
        # Basic statistics
        total_nodes = len(self.nodes)
        total_cash_flows = sum(len(node.cash_flow_series.amounts) for node in self.nodes.values())
        
        # Risk and utility distributions
        risk_scores = [node.risk_score for node in self.nodes.values()]
        utility_scores = [node.utility_score for node in self.nodes.values()]
        
        # Net worth distribution
        net_worths = [node.financial_state.net_worth for node in self.nodes.values()]
        
        # Cash flow series lengths
        series_lengths = [len(node.cash_flow_series.amounts) for node in self.nodes.values()]
        
        return {
            'total_nodes': total_nodes,
            'total_cash_flows': total_cash_flows,
            'average_series_length': np.mean(series_lengths) if series_lengths else 0,
            'risk_score_stats': {
                'mean': np.mean(risk_scores),
                'std': np.std(risk_scores),
                'min': np.min(risk_scores),
                'max': np.max(risk_scores)
            },
            'utility_score_stats': {
                'mean': np.mean(utility_scores),
                'std': np.std(utility_scores),
                'min': np.min(utility_scores),
                'max': np.max(utility_scores)
            },
            'net_worth_stats': {
                'mean': np.mean(net_worths),
                'std': np.std(net_worths),
                'min': np.min(net_worths),
                'max': np.max(net_worths)
            }
        }
    
    def export_mesh(self, filepath: str):
        """Export mesh to JSON file"""
        mesh_data = {
            'initial_state': self.initial_state,
            'current_position': self.current_position,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(mesh_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported mesh to {filepath}")
    
    def import_mesh(self, filepath: str):
        """Import mesh from JSON file"""
        with open(filepath, 'r') as f:
            mesh_data = json.load(f)
        
        self.initial_state = mesh_data['initial_state']
        self.current_position = mesh_data['current_position']
        
        # Reconstruct nodes
        self.nodes = {}
        for node_id, node_data in mesh_data['nodes'].items():
            self.nodes[node_id] = EnhancedMeshNode.from_dict(node_data)
        
        self.logger.info(f"Imported mesh from {filepath} with {len(self.nodes)} nodes")

def demo_enhanced_mesh():
    """Demonstrate the enhanced mesh system"""
    print("🚀 Enhanced Mesh System Demo")
    print("=" * 50)
    
    # Initialize mesh
    initial_state = {
        'cash': 100000,
        'investments': 500000,
        'income': 150000,
        'expenses': 60000
    }
    
    mesh = EnhancedMeshEngine(initial_state)
    
    # Create initial node
    initial_node_id = mesh.create_initial_node()
    print(f"✅ Created initial node: {initial_node_id}")
    
    # Add some cash flow events
    events = [
        ('salary_payment', 12500, 'income', 'income', 'cash', '2024-01-15'),
        ('mortgage_payment', -3000, 'expense', 'cash', 'expenses', '2024-01-15'),
        ('investment_dividend', 2000, 'income', 'investments', 'cash', '2024-02-01'),
        ('car_payment', -500, 'expense', 'cash', 'expenses', '2024-02-15'),
        ('bonus_payment', 5000, 'income', 'income', 'cash', '2024-03-01')
    ]
    
    for event_id, amount, category, source, target, date_str in events:
        timestamp = datetime.fromisoformat(date_str)
        node_id = mesh.add_cash_flow_event(event_id, amount, category, source, target, timestamp)
        print(f"📊 Added event {event_id}: ${amount} -> Node {node_id}")
    
    # Get mesh statistics
    stats = mesh.get_mesh_statistics()
    print(f"\n📈 Mesh Statistics:")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total cash flows: {stats['total_cash_flows']}")
    print(f"Average series length: {stats['average_series_length']:.1f}")
    
    # Get path analysis for current node
    current_node_id = mesh.current_position
    path_analysis = mesh.get_path_analysis(current_node_id)
    
    print(f"\n🔍 Path Analysis for Node {current_node_id}:")
    print(f"Net worth: ${path_analysis['state_analysis']['net_worth_change']:,.2f}")
    print(f"Risk score: {path_analysis['state_analysis']['risk_score']:.3f}")
    print(f"Utility score: {path_analysis['state_analysis']['utility_score']:.3f}")
    
    # Find similar nodes
    similar_nodes = mesh.find_similar_nodes(current_node_id, threshold=0.5)
    print(f"\n🔗 Similar nodes (threshold 0.5): {len(similar_nodes)}")
    
    print("\n✅ Enhanced mesh demo completed!")

if __name__ == "__main__":
    demo_enhanced_mesh() 