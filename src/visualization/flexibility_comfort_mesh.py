"""
Flexibility vs. Comfort Mesh Engine

Specialized mesh engine that visualizes financial states from the perspective of:
- Flexibility: Ability to adapt to future changes
- Comfort: Low risk, high solvency, highly solvable states

This creates a 2D visualization where each point represents a financial scenario
plotted by (flexibility, comfort) coordinates.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from enum import Enum


class PaymentType(Enum):
    """Types of payment structures"""
    PLANNED = "planned"
    PV_CALCULATED = "pv_calculated"
    LUMP_SUM = "lump_sum"
    ANNUITY = "annuity"


@dataclass
class ClientProfile:
    """Client financial profile"""
    name: str
    age: int
    current_assets: Dict[str, float]
    income: float
    expenses: float
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    time_horizon: int  # years
    financial_goals: List[Dict]
    payment_preferences: Dict[str, PaymentType]
    metadata: Dict = field(default_factory=dict)


@dataclass
class FinancialScenario:
    """A financial scenario with flexibility and comfort metrics"""
    scenario_id: str
    timestamp: datetime
    assets: Dict[str, float]
    income: float
    expenses: float
    flexibility_score: float  # 0-1, ability to adapt
    comfort_score: float  # 0-1, low risk, high solvency
    risk_metrics: Dict[str, float]
    feasibility_score: float
    distance_to_target: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MeshVisualization:
    """2D mesh visualization data"""
    x_coords: List[float]  # Flexibility scores
    y_coords: List[float]  # Comfort scores
    scenario_ids: List[str]
    colors: List[str]
    sizes: List[float]
    hover_data: List[Dict]
    optimal_path: List[Tuple[float, float]]
    target_zone: List[Tuple[float, float]]
    metadata: Dict = field(default_factory=dict)


class FlexibilityComfortMeshEngine:
    """
    Mesh engine that visualizes financial states from flexibility vs. comfort perspective
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.client_profiles: Dict[str, ClientProfile] = {}
        self.scenarios: List[FinancialScenario] = []
        self.visualizations: Dict[str, MeshVisualization] = {}
        
    def add_client_profile(self, profile: ClientProfile):
        """Add a client profile to the mesh engine"""
        self.client_profiles[profile.name] = profile
        
    def compute_flexibility_score(self, assets: Dict[str, float], 
                                income: float, expenses: float) -> float:
        """
        Compute flexibility score based on:
        - Liquidity (cash ratio)
        - Diversification (number of asset classes)
        - Income-to-expense ratio
        - Asset reallocation potential
        """
        total_assets = sum(assets.values())
        if total_assets == 0:
            return 0.0
        
        # Liquidity component (0-0.4)
        cash_ratio = assets.get('cash', 0) / total_assets
        liquidity_score = min(0.4, cash_ratio * 2)  # Cap at 0.4
        
        # Diversification component (0-0.3)
        asset_ratios = [v / total_assets for v in assets.values() if v > 0]
        if len(asset_ratios) > 1:
            # Herfindahl index for diversification
            diversification = 1 - sum(r**2 for r in asset_ratios)
            diversification_score = min(0.3, diversification * 0.5)
        else:
            diversification_score = 0.0
        
        # Income flexibility component (0-0.2)
        if expenses > 0:
            income_ratio = income / expenses
            income_score = min(0.2, (income_ratio - 1) * 0.1)
        else:
            income_score = 0.2
        
        # Asset reallocation potential (0-0.1)
        # Higher score if assets are more evenly distributed
        if len(asset_ratios) > 1:
            reallocation_score = min(0.1, 0.1 * (1 - max(asset_ratios)))
        else:
            reallocation_score = 0.0
        
        total_flexibility = liquidity_score + diversification_score + income_score + reallocation_score
        return min(1.0, max(0.0, total_flexibility))
    
    def compute_comfort_score(self, assets: Dict[str, float], 
                            risk_metrics: Dict[str, float],
                            feasibility_score: float) -> float:
        """
        Compute comfort score based on:
        - Low risk (inverse of volatility)
        - High feasibility
        - Adequate cash reserves
        - Debt-to-asset ratio
        """
        total_assets = sum(assets.values())
        if total_assets == 0:
            return 0.0
        
        # Risk component (0-0.4)
        volatility = risk_metrics.get('volatility', 0.5)
        risk_score = max(0.0, 0.4 * (1 - volatility))
        
        # Feasibility component (0-0.3)
        feasibility_score_component = 0.3 * feasibility_score
        
        # Cash adequacy component (0-0.2)
        cash_ratio = assets.get('cash', 0) / total_assets
        cash_score = min(0.2, cash_ratio * 2)
        
        # Debt component (0-0.1) - assuming no debt for now
        debt_score = 0.1
        
        total_comfort = risk_score + feasibility_score_component + cash_score + debt_score
        return min(1.0, max(0.0, total_comfort))
    
    def compute_risk_metrics(self, assets: Dict[str, float]) -> Dict[str, float]:
        """Compute risk metrics for a given asset allocation"""
        total_assets = sum(assets.values())
        if total_assets == 0:
            return {'volatility': 0.0, 'var_95': 0.0, 'max_drawdown': 0.0}
        
        # Simplified risk calculation
        cash_ratio = assets.get('cash', 0) / total_assets
        stock_ratio = assets.get('stocks', 0) / total_assets
        bond_ratio = assets.get('bonds', 0) / total_assets
        real_estate_ratio = assets.get('real_estate', 0) / total_assets
        
        # Volatility based on asset allocation
        volatility = (stock_ratio * 0.6 + 
                     bond_ratio * 0.2 + 
                     real_estate_ratio * 0.4 + 
                     cash_ratio * 0.0)
        
        # Value at Risk (simplified)
        var_95 = volatility * 1.65
        
        # Maximum drawdown (simplified)
        max_drawdown = volatility * 0.5
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown
        }
    
    def generate_scenarios(self, client_name: str, num_scenarios: int = 1000) -> List[FinancialScenario]:
        """Generate financial scenarios for a client"""
        if client_name not in self.client_profiles:
            raise ValueError(f"Client {client_name} not found")
        
        client = self.client_profiles[client_name]
        scenarios = []
        
        # Generate scenarios by varying asset allocations
        for i in range(num_scenarios):
            # Create varied asset allocation
            total_assets = sum(client.current_assets.values())
            
            # Generate random allocation (with some constraints)
            if client.risk_tolerance == 'conservative':
                cash_range = (0.2, 0.4)
                stock_range = (0.1, 0.3)
                bond_range = (0.3, 0.5)
                real_estate_range = (0.0, 0.2)
            elif client.risk_tolerance == 'moderate':
                cash_range = (0.1, 0.3)
                stock_range = (0.2, 0.5)
                bond_range = (0.2, 0.4)
                real_estate_range = (0.1, 0.3)
            else:  # aggressive
                cash_range = (0.05, 0.2)
                stock_range = (0.4, 0.7)
                bond_range = (0.1, 0.3)
                real_estate_range = (0.1, 0.3)
            
            # Generate random allocation within ranges
            cash_ratio = np.random.uniform(*cash_range)
            stock_ratio = np.random.uniform(*stock_range)
            bond_ratio = np.random.uniform(*bond_range)
            real_estate_ratio = np.random.uniform(*real_estate_range)
            
            # Normalize to sum to 1
            total_ratio = cash_ratio + stock_ratio + bond_ratio + real_estate_ratio
            cash_ratio /= total_ratio
            stock_ratio /= total_ratio
            bond_ratio /= total_ratio
            real_estate_ratio /= total_ratio
            
            # Create asset allocation
            assets = {
                'cash': cash_ratio * total_assets,
                'stocks': stock_ratio * total_assets,
                'bonds': bond_ratio * total_assets,
                'real_estate': real_estate_ratio * total_assets
            }
            
            # Vary income and expenses slightly
            income_variation = np.random.normal(1.0, 0.1)
            expense_variation = np.random.normal(1.0, 0.1)
            income = client.income * income_variation
            expenses = client.expenses * expense_variation
            
            # Compute metrics
            risk_metrics = self.compute_risk_metrics(assets)
            flexibility_score = self.compute_flexibility_score(assets, income, expenses)
            
            # Simplified feasibility score
            feasibility_score = 1.0 - risk_metrics['volatility']
            
            comfort_score = self.compute_comfort_score(assets, risk_metrics, feasibility_score)
            
            # Distance to target (simplified)
            target_flexibility = 0.7
            target_comfort = 0.8
            distance_to_target = np.sqrt((flexibility_score - target_flexibility)**2 + 
                                       (comfort_score - target_comfort)**2)
            
            scenario = FinancialScenario(
                scenario_id=f"{client_name}_scenario_{i}",
                timestamp=datetime.now() + timedelta(days=i),
                assets=assets,
                income=income,
                expenses=expenses,
                flexibility_score=flexibility_score,
                comfort_score=comfort_score,
                risk_metrics=risk_metrics,
                feasibility_score=feasibility_score,
                distance_to_target=distance_to_target,
                metadata={'client_name': client_name}
            )
            
            scenarios.append(scenario)
        
        self.scenarios.extend(scenarios)
        return scenarios
    
    def create_mesh_visualization(self, client_name: str) -> MeshVisualization:
        """Create 2D mesh visualization for a client"""
        # Filter scenarios for this client
        client_scenarios = [s for s in self.scenarios if s.metadata.get('client_name') == client_name]
        
        if not client_scenarios:
            # Generate scenarios if none exist
            client_scenarios = self.generate_scenarios(client_name)
        
        # Extract coordinates
        x_coords = [s.flexibility_score for s in client_scenarios]
        y_coords = [s.comfort_score for s in client_scenarios]
        scenario_ids = [s.scenario_id for s in client_scenarios]
        
        # Color by distance to target
        colors = []
        for s in client_scenarios:
            if s.distance_to_target < 0.2:
                colors.append('green')  # Close to target
            elif s.distance_to_target < 0.4:
                colors.append('yellow')  # Medium distance
            else:
                colors.append('red')  # Far from target
        
        # Size by total assets
        sizes = [sum(s.assets.values()) / 10000 for s in client_scenarios]  # Scale down
        
        # Hover data
        hover_data = []
        for s in client_scenarios:
            hover_data.append({
                'scenario_id': s.scenario_id,
                'total_assets': f"${sum(s.assets.values()):,.0f}",
                'cash': f"${s.assets.get('cash', 0):,.0f}",
                'stocks': f"${s.assets.get('stocks', 0):,.0f}",
                'bonds': f"${s.assets.get('bonds', 0):,.0f}",
                'real_estate': f"${s.assets.get('real_estate', 0):,.0f}",
                'flexibility': f"{s.flexibility_score:.3f}",
                'comfort': f"{s.comfort_score:.3f}",
                'risk': f"{s.risk_metrics['volatility']:.3f}",
                'income': f"${s.income:,.0f}",
                'expenses': f"${s.expenses:,.0f}"
            })
        
        # Optimal path (simplified - from current to target)
        optimal_path = [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]  # Example path
        
        # Target zone (high flexibility, high comfort)
        target_zone = [(0.6, 0.7), (0.8, 0.7), (0.8, 0.9), (0.6, 0.9)]
        
        visualization = MeshVisualization(
            x_coords=x_coords,
            y_coords=y_coords,
            scenario_ids=scenario_ids,
            colors=colors,
            sizes=sizes,
            hover_data=hover_data,
            optimal_path=optimal_path,
            target_zone=target_zone,
            metadata={'client_name': client_name}
        )
        
        self.visualizations[client_name] = visualization
        return visualization
    
    def generate_recommendations(self, client_name: str) -> List[Dict]:
        """Generate recommendations based on current position in mesh"""
        if client_name not in self.visualizations:
            self.create_mesh_visualization(client_name)
        
        viz = self.visualizations[client_name]
        
        # Find current position (average of scenarios)
        avg_flexibility = np.mean(viz.x_coords)
        avg_comfort = np.mean(viz.y_coords)
        
        recommendations = []
        
        # Flexibility recommendations
        if avg_flexibility < 0.5:
            recommendations.append({
                'type': 'flexibility',
                'priority': 'high',
                'description': 'Increase flexibility by diversifying assets and maintaining higher cash reserves',
                'actions': [
                    'Increase cash allocation to 20-30%',
                    'Diversify across more asset classes',
                    'Consider liquid investments'
                ]
            })
        
        # Comfort recommendations
        if avg_comfort < 0.6:
            recommendations.append({
                'type': 'comfort',
                'priority': 'high',
                'description': 'Improve comfort by reducing risk and increasing stability',
                'actions': [
                    'Reduce stock allocation',
                    'Increase bond allocation',
                    'Build emergency fund'
                ]
            })
        
        # Target zone recommendations
        if avg_flexibility < 0.7 or avg_comfort < 0.8:
            recommendations.append({
                'type': 'target_zone',
                'priority': 'medium',
                'description': 'Move toward optimal flexibility-comfort zone',
                'actions': [
                    'Gradually rebalance toward target allocation',
                    'Monitor progress monthly',
                    'Adjust based on changing circumstances'
                ]
            })
        
        return recommendations
    
    def create_interactive_dashboard(self, client_name: str) -> str:
        """Create interactive dashboard HTML"""
        viz = self.create_mesh_visualization(client_name)
        recommendations = self.generate_recommendations(client_name)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scenarios
        fig.add_trace(go.Scatter(
            x=viz.x_coords,
            y=viz.y_coords,
            mode='markers',
            marker=dict(
                size=viz.sizes,
                color=viz.colors,
                opacity=0.7
            ),
            text=[f"Scenario {i+1}" for i in range(len(viz.scenario_ids))],
            hovertemplate='<b>%{text}</b><br>' +
                        'Flexibility: %{x:.3f}<br>' +
                        'Comfort: %{y:.3f}<br>' +
                        '<extra></extra>',
            name='Financial Scenarios'
        ))
        
        # Add target zone
        target_x = [p[0] for p in viz.target_zone]
        target_y = [p[1] for p in viz.target_zone]
        fig.add_trace(go.Scatter(
            x=target_x + [target_x[0]],  # Close the polygon
            y=target_y + [target_y[0]],
            mode='lines',
            line=dict(color='green', width=3, dash='dash'),
            name='Target Zone'
        ))
        
        # Add optimal path
        path_x = [p[0] for p in viz.optimal_path]
        path_y = [p[1] for p in viz.optimal_path]
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue'),
            name='Optimal Path'
        ))
        
        fig.update_layout(
            title=f'Financial Mesh: Flexibility vs. Comfort - {client_name}',
            xaxis_title='Flexibility Score',
            yaxis_title='Comfort Score',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=800,
            height=600
        )
        
        # Create recommendations HTML
        rec_html = '<div class="recommendations">'
        rec_html += '<h3>Recommendations</h3>'
        for rec in recommendations:
            rec_html += f'<div class="recommendation {rec["priority"]}">'
            rec_html += f'<h4>{rec["type"].title()}</h4>'
            rec_html += f'<p>{rec["description"]}</p>'
            rec_html += '<ul>'
            for action in rec['actions']:
                rec_html += f'<li>{action}</li>'
            rec_html += '</ul></div>'
        rec_html += '</div>'
        
        # Combine into full dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Mesh Dashboard - {client_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: flex; gap: 20px; }}
                .chart {{ flex: 2; }}
                .recommendations {{ flex: 1; }}
                .recommendation {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .high {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                .medium {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .low {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Financial Mesh Dashboard</h1>
            <div class="dashboard">
                <div class="chart">
                    {fig.to_html(full_html=False)}
                </div>
                <div class="recommendations">
                    {rec_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return dashboard_html 