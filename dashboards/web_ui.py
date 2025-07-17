#!/usr/bin/env python3
"""
Web UI for Unified Cash Flow Model

Interactive web interface to explore:
- Time uncertainty mesh visualization
- Key decision moments
- Cash flow state evolution
- Accounting metrics over time
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, jsonify
from src.unified_cash_flow_model import UnifiedCashFlowModel, CashFlowEvent
from src.accounting_debugger import AccountingDebugger
from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine

app = Flask(__name__)

# Global model instance
model = None
debugger = None
mesh_data = None
risk_analysis = None

def initialize_model():
    """Initialize the unified cash flow model with case data"""
    global model, debugger, mesh_data, risk_analysis
    
    # Initialize with case-specific data
    initial_state = {
        'total_wealth': 764560.97,
        'cash': 764560.97 * 0.0892,
        'investments': 764560.97 * 0.9554,
        'income': 150000,
        'expenses': 60000
    }
    
    model = UnifiedCashFlowModel(initial_state)
    
    # Add case-specific events
    case_events = model.create_case_events_from_analysis()
    for event in case_events:
        model.add_cash_flow_event(event)
    
    # Initialize time uncertainty mesh
    mesh_data, risk_analysis = model.initialize_time_uncertainty_mesh(
        num_scenarios=1000,  # Reduced for web performance
        time_horizon_years=5
    )
    
    # Initialize debugger
    debugger = AccountingDebugger(model.accounting_engine, model)
    
    return model, debugger, mesh_data, risk_analysis

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/model_info')
def get_model_info():
    """Get basic model information"""
    if model is None:
        initialize_model()
    
    return jsonify({
        'total_events': len(model.cash_flow_events),
        'initial_wealth': model.initial_state['total_wealth'],
        'events': [
            {
                'id': event.event_id,
                'description': event.description,
                'date': event.estimated_date,
                'amount': event.amount,
                'type': event.event_type,
                'category': event.category
            }
            for event in model.cash_flow_events
        ]
    })

@app.route('/api/cash_flow_timeline')
def get_cash_flow_timeline():
    """Get cash flow timeline data"""
    if model is None:
        initialize_model()
    
    # Simulate cash flows
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    states = model.simulate_cash_flows_over_time(start_date, end_date)
    
    timeline_data = []
    for state in states:
        timeline_data.append({
            'timestamp': state.timestamp.isoformat(),
            'net_worth': float(state.net_worth),
            'total_assets': float(state.total_assets),
            'total_liabilities': float(state.total_liabilities),
            'net_cash_flow': float(state.net_cash_flow),
            'liquidity_ratio': state.liquidity_ratio,
            'stress_level': state.stress_level,
            'cash_checking': float(state.account_balances.get('cash_checking', 0)),
            'cash_savings': float(state.account_balances.get('cash_savings', 0)),
            'investments_stocks': float(state.account_balances.get('investments_stocks', 0)),
            'investments_bonds': float(state.account_balances.get('investments_bonds', 0))
        })
    
    return jsonify(timeline_data)

@app.route('/api/decision_moments')
def get_decision_moments():
    """Get key decision moments"""
    if model is None:
        initialize_model()
    
    decision_moments = []
    
    for event in model.cash_flow_events:
        # Calculate state before and after event
        event_date = pd.to_datetime(event.estimated_date)
        
        # Get state before event
        before_date = event_date - timedelta(days=30)
        before_state = model.calculate_cash_flow_state(before_date)
        
        # Get state after event
        after_date = event_date + timedelta(days=30)
        after_state = model.calculate_cash_flow_state(after_date)
        
        decision_moments.append({
            'event_id': event.event_id,
            'description': event.description,
            'date': event.estimated_date,
            'amount': event.amount,
            'category': event.category,
            'before': {
                'net_worth': float(before_state.net_worth),
                'liquidity_ratio': before_state.liquidity_ratio,
                'stress_level': before_state.stress_level,
                'cash_balance': float(before_state.account_balances.get('cash_checking', 0) + 
                                    before_state.account_balances.get('cash_savings', 0))
            },
            'after': {
                'net_worth': float(after_state.net_worth),
                'liquidity_ratio': after_state.liquidity_ratio,
                'stress_level': after_state.stress_level,
                'cash_balance': float(after_state.account_balances.get('cash_checking', 0) + 
                                    after_state.account_balances.get('cash_savings', 0))
            },
            'impact': {
                'net_worth_change': float(after_state.net_worth - before_state.net_worth),
                'liquidity_change': after_state.liquidity_ratio - before_state.liquidity_ratio,
                'stress_change': after_state.stress_level - before_state.stress_level
            }
        })
    
    return jsonify(decision_moments)

@app.route('/api/mesh_visualization')
def get_mesh_visualization():
    """Get mesh visualization data"""
    if mesh_data is None:
        initialize_model()
    
    # Extract key mesh data for visualization
    mesh_viz_data = {
        'scenarios': len(mesh_data.get('scenarios', [])),
        'time_steps': len(mesh_data.get('time_steps', [])),
        'events': len(mesh_data.get('events', [])),
        'risk_metrics': {
            'total_risk': risk_analysis.get('total_risk', 0),
            'timing_risk': risk_analysis.get('timing_risk', 0),
            'amount_risk': risk_analysis.get('amount_risk', 0),
            'correlation_risk': risk_analysis.get('correlation_risk', 0)
        }
    }
    
    return jsonify(mesh_viz_data)

@app.route('/api/accounting_metrics')
def get_accounting_metrics():
    """Get current accounting metrics"""
    if debugger is None:
        initialize_model()
    
    metrics = debugger.calculate_accounting_metrics()
    validation = debugger.validate_accounting_state()
    
    return jsonify({
        'metrics': {
            'total_assets': float(metrics.total_assets),
            'total_liabilities': float(metrics.total_liabilities),
            'net_worth': float(metrics.net_worth),
            'liquidity_ratio': metrics.liquidity_ratio,
            'debt_to_asset_ratio': metrics.debt_to_asset_ratio,
            'cash_flow_coverage': metrics.cash_flow_coverage,
            'stress_level': metrics.stress_level
        },
        'validation': {
            'is_valid': validation.is_valid,
            'errors': validation.errors,
            'warnings': validation.warnings
        }
    })

@app.route('/api/run_simulation')
def run_simulation():
    """Run a custom simulation"""
    data = request.get_json()
    
    # Create new model with custom parameters
    initial_wealth = data.get('initial_wealth', 1000000)
    events = data.get('events', [])
    
    custom_model = UnifiedCashFlowModel({'total_wealth': initial_wealth})
    
    # Add custom events
    for event_data in events:
        event = CashFlowEvent(
            event_id=event_data['id'],
            description=event_data['description'],
            estimated_date=event_data['date'],
            amount=event_data['amount'],
            source_account=event_data['source_account'],
            target_account=event_data['target_account'],
            event_type=event_data['type']
        )
        custom_model.add_cash_flow_event(event)
    
    # Run simulation
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    states = custom_model.simulate_cash_flows_over_time(start_date, end_date)
    
    # Return results
    timeline_data = []
    for state in states:
        timeline_data.append({
            'timestamp': state.timestamp.isoformat(),
            'net_worth': float(state.net_worth),
            'net_cash_flow': float(state.net_cash_flow),
            'liquidity_ratio': state.liquidity_ratio,
            'stress_level': state.stress_level
        })
    
    return jsonify({
        'timeline': timeline_data,
        'summary': custom_model.get_cash_flow_summary()
    })

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified Cash Flow Model Web UI')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    # Initialize model on startup
    initialize_model()
    print("🚀 Starting Unified Cash Flow Model Web UI...")
    print("📊 Model initialized with case data")
    print(f"🌐 Web UI available at http://localhost:{args.port}")
    app.run(debug=True, host='0.0.0.0', port=args.port) 