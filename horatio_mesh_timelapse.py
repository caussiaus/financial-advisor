#!/usr/bin/env python3
"""
Horatio Mesh Timelapse Analysis
- Loads Horatio's profile data
- Runs mesh for 10 years with monthly snapshots
- Exports each snapshot showing mesh evolution
- Demonstrates the "sculpture" of financial possibilities
"""

import os
import json
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.enhanced_pdf_processor import FinancialMilestone

HORATIO_FILE = 'data/inputs/people/archived/horatio_profile.json'
SNAPSHOT_INTERVAL_MONTHS = 1
MESH_HORIZON_YEARS = 10


def load_horatio_data():
    """Load Horatio's profile data"""
    with open(HORATIO_FILE, 'r') as f:
        horatio_data = json.load(f)
    return horatio_data


def convert_horatio_to_mesh_data(horatio_data):
    """Convert Horatio's data to mesh-compatible format"""
    # Initial financial state
    financial_profile = horatio_data['financial_profile']
    initial_state = {
        'cash': financial_profile['liquid_assets'],
        'investments': financial_profile['investment_portfolio'],
        'real_estate': financial_profile['real_estate'],
        'debts': financial_profile['total_liabilities']
    }
    initial_state['total_wealth'] = (
        initial_state['cash'] + initial_state['investments'] + 
        initial_state['real_estate'] - initial_state['debts']
    )
    
    # Convert lifestyle events to milestones
    milestones = []
    for event in horatio_data['lifestyle_events']:
        milestone = FinancialMilestone(
            timestamp=datetime.fromisoformat(event['estimated_date']),
            event_type=event['event_type'],
            description=event['description'],
            financial_impact=event['amount'],
            probability=event.get('probability', 0.5)
        )
        milestones.append(milestone)
    
    return initial_state, milestones


def get_mesh_snapshot(mesh_engine, snapshot_time):
    """Get a snapshot of the mesh at a specific time"""
    # Get all nodes from the mesh
    all_nodes = []
    all_edges = []
    
    # Try to get all node IDs from memory manager
    try:
        # Get all stored nodes
        stored_nodes = []
        for node_id in mesh_engine.omega_mesh.nodes():
            node = mesh_engine.memory_manager.batch_retrieve([node_id])[0]
            if node and node.timestamp <= snapshot_time:
                stored_nodes.append(node)
        
        # Convert to snapshot format
        for node in stored_nodes:
            all_nodes.append({
                'node_id': node.node_id,
                'timestamp': node.timestamp.isoformat(),
                'financial_state': node.financial_state,
                'event_triggers': node.event_triggers,
                'probability': node.probability,
                'is_solidified': node.is_solidified
            })
        
        # Get edges
        for edge in mesh_engine.omega_mesh.edges():
            source_node = mesh_engine.memory_manager.batch_retrieve([edge[0]])[0]
            target_node = mesh_engine.memory_manager.batch_retrieve([edge[1]])[0]
            if (source_node and target_node and 
                source_node.timestamp <= snapshot_time and 
                target_node.timestamp <= snapshot_time):
                all_edges.append({
                    'source': edge[0],
                    'target': edge[1]
                })
    
    except Exception as e:
        print(f"Warning: Could not get full mesh snapshot: {e}")
        # Fallback: just get current position
        if mesh_engine.current_position:
            current_node = mesh_engine.memory_manager.batch_retrieve([mesh_engine.current_position])[0]
            if current_node and current_node.timestamp <= snapshot_time:
                all_nodes.append({
                    'node_id': current_node.node_id,
                    'timestamp': current_node.timestamp.isoformat(),
                    'financial_state': current_node.financial_state,
                    'event_triggers': current_node.event_triggers,
                    'probability': current_node.probability,
                    'is_solidified': current_node.is_solidified
                })
    
    return {
        'snapshot_time': snapshot_time.isoformat(),
        'nodes': all_nodes,
        'edges': all_edges,
        'total_nodes': len(all_nodes),
        'total_edges': len(all_edges)
    }


def main():
    print("\n🎭 Horatio Mesh Timelapse Analysis")
    print("=" * 50)
    
    # Load Horatio's data
    horatio_data = load_horatio_data()
    print(f"✅ Loaded Horatio's profile")
    print(f"   Age: {horatio_data['profile']['age']}")
    print(f"   Income: ${horatio_data['profile']['base_income']:,.0f}")
    print(f"   Total Assets: ${horatio_data['financial_profile']['total_assets']:,.0f}")
    
    # Convert to mesh format
    initial_state, milestones = convert_horatio_to_mesh_data(horatio_data)
    print(f"✅ Converted to mesh format")
    print(f"   Initial wealth: ${initial_state['total_wealth']:,.0f}")
    print(f"   Milestones: {len(milestones)} events")
    
    # Initialize mesh
    mesh_engine = StochasticMeshEngine(initial_state)
    mesh_engine.initialize_mesh(milestones, time_horizon_years=MESH_HORIZON_YEARS)
    print(f"✅ Mesh initialized")
    
    # Generate snapshot times (monthly for 10 years)
    start_time = datetime.now()
    snapshot_times = []
    for year in range(MESH_HORIZON_YEARS + 1):
        for month in range(12):
            snapshot_time = start_time + timedelta(days=365 * year + 30 * month)
            snapshot_times.append(snapshot_time)
    
    # Take snapshots
    snapshots = []
    print(f"\n📸 Taking {len(snapshot_times)} snapshots...")
    
    for i, snapshot_time in enumerate(snapshot_times):
        if i % 12 == 0:  # Print progress yearly
            print(f"   Year {i//12}: {snapshot_time.strftime('%Y-%m')}")
        
        snapshot = get_mesh_snapshot(mesh_engine, snapshot_time)
        snapshots.append(snapshot)
    
    # Export snapshots
    print(f"\n💾 Exporting snapshots...")
    
    # Export as JSON
    with open('horatio_mesh_timelapse.json', 'w') as f:
        json.dump({
            'horatio_profile': horatio_data,
            'mesh_config': {
                'horizon_years': MESH_HORIZON_YEARS,
                'snapshot_interval_months': SNAPSHOT_INTERVAL_MONTHS,
                'initial_state': initial_state
            },
            'snapshots': snapshots
        }, f, indent=2, default=str)
    
    # Export summary CSV
    with open('horatio_mesh_summary.csv', 'w') as f:
        f.write('timestamp,total_nodes,total_edges,avg_wealth,events_triggered\n')
        for snapshot in snapshots:
            if snapshot['nodes']:
                avg_wealth = sum(
                    node['financial_state'].get('total_wealth', 0) 
                    for node in snapshot['nodes']
                ) / len(snapshot['nodes'])
                events = sum(
                    len(node['event_triggers']) 
                    for node in snapshot['nodes']
                )
            else:
                avg_wealth = 0
                events = 0
            
            f.write(f"{snapshot['snapshot_time']},{snapshot['total_nodes']},"
                   f"{snapshot['total_edges']},{avg_wealth:.2f},{events}\n")
    
    print(f"✅ Exported:")
    print(f"   horatio_mesh_timelapse.json - Full timelapse data")
    print(f"   horatio_mesh_summary.csv - Summary statistics")
    
    # Print sample statistics
    print(f"\n📊 Sample Statistics:")
    for i, snapshot in enumerate(snapshots[::12]):  # Yearly samples
        year = i
        print(f"   Year {year}: {snapshot['total_nodes']} nodes, "
              f"{snapshot['total_edges']} edges")


if __name__ == "__main__":
    main() 