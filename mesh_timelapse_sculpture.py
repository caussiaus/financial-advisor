#!/usr/bin/env python3
"""
Mesh Timelapse Sculpture
- Loads all people in data/inputs/people/current/
- For each, runs the mesh for 10 years
- Snapshots the mesh at each year (timestamp)
- Exports each snapshot as a graph (JSON)
- (Ready for later animation/visualization)
"""

import os
import json
from datetime import datetime, timedelta
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.enhanced_pdf_processor import FinancialMilestone

PEOPLE_DIR = os.path.join('data', 'inputs', 'people', 'current')
SNAPSHOT_INTERVAL_YEARS = 1
MESH_HORIZON_YEARS = 10


def load_person_data(person_id):
    person_path = os.path.join(PEOPLE_DIR, person_id)
    with open(os.path.join(person_path, 'financial_state.json'), 'r') as f:
        financial_state = json.load(f)
    with open(os.path.join(person_path, 'goals.json'), 'r') as f:
        goals = json.load(f)
    with open(os.path.join(person_path, 'life_events.json'), 'r') as f:
        life_events = json.load(f)
    return financial_state, goals, life_events


def milestones_from_life_events(life_events):
    milestones = []
    for event in life_events['planned_events']:
        milestones.append(FinancialMilestone(
            timestamp=datetime.fromisoformat(event['date']),
            event_type=event['category'],
            description=event['description'],
            financial_impact=event['expected_impact'],
            probability=event.get('probability', 0.5)
        ))
    return milestones


def snapshot_mesh(mesh_engine, snapshot_times, person_id):
    """
    For each snapshot_time, collect all nodes with timestamp <= snapshot_time.
    Return a dict: {snapshot_time: {nodes: [...], edges: [...]}}
    """
    all_snapshots = {}
    # Collect all nodes from memory manager
    all_node_ids = mesh_engine.memory_manager.get_all_node_ids()
    all_nodes = mesh_engine.memory_manager.batch_retrieve(all_node_ids)
    # Build edges
    edges = []
    node_dict = {}
    for node in all_nodes:
        if node is None:
            continue
        node_dict[node.node_id] = node
        for child in node.child_nodes:
            edges.append((node.node_id, child))
    # For each snapshot time, filter nodes/edges
    for snap_time in snapshot_times:
        nodes_snap = []
        edges_snap = []
        for node in all_nodes:
            if node is None:
                continue
            if node.timestamp <= snap_time:
                nodes_snap.append({
                    'node_id': node.node_id,
                    'timestamp': node.timestamp.isoformat(),
                    'financial_state': node.financial_state,
                    'event_triggers': node.event_triggers,
                    'person_id': person_id
                })
        node_ids_snap = set(n['node_id'] for n in nodes_snap)
        for src, tgt in edges:
            if src in node_ids_snap and tgt in node_ids_snap:
                edges_snap.append({'source': src, 'target': tgt})
        all_snapshots[snap_time.isoformat()] = {
            'nodes': nodes_snap,
            'edges': edges_snap
        }
    return all_snapshots


def main():
    print("\n🚀 Mesh Timelapse Sculpture: Multi-Person Mesh Evolution")
    people = [d for d in os.listdir(PEOPLE_DIR) if os.path.isdir(os.path.join(PEOPLE_DIR, d))]
    print(f"Found {len(people)} people: {people}")
    
    # For each person, run mesh and snapshot
    global_snapshots = defaultdict(lambda: {'nodes': [], 'edges': []})
    for person_id in people:
        print(f"\nProcessing {person_id}...")
        financial_state, goals, life_events = load_person_data(person_id)
        initial_state = {
            'cash': financial_state['assets']['cash'],
            'investments': financial_state['assets']['investments'],
            'real_estate': financial_state['assets']['real_estate'],
            'debts': sum(financial_state['liabilities'].values())
        }
        initial_state['total_wealth'] = (
            initial_state['cash'] + initial_state['investments'] + initial_state['real_estate'] - initial_state['debts']
        )
        milestones = milestones_from_life_events(life_events)
        mesh_engine = StochasticMeshEngine(initial_state)
        mesh_engine.initialize_mesh(milestones, time_horizon_years=MESH_HORIZON_YEARS)
        # Determine snapshot times (every year from now)
        start_time = datetime.now()
        snapshot_times = [start_time + timedelta(days=365 * i) for i in range(MESH_HORIZON_YEARS + 1)]
        person_snapshots = snapshot_mesh(mesh_engine, snapshot_times, person_id)
        # Merge into global snapshots
        for snap_time, snap_data in person_snapshots.items():
            global_snapshots[snap_time]['nodes'].extend(snap_data['nodes'])
            global_snapshots[snap_time]['edges'].extend(snap_data['edges'])
        print(f"  Snapshots taken at: {[t.strftime('%Y-%m-%d') for t in snapshot_times]}")
    # Export each snapshot
    for snap_time, snap_data in global_snapshots.items():
        fname = f"mesh_snapshot_{snap_time.replace(':','-')}.json"
        with open(fname, 'w') as f:
            json.dump(snap_data, f, indent=2, default=str)
        print(f"  Exported snapshot: {fname}")
    print("\n✅ All snapshots exported. Ready for visualization/animation.")

if __name__ == "__main__":
    main() 