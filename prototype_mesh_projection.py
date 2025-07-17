#!/usr/bin/env python3
"""
Prototype: Mesh Cash Flow Projection
Demonstrates the essence of the mesh system:
- Loads a sample person's JSON
- Runs the mesh for 10 years (no lifestyle inflation)
- Extracts a single path's cash flow projection
- Exports results as CSV and JSON
"""

import os
import json
from datetime import datetime
import csv
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.stochastic_mesh_engine import StochasticMeshEngine
from src.enhanced_pdf_processor import FinancialMilestone

PERSON_ID = 'person_001'
PEOPLE_DIR = os.path.join('data', 'inputs', 'people', 'current')


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


def main():
    print(f"\n🚀 Mesh Prototype: Cash Flow Projection for {PERSON_ID}")
    financial_state, goals, life_events = load_person_data(PERSON_ID)

    # Initial state: cash, investments, real_estate, debts (sum liabilities)
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

    # Run mesh for 10 years
    mesh_engine = StochasticMeshEngine(initial_state)
    mesh_engine.initialize_mesh(milestones, time_horizon_years=10)

    # Pick a single path: for prototype, just follow the current_position node back to root
    node_id = mesh_engine.current_position
    path_nodes = []
    while node_id:
        node = mesh_engine.memory_manager.batch_retrieve([node_id])[0]
        path_nodes.append(node)
        if node.parent_nodes:
            node_id = node.parent_nodes[0]
        else:
            break
    path_nodes = list(reversed(path_nodes))

    # Export cash flow projection as CSV and JSON
    csv_filename = f"mesh_projection_{PERSON_ID}.csv"
    json_filename = f"mesh_projection_{PERSON_ID}.json"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'cash', 'investments', 'real_estate', 'debts', 'net_worth', 'event_triggers'])
        for node in path_nodes:
            fs = node.financial_state
            net_worth = fs.get('cash', 0) + fs.get('investments', 0) + fs.get('real_estate', 0) - fs.get('debts', 0)
            writer.writerow([
                node.timestamp.strftime('%Y-%m-%d'),
                fs.get('cash', 0),
                fs.get('investments', 0),
                fs.get('real_estate', 0),
                fs.get('debts', 0),
                net_worth,
                ';'.join(node.event_triggers) if node.event_triggers else ''
            ])
    with open(json_filename, 'w') as f:
        json.dump([
            {
                'timestamp': node.timestamp.isoformat(),
                'financial_state': node.financial_state,
                'event_triggers': node.event_triggers,
                'parent_nodes': node.parent_nodes
            }
            for node in path_nodes
        ], f, indent=2, default=str)
    print(f"\n✅ Exported cash flow projection to {csv_filename} and {json_filename}")
    print(f"\nSample rows:")
    for node in path_nodes[:5]:
        print(f"  {node.timestamp.date()} | cash: {node.financial_state.get('cash', 0):,.2f} | investments: {node.financial_state.get('investments', 0):,.2f} | debts: {node.financial_state.get('debts', 0):,.2f}")

if __name__ == "__main__":
    main() 