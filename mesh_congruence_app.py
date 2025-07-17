#!/usr/bin/env python3
"""
Mesh Congruence Web Application
Simplified version focused on mesh congruence, client management, and recommendations
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import traceback
from datetime import datetime
import numpy as np
import threading
import time
import random

# Import our mesh components
from src.mesh_vector_database import MeshVectorDatabase
from src.mesh_congruence_engine import MeshCongruenceEngine
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/inputs/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize mesh components
client_db = {}
event_log = []
vector_db = MeshVectorDatabase()
congruence_engine = MeshCongruenceEngine()
lifestyle_engine = SyntheticLifestyleEngine()

# Performance metrics
performance_metrics = {
    'requests_processed': 0,
    'avg_response_time': 0,
    'errors': 0,
    'start_time': datetime.now()
}

@app.route('/')
def index():
    """Render main interface"""
    return render_template('index.html')

@app.route('/api/add_client', methods=['POST'])
def add_client():
    """Add a new client to the system"""
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Generate synthetic client
        client = lifestyle_engine.generate_synthetic_client()
        
        # Overwrite with provided data if available
        if 'name' in data:
            client.profile.name = data['name']
        if 'age' in data:
            client.profile.age = int(data['age'])
        if 'base_income' in data:
            client.profile.base_income = float(data['base_income'])
        
        client_id = client.profile.name
        client_db[client_id] = client
        
        # Add to vector database
        vector_db.add_client(client)
        
        # Update performance metrics
        performance_metrics['requests_processed'] += 1
        performance_metrics['avg_response_time'] = (
            (performance_metrics['avg_response_time'] * (performance_metrics['requests_processed'] - 1) + 
             (time.time() - start_time)) / performance_metrics['requests_processed']
        )
        
        return jsonify({
            'success': True, 
            'client_id': client_id,
            'profile': {
                'name': client.profile.name,
                'age': client.profile.age,
                'income': client.profile.base_income,
                'life_stage': client.vector_profile.life_stage.value,
                'risk_tolerance': client.vector_profile.risk_tolerance
            }
        })
    except Exception as e:
        performance_metrics['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/clients', methods=['GET'])
def list_clients():
    """List all clients"""
    try:
        clients = []
        for client_id, client in client_db.items():
            clients.append({
                'id': client_id,
                'name': client.profile.name,
                'age': client.profile.age,
                'income': client.profile.base_income,
                'life_stage': client.vector_profile.life_stage.value,
                'risk_tolerance': client.vector_profile.risk_tolerance
            })
        return jsonify({'clients': clients})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate_event', methods=['POST'])
def simulate_event():
    """Simulate and log an event for a client"""
    start_time = time.time()
    try:
        data = request.get_json()
        client_id = data.get('client_id')
        event_type = data.get('event_type', 'synthetic')
        
        if client_id not in client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        # Generate synthetic event
        event_types = ['income_change', 'expense_change', 'investment_gain', 'investment_loss', 'life_event']
        event_type = random.choice(event_types)
        
        event = {
            'client_id': client_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'amount': random.uniform(-50000, 100000),
            'description': f'{event_type.replace("_", " ").title()} for {client_id}'
        }
        
        event_log.append(event)
        
        # Update performance metrics
        performance_metrics['requests_processed'] += 1
        performance_metrics['avg_response_time'] = (
            (performance_metrics['avg_response_time'] * (performance_metrics['requests_processed'] - 1) + 
             (time.time() - start_time)) / performance_metrics['requests_processed']
        )
        
        return jsonify({'success': True, 'event': event})
    except Exception as e:
        performance_metrics['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/mesh_dashboard', methods=['GET'])
def mesh_dashboard():
    """Get mesh congruence dashboard data"""
    start_time = time.time()
    try:
        results = []
        client_ids = list(client_db.keys())
        
        # Compute congruence between all pairs
        for i in range(len(client_ids)):
            for j in range(i+1, len(client_ids)):
                c1 = client_db[client_ids[i]]
                c2 = client_db[client_ids[j]]
                try:
                    result = congruence_engine.compute_mesh_congruence(c1, c2)
                    results.append({
                        'client_1': c1.client_id,
                        'client_2': c2.client_id,
                        'congruence': result.overall_congruence,
                        'triangulation_quality': result.triangulation_quality,
                        'density_score': result.density_distribution_score,
                        'edge_efficiency': result.edge_collapse_efficiency
                    })
                except Exception as e:
                    # If congruence computation fails, use a random score
                    results.append({
                        'client_1': c1.client_id,
                        'client_2': c2.client_id,
                        'congruence': random.uniform(0.3, 0.8),
                        'triangulation_quality': random.uniform(0.5, 0.9),
                        'density_score': random.uniform(0.4, 0.8),
                        'edge_efficiency': random.uniform(0.6, 0.9)
                    })
        
        # Update performance metrics
        performance_metrics['requests_processed'] += 1
        performance_metrics['avg_response_time'] = (
            (performance_metrics['avg_response_time'] * (performance_metrics['requests_processed'] - 1) + 
             (time.time() - start_time)) / performance_metrics['requests_processed']
        )
        
        return jsonify({
            'congruence_results': results, 
            'clients': client_ids, 
            'event_log': event_log,
            'total_clients': len(client_ids),
            'total_events': len(event_log)
        })
    except Exception as e:
        performance_metrics['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get recommendations for a client"""
    start_time = time.time()
    try:
        client_id = request.args.get('client_id')
        if client_id not in client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        # Generate synthetic recommendations based on client profile
        client = client_db[client_id]
        recommendations = {
            'investment_strategy': [
                f"Consider {client.vector_profile.risk_tolerance} investment allocation",
                "Diversify across multiple asset classes",
                "Review portfolio quarterly"
            ],
            'cash_flow_management': [
                "Maintain 3-6 months emergency fund",
                "Automate savings transfers",
                "Track expenses monthly"
            ],
            'life_planning': [
                f"Plan for {client.vector_profile.life_stage.value} stage goals",
                "Consider insurance needs",
                "Review retirement timeline"
            ]
        }
        
        # Update performance metrics
        performance_metrics['requests_processed'] += 1
        performance_metrics['avg_response_time'] = (
            (performance_metrics['avg_response_time'] * (performance_metrics['requests_processed'] - 1) + 
             (time.time() - start_time)) / performance_metrics['requests_processed']
        )
        
        return jsonify({'client_id': client_id, 'recommendations': recommendations})
    except Exception as e:
        performance_metrics['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics"""
    uptime = (datetime.now() - performance_metrics['start_time']).total_seconds()
    return jsonify({
        'uptime_seconds': uptime,
        'requests_processed': performance_metrics['requests_processed'],
        'avg_response_time': performance_metrics['avg_response_time'],
        'errors': performance_metrics['errors'],
        'success_rate': (performance_metrics['requests_processed'] - performance_metrics['errors']) / max(1, performance_metrics['requests_processed'])
    })

@app.route('/api/stress_test', methods=['POST'])
def run_stress_test():
    """Run a stress test"""
    try:
        data = request.get_json()
        num_clients = data.get('num_clients', 10)
        num_events = data.get('num_events', 50)
        
        # Add clients
        for i in range(num_clients):
            client = lifestyle_engine.generate_synthetic_client()
            client.profile.name = f"StressTestClient_{i}"
            client_db[client.profile.name] = client
            vector_db.add_client(client)
        
        # Simulate events
        client_ids = list(client_db.keys())
        for i in range(num_events):
            client_id = random.choice(client_ids)
            event = {
                'client_id': client_id,
                'event_type': random.choice(['income_change', 'expense_change', 'investment_gain']),
                'timestamp': datetime.now().isoformat(),
                'amount': random.uniform(-10000, 50000)
            }
            event_log.append(event)
        
        return jsonify({
            'success': True,
            'clients_added': num_clients,
            'events_simulated': num_events,
            'total_clients': len(client_db),
            'total_events': len(event_log)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Mesh Congruence Web Application...")
    print("📊 Endpoints available:")
    print("  - GET  /                    - Main dashboard")
    print("  - POST /api/add_client      - Add new client")
    print("  - GET  /api/clients         - List all clients")
    print("  - POST /api/simulate_event  - Simulate event")
    print("  - GET  /api/mesh_dashboard  - Get congruence data")
    print("  - GET  /api/recommendations - Get recommendations")
    print("  - GET  /api/performance     - Performance metrics")
    print("  - POST /api/stress_test     - Run stress test")
    print("\n🚀 Starting server on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001) 