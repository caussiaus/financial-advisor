#!/usr/bin/env python3
"""
Omega Mesh Web Application
"""
from flask import Flask, render_template, request, jsonify
import os
import json
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename

from src.omega_mesh_integration import OmegaMeshIntegration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/inputs/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize with default financial state
initial_state = {
    'total_wealth': 2000000,
    'cash': 400000,
    'savings': 600000,
    'investments': 1000000,
    'debts': 0
}

# Create Omega mesh system
omega_system = OmegaMeshIntegration(initial_state)

@app.route('/')
def index():
    """Render main interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process milestones and entities"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
        
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file with secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process file
        try:
            milestones, entities = omega_system.process_ips_document(file_path)
        except Exception as e:
            print(f"Error processing document: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500
        
        # Check if milestones were extracted
        if not milestones:
            return jsonify({'error': 'No milestones found in the document'}), 400
            
        # Initialize mesh with milestones
        try:
            status = omega_system.mesh_engine.initialize_mesh(milestones)
        except Exception as e:
            print(f"Error initializing mesh: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error initializing mesh: {str(e)}'}), 500
            
        return jsonify({
            'success': True,
            'status': status,
            'milestones': [
                {
                    'event_type': m.event_type,
                    'description': m.description,
                    'timestamp': m.timestamp.isoformat(),
                    'financial_impact': m.financial_impact,
                    'probability': m.probability,
                    'entity': m.entity
                }
                for m in milestones
            ],
            'entities': [
                {
                    'name': e.name,
                    'entity_type': e.entity_type,
                    'initial_balances': e.initial_balances,
                    'metadata': e.metadata
                }
                for e in entities
            ]
        })
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/demo')
def run_demo():
    """Run demonstration with sample data"""
    try:
        # Use sample milestones and entities
        try:
            milestones = omega_system._create_sample_milestones()
            entities = omega_system._create_sample_entities()
        except Exception as e:
            print(f"Error creating sample data: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error creating sample data: {str(e)}'}), 500
        
        # Initialize mesh
        try:
            status = omega_system.mesh_engine.initialize_mesh(milestones)
        except Exception as e:
            print(f"Error initializing mesh: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error initializing mesh: {str(e)}'}), 500
        
        # Generate some sample payments
        try:
            payment_demo = omega_system.demonstrate_flexible_payment()
        except Exception as e:
            print(f"Error generating payment demo: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error generating payment demo: {str(e)}'}), 500
        
        # Show mesh evolution
        try:
            evolution = omega_system.show_omega_mesh_evolution()
        except Exception as e:
            print(f"Error showing mesh evolution: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error showing mesh evolution: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'status': status,
            'payment_demo': payment_demo,
            'evolution': evolution,
            'milestones': [
                {
                    'event_type': m.event_type,
                    'description': m.description,
                    'timestamp': m.timestamp.isoformat(),
                    'financial_impact': m.financial_impact,
                    'probability': m.probability,
                    'entity': m.entity
                }
                for m in milestones
            ],
            'entities': [
                {
                    'name': e.name,
                    'entity_type': e.entity_type,
                    'initial_balances': e.initial_balances,
                    'metadata': e.metadata
                }
                for e in entities
            ]
        })
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Omega Mesh Web Application...")
    print("🚀 Navigate to http://localhost:8081")
    print(f"💰 Accounting initialized with ${initial_state['total_wealth']:,.2f} total wealth")
    app.run(host='0.0.0.0', port=8081, debug=True) 