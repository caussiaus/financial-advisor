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
            print(f"✅ Processed document: {len(milestones)} milestones, {len(entities)} entities")
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
            print(f"✅ Initialized mesh with status: {type(status)}")
        except Exception as e:
            print(f"Error initializing mesh: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error initializing mesh: {str(e)}'}), 500
            
        # Ensure status is JSON serializable
        if status:
            status = omega_system.mesh_engine._convert_to_json_serializable(status)
            print(f"✅ Converted status to JSON serializable")
            
        # Convert milestones and entities to ensure JSON serializable
        serializable_milestones = []
        for m in milestones:
            try:
                serializable_milestones.append({
                    'event_type': m.event_type,
                    'description': m.description,
                    'timestamp': m.timestamp.isoformat(),
                    'financial_impact': float(m.financial_impact) if m.financial_impact is not None else 0.0,
                    'probability': float(m.probability) if m.probability is not None else 0.7,
                    'entity': m.entity
                })
            except Exception as e:
                print(f"⚠️ Error converting milestone {m}: {e}")
                continue
                
        serializable_entities = []
        for e in entities:
            try:
                serializable_entities.append({
                    'name': e.name,
                    'entity_type': e.entity_type,
                    'initial_balances': {k: float(v) if v is not None else 0.0 for k, v in e.initial_balances.items()},
                    'metadata': e.metadata
                })
            except Exception as e:
                print(f"⚠️ Error converting entity {e}: {e}")
                continue
                
        print(f"✅ Converted {len(serializable_milestones)} milestones and {len(serializable_entities)} entities")
            
        try:
            response_data = {
                'success': True,
                'status': status,
                'milestones': serializable_milestones,
                'entities': serializable_entities
            }
            
            # Use DL-friendly storage to ensure serialization
            from src.dl_friendly_storage import DLFriendlyStorage
            storage = DLFriendlyStorage()
            serialized_response = storage._convert_to_json_serializable(response_data)
            
            return jsonify(serialized_response)
        except Exception as e:
            print(f"❌ JSON serialization error: {e}")
            print(f"❌ Error type: {type(e)}")
            # Try to identify the problematic data
            for key, value in response_data.items():
                try:
                    json.dumps(value)
                    print(f"✅ {key} is JSON serializable")
                except Exception as json_error:
                    print(f"❌ {key} is NOT JSON serializable: {json_error}")
            return jsonify({'error': f'JSON serialization error: {str(e)}'}), 500
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
        
        # Ensure status is JSON serializable
        if status:
            status = omega_system.mesh_engine._convert_to_json_serializable(status)
        
        # Generate some sample payments
        try:
            payment_demo = omega_system.demonstrate_flexible_payment()
        except Exception as e:
            print(f"Error generating payment demo: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error generating payment demo: {str(e)}'}), 500
        
        # Get system status and recommendations
        try:
            system_status = omega_system.get_system_status()
            recommendations = omega_system.generate_monthly_recommendations(months_ahead=6)
        except Exception as e:
            print(f"Error getting system status: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Error getting system status: {str(e)}'}), 500
        
        # Ensure all data is JSON serializable
        if payment_demo:
            payment_demo = omega_system.mesh_engine._convert_to_json_serializable(payment_demo)
        if system_status:
            system_status = omega_system.mesh_engine._convert_to_json_serializable(system_status)
        if recommendations:
            recommendations = omega_system.mesh_engine._convert_to_json_serializable(recommendations)
        
        return jsonify({
            'success': True,
            'status': status,
            'payment_demo': payment_demo,
            'system_status': system_status,
            'recommendations': recommendations,
            'milestones': [
                {
                    'event_type': m.event_type,
                    'description': m.description,
                    'timestamp': m.timestamp.isoformat(),
                    'financial_impact': float(m.financial_impact) if m.financial_impact is not None else 0.0,
                    'probability': float(m.probability) if m.probability is not None else 0.7,
                    'entity': m.entity
                }
                for m in milestones
            ],
            'entities': [
                {
                    'name': e.name,
                    'entity_type': e.entity_type,
                    'initial_balances': {k: float(v) if v is not None else 0.0 for k, v in e.initial_balances.items()},
                    'metadata': e.metadata
                }
                for e in entities
            ]
        })
    except Exception as e:
        print(f"Unexpected error in demo: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/commutator/optimize', methods=['POST'])
def optimize_with_commutators():
    """Optimize financial state using commutator algorithms"""
    try:
        data = request.get_json() or {}
        target_metrics = data.get('target_metrics', None)
        
        # Optimize state using commutator algorithms
        result = omega_system.optimize_financial_state_with_commutators(target_metrics)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in commutator optimization: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/commutator/analysis', methods=['GET'])
def get_commutator_analysis():
    """Get commutator analysis of current state"""
    try:
        analysis = omega_system.get_commutator_analysis()
        return jsonify(analysis)
    except Exception as e:
        print(f"Error getting commutator analysis: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/commutator/execute', methods=['POST'])
def execute_commutator_sequence():
    """Execute specific commutator operations"""
    try:
        data = request.get_json() or {}
        operation_types = data.get('operation_types', None)
        
        result = omega_system.execute_commutator_sequence(operation_types)
        return jsonify(result)
    except Exception as e:
        print(f"Error executing commutator sequence: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Omega Mesh Web Application...")
    print("🚀 Navigate to http://localhost:8081")
    print(f"💰 Accounting initialized with ${initial_state['total_wealth']:,.2f} total wealth")
    app.run(host='0.0.0.0', port=8081, debug=True) 