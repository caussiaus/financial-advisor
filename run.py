#!/usr/bin/env python
"""
IPS Toolkit - Main Application Runner
Author: ChatGPT 2025-07-16

A unified runner for the Investment Policy Statement (IPS) toolkit.
This script provides multiple modes of operation:
- Web Service: Runs a Flask-based web interface for PDF processing and analysis.
- Full Analysis: Executes a comprehensive, non-interactive analysis.
- Interactive Console: A command-line interface for iterative client input.
"""

import sys
import os
import subprocess
import webbrowser
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify
import traceback
import json
import numpy as np

# --- Configuration ---
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="IPS Toolkit Runner")
    parser.add_argument(
        "mode",
        choices=["web", "full", "interactive"],
        default="web",
        nargs="?",
        help="The mode to run the application in."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the web service."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host for the web service."
    )
    return parser.parse_args()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Application Modes ---
def run_web_service(host, port):
    """Starts the Flask web service."""
    print("🚀 Starting Web Service Mode...")
    app = Flask(__name__, template_folder='templates')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.json_encoder = NumpyEncoder

    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('results', filename=filename))

    @app.route('/api/process-pdf', methods=['POST'])
    def process_pdf():
        """API endpoint for processing PDF files"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Only PDF and TXT files are allowed.'}), 400
            
            # Save the file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the file using enhanced chunked processor
            try:
                # Import and use the enhanced chunked processor
                sys.path.append('src')
                from enhanced_chunked_processor import EnhancedChunkedProcessor
                from fsqca_dashboard import fsQCADashboard
                
                client_id = f"CLIENT_{filename.replace('.', '_').replace(' ', '_')}"
                processor = EnhancedChunkedProcessor(client_id)
                results = processor.process_document(file_path)
                
                # Generate fsQCA dashboard
                dashboard = fsQCADashboard(client_id)
                dashboard.load_results(results)
                dashboard_path = dashboard.save_dashboard()
                
                # Generate fsQCA report
                fsqca_report = dashboard.generate_fsqca_report()
                
                # Save comprehensive results
                output_file = f"enhanced_chunked_analysis_{client_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Convert results to JSON-serializable format
                def convert_to_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    else:
                        return obj
                
                serializable_results = convert_to_serializable(results)
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': 'File processed successfully with enhanced chunked processing and fsQCA analysis',
                    'results': serializable_results,
                    'dashboard_path': dashboard_path,
                    'fsqca_report': fsqca_report,
                    'processing_method': 'chunked_tree_coordination',
                    'happiness_estimate': results.get('fsqca_analysis', {}).get('happiness_estimate', 0.0)
                })
                
            except Exception as e:
                return jsonify({
                    'error': f'Processing failed: {str(e)}'
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': f'Upload failed: {str(e)}'
            }), 500

    @app.route('/results/<filename>')
    def results(filename):
        # Derive client_id from filename
        client_id = f"CLIENT_{filename.replace('.', '_').replace(' ', '_')}"
        output_file = f"enhanced_chunked_analysis_{client_id}.json"
        results = None
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
        return render_template('results.html', filename=filename, results=results)
        
    @app.route('/fsqca-dashboard/<client_id>')
    def fsqca_dashboard(client_id):
        """Serve fsQCA dashboard"""
        dashboard_path = f"docs/fsqca_dashboard_{client_id}.html"
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r') as f:
                return f.read()
        else:
            return "Dashboard not found", 404

    @app.route('/d3-vis')
    def d3_vis():
        """Serves the D3.js visualization."""
        return render_template('d3_visualization.html')

    @app.route('/api/house-purchase-analysis', methods=['POST'])
    def house_purchase_analysis():
        """API endpoint for house purchase scenario analysis using transaction mesh"""
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            client_filename = data.get('filename')
            house_price = data.get('house_price')  # Optional
            
            if not client_filename:
                return jsonify({'error': 'Client filename required'}), 400
            
            # Import required components
            sys.path.append('src')
            from transaction_mesh_analyzer import TransactionMeshAnalyzer
            from spending_vector_database import SpendingPatternVectorDB
            from spending_surface_modeler import SpendingSurfaceModeler
            from temporal_fsqca_integration import TemporalfsQCAIntegrator
            from timeline_bias_engine import TimelineBiasEngine
            from continuous_configuration_mesh import ContinuousConfigurationMesh
            
            print(f"🏠 Analyzing house purchase scenario for {client_filename}")
            
            # Initialize components
            try:
                vector_db = SpendingPatternVectorDB()
                surface_modeler = SpendingSurfaceModeler(vector_db)
                timeline_engine = TimelineBiasEngine()
                config_mesh = ContinuousConfigurationMesh({
                    'income': [40000, 150000, 10],
                    'age': [25, 65, 5],
                    'risk_tolerance': [0.2, 0.8, 0.1]
                })
                temporal_integrator = TemporalfsQCAIntegrator(
                    timeline_engine, surface_modeler, config_mesh
                )
                
                # Initialize transaction mesh analyzer
                mesh_analyzer = TransactionMeshAnalyzer(
                    vector_db, surface_modeler, temporal_integrator
                )
                
                print("✅ Components initialized successfully")
                
            except Exception as e:
                print(f"⚠️ Component initialization failed, using demo mode: {e}")
                # Return demo response if components can't be initialized
                return jsonify({
                    'success': True,
                    'demo_mode': True,
                    'analysis': _generate_demo_house_analysis(client_filename, house_price),
                    'message': 'Demo analysis generated (components not fully initialized)'
                })
            
            # Process the uploaded case file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], client_filename)
            
            if not os.path.exists(file_path):
                return jsonify({'error': f'File {client_filename} not found'}), 404
            
            # Process case and create mesh state
            case_results = mesh_analyzer.process_uploaded_case(file_path)
            mesh_state = case_results['mesh_state']
            
            print(f"📊 Mesh state: {len(mesh_state.available_poles)} available, {len(mesh_state.removed_poles)} eliminated")
            
            # Analyze house purchase scenario
            house_scenario = mesh_analyzer.analyze_house_purchase_scenario(
                mesh_state, house_price
            )
            
            print(f"🏠 House scenario: ${house_scenario.house_price:,} with {len(house_scenario.remaining_poles)} remaining options")
            
            # Generate complete response
            analysis_response = mesh_analyzer.generate_interactive_demo_response(
                mesh_state, house_scenario, case_results['client_data']
            )
            
            return jsonify({
                'success': True,
                'filename': client_filename,
                'house_price': house_scenario.house_price,
                'analysis': analysis_response,
                'processing_timestamp': datetime.now().isoformat(),
                'methodology': 'transaction_mesh_configuration_analysis'
            })
            
        except Exception as e:
            print(f"❌ House purchase analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return demo response as fallback
            return jsonify({
                'success': True,
                'demo_mode': True,
                'analysis': _generate_demo_house_analysis(
                    data.get('filename', 'demo'), 
                    data.get('house_price')
                ),
                'error_details': str(e),
                'message': 'Demo analysis generated due to processing error'
            })

    def _generate_demo_house_analysis(filename, house_price=None):
        """Generate demo house purchase analysis for demonstration"""
        if house_price is None:
            house_price = 350000
        
        return {
            'user_question': "I'm thinking of buying a house, what should I do with the rest of my finances?",
            'mesh_analysis': {
                'total_decision_poles': 12,
                'available_poles': 7,
                'eliminated_by_history': 5,
                'constraint_percentage': 0.42
            },
            'current_position': {
                'welfare_state': {
                    'financial_security': 0.65,
                    'stress_level': 0.45,
                    'quality_of_life': 0.60,
                    'flexibility': 0.75,
                    'growth_potential': 0.70,
                    'social_status': 0.55
                },
                'resources': {
                    'annual_income': 85000,
                    'liquid_savings': 95000,
                    'investment_portfolio': 45000,
                    'credit_score': 750,
                    'emergency_fund': 25000
                },
                'established_path': [
                    'DEMO_001_EMERGENCY_FUND',
                    'DEMO_002_GRAD_DEGREE', 
                    'DEMO_003_CAREER_ESTABLISH',
                    'DEMO_004_INVESTMENT_START',
                    'DEMO_005_CREDIT_BUILDING'
                ]
            },
            'house_purchase_analysis': {
                'scenario': {
                    'house_price': house_price,
                    'down_payment': house_price * 0.20,
                    'monthly_payment': house_price * 0.006,
                    'remaining_options': 6
                },
                'affordability': {
                    'down_payment_coverage': 1.35,
                    'income_ratio': 0.25,
                    'affordability_score': 0.85,
                    'recommendation': 'affordable'
                },
                'welfare_projection': {
                    'financial_security': 0.85,
                    'stress_level': 0.65,
                    'quality_of_life': 0.90,
                    'flexibility': 0.35,
                    'growth_potential': 0.60,
                    'social_status': 0.75
                }
            },
            'optimization_strategy': {
                'immediate_actions': [
                    {
                        'category': 'risk_management',
                        'priority': 'high',
                        'description': 'Rebuild emergency fund to account for homeownership risks',
                        'target_amount': house_price * 0.05,
                        'rationale': 'Homeownership creates new risk categories'
                    },
                    {
                        'category': 'growth_optimization',
                        'priority': 'high', 
                        'description': 'Focus on tax-advantaged growth investments',
                        'rationale': 'Reduced flexibility requires efficient growth strategies'
                    }
                ],
                'medium_term_strategy': [
                    {
                        'category': 'tax_optimization',
                        'priority': 'medium',
                        'description': 'Leverage mortgage interest deduction and homeowner tax benefits'
                    }
                ],
                'risk_mitigation': [
                    {
                        'risk': 'flexibility_risk',
                        'level': 0.9,
                        'mitigation': 'Focus on reversible investments and maintain career optionality',
                        'timeline': 'ongoing'
                    }
                ]
            },
            'configuration_mesh_visualization': {
                'nodes': [
                    {'id': 'house_purchase_scenario', 'status': 'considering', 'size': 25},
                    {'id': 'investment_growth', 'status': 'available', 'size': 15},
                    {'id': 'career_advancement', 'status': 'available', 'size': 12},
                    {'id': 'family_planning', 'status': 'available', 'size': 18},
                    {'id': 'luxury_lifestyle', 'status': 'removed', 'size': 5},
                    {'id': 'high_risk_investment', 'status': 'removed', 'size': 5}
                ],
                'edges': [
                    {'source': 'house_purchase_scenario', 'target': 'luxury_lifestyle', 'type': 'exclusion'},
                    {'source': 'investment_growth', 'target': 'career_advancement', 'type': 'dependency'}
                ],
                'legends': {
                    'green': 'Available Options',
                    'red': 'Eliminated by History', 
                    'yellow': 'Current Consideration'
                }
            },
            'similar_case_insights': {
                'similarity_score': 0.73,
                'interpretation': 'high',
                'insights': [
                    'Your profile matches 73% of successful house purchasers',
                    'Similar cases typically focus on growth investments post-purchase',
                    'Timeline optimization suggests 3-5 year planning horizon for major decisions'
                ]
            }
        }

    print(f"🌐 Navigate to http://{host}:{port}")
    app.run(host=host, port=port, debug=True)

def run_full_analysis():
    """Runs a full, non-interactive analysis."""
    print("🚀 Running Full Analysis Mode...")
    try:
        sys.path.append('src')
        from dynamic_portfolio_engine import DynamicPortfolioEngine
        from life_choice_optimizer import LifeChoiceOptimizer
        from enhanced_dashboard_with_optimization import EnhancedDashboardWithOptimization

        client_config = {
            'income': 250000, 'disposable_cash': 8000, 'allowable_var': 0.15,
            'age': 42, 'risk_profile': 3, 'portfolio_value': 1500000,
            'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
        }

        portfolio_engine = DynamicPortfolioEngine(client_config)
        optimizer = LifeChoiceOptimizer(portfolio_engine)
        dashboard = EnhancedDashboardWithOptimization(portfolio_engine)

        main_dashboard = dashboard.create_enhanced_dashboard()
        optimization_dashboard = optimizer.create_optimization_dashboard()
        interactive_html = dashboard.generate_interactive_html()
        report = optimizer.generate_optimization_report('financial_growth')

        main_dashboard.write_html("docs/full_app_dashboard.html")
        optimization_dashboard.write_html("docs/full_app_optimization.html")
        with open("docs/full_app_interactive.html", "w", encoding="utf-8") as f:
            f.write(interactive_html)
        with open("docs/full_app_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("✅ Full analysis complete. Reports generated in 'docs/'.")
        
    except Exception as e:
        print(f"❌ Error during full analysis: {e}")
        import traceback
        traceback.print_exc()

def run_interactive_console():
    """Runs an interactive console for client input."""
    print("🚀 Starting Interactive Console Mode...")
    # Placeholder for start_service.py logic
    print("Interactive mode is under construction.")

# --- Main Execution ---
def main():
    """Main execution block."""
    args = parse_arguments()

    if args.mode == "web":
        run_web_service(args.host, args.port)
    elif args.mode == "full":
        run_full_analysis()
    elif args.mode == "interactive":
        run_interactive_console()

if __name__ == "__main__":
    main() 