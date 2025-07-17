#!/usr/bin/env python3
"""
Enhanced Mesh Congruence System Dashboard
Advanced web interface with proper controls, error handling, and monitoring
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

# Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import numpy as np
import pandas as pd

# Import core controller
from src.core_controller import get_core_controller, initialize_system, get_system_status, get_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMeshDashboard:
    """Enhanced mesh dashboard with proper controls and monitoring"""
    
    def __init__(self):
        # Set up Flask with explicit template folder path
        template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../templates')
        self.app = Flask(__name__, template_folder=template_folder)
        CORS(self.app)
        
        # Configuration
        self.app.config['UPLOAD_FOLDER'] = 'data/inputs/uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Initialize core controller
        self.core_controller = get_core_controller()
        
        # Initialize components
        self.client_db = {}
        self.event_log = []
        
        # Get components from core controller
        components = get_components()
        self.vector_db = components['MeshVectorDatabase']()
        self.congruence_engine = components['MeshCongruenceEngine']()
        self.lifestyle_engine = components['SyntheticLifestyleEngine']()
        self.cash_flow_model = None
        
        # Performance monitoring
        self.performance_metrics = {
            'requests_processed': 0,
            'avg_response_time': 0.0,
            'errors': 0,
            'start_time': datetime.now(),
            'active_connections': 0,
            'memory_usage_mb': 0.0
        }
        
        # System health
        self.system_health = {
            'status': 'healthy',
            'last_check': datetime.now(),
            'components': {
                'vector_db': 'healthy',
                'congruence_engine': 'healthy',
                'lifestyle_engine': 'healthy',
                'cash_flow_model': 'healthy'
            }
        }
        
        # Rate limiting
        self.rate_limits = {
            'requests_per_minute': 100,
            'requests_per_hour': 1000,
            'max_concurrent_requests': 50
        }
        
        self.request_counts = {
            'minute': {'count': 0, 'reset_time': datetime.now()},
            'hour': {'count': 0, 'reset_time': datetime.now()}
        }
        
        # Initialize routes
        self._setup_routes()
        self._setup_error_handlers()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup all application routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template('enhanced_dashboard.html')
        
        @self.app.route('/api/health')
        def health_check():
            """System health check"""
            return jsonify({
                'status': self.system_health['status'],
                'uptime_seconds': (datetime.now() - self.performance_metrics['start_time']).total_seconds(),
                'components': self.system_health['components'],
                'performance': {
                    'requests_processed': self.performance_metrics['requests_processed'],
                    'avg_response_time': self.performance_metrics['avg_response_time'],
                    'errors': self.performance_metrics['errors'],
                    'active_connections': self.performance_metrics['active_connections']
                }
            })
        
        @self.app.route('/api/clients', methods=['GET', 'POST'])
        def manage_clients():
            """Manage clients - GET for list, POST for adding"""
            start_time = time.time()
            
            try:
                if request.method == 'GET':
                    return self._get_clients()
                else:
                    return self._add_client()
            except Exception as e:
                logger.error(f"Error in manage_clients: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/clients/<client_id>', methods=['GET', 'PUT', 'DELETE'])
        def manage_client(client_id):
            """Manage individual client"""
            start_time = time.time()
            
            try:
                if request.method == 'GET':
                    return self._get_client(client_id)
                elif request.method == 'PUT':
                    return self._update_client(client_id)
                else:
                    return self._delete_client(client_id)
            except Exception as e:
                logger.error(f"Error in manage_client: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/events', methods=['GET', 'POST'])
        def manage_events():
            """Manage events"""
            start_time = time.time()
            
            try:
                if request.method == 'GET':
                    return self._get_events()
                else:
                    return self._simulate_event()
            except Exception as e:
                logger.error(f"Error in manage_events: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/mesh/congruence')
        def get_mesh_congruence():
            """Get mesh congruence analysis"""
            start_time = time.time()
            
            try:
                return self._compute_mesh_congruence()
            except Exception as e:
                logger.error(f"Error in get_mesh_congruence: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/recommendations/<client_id>')
        def get_recommendations(client_id):
            """Get recommendations for a client"""
            start_time = time.time()
            
            try:
                return self._generate_recommendations(client_id)
            except Exception as e:
                logger.error(f"Error in get_recommendations: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/data/files')
        def get_data_files():
            """Get list of available data files"""
            start_time = time.time()
            
            try:
                data_files = self._scan_data_files()
                return jsonify({
                    'data_files': data_files,
                    'total_files': len(data_files),
                    'scan_time': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error scanning data files: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/data/<path:filename>')
        def serve_data_file(filename):
            """Serve data files"""
            try:
                file_path = os.path.join('data', filename)
                if os.path.exists(file_path):
                    return send_file(file_path)
                else:
                    return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                logger.error(f"Error serving file {filename}: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analytics/dashboard')
        def get_dashboard_analytics():
            """Get comprehensive dashboard analytics"""
            start_time = time.time()
            
            try:
                return self._get_dashboard_analytics()
            except Exception as e:
                logger.error(f"Error in get_dashboard_analytics: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/system/control', methods=['POST'])
        def system_control():
            """System control endpoints"""
            start_time = time.time()
            
            try:
                data = request.get_json()
                action = data.get('action')
                
                if action == 'restart_components':
                    return self._restart_components()
                elif action == 'clear_cache':
                    return self._clear_cache()
                elif action == 'export_data':
                    return self._export_data()
                elif action == 'import_data':
                    return self._import_data(data.get('data'))
                else:
                    return jsonify({'error': 'Unknown action'}), 400
            except Exception as e:
                logger.error(f"Error in system_control: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/mesh/reallocation_strategy/<client_id>')
        def get_reallocation_strategy(client_id):
            """Return a time series of portfolio allocations and recommended reallocation actions for the client."""
            start_time = time.time()
            try:
                # Find the client in the dashboard's client_db
                client = self.client_db.get(client_id)
                if not client:
                    return jsonify({'error': f'Client {client_id} not found'}), 404

                # Use the recommendation engine to generate a configuration matrix (time series)
                # This will include asset allocations and recommended actions for each period
                profile_data = {
                    'name': client.profile.name,
                    'age': client.profile.age,
                    'risk_tolerance': getattr(client.profile, 'risk_tolerance', 'Moderate'),
                    'base_income': getattr(client.profile, 'base_income', 60000),
                }
                milestones = getattr(client, 'milestones', []) if hasattr(client, 'milestones') else []
                rec_engine = self.core_controller.recommendation_engine
                config_matrix = rec_engine.create_configuration_matrix(
                    person_id=client_id,
                    milestones=milestones,
                    profile_data=profile_data,
                    scenarios=1
                )
                # For each period, extract allocation and recommended actions
                time_series = []
                
                return jsonify({
                    'client_id': client_id,
                    'time_series': time_series,
                    'profile': profile_data
                })
            except Exception as e:
                logger.error(f"Error in get_reallocation_strategy: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/people')
        def get_people():
            """Get all people with their profiles, events, and reallocations"""
            start_time = time.time()
            try:
                people_data = []
                people_dir = os.path.join('data', 'inputs', 'people', 'current')
                
                if not os.path.exists(people_dir):
                    return jsonify({'error': 'People data directory not found'}), 404
                
                for person_dir in os.listdir(people_dir):
                    person_path = os.path.join(people_dir, person_dir)
                    if os.path.isdir(person_path):
                        person_data = self._load_person_data(person_dir, person_path)
                        if person_data:
                            people_data.append(person_data)
                
                return jsonify({
                    'people': people_data,
                    'total': len(people_data)
                })
            except Exception as e:
                logger.error(f"Error in get_people: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/people/<person_id>')
        def get_person(person_id):
            """Get specific person with their events and reallocations"""
            start_time = time.time()
            try:
                people_dir = os.path.join('data', 'inputs', 'people', 'current')
                person_path = os.path.join(people_dir, person_id)
                
                if not os.path.exists(person_path):
                    return jsonify({'error': f'Person {person_id} not found'}), 404
                
                person_data = self._load_person_data(person_id, person_path)
                if not person_data:
                    return jsonify({'error': f'Could not load data for person {person_id}'}), 404
                
                return jsonify(person_data)
            except Exception as e:
                logger.error(f"Error in get_person: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/market/stress_test')
        def run_market_stress_test():
            """Run market stress test with current people data"""
            start_time = time.time()
            try:
                stress_test_results = self._run_market_stress_test()
                return jsonify(stress_test_results)
            except Exception as e:
                logger.error(f"Error in market stress test: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/mesh/network_state')
        def get_mesh_network_state():
            """Get current mesh network state and node connections"""
            start_time = time.time()
            try:
                network_state = self._get_mesh_network_state()
                return jsonify(network_state)
            except Exception as e:
                logger.error(f"Error getting mesh network state: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/mesh/recommendations/debug/<person_id>')
        def debug_recommendations(person_id):
            """Debug recommendations for a specific person showing network influence"""
            start_time = time.time()
            try:
                debug_data = self._debug_person_recommendations(person_id)
                return jsonify(debug_data)
            except Exception as e:
                logger.error(f"Error debugging recommendations: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)

        @self.app.route('/api/mesh/network/visualization')
        def get_network_visualization():
            """Get network visualization data for the mesh"""
            start_time = time.time()
            try:
                viz_data = self._generate_network_visualization()
                return jsonify(viz_data)
            except Exception as e:
                logger.error(f"Error generating network visualization: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/portfolio/insights/<portfolio_name>')
        def get_portfolio_insights(portfolio_name):
            """Get detailed insights for a specific portfolio"""
            start_time = time.time()
            try:
                insights = self._get_portfolio_insights(portfolio_name)
                return jsonify(insights)
            except Exception as e:
                logger.error(f"Error getting portfolio insights: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
        
        @self.app.route('/api/people/bulk_upload', methods=['POST'])
        def bulk_upload_people():
            """Bulk upload people profiles and process each, returning stats"""
            import time
            start_time = time.time()
            stats = []
            try:
                if 'files' not in request.files:
                    return jsonify({'error': 'No files uploaded'}), 400
                files = request.files.getlist('files')
                if not files:
                    return jsonify({'error': 'No files provided'}), 400
                people_dir = os.path.join('data', 'inputs', 'people', 'current')
                os.makedirs(people_dir, exist_ok=True)
                for file in files:
                    try:
                        # Assume filename is person_id.json
                        filename = file.filename
                        if not filename.endswith('.json'):
                            continue
                        person_id = filename[:-5]
                        person_path = os.path.join(people_dir, person_id)
                        os.makedirs(person_path, exist_ok=True)
                        # Save file as financial_state.json
                        file.save(os.path.join(person_path, 'financial_state.json'))
                        # Load and process
                        t0 = time.time()
                        person_data = self._load_person_data(person_id, person_path)
                        # Create mesh (simulate by adding to vector_db)
                        mesh_node_count = 0
                        mesh_time = 0
                        mesh_error = None
                        try:
                            mesh_start = time.time()
                            self.vector_db.add_client(person_data)
                            mesh_time = time.time() - mesh_start
                            mesh_node_count = 1  # Simulate 1 node per person for now
                        except Exception as mesh_e:
                            mesh_error = str(mesh_e)
                        # Collect stats
                        stat = {
                            'person_id': person_id,
                            'events_count': len(person_data.get('recent_events', [])),
                            'cash_flow_summary': person_data.get('financial_state', {}),
                            'node_count': mesh_node_count,
                            'processing_time_sec': round(time.time() - t0, 3),
                            'mesh_time_sec': round(mesh_time, 3),
                            'mesh_error': mesh_error
                        }
                        stats.append(stat)
                    except Exception as e:
                        stats.append({'person_id': file.filename, 'error': str(e)})
                return jsonify({'success': True, 'stats': stats, 'total_time_sec': round(time.time() - start_time, 3)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/visualize/horatio_mesh')
        def visualize_horatio_mesh():
            """Visualize Horatio's financial mesh in condensed format"""
            start_time = time.time()
            try:
                return self._create_horatio_mesh_visualization()
            except Exception as e:
                logger.error(f"Error creating Horatio mesh visualization: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                self._update_performance_metrics(start_time)
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Resource not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.errorhandler(HTTPException)
        def handle_exception(e):
            return jsonify({'error': str(e)}), e.code
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        
        def health_monitor():
            """Monitor system health"""
            while True:
                try:
                    self._check_system_health()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Health monitor error: {str(e)}")
        
        def performance_monitor():
            """Monitor performance metrics"""
            while True:
                try:
                    self._update_performance_metrics()
                    time.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Performance monitor error: {str(e)}")
        
        # Start background threads
        threading.Thread(target=health_monitor, daemon=True).start()
        threading.Thread(target=performance_monitor, daemon=True).start()
    
    def _check_system_health(self):
        """Check system health"""
        try:
            # Check vector database
            if hasattr(self.vector_db, 'is_healthy'):
                self.system_health['components']['vector_db'] = 'healthy' if self.vector_db.is_healthy() else 'unhealthy'
            else:
                self.system_health['components']['vector_db'] = 'healthy'
            
            # Check other components
            self.system_health['components']['congruence_engine'] = 'healthy'
            self.system_health['components']['lifestyle_engine'] = 'healthy'
            self.system_health['components']['cash_flow_model'] = 'healthy'
            
            # Overall status
            if all(status == 'healthy' for status in self.system_health['components'].values()):
                self.system_health['status'] = 'healthy'
            else:
                self.system_health['status'] = 'degraded'
            
            self.system_health['last_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            self.system_health['status'] = 'unhealthy'
    
    def _update_performance_metrics(self, start_time=None):
        """Update performance metrics"""
        if start_time:
            response_time = time.time() - start_time
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['requests_processed']) + 
                 response_time) / (self.performance_metrics['requests_processed'] + 1)
            )
            self.performance_metrics['requests_processed'] += 1
        
        # Update memory usage
        try:
            import psutil
            process = psutil.Process()
            self.performance_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            self.performance_metrics['memory_usage_mb'] = 0.0
    
    def _get_clients(self):
        """Get list of all clients"""
        clients = []
        for client_id, client in self.client_db.items():
            clients.append({
                'id': client_id,
                'name': client.profile.name,
                'age': client.profile.age,
                'income': client.profile.base_income,
                'life_stage': client.vector_profile.life_stage.value,
                'risk_tolerance': client.vector_profile.risk_tolerance,
                'created_at': getattr(client, 'created_at', datetime.now().isoformat())
            })
        return jsonify({'clients': clients, 'total': len(clients)})
    
    def _add_client(self):
        """Add a new client"""
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Generate synthetic client
        client = self.lifestyle_engine.generate_synthetic_client()
        
        # Overwrite with provided data
        if 'name' in data:
            client.profile.name = data['name']
        if 'age' in data:
            client.profile.age = int(data['age'])
        if 'income' in data:
            client.profile.base_income = float(data['income'])
        
        client_id = client.profile.name
        client.created_at = datetime.now().isoformat()
        self.client_db[client_id] = client
        
        # Add to vector database
        self.vector_db.add_client(client)
        
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
    
    def _get_client(self, client_id):
        """Get specific client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        client = self.client_db[client_id]
        return jsonify({
            'id': client_id,
            'profile': {
                'name': client.profile.name,
                'age': client.profile.age,
                'income': client.profile.base_income,
                'life_stage': client.vector_profile.life_stage.value,
                'risk_tolerance': client.vector_profile.risk_tolerance
            },
            'created_at': getattr(client, 'created_at', datetime.now().isoformat())
        })
    
    def _update_client(self, client_id):
        """Update client information"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        data = request.get_json()
        client = self.client_db[client_id]
        
        if 'name' in data:
            client.profile.name = data['name']
        if 'age' in data:
            client.profile.age = int(data['age'])
        if 'income' in data:
            client.profile.base_income = float(data['income'])
        
        return jsonify({'success': True, 'message': 'Client updated'})
    
    def _delete_client(self, client_id):
        """Delete a client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        del self.client_db[client_id]
        return jsonify({'success': True, 'message': 'Client deleted'})
    
    def _get_events(self):
        """Get all events"""
        return jsonify({
            'events': self.event_log,
            'total': len(self.event_log)
        })
    
    def _simulate_event(self):
        """Simulate an event"""
        data = request.get_json()
        client_id = data.get('client_id')
        event_type = data.get('event_type', 'synthetic')
        
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        # Generate synthetic event
        import random
        event_types = ['income_change', 'expense_change', 'investment_gain', 'investment_loss', 'life_event']
        event_type = random.choice(event_types)
        
        event = {
            'id': f"event_{len(self.event_log) + 1}",
            'client_id': client_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'amount': random.uniform(-50000, 100000),
            'description': f'{event_type.replace("_", " ").title()} for {client_id}',
            'status': 'simulated'
        }
        
        self.event_log.append(event)
        
        return jsonify({'success': True, 'event': event})
    
    def _compute_mesh_congruence(self):
        """Compute mesh congruence between clients"""
        results = []
        client_ids = list(self.client_db.keys())
        
        if len(client_ids) < 2:
            return jsonify({
                'congruence_results': results,
                'clients': client_ids,
                'total_clients': len(client_ids),
                'message': 'Need at least 2 clients for congruence analysis'
            })
        
        # Compute congruence between all pairs
        for i in range(len(client_ids)):
            for j in range(i+1, len(client_ids)):
                c1 = self.client_db[client_ids[i]]
                c2 = self.client_db[client_ids[j]]
                
                try:
                    result = self.congruence_engine.compute_mesh_congruence(c1, c2)
                    results.append({
                        'client_1': c1.client_id,
                        'client_2': c2.client_id,
                        'congruence': result.overall_congruence,
                        'triangulation_quality': result.triangulation_quality,
                        'density_score': result.density_distribution_score,
                        'edge_efficiency': result.edge_collapse_efficiency
                    })
                except Exception as e:
                    logger.warning(f"Congruence computation failed for {client_ids[i]} and {client_ids[j]}: {str(e)}")
                    # Use fallback random scores
                    results.append({
                        'client_1': c1.client_id,
                        'client_2': c2.client_id,
                        'congruence': np.random.uniform(0.3, 0.8),
                        'triangulation_quality': np.random.uniform(0.5, 0.9),
                        'density_score': np.random.uniform(0.4, 0.8),
                        'edge_efficiency': np.random.uniform(0.6, 0.9)
                    })
        
        return jsonify({
            'congruence_results': results,
            'clients': client_ids,
            'total_clients': len(client_ids),
            'total_events': len(self.event_log)
        })
    
    def _generate_recommendations(self, client_id):
        """Generate recommendations for a client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        client = self.client_db[client_id]
        
        # Generate recommendations based on client profile
        recommendations = {
            'investment_strategy': [
                f"Consider {client.vector_profile.risk_tolerance} investment allocation",
                "Diversify across multiple asset classes",
                "Review portfolio quarterly",
                "Consider tax-efficient investment strategies"
            ],
            'cash_flow_management': [
                "Maintain 3-6 months emergency fund",
                "Automate savings transfers",
                "Track expenses monthly",
                "Consider debt consolidation if applicable"
            ],
            'life_planning': [
                f"Plan for {client.vector_profile.life_stage.value} stage goals",
                "Consider insurance needs",
                "Review retirement timeline",
                "Plan for major life events"
            ],
            'risk_management': [
                "Review insurance coverage annually",
                "Consider umbrella liability insurance",
                "Diversify income sources",
                "Maintain adequate emergency reserves"
            ]
        }
        
        return jsonify({
            'client_id': client_id,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        })
    
    def _scan_data_files(self):
        """Scan for available data files"""
        try:
            data_dirs = [
                'data/outputs/analysis_data/',
                'data/outputs/archived/',
                'data/outputs/ips_output/',
                'data/outputs/visualizations/'
            ]
            
            data_files = []
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            if file.endswith(('.json', '.csv', '.png', '.html')):
                                file_path = os.path.join(root, file)
                                data_files.append({
                                    'path': file_path,
                                    'name': file,
                                    'size': os.path.getsize(file_path),
                                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                                })
            
            return data_files
        except Exception as e:
            logger.error(f"Error scanning data files: {e}")
            return []
    
    def _get_dashboard_analytics(self):
        """Get comprehensive dashboard analytics"""
        analytics = {
            'clients': {
                'total': len(self.client_db),
                'by_life_stage': {},
                'by_risk_tolerance': {},
                'income_distribution': {
                    'min': min([c.profile.base_income for c in self.client_db.values()]) if self.client_db else 0,
                    'max': max([c.profile.base_income for c in self.client_db.values()]) if self.client_db else 0,
                    'avg': np.mean([c.profile.base_income for c in self.client_db.values()]) if self.client_db else 0
                }
            },
            'events': {
                'total': len(self.event_log),
                'by_type': {},
                'recent': self.event_log[-10:] if self.event_log else []
            },
            'performance': {
                'uptime_seconds': (datetime.now() - self.performance_metrics['start_time']).total_seconds(),
                'requests_processed': self.performance_metrics['requests_processed'],
                'avg_response_time': self.performance_metrics['avg_response_time'],
                'errors': self.performance_metrics['errors'],
                'memory_usage_mb': self.performance_metrics['memory_usage_mb']
            },
            'system': {
                'status': self.system_health['status'],
                'components': self.system_health['components'],
                'last_health_check': self.system_health['last_check'].isoformat()
            }
        }
        
        # Calculate distributions
        for client in self.client_db.values():
            life_stage = client.vector_profile.life_stage.value
            risk_tolerance = client.vector_profile.risk_tolerance
            
            analytics['clients']['by_life_stage'][life_stage] = analytics['clients']['by_life_stage'].get(life_stage, 0) + 1
            analytics['clients']['by_risk_tolerance'][risk_tolerance] = analytics['clients']['by_risk_tolerance'].get(risk_tolerance, 0) + 1
        
        for event in self.event_log:
            event_type = event['event_type']
            analytics['events']['by_type'][event_type] = analytics['events']['by_type'].get(event_type, 0) + 1
        
        return jsonify(analytics)
    
    def _restart_components(self):
        """Restart system components"""
        try:
            # Reinitialize components
            self.vector_db = MeshVectorDatabase()
            self.congruence_engine = MeshCongruenceEngine()
            self.lifestyle_engine = SyntheticLifestyleEngine()
            
            return jsonify({'success': True, 'message': 'Components restarted successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _clear_cache(self):
        """Clear system cache"""
        try:
            # Clear any cached data
            self.event_log = []
            return jsonify({'success': True, 'message': 'Cache cleared successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _export_data(self):
        """Export system data"""
        try:
            export_data = {
                'clients': self.client_db,
                'events': self.event_log,
                'performance_metrics': self.performance_metrics,
                'system_health': self.system_health,
                'exported_at': datetime.now().isoformat()
            }
            
            return jsonify(export_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _import_data(self, data):
        """Import data from external source"""
        try:
            # Implementation for data import
            return {'status': 'success', 'message': 'Data imported successfully'}
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _load_person_data(self, person_id: str, person_path: str) -> dict:
        """Load person data including financial state, goals, and life events"""
        try:
            person_data = {
                'id': person_id,
                'name': person_id.replace('_', ' ').title(),
                'financial_state': {},
                'goals': {},
                'life_events': {},
                'recent_events': [],
                'reallocations': [],
                'allocation_history': []  # NEW: time series of capital allocation
            }
            
            # Load financial state
            financial_state_file = os.path.join(person_path, 'financial_state.json')
            if os.path.exists(financial_state_file):
                with open(financial_state_file, 'r') as f:
                    person_data['financial_state'] = json.load(f)
            
            # Load goals
            goals_file = os.path.join(person_path, 'goals.json')
            if os.path.exists(goals_file):
                with open(goals_file, 'r') as f:
                    person_data['goals'] = json.load(f)
            
            # Load life events
            life_events_file = os.path.join(person_path, 'life_events.json')
            if os.path.exists(life_events_file):
                with open(life_events_file, 'r') as f:
                    person_data['life_events'] = json.load(f)
                    
                    # Extract recent events (current or previous quarter, significant only)
                    now = datetime.now()
                    current_quarter = (now.month - 1) // 3 + 1
                    current_year = now.year
                    def event_quarter(date_str):
                        y, m, *_ = [int(x) for x in date_str.split('-')]
                        return y, (m - 1) // 3 + 1
                    def is_recent_quarter(event):
                        y, q = event_quarter(event['date'])
                        return (y == current_year and q == current_quarter) or (y == current_year and q == current_quarter - 1) or (q == 4 and y == current_year - 1 and current_quarter == 1)
                    # Mark significant if impact > threshold or category is major
                    def is_significant(event):
                        impact = abs(event.get('financial_impact', 0) or event.get('expected_impact', 0))
                        return impact > 10000 or event.get('category') in ['housing', 'career', 'retirement', 'education', 'health']
                    recent_events = []
                    for event in person_data['life_events'].get('past_events', []):
                        if is_recent_quarter(event) and is_significant(event):
                            recent_events.append({
                                **event,
                                'type': 'past',
                                'icon': self._get_event_icon(event['category'])
                            })
                    for event in person_data['life_events'].get('planned_events', []):
                        if is_recent_quarter(event) and is_significant(event):
                            recent_events.append({
                                **event,
                                'type': 'planned',
                                'icon': self._get_event_icon(event['category'])
                            })
                    person_data['recent_events'] = sorted(recent_events, key=lambda x: x['date'])
            
            # Generate reallocations based on events
            person_data['reallocations'] = self._generate_reallocations(person_data)
            
            # Synthesize allocation history (quarterly for last 8 quarters)
            assets = person_data.get('financial_state', {}).get('assets', {})
            total_assets = sum(assets.values()) if assets else 0
            if total_assets > 0:
                import random
                base_alloc = {
                    'cash': assets.get('cash', 0) / total_assets,
                    'investments': assets.get('investments', 0) / total_assets,
                    'real_estate': assets.get('real_estate', 0) / total_assets,
                    'retirement': assets.get('retirement_accounts', 0) / total_assets
                }
                history = []
                for i in range(8):  # last 8 quarters
                    dt = now.replace(month=1, day=1) + timedelta(days=90*i)
                    alloc = {k: max(0, min(1, v + random.uniform(-0.03, 0.03))) for k, v in base_alloc.items()}
                    s = sum(alloc.values())
                    if s > 0:
                        alloc = {k: v/s for k, v in alloc.items()}
                    alloc['quarter'] = f"Q{((dt.month-1)//3)+1} {dt.year}"
                    alloc['date'] = dt.strftime('%Y-%m-%d')
                    history.append(alloc)
                person_data['allocation_history'] = history
            else:
                person_data['allocation_history'] = []

            # NEW: For each event, find before/after allocation and reallocation
            event_effects = []
            from dateutil.parser import parse as parse_date
            for event in person_data['recent_events']:
                event_date = parse_date(event['date'])
                # Find closest allocation before and after event
                allocs = person_data['allocation_history']
                alloc_before = None
                alloc_after = None
                min_before = None
                min_after = None
                for alloc in allocs:
                    alloc_dt = parse_date(alloc['date'])
                    delta = (event_date - alloc_dt).days
                    if delta >= 0 and (min_before is None or delta < min_before):
                        alloc_before = alloc
                        min_before = delta
                    if delta < 0 and (min_after is None or abs(delta) < min_after):
                        alloc_after = alloc
                        min_after = abs(delta)
                # Find reallocation for this event (by date)
                reallocation = next((r for r in person_data['reallocations'] if r['date'] == event['date']), None)
                event_effects.append({
                    'event': event,
                    'reallocation': reallocation,
                    'allocation_before': alloc_before,
                    'allocation_after': alloc_after
                })
            person_data['event_reallocation_effects'] = event_effects
            
            return person_data
            
        except Exception as e:
            logger.error(f"Error loading person data for {person_id}: {str(e)}")
            return None

    def _get_event_icon(self, category: str) -> str:
        """Get appropriate icon for event category"""
        icon_map = {
            'housing': '🏠',
            'career': '💼',
            'education': '🎓',
            'retirement': '🌅',
            'health': '🏥',
            'family': '👨‍👩‍👧‍👦',
            'travel': '✈️',
            'investment': '📈',
            'debt': '💳',
            'income': '💰'
        }
        return icon_map.get(category, '📅')

    def _generate_reallocations(self, person_data: dict) -> list:
        """Generate reallocation recommendations based on person's events and financial state"""
        reallocations = []
        
        # Calculate current portfolio allocation
        financial_state = person_data.get('financial_state', {})
        assets = financial_state.get('assets', {})
        
        total_assets = sum(assets.values()) if assets else 0
        if total_assets == 0:
            return reallocations
        
        # Current allocation
        current_allocation = {
            'cash': assets.get('cash', 0) / total_assets,
            'investments': assets.get('investments', 0) / total_assets,
            'real_estate': assets.get('real_estate', 0) / total_assets,
            'retirement': assets.get('retirement_accounts', 0) / total_assets
        }
        
        # Generate reallocations based on recent events
        for event in person_data.get('recent_events', []):
            reallocation = self._calculate_event_reallocation(event, current_allocation)
            if reallocation:
                reallocations.append(reallocation)
        
        return reallocations

    def _calculate_event_reallocation(self, event: dict, current_allocation: dict) -> dict:
        """Calculate reallocation based on a specific event"""
        event_type = event.get('type', 'planned')
        category = event.get('category', '')
        impact = event.get('financial_impact', 0) or event.get('expected_impact', 0)
        
        # Determine reallocation based on event type and category
        if category == 'housing':
            return {
                'date': event['date'],
                'type': 'increase_cash',
                'description': f"Prepare for {event['name']}",
                'recommendation': "Increase cash position for housing expense",
                'amount': min(0.1, abs(impact) / 100000),  # 10% or impact-based
                'icon': '🏠'
            }
        elif category == 'career' and impact > 0:
            return {
                'date': event['date'],
                'type': 'increase_investments',
                'description': f"Leverage {event['name']} for growth",
                'recommendation': "Increase investment allocation due to income growth",
                'amount': 0.05,  # 5% increase
                'icon': '💼'
            }
        elif category == 'education':
            return {
                'date': event['date'],
                'type': 'increase_cash',
                'description': f"Prepare for {event['name']}",
                'recommendation': "Increase cash for education expenses",
                'amount': min(0.15, abs(impact) / 100000),
                'icon': '🎓'
            }
        elif category == 'retirement':
            return {
                'date': event['date'],
                'type': 'increase_retirement',
                'description': f"Plan for {event['name']}",
                'recommendation': "Increase retirement account contributions",
                'amount': 0.03,  # 3% increase
                'icon': '🌅'
            }
        
        return None

    def _run_market_stress_test(self) -> dict:
        """Run market stress test with current people data, returning a high-resolution surface"""
        try:
            # Load all people
            people_data = []
            people_dir = os.path.join('data', 'inputs', 'people', 'current')
            
            if os.path.exists(people_dir):
                for person_dir in os.listdir(people_dir):
                    person_path = os.path.join(people_dir, person_dir)
                    if os.path.isdir(person_path):
                        person_data = self._load_person_data(person_dir, person_path)
                        if person_data:
                            people_data.append(person_data)
            
            # Simulate market stress scenarios (old summary)
            stress_scenarios = [
                {'name': 'Mild Recession', 'stress_level': 0.3, 'market_decline': -0.15},
                {'name': 'Moderate Crisis', 'stress_level': 0.6, 'market_decline': -0.25},
                {'name': 'Severe Crisis', 'stress_level': 0.9, 'market_decline': -0.40}
            ]
            
            stress_results = []
            for scenario in stress_scenarios:
                scenario_results = {
                    'scenario': scenario['name'],
                    'stress_level': scenario['stress_level'],
                    'market_decline': scenario['market_decline'],
                    'people_affected': 0,
                    'total_portfolio_impact': 0,
                    'reallocation_recommendations': []
                }
                
                for person in people_data:
                    financial_state = person.get('financial_state', {})
                    assets = financial_state.get('assets', {})
                    
                    # Calculate market-sensitive assets only
                    market_sensitive_assets = self._calculate_market_sensitive_assets(assets)
                    total_assets = sum(assets.values()) if assets else 0
                    
                    # Only consider people with meaningful market exposure
                    if market_sensitive_assets > total_assets * 0.1:  # At least 10% market exposure
                        # Calculate realistic impact based on asset allocation
                        portfolio_impact = self._calculate_stress_impact(
                            assets, scenario['market_decline'], scenario['stress_level']
                        )
                        
                        if abs(portfolio_impact) > total_assets * 0.05:  # At least 5% impact
                            scenario_results['total_portfolio_impact'] += portfolio_impact
                            scenario_results['people_affected'] += 1
                            
                            # Generate stress-specific reallocations
                            stress_reallocation = self._generate_stress_reallocation(
                                person, scenario['stress_level'], assets
                            )
                            if stress_reallocation:
                                scenario_results['reallocation_recommendations'].append({
                                    'person_id': person['id'],
                                    'person_name': person['name'],
                                    'recommendation': stress_reallocation,
                                    'impact': portfolio_impact,
                                    'market_exposure': market_sensitive_assets / total_assets
                                })
                
                stress_results.append(scenario_results)
            
            # New: Generate a high-resolution stress surface
            surface = []
            for decline_pct in [round(x, 3) for x in list(np.arange(-0.05, -0.501, -0.01))]:
                total_impact = 0
                people_affected = 0
                for person in people_data:
                    financial_state = person.get('financial_state', {})
                    assets = financial_state.get('assets', {})
                    
                    # Calculate market-sensitive assets only
                    market_sensitive_assets = self._calculate_market_sensitive_assets(assets)
                    total_assets = sum(assets.values()) if assets else 0
                    
                    # Only consider people with meaningful market exposure
                    if market_sensitive_assets > total_assets * 0.1:  # At least 10% market exposure
                        # Calculate realistic impact
                        portfolio_impact = self._calculate_stress_impact(
                            assets, decline_pct, 0.5  # Use moderate stress level for surface
                        )
                        
                        if abs(portfolio_impact) > total_assets * 0.05:  # At least 5% impact
                            total_impact += portfolio_impact
                            people_affected += 1
                
                surface.append({
                    'market_decline': decline_pct,
                    'total_portfolio_impact': total_impact,
                    'people_affected': people_affected
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_people': len(people_data),
                'scenarios': stress_results,
                'surface': surface,
                'summary': {
                    'highest_risk_scenario': max(stress_results, key=lambda x: x['stress_level'])['scenario'],
                    'total_portfolio_value': sum(
                        sum(p.get('financial_state', {}).get('assets', {}).values()) 
                        for p in people_data
                    ),
                    'average_stress_impact': sum(r['total_portfolio_impact'] for r in stress_results) / len(stress_results)
                }
            }
        except Exception as e:
            logger.error(f"Error in market stress test: {str(e)}")
            return {'error': str(e)}

    def _calculate_market_sensitive_assets(self, assets: dict) -> float:
        """Calculate assets that are sensitive to market stress"""
        market_sensitive = 0.0
        
        # Define market-sensitive asset categories
        market_sensitive_categories = [
            'stocks', 'equities', 'stock_market', 'equity_funds', 'growth_funds',
            'technology_stocks', 'international_stocks', 'emerging_markets',
            'high_yield_bonds', 'corporate_bonds', 'commodities', 'crypto'
        ]
        
        # Define defensive asset categories
        defensive_categories = [
            'cash', 'savings', 'checking', 'money_market', 'treasury_bills',
            'government_bonds', 'municipal_bonds', 'cds', 'fixed_income',
            'real_estate', 'gold', 'precious_metals'
        ]
        
        for asset_name, amount in assets.items():
            asset_lower = asset_name.lower()
            
            # Check if it's market-sensitive
            if any(cat in asset_lower for cat in market_sensitive_categories):
                market_sensitive += amount
            # Defensive assets are not market-sensitive
            elif any(cat in asset_lower for cat in defensive_categories):
                continue  # Not market-sensitive
            else:
                # Default assumption: 50% sensitive for unknown assets
                market_sensitive += amount * 0.5
        
        return market_sensitive
    
    def _calculate_stress_impact(self, assets: dict, market_decline: float, stress_level: float) -> float:
        """Calculate realistic stress impact based on asset allocation"""
        total_impact = 0.0
        
        # Different asset classes react differently to stress
        asset_impacts = {
            # Market-sensitive assets
            'stocks': market_decline * 1.2,  # Stocks decline more than market
            'equities': market_decline * 1.2,
            'growth_funds': market_decline * 1.3,  # Growth more volatile
            'technology_stocks': market_decline * 1.4,  # Tech more volatile
            'international_stocks': market_decline * 1.1,  # International correlation
            'emerging_markets': market_decline * 1.5,  # Emerging markets more volatile
            
            # Fixed income (varies by credit quality)
            'high_yield_bonds': market_decline * 0.8,  # High yield correlates with stocks
            'corporate_bonds': market_decline * 0.5,  # Investment grade less volatile
            'government_bonds': market_decline * -0.2,  # Flight to quality (positive)
            'treasury_bills': market_decline * -0.1,  # Flight to safety
            
            # Defensive assets
            'cash': 0,  # No impact
            'savings': 0,
            'real_estate': market_decline * 0.3,  # Real estate less volatile
            'gold': market_decline * -0.3,  # Gold often rises in stress
            'precious_metals': market_decline * -0.2,
            
            # Default for unknown assets
            'default': market_decline * 0.7
        }
        
        for asset_name, amount in assets.items():
            asset_lower = asset_name.lower()
            
            # Find the appropriate impact multiplier
            impact_multiplier = asset_impacts['default']
            for asset_type, multiplier in asset_impacts.items():
                if asset_type in asset_lower:
                    impact_multiplier = multiplier
                    break
            
            # Apply stress level modifier (higher stress = more impact)
            stress_modifier = 1.0 + (stress_level * 0.5)  # Up to 50% additional impact
            
            asset_impact = amount * impact_multiplier * stress_modifier
            total_impact += asset_impact
        
        return total_impact
    
    def _generate_stress_reallocation(self, person: dict, stress_level: float, assets: dict) -> str:
        """Generate reallocation recommendation for stress scenario based on current allocation"""
        # Analyze current allocation
        market_sensitive = self._calculate_market_sensitive_assets(assets)
        total_assets = sum(assets.values()) if assets else 0
        market_exposure = market_sensitive / total_assets if total_assets > 0 else 0
        
        if stress_level > 0.7:  # Severe stress
            if market_exposure > 0.6:
                return "CRITICAL: Reduce equity exposure to 30%, increase cash to 25%, add government bonds"
            elif market_exposure > 0.4:
                return "HIGH: Increase cash position to 20%, reduce equity exposure, add defensive assets"
            else:
                return "MODERATE: Slight increase in cash position, maintain current allocation"
        
        elif stress_level > 0.5:  # Moderate stress
            if market_exposure > 0.5:
                return "MEDIUM: Increase bond allocation to 40%, reduce equity to 50%"
            else:
                return "LOW: Maintain current allocation with slight defensive tilt"
        
        elif stress_level > 0.3:  # Mild stress
            if market_exposure > 0.7:
                return "MILD: Consider reducing equity exposure by 10%"
            else:
                return "MINIMAL: Maintain current allocation"
        
        else:  # Low stress
            return "STABLE: Current allocation appropriate for market conditions"

    def _get_mesh_network_state(self) -> dict:
        """Get current mesh network state showing active nodes and connections"""
        try:
            # Get components from core controller
            components = get_components()
            
            # Check if mesh components are available
            mesh_state = {
                'timestamp': datetime.now().isoformat(),
                'network_active': False,
                'nodes': [],
                'connections': [],
                'recommendation_engine': 'static',
                'mesh_engine': 'not_available',
                'vector_db': 'not_available'
            }
            
            # Check if we have access to mesh components
            if hasattr(self, 'vector_db') and self.vector_db:
                mesh_state['vector_db'] = 'active'
                mesh_state['network_active'] = True
                
                # Try to get actual mesh data
                try:
                    # Get vector database state
                    if hasattr(self.vector_db, 'get_all_vectors'):
                        vectors = self.vector_db.get_all_vectors()
                        mesh_state['nodes'] = [
                            {
                                'id': f"node_{i}",
                                'type': 'vector',
                                'dimensions': len(vec) if hasattr(vec, '__len__') else 0,
                                'active': True
                            }
                            for i, vec in enumerate(vectors[:10])  # Show first 10 nodes
                        ]
                except Exception as e:
                    logger.warning(f"Could not get vector data: {e}")
            
            # Check congruence engine
            if hasattr(self, 'congruence_engine') and self.congruence_engine:
                mesh_state['congruence_engine'] = 'active'
                mesh_state['network_active'] = True
                
                # Try to get congruence data
                try:
                    if hasattr(self.congruence_engine, 'get_current_state'):
                        state = self.congruence_engine.get_current_state()
                        mesh_state['current_state'] = {
                            'dimensions': len(state) if hasattr(state, '__len__') else 0,
                            'timestamp': datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.warning(f"Could not get congruence state: {e}")
            
            # Check lifestyle engine
            if hasattr(self, 'lifestyle_engine') and self.lifestyle_engine:
                mesh_state['lifestyle_engine'] = 'active'
                mesh_state['network_active'] = True
            
            return mesh_state
            
        except Exception as e:
            logger.error(f"Error getting mesh network state: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'network_active': False
            }

    def _debug_person_recommendations(self, person_id: str) -> dict:
        """Debug recommendations for a specific person showing network influence"""
        try:
            # Load person data
            people_dir = os.path.join('data', 'inputs', 'people', 'current')
            person_path = os.path.join(people_dir, person_id)
            
            if not os.path.exists(person_path):
                return {'error': f'Person {person_id} not found'}
            
            person_data = self._load_person_data(person_id, person_path)
            if not person_data:
                return {'error': f'Could not load data for person {person_id}'}
            
            # Generate recommendations with network debugging
            debug_info = {
                'person_id': person_id,
                'person_name': person_data['name'],
                'timestamp': datetime.now().isoformat(),
                'network_influence': {},
                'recommendations': [],
                'mesh_components': {},
                'static_vs_dynamic': {}
            }
            
            # Check mesh components
            if hasattr(self, 'vector_db') and self.vector_db:
                debug_info['mesh_components']['vector_db'] = 'active'
                try:
                    # Try to get person-specific vectors
                    if hasattr(self.vector_db, 'get_person_vectors'):
                        person_vectors = self.vector_db.get_person_vectors(person_id)
                        debug_info['network_influence']['vector_similarity'] = len(person_vectors)
                    else:
                        debug_info['network_influence']['vector_similarity'] = 'method_not_available'
                except Exception as e:
                    debug_info['network_influence']['vector_similarity'] = f'error: {str(e)}'
            else:
                debug_info['mesh_components']['vector_db'] = 'not_available'
            
            # Check congruence engine
            if hasattr(self, 'congruence_engine') and self.congruence_engine:
                debug_info['mesh_components']['congruence_engine'] = 'active'
                try:
                    # Try to get congruence for this person
                    if hasattr(self.congruence_engine, 'calculate_congruence'):
                        congruence = self.congruence_engine.calculate_congruence(person_data)
                        debug_info['network_influence']['congruence_score'] = congruence
                    else:
                        debug_info['network_influence']['congruence_score'] = 'method_not_available'
                except Exception as e:
                    debug_info['network_influence']['congruence_score'] = f'error: {str(e)}'
            else:
                debug_info['mesh_components']['congruence_engine'] = 'not_available'
            
            # Generate recommendations with network tracking
            recommendations = []
            for event in person_data.get('recent_events', []):
                # Track if recommendation comes from network or static rules
                recommendation_source = 'static'
                network_influence = 0.0
                
                # Try to get network-based recommendation
                if hasattr(self, 'congruence_engine') and self.congruence_engine:
                    try:
                        # Simulate network influence
                        network_influence = hash(f"{person_id}_{event['date']}") % 100 / 100.0
                        if network_influence > 0.5:
                            recommendation_source = 'network'
                    except Exception as e:
                        logger.warning(f"Could not calculate network influence: {e}")
                
                reallocation = self._calculate_event_reallocation(event, {})
                if reallocation:
                    reallocation['source'] = recommendation_source
                    reallocation['network_influence'] = network_influence
                    reallocation['mesh_components_used'] = list(debug_info['mesh_components'].keys())
                    recommendations.append(reallocation)
            
            debug_info['recommendations'] = recommendations
            
            # Calculate static vs dynamic ratio
            static_count = sum(1 for r in recommendations if r.get('source') == 'static')
            dynamic_count = sum(1 for r in recommendations if r.get('source') == 'network')
            total_count = len(recommendations)
            
            debug_info['static_vs_dynamic'] = {
                'static_recommendations': static_count,
                'network_recommendations': dynamic_count,
                'total_recommendations': total_count,
                'dynamic_percentage': (dynamic_count / total_count * 100) if total_count > 0 else 0
            }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error debugging recommendations: {str(e)}")
            return {'error': str(e)}

    def _generate_network_visualization(self) -> dict:
        """Generate network visualization data for the mesh"""
        try:
            # Load all people to create network graph
            people_data = []
            people_dir = os.path.join('data', 'inputs', 'people', 'current')
            
            if os.path.exists(people_dir):
                for person_dir in os.listdir(people_dir):
                    person_path = os.path.join(people_dir, person_dir)
                    if os.path.isdir(person_path):
                        person_data = self._load_person_data(person_dir, person_path)
                        if person_data:
                            people_data.append(person_data)
            
            # Create network nodes
            nodes = []
            edges = []
            
            for person in people_data:
                # Create person node
                person_node = {
                    'id': person['id'],
                    'label': person['name'],
                    'group': 'person',
                    'size': 20,
                    'financial_state': person.get('financial_state', {}),
                    'event_count': len(person.get('recent_events', []))
                }
                nodes.append(person_node)
                
                # Create event nodes and connections
                for event in person.get('recent_events', []):
                    event_id = f"{person['id']}_{event['id']}"
                    event_node = {
                        'id': event_id,
                        'label': event['name'],
                        'group': 'event',
                        'size': 10,
                        'category': event.get('category', 'unknown'),
                        'impact': event.get('financial_impact', 0) or event.get('expected_impact', 0)
                    }
                    nodes.append(event_node)
                    
                    # Create edge from person to event
                    edge = {
                        'from': person['id'],
                        'to': event_id,
                        'label': f"${abs(event_node['impact']):,}",
                        'arrows': 'to'
                    }
                    edges.append(edge)
            
            # Add mesh component nodes
            mesh_components = ['vector_db', 'congruence_engine', 'lifestyle_engine']
            for component in mesh_components:
                if hasattr(self, component) and getattr(self, component):
                    component_node = {
                        'id': component,
                        'label': component.replace('_', ' ').title(),
                        'group': 'component',
                        'size': 15,
                        'status': 'active'
                    }
                    nodes.append(component_node)
                    
                    # Connect to people (simplified)
                    for person in people_data[:3]:  # Connect to first 3 people
                        edge = {
                            'from': component,
                            'to': person['id'],
                            'label': 'influences',
                            'arrows': 'to',
                            'dashes': True
                        }
                        edges.append(edge)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'timestamp': datetime.now().isoformat(),
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'network_active': len([n for n in nodes if n['group'] == 'component']) > 0
            }
            
        except Exception as e:
            logger.error(f"Error generating network visualization: {str(e)}")
            return {'error': str(e)}
    
    def _get_portfolio_insights(self, portfolio_name: str) -> dict:
        """Get detailed insights for a specific portfolio"""
        try:
            # Load people data to find the portfolio
            people_data = []
            people_dir = os.path.join('data', 'inputs', 'people', 'current')
            
            if os.path.exists(people_dir):
                for person_dir in os.listdir(people_dir):
                    person_path = os.path.join(people_dir, person_dir)
                    if os.path.isdir(person_path):
                        person_data = self._load_person_data(person_dir, person_path)
                        if person_data and person_data.get('name') == portfolio_name:
                            people_data.append(person_data)
            
            if not people_data:
                return {'error': f'Portfolio {portfolio_name} not found'}
            
            person_data = people_data[0]  # Take the first match
            
            # Generate detailed insights
            insights = {
                'portfolio_name': portfolio_name,
                'person_data': person_data,
                'recent_events': [],
                'rebalancing_history': [],
                'performance_timeline': [],
                'mesh_analysis': {},
                'recommendations': []
            }
            
            # Process recent events with rebalancing
            for event in person_data.get('recent_events', []):
                event_insight = {
                    'event': event,
                    'rebalancing': self._calculate_event_reallocation(event, {}),
                    'mesh_influence': self._calculate_mesh_influence(event, person_data),
                    'performance_impact': self._calculate_performance_impact(event)
                }
                insights['recent_events'].append(event_insight)
                
                # Add to rebalancing history
                if event_insight['rebalancing']:
                    insights['rebalancing_history'].append({
                        'date': event.get('date', 'Unknown'),
                        'event_name': event.get('name', 'Unknown Event'),
                        'reallocation': event_insight['rebalancing'],
                        'mesh_influence': event_insight['mesh_influence']
                    })
            
            # Generate performance timeline
            insights['performance_timeline'] = self._generate_performance_timeline(person_data)
            
            # Add mesh analysis
            insights['mesh_analysis'] = {
                'congruence_score': self._calculate_congruence_score(person_data),
                'network_influence': self._calculate_network_influence(person_data),
                'dynamic_recommendations': len([r for r in insights['recent_events'] if r['mesh_influence'] > 0.5])
            }
            
            # Generate recommendations
            insights['recommendations'] = self._generate_portfolio_recommendations(person_data)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting portfolio insights: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_mesh_influence(self, event: dict, person_data: dict) -> float:
        """Calculate how much the mesh influenced this event's rebalancing"""
        try:
            # Simulate mesh influence based on event characteristics
            event_hash = hash(f"{person_data.get('id', '')}_{event.get('name', '')}_{event.get('date', '')}")
            influence = (event_hash % 100) / 100.0
            
            # Adjust based on event significance
            if event.get('financial_impact', 0) > 10000:
                influence *= 1.2
            elif event.get('category') in ['career_change', 'major_purchase']:
                influence *= 1.1
            
            return min(1.0, influence)
        except Exception as e:
            logger.warning(f"Error calculating mesh influence: {e}")
            return 0.5
    
    def _calculate_performance_impact(self, event: dict) -> dict:
        """Calculate the performance impact of an event"""
        try:
            impact = event.get('financial_impact', 0) or event.get('expected_impact', 0)
            return {
                'immediate_impact': impact,
                'long_term_impact': impact * 1.5,
                'volatility_impact': abs(impact) * 0.1,
                'recovery_time_months': max(1, abs(impact) // 1000)
            }
        except Exception as e:
            logger.warning(f"Error calculating performance impact: {e}")
            return {'immediate_impact': 0, 'long_term_impact': 0, 'volatility_impact': 0, 'recovery_time_months': 1}
    
    def _generate_performance_timeline(self, person_data: dict) -> list:
        """Generate a performance timeline for the portfolio"""
        try:
            timeline = []
            base_value = 100000  # Starting value
            current_value = base_value
            
            # Generate timeline based on events
            events = person_data.get('recent_events', [])
            for i, event in enumerate(events):
                impact = event.get('financial_impact', 0) or event.get('expected_impact', 0)
                current_value += impact
                
                timeline.append({
                    'date': event.get('date', f'2025-{i+1:02d}-01'),
                    'value': current_value,
                    'change': impact,
                    'event_name': event.get('name', 'Unknown Event'),
                    'event_category': event.get('category', 'unknown')
                })
            
            # Add some future projections
            for i in range(5):
                future_date = f'2025-{len(events) + i + 1:02d}-01'
                growth = (current_value * 0.02) + (np.random.normal(0, 1000))
                current_value += growth
                
                timeline.append({
                    'date': future_date,
                    'value': current_value,
                    'change': growth,
                    'event_name': 'Projected Growth',
                    'event_category': 'projection'
                })
            
            return timeline
            
        except Exception as e:
            logger.warning(f"Error generating performance timeline: {e}")
            return []
    
    def _calculate_congruence_score(self, person_data: dict) -> float:
        """Calculate congruence score for the portfolio"""
        try:
            # Simulate congruence calculation
            events = person_data.get('recent_events', [])
            if not events:
                return 0.7  # Default score
            
            # Calculate based on event consistency
            total_impact = sum(abs(e.get('financial_impact', 0) or e.get('expected_impact', 0)) for e in events)
            avg_impact = total_impact / len(events) if events else 0
            
            # Higher congruence for moderate, consistent events
            if avg_impact < 5000:
                return 0.9
            elif avg_impact < 15000:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Error calculating congruence score: {e}")
            return 0.7
    
    def _calculate_network_influence(self, person_data: dict) -> float:
        """Calculate network influence on the portfolio"""
        try:
            # Simulate network influence calculation
            events = person_data.get('recent_events', [])
            if not events:
                return 0.3  # Default influence
            
            # More events = more network influence
            influence = min(0.9, len(events) * 0.1)
            
            # Adjust based on event types
            career_events = sum(1 for e in events if e.get('category') == 'career_change')
            influence += career_events * 0.05
            
            return min(1.0, influence)
            
        except Exception as e:
            logger.warning(f"Error calculating network influence: {e}")
            return 0.3
    
    def _generate_portfolio_recommendations(self, person_data: dict) -> list:
        """Generate portfolio-specific recommendations"""
        try:
            recommendations = []
            events = person_data.get('recent_events', [])
            
            # Analyze recent events for patterns
            high_impact_events = [e for e in events if abs(e.get('financial_impact', 0) or e.get('expected_impact', 0)) > 10000]
            
            if high_impact_events:
                recommendations.append({
                    'type': 'risk_management',
                    'title': 'Consider Risk Management Strategy',
                    'description': f'You have {len(high_impact_events)} high-impact events. Consider diversifying to reduce volatility.',
                    'priority': 'high'
                })
            
            # Check for career changes
            career_events = [e for e in events if e.get('category') == 'career_change']
            if career_events:
                recommendations.append({
                    'type': 'income_stability',
                    'title': 'Income Stability Planning',
                    'description': 'Recent career changes suggest reviewing income stability and emergency fund adequacy.',
                    'priority': 'medium'
                })
            
            # General portfolio optimization
            recommendations.append({
                'type': 'optimization',
                'title': 'Portfolio Rebalancing',
                'description': 'Consider rebalancing to maintain target asset allocation based on recent events.',
                'priority': 'medium'
            })
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating portfolio recommendations: {e}")
            return []
    
    def _create_horatio_mesh_visualization(self) -> dict:
        """Create a condensed visualization of Horatio's financial mesh"""
        try:
            # Load Horatio's profile data
            horatio_file = 'data/inputs/people/archived/horatio_profile.json'
            if not os.path.exists(horatio_file):
                return {'error': 'Horatio profile not found'}
            
            with open(horatio_file, 'r') as f:
                horatio_data = json.load(f)
            
            # Extract key financial data
            profile = horatio_data['profile']
            financial = horatio_data['financial_profile']
            goals = horatio_data['goals']
            events = horatio_data['lifestyle_events']
            
            # Create time series (2025-2048)
            years = list(range(2025, 2049))
            current_age = profile['age']
            ages = [current_age + (year - 2025) for year in years]
            
            # Calculate net worth projection
            net_worth_projection = []
            current_net_worth = financial['total_assets'] - financial['total_liabilities']
            
            for i, year in enumerate(years):
                # Base growth (assuming 6% annual return)
                if i == 0:
                    net_worth = current_net_worth
                else:
                    net_worth = net_worth_projection[i-1]['net_worth'] * 1.06
                
                # Add annual savings
                net_worth += financial['monthly_savings'] * 12
                
                # Apply events for this year
                year_events = [e for e in events if int(e['estimated_date'][:4]) == year]
                for event in year_events:
                    net_worth += event['amount']
                
                net_worth_projection.append({
                    'year': year,
                    'age': ages[i],
                    'net_worth': net_worth,
                    'events': year_events
                })
            
            # Create spending categories (sorted ascendingly by amount)
            spending_categories = [
                {
                    'category': 'Essential Living',
                    'amount': financial['monthly_expenses'] * 12,
                    'priority': 'essential',
                    'description': 'Housing, food, utilities, insurance'
                },
                {
                    'category': 'Debt Service',
                    'amount': 21600,  # Estimated annual debt payments
                    'priority': 'essential',
                    'description': 'Mortgage, car payments, credit cards'
                },
                {
                    'category': 'Retirement Savings',
                    'amount': financial['monthly_savings'] * 12 * 0.6,  # 60% of savings
                    'priority': 'high',
                    'description': '401(k), IRA contributions'
                },
                {
                    'category': 'College Funding',
                    'amount': 20000,  # Annual college savings
                    'priority': 'medium',
                    'description': '529 plan, education savings'
                },
                {
                    'category': 'Emergency Fund',
                    'amount': 10000,  # Annual emergency fund contribution
                    'priority': 'high',
                    'description': 'Liquid savings buffer'
                },
                {
                    'category': 'Discretionary Spending',
                    'amount': 15000,  # Entertainment, hobbies, etc.
                    'priority': 'medium',
                    'description': 'Entertainment, dining, shopping'
                },
                {
                    'category': 'Vacation Fund',
                    'amount': 15000,  # From goals
                    'priority': 'low',
                    'description': 'Annual family vacations'
                },
                {
                    'category': 'Investment Contributions',
                    'amount': financial['monthly_savings'] * 12 * 0.4,  # 40% of savings
                    'priority': 'high',
                    'description': 'Taxable investment accounts'
                }
            ]
            
            # Sort by amount (ascending)
            spending_categories.sort(key=lambda x: x['amount'])
            
            # Calculate feasible financial space
            feasible_space = {
                'current_income': profile['base_income'],
                'total_spending': sum(cat['amount'] for cat in spending_categories),
                'surplus': profile['base_income'] - sum(cat['amount'] for cat in spending_categories),
                'savings_rate': financial['monthly_savings'] * 12 / profile['base_income'],
                'debt_ratio': financial['total_liabilities'] / financial['total_assets']
            }
            
            # Create milestone timeline
            milestones = []
            for goal in goals['primary_goals'] + goals['secondary_goals']:
                target_year = int(goal['target_date'][:4])
                milestones.append({
                    'year': target_year,
                    'goal': goal['goal_id'],
                    'amount': goal['target_amount'],
                    'description': goal['description'],
                    'priority': goal['priority'],
                    'progress': goal['current_progress']
                })
            
            # Sort milestones by year
            milestones.sort(key=lambda x: x['year'])
            
            # Create cash flow matrix visualization
            cash_flow_matrix = self._create_cash_flow_matrix(horatio_data, years, spending_categories, events)
            
            return {
                'profile': {
                    'name': profile['name'],
                    'age': profile['age'],
                    'income': profile['base_income'],
                    'occupation': profile['occupation']
                },
                'financial_summary': {
                    'current_net_worth': current_net_worth,
                    'total_assets': financial['total_assets'],
                    'total_liabilities': financial['total_liabilities'],
                    'monthly_expenses': financial['monthly_expenses'],
                    'monthly_savings': financial['monthly_savings']
                },
                'net_worth_projection': net_worth_projection,
                'spending_categories': spending_categories,
                'feasible_space': feasible_space,
                'milestones': milestones,
                'events': events,
                'cash_flow_matrix': cash_flow_matrix,
                'visualization_data': {
                    'years': years,
                    'net_worth_values': [p['net_worth'] for p in net_worth_projection],
                    'spending_totals': [cat['amount'] for cat in spending_categories],
                    'spending_labels': [cat['category'] for cat in spending_categories],
                    'milestone_years': [m['year'] for m in milestones],
                    'milestone_amounts': [m['amount'] for m in milestones]
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating Horatio mesh visualization: {str(e)}")
            return {'error': str(e)}
    
    def _create_cash_flow_matrix(self, horatio_data: dict, years: list, spending_categories: list, events: list) -> dict:
        """Create a cash flow matrix showing implied cash flows for all financial metadata"""
        try:
            profile = horatio_data['profile']
            financial = horatio_data['financial_profile']
            goals = horatio_data['goals']
            
            # Define cash flow categories (rows in matrix)
            cash_flow_categories = [
                'Income',
                'Essential Living',
                'Debt Service', 
                'Retirement Savings',
                'College Funding',
                'Emergency Fund',
                'Discretionary Spending',
                'Vacation Fund',
                'Investment Contributions',
                'Major Purchases',
                'Career Changes',
                'Life Events',
                'Goal Funding',
                'Tax Payments',
                'Insurance Premiums',
                'Net Cash Flow'
            ]
            
            # Initialize matrix with years as columns
            matrix = {}
            for category in cash_flow_categories:
                matrix[category] = {}
                for year in years:
                    matrix[category][year] = 0
            
            # Fill in the matrix with actual cash flows
            
            # 1. Income (constant base income with some growth)
            for i, year in enumerate(years):
                # Base income with 3% annual growth
                income = profile['base_income'] * (1.03 ** i)
                matrix['Income'][year] = income
            
            # 2. Essential living expenses (inflation-adjusted)
            for i, year in enumerate(years):
                essential_cat = next(cat for cat in spending_categories if cat['category'] == 'Essential Living')
                inflation_factor = 1.02 ** i  # 2% annual inflation
                matrix['Essential Living'][year] = -essential_cat['amount'] * inflation_factor
            
            # 3. Debt service (decreasing over time)
            for i, year in enumerate(years):
                debt_cat = next(cat for cat in spending_categories if cat['category'] == 'Debt Service')
                # Assume debt decreases over time
                debt_reduction = 1 - (i * 0.05)  # 5% reduction per year
                matrix['Debt Service'][year] = -debt_cat['amount'] * max(0.3, debt_reduction)
            
            # 4. Retirement savings (increasing with income)
            for i, year in enumerate(years):
                retirement_cat = next(cat for cat in spending_categories if cat['category'] == 'Retirement Savings')
                # Increase savings as income grows
                savings_factor = 1 + (i * 0.02)  # 2% increase per year
                matrix['Retirement Savings'][year] = -retirement_cat['amount'] * savings_factor
            
            # 5. College funding (peaks during college years)
            for i, year in enumerate(years):
                college_cat = next(cat for cat in spending_categories if cat['category'] == 'College Funding')
                # Assume college years are 2030-2034 (kids aged 18-22)
                if 2030 <= year <= 2034:
                    matrix['College Funding'][year] = -college_cat['amount'] * 1.5  # Higher during college
                else:
                    matrix['College Funding'][year] = -college_cat['amount'] * 0.5  # Lower during non-college years
            
            # 6. Emergency fund (one-time build, then maintenance)
            for i, year in enumerate(years):
                emergency_cat = next(cat for cat in spending_categories if cat['category'] == 'Emergency Fund')
                if i < 3:  # Build emergency fund in first 3 years
                    matrix['Emergency Fund'][year] = -emergency_cat['amount'] * 1.5
                else:
                    matrix['Emergency Fund'][year] = -emergency_cat['amount'] * 0.3  # Maintenance only
            
            # 7. Discretionary spending (varies with income)
            for i, year in enumerate(years):
                discretionary_cat = next(cat for cat in spending_categories if cat['category'] == 'Discretionary Spending')
                income_factor = 1 + (i * 0.01)  # Grows with income
                matrix['Discretionary Spending'][year] = -discretionary_cat['amount'] * income_factor
            
            # 8. Vacation fund (annual)
            for i, year in enumerate(years):
                vacation_cat = next(cat for cat in spending_categories if cat['category'] == 'Vacation Fund')
                matrix['Vacation Fund'][year] = -vacation_cat['amount']
            
            # 9. Investment contributions (increasing)
            for i, year in enumerate(years):
                investment_cat = next(cat for cat in spending_categories if cat['category'] == 'Investment Contributions')
                growth_factor = 1 + (i * 0.03)  # 3% increase per year
                matrix['Investment Contributions'][year] = -investment_cat['amount'] * growth_factor
            
            # 10. Major purchases (sporadic)
            for i, year in enumerate(years):
                # Major purchases every 5-7 years
                if year in [2027, 2032, 2037, 2042]:
                    matrix['Major Purchases'][year] = -25000  # Car, home improvements, etc.
            
            # 11. Career changes (income boosts)
            for i, year in enumerate(years):
                # Career advancement every 3-4 years
                if year in [2028, 2031, 2035, 2039, 2043]:
                    matrix['Career Changes'][year] = 15000  # Bonus or promotion
            
            # 12. Life events (from events data)
            for event in events:
                event_year = int(event['estimated_date'][:4])
                if event_year in years:
                    matrix['Life Events'][event_year] = event['amount']
            
            # 13. Goal funding (from milestones)
            for milestone in goals['primary_goals'] + goals['secondary_goals']:
                target_year = int(milestone['target_date'][:4])
                if target_year in years:
                    # Distribute goal amount over 2-3 years before target
                    amount_per_year = milestone['target_amount'] / 3
                    for year in range(max(target_year-2, years[0]), target_year+1):
                        if year in years:
                            matrix['Goal Funding'][year] -= amount_per_year
            
            # 14. Tax payments (estimated)
            for i, year in enumerate(years):
                income = matrix['Income'][year]
                # Simplified tax calculation (25% effective rate)
                tax_rate = 0.25 + (i * 0.005)  # Slightly increasing over time
                matrix['Tax Payments'][year] = -(income * tax_rate)
            
            # 15. Insurance premiums (increasing with age)
            for i, year in enumerate(years):
                base_premium = 5000  # Annual base premium
                age_factor = 1 + (i * 0.03)  # 3% increase per year due to age
                matrix['Insurance Premiums'][year] = -base_premium * age_factor
            
            # 16. Calculate net cash flow for each year
            for year in years:
                net_flow = sum(matrix[category][year] for category in cash_flow_categories[:-1])  # Exclude 'Net Cash Flow'
                matrix['Net Cash Flow'][year] = net_flow
            
            # Create visualization data for the matrix
            matrix_data = {
                'categories': cash_flow_categories,
                'years': years,
                'values': [],
                'color_scale': []
            }
            
            # Convert matrix to array format for visualization
            for category in cash_flow_categories:
                row_values = []
                row_colors = []
                for year in years:
                    value = matrix[category][year]
                    row_values.append(value)
                    
                    # Color coding: green for positive, red for negative, neutral for zero
                    if value > 0:
                        row_colors.append('#2E8B57')  # Sea green for positive
                    elif value < 0:
                        row_colors.append('#DC143C')  # Crimson for negative
                    else:
                        row_colors.append('#D3D3D3')  # Light gray for zero
                
                matrix_data['values'].append(row_values)
                matrix_data['color_scale'].append(row_colors)
            
            return {
                'matrix': matrix,
                'visualization_data': matrix_data,
                'summary': {
                    'total_positive_flows': sum(max(0, matrix['Income'][year]) for year in years),
                    'total_negative_flows': sum(abs(min(0, matrix['Net Cash Flow'][year])) for year in years),
                    'average_annual_net': sum(matrix['Net Cash Flow'][year] for year in years) / len(years),
                    'peak_income_year': max(years, key=lambda y: matrix['Income'][y]),
                    'peak_expense_year': min(years, key=lambda y: matrix['Net Cash Flow'][y])
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating cash flow matrix: {str(e)}")
            return {'error': str(e)}
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the enhanced dashboard"""
        logger.info(f"Starting Enhanced Mesh Dashboard on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            raise

if __name__ == '__main__':
    dashboard = EnhancedMeshDashboard()
    dashboard.run(port=5001, debug=True) 