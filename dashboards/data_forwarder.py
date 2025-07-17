"""
Simple Data Forwarding Service

This service takes existing working components and forwards their data to the frontend.
No complex imports, just data forwarding.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_forwarder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataForwarder:
    """Simple data forwarding service for the frontend"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.app.config['UPLOAD_FOLDER'] = 'data/inputs/uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Initialize data storage
        self.client_db = {}
        self.event_log = []
        self.system_status = {
            'status': 'healthy',
            'last_check': datetime.now(),
            'components': {
                'data_forwarder': 'healthy',
                'mesh_engine': 'available',
                'vector_db': 'available',
                'recommendation_engine': 'available'
            }
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'requests_processed': 0,
            'avg_response_time': 0.0,
            'errors': 0,
            'start_time': datetime.now(),
            'active_connections': 0
        }
        
        # Initialize routes
        self._setup_routes()
        self._setup_error_handlers()
        
        logger.info("✅ Data Forwarder initialized successfully")
    
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
                'status': self.system_status['status'],
                'uptime_seconds': (datetime.now() - self.performance_metrics['start_time']).total_seconds(),
                'components': self.system_status['components'],
                'performance': {
                    'requests_processed': self.performance_metrics['requests_processed'],
                    'avg_response_time': self.performance_metrics['avg_response_time'],
                    'errors': self.performance_metrics['errors'],
                    'active_connections': self.performance_metrics['active_connections']
                },
                'service_type': 'data_forwarder'
            })
        
        @self.app.route('/api/system/status')
        def system_status():
            """Get detailed system status"""
            return jsonify({
                'service_type': 'data_forwarder',
                'status': 'operational',
                'components_available': {
                    'mesh_engine': True,
                    'vector_db': True,
                    'recommendation_engine': True,
                    'pdf_processor': True,
                    'accounting_engine': True
                },
                'system_health': self.system_status,
                'performance_metrics': self.performance_metrics,
                'data_sources': [
                    'data/outputs/analysis_data/',
                    'data/outputs/archived/',
                    'data/outputs/ips_output/',
                    'data/outputs/visualizations/'
                ]
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
        
        @self.app.route('/api/mesh/status')
        def get_mesh_status():
            """Get mesh engine status from data files"""
            try:
                # Read mesh status from data files
                mesh_data = self._load_mesh_data()
                return jsonify({
                    'mesh_status': mesh_data,
                    'total_nodes': mesh_data.get('total_nodes', 0),
                    'visible_future_nodes': mesh_data.get('visible_future_nodes', 0),
                    'mesh_efficiency': mesh_data.get('efficiency', 0.0),
                    'data_source': 'file_system'
                })
            except Exception as e:
                logger.error(f"Error getting mesh status: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
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
        
        @self.app.route('/api/data/files')
        def get_data_files():
            """Get list of available data files"""
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
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.errorhandler(HTTPException)
        def handle_exception(e):
            return jsonify({'error': str(e)}), e.code
    
    def _update_performance_metrics(self, start_time=None):
        """Update performance metrics"""
        if start_time:
            response_time = time.time() - start_time
            self.performance_metrics['requests_processed'] += 1
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['requests_processed'] - 1) + response_time) /
                self.performance_metrics['requests_processed']
            )
    
    def _get_clients(self):
        """Get all clients"""
        return jsonify({
            'clients': list(self.client_db.keys()),
            'total_clients': len(self.client_db)
        })
    
    def _add_client(self):
        """Add a new client"""
        try:
            data = request.get_json()
            client_id = data.get('client_id', f'client_{len(self.client_db) + 1}')
            
            # Create sample client data
            client_data = {
                'id': client_id,
                'name': data.get('name', f'Client {client_id}'),
                'age': data.get('age', 35),
                'income': data.get('income', 80000),
                'risk_tolerance': data.get('risk_tolerance', 'moderate'),
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.client_db[client_id] = client_data
            
            return jsonify({
                'message': f'Client {client_id} added successfully',
                'client_id': client_id,
                'client_data': client_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    def _get_client(self, client_id):
        """Get a specific client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        return jsonify({
            'client': self.client_db[client_id]
        })
    
    def _update_client(self, client_id):
        """Update a client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        try:
            data = request.get_json()
            self.client_db[client_id].update(data)
            self.client_db[client_id]['updated_at'] = datetime.now().isoformat()
            
            return jsonify({
                'message': f'Client {client_id} updated successfully',
                'client_data': self.client_db[client_id]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    def _delete_client(self, client_id):
        """Delete a client"""
        if client_id not in self.client_db:
            return jsonify({'error': 'Client not found'}), 404
        
        del self.client_db[client_id]
        
        return jsonify({
            'message': f'Client {client_id} deleted successfully'
        })
    
    def _get_events(self):
        """Get all events"""
        return jsonify({
            'events': self.event_log,
            'total_events': len(self.event_log)
        })
    
    def _simulate_event(self):
        """Simulate a new event"""
        try:
            data = request.get_json()
            event_id = f'event_{len(self.event_log) + 1}'
            
            event = {
                'id': event_id,
                'type': data.get('type', 'general'),
                'description': data.get('description', 'Simulated event'),
                'timestamp': datetime.now().isoformat(),
                'data': data.get('data', {})
            }
            
            self.event_log.append(event)
            
            return jsonify({
                'message': f'Event {event_id} created successfully',
                'event': event
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    def _generate_recommendations(self, client_id):
        """Generate recommendations for a client"""
        try:
            if client_id not in self.client_db:
                return jsonify({'error': 'Client not found'}), 404
            
            client_data = self.client_db[client_id]
            
            # Generate sample recommendations based on client data
            recommendations = []
            for month in range(1, 13):
                recommendation = {
                    'month': month,
                    'year': datetime.now().year,
                    'type': 'investment',
                    'description': f'Monthly investment recommendation for {client_data["name"]}',
                    'suggested_amount': client_data.get('income', 80000) * 0.1,
                    'priority': 'medium',
                    'rationale': 'Based on income and risk tolerance',
                    'expected_outcome': 'Portfolio growth',
                    'risk_level': client_data.get('risk_tolerance', 'moderate')
                }
                recommendations.append(recommendation)
            
            return jsonify({
                'client_id': client_id,
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _get_dashboard_analytics(self):
        """Get comprehensive dashboard analytics"""
        try:
            # Load analytics from data files
            analytics_data = self._load_analytics_data()
            
            return jsonify({
                'analytics': analytics_data,
                'summary': {
                    'total_clients': len(self.client_db),
                    'total_events': len(self.event_log),
                    'system_status': self.system_status['status'],
                    'uptime_seconds': (datetime.now() - self.performance_metrics['start_time']).total_seconds()
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def _load_mesh_data(self):
        """Load mesh data from files"""
        try:
            # Look for mesh data files
            mesh_files = [
                'data/outputs/analysis_data/simple_analysis_results_20250717_072502.json',
                'data/outputs/archived/analysis_data/comprehensive_simulation_results.json'
            ]
            
            for file_path in mesh_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        return {
                            'total_nodes': data.get('total_nodes', 1000),
                            'visible_future_nodes': data.get('visible_nodes', 800),
                            'efficiency': data.get('efficiency', 0.8),
                            'data_source': file_path
                        }
            
            # Return default data if no files found
            return {
                'total_nodes': 1000,
                'visible_future_nodes': 800,
                'efficiency': 0.8,
                'data_source': 'default'
            }
        except Exception as e:
            logger.error(f"Error loading mesh data: {e}")
            return {
                'total_nodes': 0,
                'visible_future_nodes': 0,
                'efficiency': 0.0,
                'data_source': 'error'
            }
    
    def _load_analytics_data(self):
        """Load analytics data from files"""
        try:
            analytics_files = [
                'data/outputs/analysis_data/simple_analysis_results_20250717_072502.json',
                'data/outputs/archived/analysis_data/comprehensive_simulation_results.json'
            ]
            
            analytics_data = {}
            for file_path in analytics_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        analytics_data[os.path.basename(file_path)] = data
            
            return analytics_data
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")
            return {}
    
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
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the data forwarder"""
        logger.info(f"Starting Data Forwarder on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
        except Exception as e:
            logger.error(f"Failed to start data forwarder: {str(e)}")
            raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Forwarder Service')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    
    args = parser.parse_args()
    
    try:
        forwarder = DataForwarder()
        forwarder.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start data forwarder: {e}")
        sys.exit(1) 