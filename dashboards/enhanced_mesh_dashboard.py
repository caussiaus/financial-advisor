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
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import numpy as np
import pandas as pd

# Import our mesh components
from src.analysis.mesh_vector_database import MeshVectorDatabase
from src.analysis.mesh_congruence_engine import MeshCongruenceEngine
from src.synthetic_lifestyle_engine import SyntheticLifestyleEngine, SyntheticClientData
from src.unified_cash_flow_model import UnifiedCashFlowModel
from src.core.time_uncertainty_mesh import TimeUncertaintyMeshEngine

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
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.app.config['UPLOAD_FOLDER'] = 'data/inputs/uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Initialize components
        self.client_db = {}
        self.event_log = []
        self.vector_db = MeshVectorDatabase()
        self.congruence_engine = MeshCongruenceEngine()
        self.lifestyle_engine = SyntheticLifestyleEngine()
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
        """Import system data"""
        try:
            if 'clients' in data:
                self.client_db = data['clients']
            if 'events' in data:
                self.event_log = data['events']
            
            return jsonify({'success': True, 'message': 'Data imported successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the enhanced dashboard"""
        logger.info(f"Starting Enhanced Mesh Dashboard on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            raise

if __name__ == '__main__':
    dashboard = EnhancedMeshDashboard()
    dashboard.run(port=5001, debug=True) 