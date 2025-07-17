#!/usr/bin/env python3
"""
System Control Script for Enhanced Mesh Dashboard
Provides comprehensive control and monitoring capabilities
"""

import os
import sys
import json
import time
import requests
import psutil
import subprocess
from datetime import datetime, timedelta
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemController:
    """Comprehensive system controller for the enhanced dashboard"""
    
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10
    
    def check_health(self):
        """Check system health"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'healthy',
                    'data': data,
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}",
                    'response_time': response.elapsed.total_seconds()
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unreachable',
                'error': str(e),
                'response_time': None
            }
    
    def get_analytics(self):
        """Get dashboard analytics"""
        try:
            response = self.session.get(f"{self.base_url}/api/analytics/dashboard")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def add_client(self, name, age, income):
        """Add a new client"""
        try:
            data = {
                'name': name,
                'age': int(age),
                'income': float(income)
            }
            response = self.session.post(f"{self.base_url}/api/clients", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def get_clients(self):
        """Get all clients"""
        try:
            response = self.session.get(f"{self.base_url}/api/clients")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def simulate_event(self, client_id, event_type="synthetic"):
        """Simulate an event"""
        try:
            data = {
                'client_id': client_id,
                'event_type': event_type
            }
            response = self.session.post(f"{self.base_url}/api/events", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def get_mesh_congruence(self):
        """Get mesh congruence analysis"""
        try:
            response = self.session.get(f"{self.base_url}/api/mesh/congruence")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def get_recommendations(self, client_id):
        """Get recommendations for a client"""
        try:
            response = self.session.get(f"{self.base_url}/api/recommendations/{client_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def restart_components(self):
        """Restart system components"""
        try:
            data = {'action': 'restart_components'}
            response = self.session.post(f"{self.base_url}/api/system/control", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear system cache"""
        try:
            data = {'action': 'clear_cache'}
            response = self.session.post(f"{self.base_url}/api/system/control", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def export_data(self):
        """Export system data"""
        try:
            data = {'action': 'export_data'}
            response = self.session.post(f"{self.base_url}/api/system/control", json=data)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    
    def get_system_info(self):
        """Get comprehensive system information"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'health': self.check_health(),
            'analytics': self.get_analytics(),
            'system_resources': self.get_system_resources()
        }
        return info
    
    def get_system_resources(self):
        """Get system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def monitor_system(self, interval=30, duration=None):
        """Monitor system continuously"""
        start_time = datetime.now()
        print(f"🔍 Starting system monitoring (interval: {interval}s)")
        
        try:
            while True:
                if duration and (datetime.now() - start_time).total_seconds() > duration:
                    print("⏰ Monitoring duration reached")
                    break
                
                info = self.get_system_info()
                self.print_system_status(info)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def print_system_status(self, info):
        """Print formatted system status"""
        print(f"\n{'='*60}")
        print(f"📊 System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Health status
        health = info['health']
        status_icon = "✅" if health['status'] == 'healthy' else "❌" if health['status'] == 'unreachable' else "⚠️"
        print(f"{status_icon} Health: {health['status'].upper()}")
        if health['status'] == 'healthy':
            print(f"   Uptime: {health['data']['uptime_seconds']:.1f}s")
            print(f"   Requests: {health['data']['performance']['requests_processed']}")
            print(f"   Avg Response: {health['data']['performance']['avg_response_time']:.3f}s")
            print(f"   Errors: {health['data']['performance']['errors']}")
        
        # Analytics
        analytics = info['analytics']
        if 'error' not in analytics:
            print(f"📈 Analytics:")
            print(f"   Clients: {analytics['clients']['total']}")
            print(f"   Events: {analytics['events']['total']}")
            print(f"   Memory: {analytics['performance']['memory_usage_mb']:.1f}MB")
        
        # System resources
        resources = info['system_resources']
        if 'error' not in resources:
            print(f"💻 System Resources:")
            print(f"   CPU: {resources['cpu_percent']:.1f}%")
            print(f"   Memory: {resources['memory_percent']:.1f}% ({resources['memory_available_gb']:.1f}GB free)")
            print(f"   Disk: {resources['disk_percent']:.1f}% ({resources['disk_free_gb']:.1f}GB free)")
        
        print(f"{'='*60}")
    
    def stress_test(self, num_clients=10, num_events=50):
        """Run stress test"""
        print(f"🧪 Starting stress test: {num_clients} clients, {num_events} events")
        
        # Add clients
        clients = []
        for i in range(num_clients):
            name = f"stress_client_{i+1}"
            age = 25 + (i % 50)
            income = 50000 + (i * 10000)
            
            result = self.add_client(name, age, income)
            if 'error' not in result:
                clients.append(result['client_id'])
                print(f"✅ Added client: {name}")
            else:
                print(f"❌ Failed to add client {name}: {result['error']}")
        
        # Simulate events
        events_simulated = 0
        for i in range(num_events):
            if clients:
                client_id = clients[i % len(clients)]
                result = self.simulate_event(client_id, "stress_test")
                if 'error' not in result:
                    events_simulated += 1
                    if i % 10 == 0:
                        print(f"✅ Simulated event {i+1}/{num_events}")
                else:
                    print(f"❌ Failed to simulate event {i+1}: {result['error']}")
        
        print(f"🎯 Stress test completed: {len(clients)} clients, {events_simulated} events")
        
        # Get final analytics
        analytics = self.get_analytics()
        if 'error' not in analytics:
            print(f"📊 Final stats: {analytics['clients']['total']} clients, {analytics['events']['total']} events")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Mesh Dashboard System Controller')
    parser.add_argument('--url', default='http://localhost:5001', help='Dashboard URL')
    parser.add_argument('--action', choices=[
        'health', 'analytics', 'add-client', 'get-clients', 'simulate-event',
        'congruence', 'recommendations', 'restart', 'clear-cache', 'export',
        'monitor', 'stress-test', 'info'
    ], help='Action to perform')
    
    # Client management
    parser.add_argument('--name', help='Client name')
    parser.add_argument('--age', type=int, help='Client age')
    parser.add_argument('--income', type=float, help='Client income')
    parser.add_argument('--client-id', help='Client ID for operations')
    
    # Event simulation
    parser.add_argument('--event-type', default='synthetic', help='Event type')
    
    # Monitoring
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval (seconds)')
    parser.add_argument('--duration', type=int, help='Monitoring duration (seconds)')
    
    # Stress testing
    parser.add_argument('--num-clients', type=int, default=10, help='Number of clients for stress test')
    parser.add_argument('--num-events', type=int, default=50, help='Number of events for stress test')
    
    args = parser.parse_args()
    
    controller = SystemController(args.url)
    
    if args.action == 'health':
        health = controller.check_health()
        print(json.dumps(health, indent=2))
    
    elif args.action == 'analytics':
        analytics = controller.get_analytics()
        print(json.dumps(analytics, indent=2))
    
    elif args.action == 'add-client':
        if not all([args.name, args.age, args.income]):
            print("❌ --name, --age, and --income are required for add-client")
            return
        result = controller.add_client(args.name, args.age, args.income)
        print(json.dumps(result, indent=2))
    
    elif args.action == 'get-clients':
        clients = controller.get_clients()
        print(json.dumps(clients, indent=2))
    
    elif args.action == 'simulate-event':
        if not args.client_id:
            print("❌ --client-id is required for simulate-event")
            return
        result = controller.simulate_event(args.client_id, args.event_type)
        print(json.dumps(result, indent=2))
    
    elif args.action == 'congruence':
        congruence = controller.get_mesh_congruence()
        print(json.dumps(congruence, indent=2))
    
    elif args.action == 'recommendations':
        if not args.client_id:
            print("❌ --client-id is required for recommendations")
            return
        recommendations = controller.get_recommendations(args.client_id)
        print(json.dumps(recommendations, indent=2))
    
    elif args.action == 'restart':
        result = controller.restart_components()
        print(json.dumps(result, indent=2))
    
    elif args.action == 'clear-cache':
        result = controller.clear_cache()
        print(json.dumps(result, indent=2))
    
    elif args.action == 'export':
        data = controller.export_data()
        filename = f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Data exported to {filename}")
    
    elif args.action == 'monitor':
        controller.monitor_system(args.interval, args.duration)
    
    elif args.action == 'stress-test':
        controller.stress_test(args.num_clients, args.num_events)
    
    elif args.action == 'info':
        info = controller.get_system_info()
        print(json.dumps(info, indent=2))
    
    else:
        print("❌ No action specified. Use --help for options.")

if __name__ == '__main__':
    main() 