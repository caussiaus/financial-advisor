#!/usr/bin/env python3
"""
Stress Test Script for Mesh Congruence App
Tests all endpoints and measures performance under load
"""

import requests
import time
import threading
import json
import random
from datetime import datetime
import concurrent.futures

# Configuration
BASE_URL = "http://localhost:5001"
TEST_DURATION = 60  # seconds
CONCURRENT_USERS = 5
REQUESTS_PER_USER = 20

class StressTester:
    def __init__(self):
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        self.lock = threading.Lock()
    
    def add_client(self, user_id):
        """Add a new client"""
        try:
            data = {
                'name': f'StressTestUser_{user_id}_{random.randint(1000, 9999)}',
                'age': random.randint(25, 65),
                'base_income': random.randint(30000, 200000)
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/add_client", json=data)
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                    return response.json().get('client_id')
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"Add client failed: {response.text}")
                    return None
                    
        except Exception as e:
            with self.lock:
                self.results['failed_requests'] += 1
                self.results['errors'].append(f"Add client exception: {str(e)}")
            return None
    
    def simulate_event(self, client_id):
        """Simulate an event for a client"""
        try:
            data = {
                'client_id': client_id,
                'event_type': random.choice(['income_change', 'expense_change', 'investment_gain', 'investment_loss'])
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/api/simulate_event", json=data)
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"Simulate event failed: {response.text}")
                    
        except Exception as e:
            with self.lock:
                self.results['failed_requests'] += 1
                self.results['errors'].append(f"Simulate event exception: {str(e)}")
    
    def get_recommendations(self, client_id):
        """Get recommendations for a client"""
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/api/recommendations?client_id={client_id}")
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"Get recommendations failed: {response.text}")
                    
        except Exception as e:
            with self.lock:
                self.results['failed_requests'] += 1
                self.results['errors'].append(f"Get recommendations exception: {str(e)}")
    
    def get_mesh_dashboard(self):
        """Get mesh dashboard data"""
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/api/mesh_dashboard")
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"Get mesh dashboard failed: {response.text}")
                    
        except Exception as e:
            with self.lock:
                self.results['failed_requests'] += 1
                self.results['errors'].append(f"Get mesh dashboard exception: {str(e)}")
    
    def get_performance(self):
        """Get performance metrics"""
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/api/performance")
            response_time = time.time() - start_time
            
            with self.lock:
                self.results['total_requests'] += 1
                self.results['response_times'].append(response_time)
                
                if response.status_code == 200:
                    self.results['successful_requests'] += 1
                    return response.json()
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"Get performance failed: {response.text}")
                    return None
                    
        except Exception as e:
            with self.lock:
                self.results['failed_requests'] += 1
                self.results['errors'].append(f"Get performance exception: {str(e)}")
            return None
    
    def run_stress_test(self, user_id):
        """Run stress test for a single user"""
        print(f"🧪 Starting stress test for user {user_id}")
        
        # Add some clients
        client_ids = []
        for i in range(3):
            client_id = self.add_client(user_id)
            if client_id:
                client_ids.append(client_id)
            time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate events
        for i in range(5):
            if client_ids:
                client_id = random.choice(client_ids)
                self.simulate_event(client_id)
            time.sleep(random.uniform(0.1, 0.3))
        
        # Get recommendations
        for client_id in client_ids:
            self.get_recommendations(client_id)
            time.sleep(random.uniform(0.1, 0.2))
        
        # Get mesh dashboard
        for i in range(2):
            self.get_mesh_dashboard()
            time.sleep(random.uniform(0.2, 0.5))
        
        print(f"✅ Completed stress test for user {user_id}")

def run_concurrent_stress_test():
    """Run stress test with multiple concurrent users"""
    print("🚀 Starting Mesh Congruence App Stress Test")
    print(f"⏱️  Duration: {TEST_DURATION} seconds")
    print(f"👥 Concurrent users: {CONCURRENT_USERS}")
    print(f"📊 Requests per user: {REQUESTS_PER_USER}")
    print("=" * 60)
    
    tester = StressTester()
    start_time = time.time()
    
    # Run stress test with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        # Submit stress test tasks
        futures = []
        for user_id in range(CONCURRENT_USERS):
            future = executor.submit(tester.run_stress_test, user_id)
            futures.append(future)
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Calculate statistics
    response_times = tester.results['response_times']
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0
    
    success_rate = (tester.results['successful_requests'] / tester.results['total_requests']) * 100 if tester.results['total_requests'] > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 STRESS TEST RESULTS")
    print("=" * 60)
    print(f"⏱️  Test duration: {test_duration:.2f} seconds")
    print(f"📊 Total requests: {tester.results['total_requests']}")
    print(f"✅ Successful requests: {tester.results['successful_requests']}")
    print(f"❌ Failed requests: {tester.results['failed_requests']}")
    print(f"📈 Success rate: {success_rate:.2f}%")
    print(f"⏱️  Average response time: {avg_response_time:.3f} seconds")
    print(f"⚡ Min response time: {min_response_time:.3f} seconds")
    print(f"🐌 Max response time: {max_response_time:.3f} seconds")
    print(f"📊 Requests per second: {tester.results['total_requests'] / test_duration:.2f}")
    
    if tester.results['errors']:
        print(f"\n❌ Errors encountered: {len(tester.results['errors'])}")
        for error in tester.results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Get server performance metrics
    server_performance = tester.get_performance()
    if server_performance:
        print(f"\n🖥️  SERVER PERFORMANCE")
        print(f"  - Uptime: {server_performance['uptime_seconds']:.2f} seconds")
        print(f"  - Server requests processed: {server_performance['requests_processed']}")
        print(f"  - Server avg response time: {server_performance['avg_response_time']:.3f} seconds")
        print(f"  - Server errors: {server_performance['errors']}")
        print(f"  - Server success rate: {server_performance['success_rate']:.2f}%")
    
    print("\n🎉 Stress test completed!")

def run_continuous_stress_test():
    """Run continuous stress test for specified duration"""
    print("🔄 Starting Continuous Stress Test")
    print(f"⏱️  Duration: {TEST_DURATION} seconds")
    
    tester = StressTester()
    start_time = time.time()
    
    while time.time() - start_time < TEST_DURATION:
        # Randomly choose an action
        action = random.choice(['add_client', 'simulate_event', 'get_recommendations', 'get_dashboard'])
        
        if action == 'add_client':
            tester.add_client(random.randint(1, 1000))
        elif action == 'simulate_event':
            # Get list of clients first
            try:
                response = requests.get(f"{BASE_URL}/api/clients")
                if response.status_code == 200:
                    clients = response.json().get('clients', [])
                    if clients:
                        client_id = random.choice(clients)['id']
                        tester.simulate_event(client_id)
            except:
                pass
        elif action == 'get_recommendations':
            try:
                response = requests.get(f"{BASE_URL}/api/clients")
                if response.status_code == 200:
                    clients = response.json().get('clients', [])
                    if clients:
                        client_id = random.choice(clients)['id']
                        tester.get_recommendations(client_id)
            except:
                pass
        elif action == 'get_dashboard':
            tester.get_mesh_dashboard()
        
        time.sleep(random.uniform(0.1, 0.5))
    
    # Print results
    end_time = time.time()
    test_duration = end_time - start_time
    
    response_times = tester.results['response_times']
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    success_rate = (tester.results['successful_requests'] / tester.results['total_requests']) * 100 if tester.results['total_requests'] > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 CONTINUOUS STRESS TEST RESULTS")
    print("=" * 60)
    print(f"⏱️  Test duration: {test_duration:.2f} seconds")
    print(f"📊 Total requests: {tester.results['total_requests']}")
    print(f"✅ Successful requests: {tester.results['successful_requests']}")
    print(f"❌ Failed requests: {tester.results['failed_requests']}")
    print(f"📈 Success rate: {success_rate:.2f}%")
    print(f"⏱️  Average response time: {avg_response_time:.3f} seconds")
    print(f"📊 Requests per second: {tester.results['total_requests'] / test_duration:.2f}")

if __name__ == "__main__":
    print("🧪 Mesh Congruence App Stress Tester")
    print("=" * 60)
    
    # Check if app is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ App is running and accessible")
        else:
            print("❌ App is not responding properly")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to app. Make sure it's running on http://localhost:5000")
        exit(1)
    
    # Run stress tests
    print("\n1️⃣ Running concurrent stress test...")
    run_concurrent_stress_test()
    
    print("\n2️⃣ Running continuous stress test...")
    run_continuous_stress_test()
    
    print("\n🎉 All stress tests completed!") 