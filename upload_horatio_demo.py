#!/usr/bin/env python3
"""
Upload Horatio Profile Demo
Demonstrates uploading a comprehensive client profile to the enhanced mesh dashboard
"""

import json
import requests
import time
from datetime import datetime

def upload_horatio_profile():
    """Upload Horatio's comprehensive profile to the dashboard"""
    
    # Load Horatio's profile
    with open('horatio_profile.json', 'r') as f:
        horatio_data = json.load(f)
    
    print("🎯 Horatio Profile Upload Demo")
    print("=" * 60)
    
    # Dashboard URL
    base_url = "http://localhost:5001"
    
    try:
        # 1. Check system health
        print("1. Checking system health...")
        health_response = requests.get(f"{base_url}/api/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ System Status: {health_data['status']}")
            print(f"   ✅ Uptime: {health_data['uptime_seconds']:.1f} seconds")
        else:
            print(f"   ❌ Health check failed: {health_response.status_code}")
            return
        
        # 2. Upload Horatio's profile
        print("\n2. Uploading Horatio's comprehensive profile...")
        
        # Convert to the format expected by the dashboard
        upload_data = {
            'name': horatio_data['profile']['name'],
            'age': horatio_data['profile']['age'],
            'income': horatio_data['profile']['base_income'],
            'comprehensive_profile': horatio_data  # Include full profile
        }
        
        upload_response = requests.post(f"{base_url}/api/clients", json=upload_data)
        
        if upload_response.status_code == 200:
            result = upload_response.json()
            print(f"   ✅ Profile uploaded successfully!")
            print(f"   ✅ Client ID: {result['client_id']}")
            print(f"   ✅ Name: {result['profile']['name']}")
            print(f"   ✅ Age: {result['profile']['age']}")
            print(f"   ✅ Income: ${result['profile']['income']:,}")
            print(f"   ✅ Life Stage: {result['profile']['life_stage']}")
            print(f"   ✅ Risk Tolerance: {result['profile']['risk_tolerance']:.2f}")
        else:
            print(f"   ❌ Upload failed: {upload_response.status_code}")
            print(f"   Error: {upload_response.text}")
            return
        
        # 3. Get updated client list
        print("\n3. Retrieving updated client list...")
        clients_response = requests.get(f"{base_url}/api/clients")
        if clients_response.status_code == 200:
            clients_data = clients_response.json()
            print(f"   ✅ Total clients: {clients_data['total']}")
            for client in clients_data['clients']:
                print(f"   📋 {client['name']} (Age: {client['age']}, Income: ${client['income']:,})")
        
        # 4. Simulate some events for Horatio
        print("\n4. Simulating life events for Horatio...")
        
        events = [
            {"event_type": "income_change", "description": "Annual salary increase"},
            {"event_type": "investment_gain", "description": "Strong market performance"},
            {"event_type": "life_event", "description": "Child's college application"}
        ]
        
        for i, event in enumerate(events, 1):
            event_data = {
                'client_id': 'horatio',
                'event_type': event['event_type']
            }
            
            event_response = requests.post(f"{base_url}/api/events", json=event_data)
            if event_response.status_code == 200:
                event_result = event_response.json()
                print(f"   ✅ Event {i}: {event['description']}")
                print(f"      Amount: ${event_result['event']['amount']:,.2f}")
            else:
                print(f"   ❌ Event {i} failed: {event_response.status_code}")
        
        # 5. Get recommendations for Horatio
        print("\n5. Generating personalized recommendations...")
        recommendations_response = requests.get(f"{base_url}/api/recommendations/horatio")
        
        if recommendations_response.status_code == 200:
            rec_data = recommendations_response.json()
            print("   📋 Investment Strategy:")
            for rec in rec_data['recommendations']['investment_strategy']:
                print(f"      • {rec}")
            
            print("   📋 Cash Flow Management:")
            for rec in rec_data['recommendations']['cash_flow_management']:
                print(f"      • {rec}")
            
            print("   📋 Life Planning:")
            for rec in rec_data['recommendations']['life_planning']:
                print(f"      • {rec}")
            
            print("   📋 Risk Management:")
            for rec in rec_data['recommendations']['risk_management']:
                print(f"      • {rec}")
        else:
            print(f"   ❌ Recommendations failed: {recommendations_response.status_code}")
        
        # 6. Get mesh congruence analysis
        print("\n6. Computing mesh congruence analysis...")
        congruence_response = requests.get(f"{base_url}/api/mesh/congruence")
        
        if congruence_response.status_code == 200:
            congruence_data = congruence_response.json()
            print(f"   ✅ Total clients: {congruence_data['total_clients']}")
            print(f"   ✅ Total events: {congruence_data['total_events']}")
            
            if congruence_data['congruence_results']:
                avg_congruence = sum(r['congruence'] for r in congruence_data['congruence_results']) / len(congruence_data['congruence_results'])
                print(f"   📊 Average congruence: {avg_congruence:.1%}")
                
                # Show top congruence pairs
                sorted_results = sorted(congruence_data['congruence_results'], key=lambda x: x['congruence'], reverse=True)
                print("   🏆 Top congruence pairs:")
                for i, result in enumerate(sorted_results[:3], 1):
                    print(f"      {i}. {result['client_1']} ↔ {result['client_2']}: {result['congruence']:.1%}")
        
        # 7. Get comprehensive analytics
        print("\n7. Retrieving comprehensive analytics...")
        analytics_response = requests.get(f"{base_url}/api/analytics/dashboard")
        
        if analytics_response.status_code == 200:
            analytics_data = analytics_response.json()
            print(f"   📈 Client Distribution:")
            print(f"      • Total clients: {analytics_data['clients']['total']}")
            print(f"      • Average income: ${analytics_data['clients']['income_distribution']['avg']:,.0f}")
            
            print(f"   📊 Event Analysis:")
            print(f"      • Total events: {analytics_data['events']['total']}")
            for event_type, count in analytics_data['events']['by_type'].items():
                print(f"      • {event_type}: {count}")
            
            print(f"   ⚡ Performance Metrics:")
            print(f"      • Requests processed: {analytics_data['performance']['requests_processed']}")
            print(f"      • Average response time: {analytics_data['performance']['avg_response_time']:.3f}s")
            print(f"      • Memory usage: {analytics_data['performance']['memory_usage_mb']:.1f}MB")
        
        print("\n" + "=" * 60)
        print("🎉 Horatio Profile Upload Demo Completed Successfully!")
        print("=" * 60)
        print("🌐 Access the enhanced dashboard at: http://localhost:5001")
        print("📊 View real-time analytics and mesh congruence analysis")
        print("🎛️ Use the system controls for advanced monitoring")
        print("📋 Horatio's comprehensive profile is now integrated into the mesh system")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to dashboard. Make sure it's running on port 5001")
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")

if __name__ == "__main__":
    upload_horatio_profile() 