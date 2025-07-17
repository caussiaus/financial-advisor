#!/usr/bin/env python3
"""
Enhanced Dashboard Startup Script
Manages port conflicts, proper initialization, and system controls
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import socket
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages the enhanced dashboard startup and monitoring"""
    
    def __init__(self, port=5001):
        self.port = port
        self.process = None
        self.start_time = None
        
    def find_free_port(self, start_port=5001, max_attempts=10):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_attempts}")
    
    def kill_process_on_port(self, port):
        """Kill any process using the specified port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.info['connections']
                    for conn in connections:
                        if conn.laddr.port == port:
                            logger.info(f"Killing process {proc.info['pid']} using port {port}")
                            proc.terminate()
                            proc.wait(timeout=5)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
        except Exception as e:
            logger.warning(f"Error killing process on port {port}: {e}")
        return False
    
    def check_port_availability(self, port):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def start_dashboard(self, debug=False):
        """Start the enhanced dashboard"""
        try:
            # Find free port
            self.port = self.find_free_port(self.port)
            logger.info(f"Using port {self.port}")
            
            # Kill any existing process on this port
            if not self.check_port_availability(self.port):
                logger.info(f"Port {self.port} is in use, attempting to free it...")
                self.kill_process_on_port(self.port)
                time.sleep(2)  # Wait for process to terminate
            
            # Start the dashboard
            cmd = [sys.executable, 'enhanced_mesh_dashboard.py']
            env = os.environ.copy()
            env['FLASK_ENV'] = 'development' if debug else 'production'
            env['FLASK_DEBUG'] = '1' if debug else '0'
            
            logger.info(f"Starting enhanced dashboard on port {self.port}")
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.start_time = datetime.now()
            logger.info(f"Dashboard started with PID {self.process.pid}")
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is None:
                logger.info("Dashboard started successfully!")
                return True
            else:
                stdout, stderr = self.process.communicate()
                logger.error(f"Dashboard failed to start. stdout: {stdout}, stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return False
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        if self.process:
            try:
                logger.info("Stopping dashboard...")
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("Dashboard stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning("Dashboard didn't stop gracefully, forcing termination")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping dashboard: {e}")
    
    def get_status(self):
        """Get dashboard status"""
        if not self.process:
            return {'status': 'stopped', 'port': self.port}
        
        is_running = self.process.poll() is None
        
        status = {
            'status': 'running' if is_running else 'stopped',
            'port': self.port,
            'pid': self.process.pid if is_running else None
        }
        
        if is_running and self.start_time:
            uptime = datetime.now() - self.start_time
            status['uptime_seconds'] = uptime.total_seconds()
        
        return status
    
    def restart_dashboard(self, debug=False):
        """Restart the dashboard"""
        logger.info("Restarting dashboard...")
        self.stop_dashboard()
        time.sleep(2)
        return self.start_dashboard(debug)
    
    def monitor_dashboard(self, check_interval=30):
        """Monitor dashboard health"""
        logger.info(f"Starting dashboard monitoring (check every {check_interval} seconds)")
        
        while True:
            try:
                status = self.get_status()
                
                if status['status'] == 'stopped':
                    logger.warning("Dashboard stopped unexpectedly, restarting...")
                    self.start_dashboard()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(check_interval)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Mesh Dashboard Manager')
    parser.add_argument('--port', type=int, default=5001, help='Port to use (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring mode')
    parser.add_argument('--stop', action='store_true', help='Stop running dashboard')
    parser.add_argument('--status', action='store_true', help='Show dashboard status')
    
    args = parser.parse_args()
    
    manager = DashboardManager(args.port)
    
    if args.stop:
        manager.stop_dashboard()
        print("Dashboard stopped")
        return
    
    if args.status:
        status = manager.get_status()
        print(f"Status: {status['status']}")
        print(f"Port: {status['port']}")
        if status['pid']:
            print(f"PID: {status['pid']}")
        if 'uptime_seconds' in status:
            print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        return
    
    # Start dashboard
    if manager.start_dashboard(args.debug):
        print(f"✅ Enhanced Mesh Dashboard started successfully!")
        print(f"🌐 Access at: http://localhost:{manager.port}")
        print(f"📊 Health check: http://localhost:{manager.port}/api/health")
        print(f"🔧 System controls available in the dashboard")
        
        if args.monitor:
            try:
                manager.monitor_dashboard()
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped")
        else:
            try:
                # Keep the process running
                while True:
                    time.sleep(1)
                    if manager.process and manager.process.poll() is not None:
                        print("❌ Dashboard process stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n🛑 Stopping dashboard...")
                manager.stop_dashboard()
    else:
        print("❌ Failed to start dashboard")
        sys.exit(1)

if __name__ == '__main__':
    main() 