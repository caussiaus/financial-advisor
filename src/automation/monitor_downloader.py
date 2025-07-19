"""
Enhanced Market Data Downloader Monitor

Monitor the progress of the enhanced market data downloader
and provide real-time status updates.
"""

import json
import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloaderMonitor:
    """Monitor the enhanced market data downloader"""
    
    def __init__(self):
        self.data_dir = Path("data/enhanced_market_data")
        self.log_dir = Path("logs")
        self.pid_file = self.log_dir / "enhanced_downloader.pid"
    
    def get_process_status(self) -> Dict:
        """Get the status of the downloader process"""
        status = {
            "running": False,
            "pid": None,
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_mb": 0.0,
            "start_time": None,
            "uptime": None
        }
        
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    content = f.read().strip()
                    # Handle both "PID: 12345" and "12345" formats
                    if content.startswith("PID: "):
                        pid = int(content.split(": ")[1])
                    else:
                        pid = int(content)
                
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    status.update({
                        "running": True,
                        "pid": pid,
                        "cpu_percent": process.cpu_percent(),
                        "memory_percent": process.memory_percent(),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                        "start_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                        "uptime": time.time() - process.create_time()
                    })
            except Exception as e:
                logger.error(f"Error getting process status: {e}")
        
        return status
    
    def get_download_progress(self) -> Dict:
        """Get the progress of data downloads"""
        progress = {
            "total_files": 0,
            "completed_files": 0,
            "file_sizes": {},
            "last_updated": None,
            "download_summary": None
        }
        
        if self.data_dir.exists():
            files = list(self.data_dir.glob("*.json"))
            progress["total_files"] = len(files)
            
            total_size = 0
            for file in files:
                if file.name != "enhanced_market_data_summary.json":
                    progress["completed_files"] += 1
                    size = file.stat().st_size
                    progress["file_sizes"][file.name] = size
                    total_size += size
            
            progress["total_size_mb"] = total_size / 1024 / 1024
            
            # Check for summary file
            summary_file = self.data_dir / "enhanced_market_data_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    progress["download_summary"] = summary
                    progress["last_updated"] = datetime.fromtimestamp(summary_file.stat().st_mtime).isoformat()
                except Exception as e:
                    logger.error(f"Error reading summary file: {e}")
        
        return progress
    
    def get_log_status(self) -> Dict:
        """Get the status of log files"""
        log_status = {
            "log_files": [],
            "latest_log": None,
            "error_logs": []
        }
        
        if self.log_dir.exists():
            log_files = list(self.log_dir.glob("enhanced_downloader_*.log"))
            error_files = list(self.log_dir.glob("enhanced_downloader_*_error.log"))
            
            for log_file in log_files:
                log_status["log_files"].append({
                    "name": log_file.name,
                    "size_mb": log_file.stat().st_size / 1024 / 1024,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
            
            for error_file in error_files:
                log_status["error_logs"].append({
                    "name": error_file.name,
                    "size_mb": error_file.stat().st_size / 1024 / 1024,
                    "modified": datetime.fromtimestamp(error_file.stat().st_mtime).isoformat()
                })
            
            if log_status["log_files"]:
                latest_log = max(log_status["log_files"], key=lambda x: x["modified"])
                log_status["latest_log"] = latest_log
        
        return log_status
    
    def print_status(self):
        """Print comprehensive status"""
        print(f"\n{'='*80}")
        print(f" ENHANCED MARKET DATA DOWNLOADER STATUS")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Process status
        process_status = self.get_process_status()
        print(f"\n{'='*60}")
        print(f" PROCESS STATUS")
        print(f"{'='*60}")
        
        if process_status["running"]:
            print(f"✅ Process is RUNNING")
            print(f"   PID: {process_status['pid']}")
            print(f"   CPU Usage: {process_status['cpu_percent']:.1f}%")
            print(f"   Memory Usage: {process_status['memory_mb']:.1f} MB ({process_status['memory_percent']:.1f}%)")
            print(f"   Start Time: {process_status['start_time']}")
            print(f"   Uptime: {process_status['uptime']/3600:.1f} hours")
        else:
            print(f"❌ Process is NOT RUNNING")
            if process_status["pid"]:
                print(f"   Last PID: {process_status['pid']}")
        
        # Download progress
        download_progress = self.get_download_progress()
        print(f"\n{'='*60}")
        print(f" DOWNLOAD PROGRESS")
        print(f"{'='*60}")
        
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"📊 Total Files: {download_progress['total_files']}")
        print(f"✅ Completed Files: {download_progress['completed_files']}")
        
        if download_progress["total_size_mb"] > 0:
            print(f"💾 Total Size: {download_progress['total_size_mb']:.1f} MB")
        
        if download_progress["last_updated"]:
            print(f"🕒 Last Updated: {download_progress['last_updated']}")
        
        # Show download summary if available
        if download_progress["download_summary"]:
            summary = download_progress["download_summary"]
            print(f"\n📈 DOWNLOAD SUMMARY:")
            print(f"   Total Investment Classes: {summary.get('total_investment_classes', 0)}")
            print(f"   Total Symbols: {summary.get('total_symbols', 0)}")
            print(f"   Successful Downloads: {summary.get('successful_downloads', 0)}")
            print(f"   Failed Downloads: {summary.get('failed_downloads', 0)}")
            
            if summary.get('total_symbols', 0) > 0:
                success_rate = (summary.get('successful_downloads', 0) / summary.get('total_symbols', 1)) * 100
                print(f"   Success Rate: {success_rate:.1f}%")
        
        # Log status
        log_status = self.get_log_status()
        print(f"\n{'='*60}")
        print(f" LOG STATUS")
        print(f"{'='*60}")
        
        print(f"📝 Log Files: {len(log_status['log_files'])}")
        print(f"⚠️  Error Logs: {len(log_status['error_logs'])}")
        
        if log_status["latest_log"]:
            print(f"📄 Latest Log: {log_status['latest_log']['name']}")
            print(f"   Size: {log_status['latest_log']['size_mb']:.1f} MB")
            print(f"   Modified: {log_status['latest_log']['modified']}")
        
        # Show recent log entries
        if log_status["latest_log"]:
            latest_log_file = self.log_dir / log_status["latest_log"]["name"]
            if latest_log_file.exists():
                print(f"\n📋 RECENT LOG ENTRIES:")
                try:
                    with open(latest_log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:  # Last 10 lines
                            print(f"   {line.strip()}")
                except Exception as e:
                    print(f"   Error reading log file: {e}")
    
    def monitor_continuously(self, interval: int = 60):
        """Monitor continuously with specified interval"""
        print(f"Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                os.system('clear')
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Run the monitor"""
    monitor = DownloaderMonitor()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor.monitor_continuously()
    else:
        monitor.print_status()


if __name__ == "__main__":
    main() 