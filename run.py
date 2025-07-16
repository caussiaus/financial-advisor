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

# --- Configuration ---
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

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

# --- Application Modes ---
def run_web_service(host, port):
    """Starts the Flask web service."""
    print("🚀 Starting Web Service Mode...")
    # This will be replaced by the consolidated Flask app
    print(f"🌐 Navigate to http://{host}:{port}")
    # For now, we'll just simulate running the web_service.py
    subprocess.run([sys.executable, "web_service.py"])

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