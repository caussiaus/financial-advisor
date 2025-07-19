#!/usr/bin/env python3
"""
Main Financial Advisor Entry Point
Consolidated interface for the integrated financial advisor system
"""

import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Import core modules
from src.core.financial_advisor import IntegratedFinancialAdvisor
from src.core.visualization_engine import FinancialVisualizationEngine, DashboardConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainFinancialAdvisor:
    """Main entry point for the integrated financial advisor system"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the main financial advisor"""
        self.config = config or {}
        
        # Initialize core components
        self.financial_advisor = IntegratedFinancialAdvisor(config)
        self.visualization_engine = FinancialVisualizationEngine()
        
        logger.info("✅ Main Financial Advisor initialized")
    
    def run_analysis(self, client_data: Dict, start_dashboard: bool = True) -> Dict:
        """Run comprehensive financial analysis"""
        try:
            logger.info("🔄 Starting comprehensive financial analysis...")
            
            # Start continuous analysis
            self.financial_advisor.start_continuous_analysis(client_data)
            
            # Wait for first analysis to complete
            import time
            time.sleep(10)  # Wait for analysis to complete
            
            # Get analysis results
            analysis_results = self.financial_advisor.get_current_analysis()
            
            # Update visualization engine
            self.visualization_engine.update_dashboard_data(analysis_results)
            
            # Start dashboard if requested
            if start_dashboard:
                self.visualization_engine.start_dashboard()
                logger.info("📊 Dashboard started")
            
            logger.info("✅ Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error running analysis: {e}")
            raise
    
    def generate_dashboard(self, analysis_data: Dict, output_file: Optional[str] = None) -> str:
        """Generate comprehensive dashboard"""
        try:
            logger.info("📊 Generating comprehensive dashboard...")
            
            # Generate dashboard HTML
            dashboard_html = self.visualization_engine.generate_comprehensive_dashboard(analysis_data)
            
            # Export to file if specified
            if output_file:
                self.visualization_engine.export_chart_to_file(dashboard_html, output_file)
                logger.info(f"✅ Dashboard exported to {output_file}")
            
            return dashboard_html
            
        except Exception as e:
            logger.error(f"❌ Error generating dashboard: {e}")
            raise
    
    def run_interactive_mode(self, client_data: Dict):
        """Run interactive mode with dashboard"""
        try:
            logger.info("🎮 Starting interactive mode...")
            
            # Run analysis
            analysis_results = self.run_analysis(client_data, start_dashboard=True)
            
            # Generate dashboard
            dashboard_html = self.generate_dashboard(analysis_results)
            
            logger.info("✅ Interactive mode started")
            logger.info(f"📊 Dashboard available at http://localhost:{self.visualization_engine.config.port}")
            
            # Keep running
            try:
                while True:
                    import time
                    time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("⏹️ Stopping interactive mode...")
                self.financial_advisor.stop_continuous_analysis()
                self.visualization_engine.stop_dashboard()
                logger.info("✅ Interactive mode stopped")
            
        except Exception as e:
            logger.error(f"❌ Error in interactive mode: {e}")
            raise
    
    def run_batch_mode(self, client_data: Dict, output_file: str):
        """Run batch mode with output to file"""
        try:
            logger.info("📋 Starting batch mode...")
            
            # Run analysis
            analysis_results = self.run_analysis(client_data, start_dashboard=False)
            
            # Generate dashboard
            dashboard_html = self.generate_dashboard(analysis_results, output_file)
            
            # Save analysis results
            results_file = output_file.replace('.html', '_results.json')
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"✅ Batch mode completed")
            logger.info(f"📊 Dashboard saved to {output_file}")
            logger.info(f"📋 Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Error in batch mode: {e}")
            raise
    
    def get_sample_client_data(self) -> Dict:
        """Get sample client data for testing"""
        return {
            "name": "Alex Johnson",
            "age": 32,
            "income": 85000,
            "expenses": 65000,
            "assets": {
                "cash": 15000,
                "investments": 45000,
                "retirement": 25000,
                "real_estate": 200000
            },
            "liabilities": {
                "student_loans": 35000,
                "credit_cards": 8000,
                "mortgage": 180000
            },
            "personality": {
                "fear_of_loss": 0.6,
                "greed_factor": 0.4,
                "social_pressure": 0.3,
                "patience": 0.7,
                "financial_literacy": 0.6
            },
            "goals": ["emergency_fund", "debt_free", "retirement"],
            "risk_tolerance": 0.6,
            "life_stage": "early_career"
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Integrated Financial Advisor System")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Run mode: interactive (with dashboard) or batch (file output)")
    parser.add_argument("--client-data", type=str, help="Path to client data JSON file")
    parser.add_argument("--output", type=str, default="financial_analysis.html",
                       help="Output file for batch mode")
    parser.add_argument("--sample", action="store_true", help="Use sample client data")
    
    args = parser.parse_args()
    
    # Initialize main advisor
    main_advisor = MainFinancialAdvisor()
    
    # Get client data
    if args.sample:
        client_data = main_advisor.get_sample_client_data()
        logger.info("📊 Using sample client data")
    elif args.client_data:
        try:
            with open(args.client_data, 'r') as f:
                client_data = json.load(f)
            logger.info(f"📊 Loaded client data from {args.client_data}")
        except Exception as e:
            logger.error(f"❌ Error loading client data: {e}")
            return
    else:
        client_data = main_advisor.get_sample_client_data()
        logger.info("📊 Using sample client data (default)")
    
    try:
        if args.mode == "interactive":
            # Run interactive mode
            main_advisor.run_interactive_mode(client_data)
        else:
            # Run batch mode
            main_advisor.run_batch_mode(client_data, args.output)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 