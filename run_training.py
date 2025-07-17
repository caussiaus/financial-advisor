#!/usr/bin/env python3
"""
Run Mesh Training Engine

Simple script to run the training engine and generate synthetic people,
apply financial shocks, and learn optimal commutator routes.

Usage:
    python run_training.py [num_scenarios]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Use the training controller for clean imports
from src.training.training_controller import (
    get_training_controller,
    run_training_session,
    get_training_status
)


def main():
    parser = argparse.ArgumentParser(description="Run Mesh Training Engine")
    parser.add_argument("num_scenarios", type=int, nargs="?", default=100,
                       help="Number of training scenarios to generate (default: 100)")
    parser.add_argument("--output-dir", default="data/outputs/training",
                       help="Output directory for training results")
    parser.add_argument("--status", action="store_true",
                       help="Show training system status")
    
    args = parser.parse_args()
    
    if args.status:
        # Show system status
        status = get_training_status()
        print("🔧 Training System Status")
        print("=" * 40)
        print(f"Import Success: {status['import_success']}")
        if not status['import_success']:
            print(f"Import Error: {status['import_error']}")
            return
        
        print(f"\nComponent Status:")
        for component, initialized in status['components_initialized'].items():
            status_icon = "✅" if initialized else "❌"
            print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        print(f"\nTraining Metrics:")
        metrics = status['training_metrics']
        print(f"  Scenarios Generated: {metrics['scenarios_generated']}")
        print(f"  Successful Routes: {metrics['successful_routes']}")
        print(f"  Failed Routes: {metrics['failed_routes']}")
        print(f"  Training Sessions: {metrics['training_sessions']}")
        return
    
    print("🚀 Starting Mesh Training Engine")
    print(f"📊 Generating {args.num_scenarios} training scenarios...")
    
    # Run training session using the controller
    result = run_training_session(num_scenarios=args.num_scenarios)
    
    print("\n✅ Training completed successfully!")
    print(f"📈 Results: {result.successful_recoveries}/{result.num_scenarios} successful recoveries")
    print(f"💾 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 