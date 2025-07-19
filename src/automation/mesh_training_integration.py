"""
Mesh Training Integration

Integrates vectorized bank statement data with stochastic mesh engine for training.
Loads cached vectorized data and prepares it for mesh simulations.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingProfile:
    """Profile for mesh training with bank statement data"""
    profile_id: str
    transactions: List[Dict]
    amounts: List[float]
    dates: List[str]
    categories: List[str]
    types: List[str]
    metadata: Dict


class SimpleMeshEngine:
    """Simplified mesh engine for training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_simulation(self, initial_value: float, drift: float, volatility: float, 
                      time_steps: int, num_paths: int) -> Dict:
        """Run simplified mesh simulation"""
        
        # Simple Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        # Generate random paths
        dt = 1.0 / time_steps
        paths = np.zeros((num_paths, time_steps + 1))
        paths[:, 0] = initial_value
        
        for i in range(time_steps):
            # Brownian motion
            dw = np.random.normal(0, np.sqrt(dt), num_paths)
            paths[:, i + 1] = paths[:, i] * (1 + drift * dt + volatility * dw)
        
        return {
            "paths": paths,
            "final_values": paths[:, -1],
            "time_steps": time_steps,
            "num_paths": num_paths
        }


class MeshTrainingIntegration:
    """Integrates bank statement data with mesh training"""
    
    def __init__(self):
        self.cache_dir = Path("data/cache/statements")
        self.training_dir = Path("data/training")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simplified mesh engine
        self.mesh_engine = SimpleMeshEngine()
        
    def load_vectorized_profiles(self) -> List[TrainingProfile]:
        """Load all vectorized profiles from cache"""
        profiles = []
        
        # Find all vectorized JSON files
        vectorized_files = list(self.cache_dir.glob("*_vectorized.json"))
        
        for file_path in vectorized_files:
            try:
                with open(file_path, 'r') as f:
                    vectorized_data = json.load(f)
                
                # Load corresponding metadata
                metadata_path = file_path.parent / f"{file_path.stem.replace('_vectorized', '')}_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Create training profile
                profile = TrainingProfile(
                    profile_id=vectorized_data["filename"],
                    transactions=[],  # Original transactions not stored in vectorized data
                    amounts=vectorized_data["amounts"],
                    dates=vectorized_data["dates"],
                    categories=vectorized_data["categories"],
                    types=vectorized_data["types"],
                    metadata=metadata
                )
                
                profiles.append(profile)
                logger.info(f"Loaded profile: {profile.profile_id} with {len(profile.amounts)} transactions")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return profiles
    
    def prepare_training_data(self, profiles: List[TrainingProfile]) -> Dict:
        """Prepare training data for mesh engine"""
        
        training_data = {
            "profiles": [],
            "total_transactions": 0,
            "amount_statistics": {},
            "category_distribution": {},
            "type_distribution": {}
        }
        
        all_amounts = []
        all_categories = []
        all_types = []
        
        for profile in profiles:
            # Convert amounts to numpy array
            amounts_array = np.array(profile.amounts, dtype=np.float32)
            
            # Calculate profile statistics
            profile_stats = {
                "profile_id": profile.profile_id,
                "num_transactions": len(profile.amounts),
                "total_amount": np.sum(amounts_array),
                "mean_amount": np.mean(amounts_array),
                "std_amount": np.std(amounts_array),
                "min_amount": np.min(amounts_array),
                "max_amount": np.max(amounts_array),
                "categories": list(set(profile.categories)),
                "types": list(set(profile.types))
            }
            
            training_data["profiles"].append(profile_stats)
            training_data["total_transactions"] += len(profile.amounts)
            
            # Collect statistics
            all_amounts.extend(profile.amounts)
            all_categories.extend(profile.categories)
            all_types.extend(profile.types)
        
        # Calculate global statistics
        amounts_array = np.array(all_amounts, dtype=np.float32)
        training_data["amount_statistics"] = {
            "total_transactions": len(amounts_array),
            "total_amount": np.sum(amounts_array),
            "mean_amount": np.mean(amounts_array),
            "std_amount": np.std(amounts_array),
            "min_amount": np.min(amounts_array),
            "max_amount": np.max(amounts_array)
        }
        
        # Category distribution
        from collections import Counter
        category_counts = Counter(all_categories)
        training_data["category_distribution"] = dict(category_counts)
        
        # Type distribution
        type_counts = Counter(all_types)
        training_data["type_distribution"] = dict(type_counts)
        
        return training_data
    
    def run_mesh_simulation_with_profiles(self, profiles: List[TrainingProfile], 
                                        num_simulations: int = 100) -> Dict:
        """Run mesh simulations using bank statement profiles"""
        
        simulation_results = {
            "profiles_simulated": len(profiles),
            "total_simulations": num_simulations,
            "simulation_date": datetime.now().isoformat(),
            "profile_results": []
        }
        
        for profile in profiles:
            logger.info(f"Running mesh simulation for profile: {profile.profile_id}")
            
            # Convert profile data to mesh-compatible format
            amounts_array = np.array(profile.amounts, dtype=np.float32)
            
            # Calculate profile parameters for mesh
            mean_amount = np.mean(amounts_array)
            std_amount = np.std(amounts_array)
            
            # Create mesh parameters based on profile characteristics
            mesh_params = {
                "initial_value": mean_amount,
                "drift": 0.0,  # No drift for transaction amounts
                "volatility": std_amount / mean_amount if mean_amount > 0 else 0.1,
                "time_steps": len(amounts_array),
                "num_paths": num_simulations
            }
            
            try:
                # Run mesh simulation
                mesh_result = self.mesh_engine.run_simulation(
                    initial_value=mesh_params["initial_value"],
                    drift=mesh_params["drift"],
                    volatility=mesh_params["volatility"],
                    time_steps=mesh_params["time_steps"],
                    num_paths=mesh_params["num_paths"]
                )
                
                # Analyze results
                profile_result = {
                    "profile_id": profile.profile_id,
                    "mesh_params": mesh_params,
                    "simulation_stats": {
                        "mean_final_value": np.mean(mesh_result["final_values"]),
                        "std_final_value": np.std(mesh_result["final_values"]),
                        "min_final_value": np.min(mesh_result["final_values"]),
                        "max_final_value": np.max(mesh_result["final_values"]),
                        "var_95": np.percentile(mesh_result["final_values"], 5),
                        "var_99": np.percentile(mesh_result["final_values"], 1)
                    },
                    "actual_stats": {
                        "mean_amount": mean_amount,
                        "std_amount": std_amount,
                        "min_amount": np.min(amounts_array),
                        "max_amount": np.max(amounts_array)
                    }
                }
                
                simulation_results["profile_results"].append(profile_result)
                logger.info(f"Completed simulation for {profile.profile_id}")
                
            except Exception as e:
                logger.error(f"Failed to simulate profile {profile.profile_id}: {e}")
        
        return simulation_results
    
    def save_training_results(self, training_data: Dict, simulation_results: Dict):
        """Save training and simulation results"""
        
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to JSON serializable types
        training_data_serializable = convert_numpy_types(training_data)
        simulation_results_serializable = convert_numpy_types(simulation_results)
        
        # Save training data summary
        training_summary_path = self.training_dir / "training_summary.json"
        with open(training_summary_path, 'w') as f:
            json.dump(training_data_serializable, f, indent=2)
        
        # Save simulation results
        simulation_path = self.training_dir / "mesh_simulation_results.json"
        with open(simulation_path, 'w') as f:
            json.dump(simulation_results_serializable, f, indent=2)
        
        # Save detailed analysis
        analysis_path = self.training_dir / "training_analysis.json"
        analysis = {
            "training_date": datetime.now().isoformat(),
            "training_data_summary": {
                "total_profiles": len(training_data["profiles"]),
                "total_transactions": training_data["total_transactions"],
                "amount_statistics": convert_numpy_types(training_data["amount_statistics"])
            },
            "simulation_summary": {
                "profiles_simulated": simulation_results["profiles_simulated"],
                "total_simulations": simulation_results["total_simulations"]
            },
            "category_distribution": training_data["category_distribution"],
            "type_distribution": training_data["type_distribution"]
        }
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Training results saved to {self.training_dir}")
    
    def run_complete_training_pipeline(self, num_simulations: int = 50):
        """Run complete training pipeline"""
        
        logger.info("Starting complete training pipeline...")
        
        # Step 1: Load vectorized profiles
        logger.info("Loading vectorized profiles...")
        profiles = self.load_vectorized_profiles()
        
        if not profiles:
            logger.error("No profiles found for training")
            return
        
        logger.info(f"Loaded {len(profiles)} profiles for training")
        
        # Step 2: Prepare training data
        logger.info("Preparing training data...")
        training_data = self.prepare_training_data(profiles)
        
        # Step 3: Run mesh simulations
        logger.info("Running mesh simulations...")
        simulation_results = self.run_mesh_simulation_with_profiles(profiles, num_simulations)
        
        # Step 4: Save results
        logger.info("Saving training results...")
        self.save_training_results(training_data, simulation_results)
        
        # Step 5: Print summary
        logger.info("Training pipeline complete!")
        logger.info(f"Profiles processed: {len(profiles)}")
        logger.info(f"Total transactions: {training_data['total_transactions']}")
        logger.info(f"Profiles simulated: {simulation_results['profiles_simulated']}")
        logger.info(f"Total simulations: {simulation_results['total_simulations']}")
        
        return {
            "training_data": training_data,
            "simulation_results": simulation_results
        }


def main():
    """Run the complete training pipeline"""
    integration = MeshTrainingIntegration()
    
    # Run complete pipeline
    results = integration.run_complete_training_pipeline(num_simulations=50)
    
    if results:
        print("Training pipeline completed successfully!")
        print(f"Profiles processed: {len(results['training_data']['profiles'])}")
        print(f"Total transactions: {results['training_data']['total_transactions']}")
        print(f"Profiles simulated: {results['simulation_results']['profiles_simulated']}")
    else:
        print("Training pipeline failed!")


if __name__ == "__main__":
    main() 