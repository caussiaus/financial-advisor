"""
Spending Surface Modeler
Creates continuous surfaces for milestone timing based on income, age, and discretionary spending patterns
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata, RegularGridInterpolator
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import pickle
import json

from .spending_vector_database import SpendingPatternVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpendingSurfaceModeler:
    """Creates continuous surfaces for milestone timing prediction based on spending patterns"""
    
    def __init__(self, vector_db: SpendingPatternVectorDB):
        self.vector_db = vector_db
        self.surfaces = {}  # Store fitted surface models
        self.scalers = {}   # Store data scalers for each surface
        self.surface_data = {}  # Store raw surface data
        
    def extract_surface_data(self, 
                            milestone: str,
                            income_range: Tuple[int, int] = (30000, 150000),
                            age_range: Tuple[int, int] = (22, 55),
                            discretionary_range: Tuple[float, float] = (0.05, 0.30),
                            grid_resolution: int = 20) -> Dict:
        """Extract surface data points for milestone timing"""
        
        logger.info(f"Extracting surface data for {milestone}")
        
        # Create grid points
        income_points = np.linspace(income_range[0], income_range[1], grid_resolution)
        age_points = np.linspace(age_range[0], age_range[1], grid_resolution)
        discretionary_points = np.linspace(discretionary_range[0], discretionary_range[1], grid_resolution)
        
        surface_points = []
        
        # Sample data across the parameter space
        for income in income_points:
            for age in age_points:
                for discretionary_ratio in discretionary_points:
                    
                    # Find patterns near this point
                    patterns = self.vector_db.find_patterns_by_criteria(
                        age_range=(int(age-2), int(age+2)),
                        income_range=(int(income-5000), int(income+5000)),
                        n_results=50
                    )
                    
                    if not patterns:
                        continue
                    
                    # Filter for patterns with similar discretionary spending
                    similar_patterns = [
                        p for p in patterns 
                        if abs(p['metadata'].get('discretionary_ratio', 0) - discretionary_ratio) < 0.05
                    ]
                    
                    if not similar_patterns:
                        continue
                    
                    # Find patterns that achieved the milestone
                    milestone_field = milestone.replace('_', '')  # e.g., 'ownshome', 'married', 'haschildren'
                    
                    achieved_patterns = [
                        p for p in similar_patterns 
                        if p['metadata'].get(milestone_field, False)
                    ]
                    
                    if not achieved_patterns:
                        continue
                    
                    # Get milestone timing data
                    original_patterns = self.vector_db.load_spending_patterns()
                    milestone_ages = []
                    
                    for pattern in achieved_patterns:
                        pattern_id = pattern['metadata']['pattern_id']
                        if pattern_id < len(original_patterns):
                            original = original_patterns[pattern_id]
                            milestone_age_field = f"{milestone}_age"
                            
                            if milestone_age_field in original and original[milestone_age_field]:
                                milestone_ages.append(original[milestone_age_field])
                    
                    if milestone_ages:
                        avg_achievement_age = np.mean(milestone_ages)
                        achievement_rate = len(achieved_patterns) / len(similar_patterns)
                        confidence = min(len(milestone_ages) / 10.0, 1.0)  # Confidence based on sample size
                        
                        surface_points.append({
                            'income': income,
                            'age': age,
                            'discretionary_ratio': discretionary_ratio,
                            'achievement_age': avg_achievement_age,
                            'achievement_rate': achievement_rate,
                            'sample_size': len(milestone_ages),
                            'confidence': confidence
                        })
        
        logger.info(f"Extracted {len(surface_points)} surface data points for {milestone}")
        
        self.surface_data[milestone] = surface_points
        return surface_points
    
    def create_gaussian_process_surface(self, 
                                       milestone: str,
                                       surface_points: List[Dict],
                                       kernel_params: Optional[Dict] = None) -> GaussianProcessRegressor:
        """Create Gaussian Process surface model for milestone timing"""
        
        if not surface_points:
            raise ValueError("No surface points provided")
        
        # Extract features and targets
        X = np.array([[p['income'], p['age'], p['discretionary_ratio']] for p in surface_points])
        y = np.array([p['achievement_age'] for p in surface_points])
        weights = np.array([p['confidence'] for p in surface_points])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define kernel
        if kernel_params is None:
            kernel_params = {
                'length_scale': 1.0,
                'length_scale_bounds': (0.1, 10.0),
                'noise_level': 0.5
            }
        
        kernel = (C(1.0, (0.01, 100.0)) * 
                 RBF(length_scale=kernel_params['length_scale'], 
                     length_scale_bounds=kernel_params['length_scale_bounds']) + 
                 WhiteKernel(noise_level=kernel_params['noise_level']))
        
        # Fit Gaussian Process
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1.0 / weights,  # Use confidence as inverse noise
            n_restarts_optimizer=5,
            normalize_y=True
        )
        
        gpr.fit(X_scaled, y)
        
        # Store scaler and model
        self.scalers[milestone] = scaler
        self.surfaces[milestone] = gpr
        
        logger.info(f"Fitted Gaussian Process surface for {milestone}")
        logger.info(f"Final kernel: {gpr.kernel_}")
        
        return gpr
    
    def create_rbf_surface(self, 
                          milestone: str,
                          surface_points: List[Dict],
                          function: str = 'thin_plate_spline') -> RBFInterpolator:
        """Create RBF interpolation surface for milestone timing"""
        
        if not surface_points:
            raise ValueError("No surface points provided")
        
        # Extract features and targets
        X = np.array([[p['income'], p['age'], p['discretionary_ratio']] for p in surface_points])
        y = np.array([p['achievement_age'] for p in surface_points])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create RBF interpolator
        rbf = RBFInterpolator(
            X_scaled,
            y,
            kernel=function,
            smoothing=0.1  # Small smoothing to handle noise
        )
        
        # Store scaler and model
        self.scalers[milestone] = scaler
        self.surfaces[milestone] = rbf
        
        logger.info(f"Fitted RBF surface for {milestone} using {function}")
        
        return rbf
    
    def predict_milestone_timing_surface(self, 
                                       milestone: str,
                                       income: float,
                                       age: float,
                                       discretionary_ratio: float,
                                       return_uncertainty: bool = False) -> Dict:
        """Predict milestone timing using fitted surface model"""
        
        if milestone not in self.surfaces:
            raise ValueError(f"No surface model fitted for {milestone}")
        
        # Scale input
        scaler = self.scalers[milestone]
        surface_model = self.surfaces[milestone]
        
        X_input = np.array([[income, age, discretionary_ratio]])
        X_scaled = scaler.transform(X_input)
        
        # Make prediction
        if isinstance(surface_model, GaussianProcessRegressor):
            if return_uncertainty:
                prediction, std = surface_model.predict(X_scaled, return_std=True)
                uncertainty = float(std[0])
            else:
                prediction = surface_model.predict(X_scaled)
                uncertainty = None
        else:  # RBF interpolator
            prediction = surface_model(X_scaled)
            uncertainty = None
        
        predicted_age = float(prediction[0])
        
        # Calculate additional metrics
        time_to_milestone = max(0, predicted_age - age)
        
        # Estimate confidence based on distance from training data
        if milestone in self.surface_data:
            training_points = self.surface_data[milestone]
            distances = []
            
            for point in training_points:
                dist = np.sqrt(
                    ((income - point['income']) / 50000) ** 2 +
                    ((age - point['age']) / 20) ** 2 +
                    ((discretionary_ratio - point['discretionary_ratio']) / 0.2) ** 2
                )
                distances.append(dist)
            
            min_distance = min(distances) if distances else 1.0
            confidence = max(0.1, 1.0 - min_distance)
        else:
            confidence = 0.5
        
        result = {
            'predicted_age': predicted_age,
            'time_to_milestone': time_to_milestone,
            'confidence': confidence,
            'model_type': 'Gaussian Process' if isinstance(surface_model, GaussianProcessRegressor) else 'RBF'
        }
        
        if uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        return result
    
    def create_spending_capacity_surface(self, 
                                       milestone: str,
                                       surface_points: List[Dict]) -> Dict:
        """Create surface showing required spending capacity for milestone achievement"""
        
        # Extract data
        data = []
        for point in surface_points:
            # Calculate required savings based on milestone type
            income = point['income']
            discretionary_ratio = point['discretionary_ratio']
            achievement_age = point['achievement_age']
            
            # Estimate required savings/spending capacity
            annual_discretionary = income * discretionary_ratio
            
            if milestone == 'home_purchase':
                # Assume 20% down payment on 3x income house
                required_savings = income * 3 * 0.20
                years_to_save = max(1, achievement_age - 25)  # Start saving at 25
                required_annual_savings = required_savings / years_to_save
                spending_capacity_ratio = required_annual_savings / annual_discretionary
            elif milestone == 'marriage':
                # Assume wedding cost of 1 year discretionary spending
                required_savings = annual_discretionary
                spending_capacity_ratio = 1.0
            elif milestone == 'first_child':
                # Assume first year child costs of 15% of income
                required_savings = income * 0.15
                spending_capacity_ratio = required_savings / annual_discretionary
            else:
                spending_capacity_ratio = 0.5  # Default
            
            data.append({
                'income': income,
                'age': point['age'],
                'discretionary_ratio': discretionary_ratio,
                'spending_capacity_ratio': min(spending_capacity_ratio, 2.0),  # Cap at 200%
                'achievement_age': achievement_age
            })
        
        return data
    
    def visualize_surface_3d(self, 
                           milestone: str,
                           income_range: Tuple[float, float] = (40000, 120000),
                           age_range: Tuple[float, float] = (25, 45),
                           discretionary_fixed: float = 0.15) -> go.Figure:
        """Create 3D visualization of milestone timing surface"""
        
        if milestone not in self.surfaces:
            raise ValueError(f"No surface model fitted for {milestone}")
        
        # Create prediction grid
        income_grid = np.linspace(income_range[0], income_range[1], 30)
        age_grid = np.linspace(age_range[0], age_range[1], 30)
        
        Income, Age = np.meshgrid(income_grid, age_grid)
        
        # Make predictions across grid
        predictions = np.zeros_like(Income)
        
        for i in range(Income.shape[0]):
            for j in range(Income.shape[1]):
                try:
                    pred = self.predict_milestone_timing_surface(
                        milestone,
                        Income[i, j],
                        Age[i, j],
                        discretionary_fixed
                    )
                    predictions[i, j] = pred['predicted_age']
                except:
                    predictions[i, j] = np.nan
        
        # Create 3D surface plot
        fig = go.Figure()
        
        fig.add_trace(go.Surface(
            x=Income,
            y=Age,
            z=predictions,
            colorscale='Viridis',
            name=f'{milestone} Achievement Age'
        ))
        
        # Add training data points if available
        if milestone in self.surface_data:
            training_data = self.surface_data[milestone]
            training_income = [p['income'] for p in training_data]
            training_age = [p['age'] for p in training_data]
            training_achievement = [p['achievement_age'] for p in training_data]
            
            fig.add_trace(go.Scatter3d(
                x=training_income,
                y=training_age,
                z=training_achievement,
                mode='markers',
                marker=dict(size=3, color='red'),
                name='Training Data'
            ))
        
        fig.update_layout(
            title=f'{milestone.replace("_", " ").title()} Achievement Age Surface<br>'
                  f'Discretionary Spending Ratio: {discretionary_fixed:.1%}',
            scene=dict(
                xaxis_title='Income ($)',
                yaxis_title='Current Age',
                zaxis_title='Achievement Age',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_spending_capacity_heatmap(self, 
                                          milestone: str,
                                          age_fixed: float = 30) -> go.Figure:
        """Create heatmap showing spending capacity requirements"""
        
        if milestone not in self.surface_data:
            raise ValueError(f"No surface data available for {milestone}")
        
        # Create spending capacity surface
        capacity_data = self.create_spending_capacity_surface(milestone, self.surface_data[milestone])
        
        # Filter for fixed age
        age_filtered = [d for d in capacity_data if abs(d['age'] - age_fixed) < 2]
        
        if not age_filtered:
            raise ValueError(f"No data available for age {age_fixed}")
        
        # Create grid
        income_values = sorted(list(set([d['income'] for d in age_filtered])))
        discretionary_values = sorted(list(set([d['discretionary_ratio'] for d in age_filtered])))
        
        # Create heatmap data
        heatmap_data = np.full((len(discretionary_values), len(income_values)), np.nan)
        
        for d in age_filtered:
            try:
                i = discretionary_values.index(d['discretionary_ratio'])
                j = income_values.index(d['income'])
                heatmap_data[i, j] = d['spending_capacity_ratio']
            except ValueError:
                continue
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=[f"${int(inc/1000)}k" for inc in income_values],
            y=[f"{disc:.1%}" for disc in discretionary_values],
            z=heatmap_data,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Spending Capacity Ratio")
        ))
        
        fig.update_layout(
            title=f'{milestone.replace("_", " ").title()} Spending Capacity Requirements<br>'
                  f'Age: {age_fixed}',
            xaxis_title='Income',
            yaxis_title='Discretionary Spending Ratio',
            width=700,
            height=500
        )
        
        return fig
    
    def optimize_spending_for_milestone(self, 
                                      milestone: str,
                                      current_age: float,
                                      income: float,
                                      target_age: float,
                                      constraints: Optional[Dict] = None) -> Dict:
        """Optimize discretionary spending to achieve milestone by target age"""
        
        if milestone not in self.surfaces:
            raise ValueError(f"No surface model fitted for {milestone}")
        
        def objective(discretionary_ratio):
            """Objective function: minimize difference between predicted and target age"""
            pred = self.predict_milestone_timing_surface(
                milestone, income, current_age, discretionary_ratio[0]
            )
            return (pred['predicted_age'] - target_age) ** 2
        
        # Set constraints
        bounds = [(0.05, 0.40)]  # Discretionary ratio between 5% and 40%
        
        if constraints:
            min_ratio = constraints.get('min_discretionary_ratio', 0.05)
            max_ratio = constraints.get('max_discretionary_ratio', 0.40)
            bounds = [(min_ratio, max_ratio)]
        
        # Optimize
        result = minimize(
            objective,
            x0=[0.15],  # Initial guess
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            optimal_ratio = result.x[0]
            
            # Get prediction with optimal ratio
            prediction = self.predict_milestone_timing_surface(
                milestone, income, current_age, optimal_ratio
            )
            
            # Calculate required changes
            annual_discretionary = income * optimal_ratio
            monthly_discretionary = annual_discretionary / 12
            
            return {
                'optimal_discretionary_ratio': optimal_ratio,
                'annual_discretionary_spending': annual_discretionary,
                'monthly_discretionary_spending': monthly_discretionary,
                'predicted_achievement_age': prediction['predicted_age'],
                'time_to_milestone': prediction['time_to_milestone'],
                'confidence': prediction['confidence'],
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'error': 'Could not find optimal spending pattern'
            }
    
    def save_surfaces(self, path: str):
        """Save fitted surface models"""
        save_data = {
            'surfaces': self.surfaces,
            'scalers': self.scalers,
            'surface_data': self.surface_data
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved surface models to {path}")
    
    def load_surfaces(self, path: str):
        """Load fitted surface models"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.surfaces = save_data['surfaces']
        self.scalers = save_data['scalers']
        self.surface_data = save_data['surface_data']
        
        logger.info(f"Loaded surface models from {path}")
    
    def get_surface_summary(self) -> Dict:
        """Get summary of fitted surfaces"""
        
        summary = {
            'fitted_surfaces': list(self.surfaces.keys()),
            'surface_details': {}
        }
        
        for milestone in self.surfaces.keys():
            model = self.surfaces[milestone]
            
            if isinstance(model, GaussianProcessRegressor):
                model_type = 'Gaussian Process'
                model_info = {
                    'kernel': str(model.kernel_),
                    'log_marginal_likelihood': model.log_marginal_likelihood()
                }
            else:
                model_type = 'RBF Interpolator'
                model_info = {}
            
            data_points = len(self.surface_data.get(milestone, []))
            
            summary['surface_details'][milestone] = {
                'model_type': model_type,
                'training_points': data_points,
                'model_info': model_info
            }
        
        return summary

# Usage example
def main():
    from .spending_pattern_scraper import SpendingDataScraper
    import asyncio
    
    async def run_demo():
        # Initialize components
        scraper = SpendingDataScraper()
        
        # Scrape data
        await scraper.scrape_all_sources()
        scraper.generate_milestone_patterns()
        
        # Set up vector database
        vector_db = SpendingPatternVectorDB()
        vector_db.vectorize_and_store_patterns()
        
        # Create surface modeler
        surface_modeler = SpendingSurfaceModeler(vector_db)
        
        # Extract and model surfaces for different milestones
        for milestone in ['home_purchase', 'marriage', 'first_child']:
            print(f"\nCreating surface for {milestone}...")
            
            # Extract surface data
            surface_points = surface_modeler.extract_surface_data(milestone)
            
            if surface_points:
                # Create Gaussian Process surface
                surface_modeler.create_gaussian_process_surface(milestone, surface_points)
                
                # Test prediction
                prediction = surface_modeler.predict_milestone_timing_surface(
                    milestone, income=75000, age=28, discretionary_ratio=0.15
                )
                print(f"Prediction for {milestone}: {prediction}")
                
                # Optimize spending
                optimization = surface_modeler.optimize_spending_for_milestone(
                    milestone, current_age=28, income=75000, target_age=32
                )
                print(f"Optimization for {milestone}: {optimization}")
        
        # Get summary
        summary = surface_modeler.get_surface_summary()
        print(f"\nSurface Summary: {summary}")
    
    asyncio.run(run_demo())

if __name__ == "__main__":
    main() 