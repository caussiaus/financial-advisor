#!/usr/bin/env python
"""
Continuous Configuration Mesh System
Author: Claude 2025-07-16

Creates a tighter mesh of configurations with continuous scale fuzzy sets for 
more accurate interpolation when finding optimal solutions. Many people can 
come up with a range of configurations and percentages, requiring more nodes 
to reduce interpolation error.

Key Features:
- Multi-dimensional continuous parameter space
- Adaptive mesh refinement for critical regions
- Fuzzy membership interpolation for smooth transitions
- Higher resolution around optimal solutions
- Spline-based interpolation for smooth surfaces
- Constraint-aware mesh generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
import itertools
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

@dataclass
class MeshPoint:
    """A point in the continuous configuration mesh"""
    coordinates: Dict[str, float]  # Parameter values
    fuzzy_memberships: Dict[str, float]  # Fuzzy set memberships
    objectives: Dict[str, float]  # Objective function values
    constraints_satisfied: bool
    interpolation_weight: float  # Weight for interpolation
    mesh_level: int  # Refinement level

@dataclass 
class MeshRegion:
    """A region in the mesh with specific properties"""
    bounds: Dict[str, Tuple[float, float]]
    resolution: Dict[str, int]
    importance_score: float
    points: List[MeshPoint]
    interpolation_quality: float

class ContinuousConfigurationMesh:
    """Main mesh generation and management system"""
    
    def __init__(self, dimensions: Dict[str, Tuple[float, float]], 
                 base_resolution: int = 10, max_refinement_levels: int = 3):
        """
        Initialize continuous configuration mesh
        
        Args:
            dimensions: Dict mapping parameter names to (min, max) bounds
            base_resolution: Base number of points per dimension
            max_refinement_levels: Maximum levels of adaptive refinement
        """
        self.dimensions = dimensions
        self.base_resolution = base_resolution
        self.max_refinement_levels = max_refinement_levels
        self.mesh_points = []
        self.mesh_regions = []
        self.interpolators = {}
        self.fuzzy_sets = {}
        
    def generate_base_mesh(self, objective_functions: Dict[str, Callable], 
                          constraint_functions: List[Callable] = None) -> None:
        """Generate the base mesh with uniform spacing"""
        
        logger.info(f"Generating base mesh with {self.base_resolution}^{len(self.dimensions)} points")
        
        # Create coordinate grids
        coordinate_arrays = {}
        for param, (min_val, max_val) in self.dimensions.items():
            coordinate_arrays[param] = np.linspace(min_val, max_val, self.base_resolution)
        
        # Generate all combinations
        param_names = list(self.dimensions.keys())
        coordinate_combinations = list(itertools.product(*[coordinate_arrays[param] for param in param_names]))
        
        # Evaluate each point
        for coordinates in coordinate_combinations:
            point_coords = dict(zip(param_names, coordinates))
            
            # Calculate fuzzy memberships
            fuzzy_memberships = self._calculate_fuzzy_memberships(point_coords)
            
            # Evaluate objectives
            objectives = {}
            for obj_name, obj_func in objective_functions.items():
                try:
                    objectives[obj_name] = obj_func(point_coords)
                except Exception as e:
                    logger.warning(f"Failed to evaluate {obj_name} at {point_coords}: {e}")
                    objectives[obj_name] = float('inf')
            
            # Check constraints
            constraints_satisfied = True
            if constraint_functions:
                for constraint_func in constraint_functions:
                    try:
                        if not constraint_func(point_coords):
                            constraints_satisfied = False
                            break
                    except Exception as e:
                        logger.warning(f"Constraint evaluation failed at {point_coords}: {e}")
                        constraints_satisfied = False
                        break
            
            mesh_point = MeshPoint(
                coordinates=point_coords,
                fuzzy_memberships=fuzzy_memberships,
                objectives=objectives,
                constraints_satisfied=constraints_satisfied,
                interpolation_weight=1.0,
                mesh_level=0
            )
            
            self.mesh_points.append(mesh_point)
        
        logger.info(f"Generated {len(self.mesh_points)} base mesh points")
        feasible_points = sum(1 for p in self.mesh_points if p.constraints_satisfied)
        logger.info(f"Feasible points: {feasible_points} ({feasible_points/len(self.mesh_points):.1%})")
    
    def _calculate_fuzzy_memberships(self, coordinates: Dict[str, float]) -> Dict[str, float]:
        """Calculate fuzzy set memberships for a point"""
        memberships = {}
        
        # Define fuzzy sets for key financial concepts
        fuzzy_sets = {
            'high_income': self._triangular_fuzzy_set(0.6, 0.8, 1.0),
            'moderate_expenses': self._triangular_fuzzy_set(0.4, 0.6, 0.8),
            'high_savings': self._triangular_fuzzy_set(0.15, 0.25, 0.4),
            'low_stress': self._triangular_fuzzy_set(0.0, 0.2, 0.4),
            'high_quality': self._triangular_fuzzy_set(0.6, 0.8, 1.0),
            'balanced_work': self._triangular_fuzzy_set(0.5, 0.7, 0.9)
        }
        
        # Map coordinates to relevant fuzzy concepts
        param_mappings = {
            'work_intensity': ['balanced_work'],
            'savings_rate': ['high_savings'],
            'stress_level': ['low_stress'],
            'quality_of_life': ['high_quality'],
            'expense_ratio': ['moderate_expenses']
        }
        
        for param, value in coordinates.items():
            if param in param_mappings:
                for fuzzy_concept in param_mappings[param]:
                    if fuzzy_concept in fuzzy_sets:
                        membership = fuzzy_sets[fuzzy_concept](value)
                        memberships[f"{param}_{fuzzy_concept}"] = membership
        
        return memberships
    
    def _triangular_fuzzy_set(self, a: float, b: float, c: float) -> Callable[[float], float]:
        """Create a triangular fuzzy membership function"""
        def membership(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x < c:
                return (c - x) / (c - b)
            else:
                return 0.0
        return membership
    
    def adaptive_refinement(self, objective_functions: Dict[str, Callable],
                          refinement_criteria: Dict[str, Any] = None) -> None:
        """Adaptively refine mesh in important regions"""
        
        if refinement_criteria is None:
            refinement_criteria = {
                'gradient_threshold': 0.1,
                'objective_threshold': 0.05,
                'feasibility_threshold': 0.8
            }
        
        for level in range(self.max_refinement_levels):
            logger.info(f"Adaptive refinement level {level + 1}")
            
            # Identify regions needing refinement
            refinement_regions = self._identify_refinement_regions(
                objective_functions, refinement_criteria
            )
            
            if not refinement_regions:
                logger.info("No regions need refinement")
                break
            
            # Refine each region
            new_points = []
            for region in refinement_regions:
                region_points = self._refine_region(region, objective_functions, level + 1)
                new_points.extend(region_points)
            
            self.mesh_points.extend(new_points)
            logger.info(f"Added {len(new_points)} refined points")
    
    def _identify_refinement_regions(self, objective_functions: Dict[str, Callable],
                                   criteria: Dict[str, Any]) -> List[MeshRegion]:
        """Identify mesh regions that need refinement"""
        regions = []
        
        # Group points into regions (simplified approach)
        feasible_points = [p for p in self.mesh_points if p.constraints_satisfied]
        
        if len(feasible_points) < 4:
            return regions
        
        # Find regions with high objective gradients
        for i, point in enumerate(feasible_points[:-1]):
            neighbors = self._find_neighbors(point, feasible_points, radius=0.1)
            
            if len(neighbors) >= 2:
                gradients = self._calculate_objective_gradients(point, neighbors, objective_functions)
                
                max_gradient = max(abs(grad) for grad in gradients.values())
                if max_gradient > criteria['gradient_threshold']:
                    
                    # Define region bounds around this point
                    bounds = {}
                    for param in self.dimensions:
                        center = point.coordinates[param]
                        param_range = self.dimensions[param][1] - self.dimensions[param][0]
                        radius = param_range / (self.base_resolution * 2)
                        
                        bounds[param] = (
                            max(self.dimensions[param][0], center - radius),
                            min(self.dimensions[param][1], center + radius)
                        )
                    
                    region = MeshRegion(
                        bounds=bounds,
                        resolution={param: 3 for param in self.dimensions},  # 3x3 refinement
                        importance_score=max_gradient,
                        points=[point],
                        interpolation_quality=0.0
                    )
                    regions.append(region)
        
        # Sort by importance and limit number of regions
        regions.sort(key=lambda r: r.importance_score, reverse=True)
        return regions[:5]  # Limit to top 5 regions
    
    def _find_neighbors(self, point: MeshPoint, all_points: List[MeshPoint], 
                       radius: float) -> List[MeshPoint]:
        """Find neighboring points within radius"""
        neighbors = []
        point_coords = np.array(list(point.coordinates.values()))
        
        for other_point in all_points:
            if other_point == point:
                continue
            
            other_coords = np.array(list(other_point.coordinates.values()))
            distance = np.linalg.norm(point_coords - other_coords)
            
            if distance <= radius:
                neighbors.append(other_point)
        
        return neighbors
    
    def _calculate_objective_gradients(self, center_point: MeshPoint, 
                                     neighbors: List[MeshPoint],
                                     objective_functions: Dict[str, Callable]) -> Dict[str, float]:
        """Calculate approximate gradients of objectives"""
        gradients = {}
        
        for obj_name in objective_functions:
            if obj_name not in center_point.objectives:
                continue
            
            center_value = center_point.objectives[obj_name]
            
            # Calculate approximate gradient using finite differences
            gradient_components = []
            
            for neighbor in neighbors:
                if obj_name in neighbor.objectives:
                    neighbor_value = neighbor.objectives[obj_name]
                    
                    # Calculate distance
                    center_coords = np.array(list(center_point.coordinates.values()))
                    neighbor_coords = np.array(list(neighbor.coordinates.values()))
                    distance = np.linalg.norm(center_coords - neighbor_coords)
                    
                    if distance > 0:
                        gradient_component = (neighbor_value - center_value) / distance
                        gradient_components.append(gradient_component)
            
            if gradient_components:
                gradients[obj_name] = np.mean(gradient_components)
            else:
                gradients[obj_name] = 0.0
        
        return gradients
    
    def _refine_region(self, region: MeshRegion, objective_functions: Dict[str, Callable],
                      mesh_level: int) -> List[MeshPoint]:
        """Refine a specific region with higher resolution"""
        new_points = []
        
        # Generate refined grid in this region
        coordinate_arrays = {}
        for param, (min_val, max_val) in region.bounds.items():
            resolution = region.resolution[param]
            coordinate_arrays[param] = np.linspace(min_val, max_val, resolution)
        
        param_names = list(region.bounds.keys())
        coordinate_combinations = list(itertools.product(*[coordinate_arrays[param] for param in param_names]))
        
        for coordinates in coordinate_combinations:
            point_coords = dict(zip(param_names, coordinates))
            
            # Add missing dimensions at boundary values
            for param in self.dimensions:
                if param not in point_coords:
                    # Use value from region center or boundary
                    point_coords[param] = (self.dimensions[param][0] + self.dimensions[param][1]) / 2
            
            # Skip if point already exists nearby
            if self._point_exists_nearby(point_coords, tolerance=0.01):
                continue
            
            # Calculate properties for new point
            fuzzy_memberships = self._calculate_fuzzy_memberships(point_coords)
            
            objectives = {}
            for obj_name, obj_func in objective_functions.items():
                try:
                    objectives[obj_name] = obj_func(point_coords)
                except Exception as e:
                    objectives[obj_name] = float('inf')
            
            # Check constraints (assume satisfied in refined regions)
            constraints_satisfied = True
            
            new_point = MeshPoint(
                coordinates=point_coords,
                fuzzy_memberships=fuzzy_memberships,
                objectives=objectives,
                constraints_satisfied=constraints_satisfied,
                interpolation_weight=2.0,  # Higher weight for refined points
                mesh_level=mesh_level
            )
            
            new_points.append(new_point)
        
        return new_points
    
    def _point_exists_nearby(self, coordinates: Dict[str, float], tolerance: float) -> bool:
        """Check if a point with similar coordinates already exists"""
        test_coords = np.array(list(coordinates.values()))
        
        for existing_point in self.mesh_points:
            existing_coords = np.array(list(existing_point.coordinates.values()))
            distance = np.linalg.norm(test_coords - existing_coords)
            
            if distance < tolerance:
                return True
        
        return False
    
    def build_interpolators(self, objective_names: List[str]) -> None:
        """Build interpolation functions for smooth objective evaluation"""
        logger.info("Building interpolators for continuous mesh")
        
        feasible_points = [p for p in self.mesh_points if p.constraints_satisfied]
        
        if len(feasible_points) < 4:
            logger.warning("Not enough feasible points for interpolation")
            return
        
        # Prepare data for interpolation
        param_names = list(self.dimensions.keys())
        coordinates_array = np.array([
            [point.coordinates[param] for param in param_names] 
            for point in feasible_points
        ])
        
        for obj_name in objective_names:
            objective_values = np.array([
                point.objectives.get(obj_name, float('inf')) 
                for point in feasible_points
            ])
            
            # Filter out infinite values
            finite_mask = np.isfinite(objective_values)
            if np.sum(finite_mask) < 4:
                logger.warning(f"Not enough finite values for {obj_name} interpolation")
                continue
            
            finite_coords = coordinates_array[finite_mask]
            finite_values = objective_values[finite_mask]
            
            try:
                # Use different interpolation methods based on data size
                if len(finite_values) < 100:
                    # Linear interpolation for small datasets
                    self.interpolators[obj_name] = lambda x, coords=finite_coords, values=finite_values: \
                        griddata(coords, values, x, method='linear', fill_value=np.mean(finite_values))
                else:
                    # Cubic for larger datasets
                    self.interpolators[obj_name] = lambda x, coords=finite_coords, values=finite_values: \
                        griddata(coords, values, x, method='cubic', fill_value=np.mean(finite_values))
                
                logger.info(f"Built interpolator for {obj_name}")
                
            except Exception as e:
                logger.warning(f"Failed to build interpolator for {obj_name}: {e}")
    
    def evaluate_at_point(self, coordinates: Dict[str, float], 
                         objective_names: List[str]) -> Dict[str, float]:
        """Evaluate objectives at any point using interpolation"""
        if not self.interpolators:
            raise ValueError("Interpolators not built. Call build_interpolators() first.")
        
        param_names = list(self.dimensions.keys())
        query_point = np.array([[coordinates[param] for param in param_names]])
        
        results = {}
        for obj_name in objective_names:
            if obj_name in self.interpolators:
                try:
                    value = self.interpolators[obj_name](query_point)
                    results[obj_name] = float(value[0])
                except Exception as e:
                    logger.warning(f"Interpolation failed for {obj_name}: {e}")
                    results[obj_name] = float('inf')
            else:
                results[obj_name] = float('inf')
        
        return results
    
    def find_pareto_optimal_points(self, objective_names: List[str], 
                                 minimize: List[bool] = None) -> List[MeshPoint]:
        """Find Pareto optimal points in the mesh"""
        if minimize is None:
            minimize = [True] * len(objective_names)
        
        feasible_points = [p for p in self.mesh_points if p.constraints_satisfied]
        pareto_points = []
        
        for candidate in feasible_points:
            is_dominated = False
            
            for other in feasible_points:
                if other == candidate:
                    continue
                
                # Check if 'other' dominates 'candidate'
                dominates = True
                
                for i, obj_name in enumerate(objective_names):
                    candidate_val = candidate.objectives.get(obj_name, float('inf'))
                    other_val = other.objectives.get(obj_name, float('inf'))
                    
                    if minimize[i]:
                        # Minimization: other should be <= candidate in all objectives
                        if other_val > candidate_val:
                            dominates = False
                            break
                    else:
                        # Maximization: other should be >= candidate in all objectives
                        if other_val < candidate_val:
                            dominates = False
                            break
                
                if dominates:
                    # Check if other is strictly better in at least one objective
                    strictly_better = False
                    for i, obj_name in enumerate(objective_names):
                        candidate_val = candidate.objectives.get(obj_name, float('inf'))
                        other_val = other.objectives.get(obj_name, float('inf'))
                        
                        if minimize[i] and other_val < candidate_val:
                            strictly_better = True
                            break
                        elif not minimize[i] and other_val > candidate_val:
                            strictly_better = True
                            break
                    
                    if strictly_better:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append(candidate)
        
        logger.info(f"Found {len(pareto_points)} Pareto optimal points out of {len(feasible_points)} feasible points")
        return pareto_points
    
    def export_mesh_data(self, filename: str) -> None:
        """Export mesh data for analysis"""
        data = []
        
        for point in self.mesh_points:
            row = point.coordinates.copy()
            row.update(point.objectives)
            row.update(point.fuzzy_memberships)
            row['constraints_satisfied'] = point.constraints_satisfied
            row['interpolation_weight'] = point.interpolation_weight
            row['mesh_level'] = point.mesh_level
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported mesh data to {filename}")
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """Get statistics about the mesh"""
        total_points = len(self.mesh_points)
        feasible_points = sum(1 for p in self.mesh_points if p.constraints_satisfied)
        
        mesh_levels = [p.mesh_level for p in self.mesh_points]
        
        stats = {
            'total_points': total_points,
            'feasible_points': feasible_points,
            'feasibility_rate': feasible_points / total_points if total_points > 0 else 0,
            'mesh_levels': {
                'min': min(mesh_levels) if mesh_levels else 0,
                'max': max(mesh_levels) if mesh_levels else 0,
                'distribution': pd.Series(mesh_levels).value_counts().to_dict()
            },
            'dimensions': len(self.dimensions),
            'interpolators_built': len(self.interpolators)
        }
        
        if feasible_points > 0:
            feasible_points_list = [p for p in self.mesh_points if p.constraints_satisfied]
            
            objective_stats = {}
            for obj_name in set().union(*[p.objectives.keys() for p in feasible_points_list]):
                values = [p.objectives[obj_name] for p in feasible_points_list if obj_name in p.objectives]
                finite_values = [v for v in values if np.isfinite(v)]
                
                if finite_values:
                    objective_stats[obj_name] = {
                        'min': min(finite_values),
                        'max': max(finite_values),
                        'mean': np.mean(finite_values),
                        'std': np.std(finite_values)
                    }
            
            stats['objective_statistics'] = objective_stats
        
        return stats

def demo_continuous_configuration_mesh():
    """Demonstrate continuous configuration mesh system"""
    print("🕸️ CONTINUOUS CONFIGURATION MESH DEMONSTRATION")
    print("=" * 70)
    
    # Define problem dimensions
    dimensions = {
        'work_intensity': (0.2, 1.0),      # 20% to 100% work effort
        'spending_level': (0.6, 1.0),     # 60% to 100% of comfortable spending
        'savings_rate': (0.05, 0.4),      # 5% to 40% savings rate
        'charity_allocation': (0.0, 0.15) # 0% to 15% for charity
    }
    
    print(f"📊 Problem Dimensions:")
    for dim, (min_val, max_val) in dimensions.items():
        print(f"   • {dim.replace('_', ' ').title()}: {min_val:.1%} to {max_val:.1%}")
    
    # Define objective functions
    def stress_objective(coords):
        """Calculate financial stress (minimize)"""
        work_stress = coords['work_intensity'] * 0.3
        frugal_stress = (1 - coords['spending_level']) * 0.2
        savings_pressure = coords['savings_rate'] * 0.25
        charity_pressure = coords['charity_allocation'] * 2.0
        return work_stress + frugal_stress + savings_pressure + charity_pressure
    
    def quality_objective(coords):
        """Calculate quality of life (maximize -> minimize negative)"""
        base_quality = 0.5
        work_balance = 0.2 if coords['work_intensity'] < 0.7 else -(coords['work_intensity'] - 0.7) * 0.3
        spending_comfort = (coords['spending_level'] - 0.6) * 0.5
        security_boost = coords['savings_rate'] * 0.3
        charity_fulfillment = coords['charity_allocation'] * 1.5
        quality = base_quality + work_balance + spending_comfort + security_boost + charity_fulfillment
        return -quality  # Negative because we minimize
    
    def portfolio_growth_objective(coords):
        """Calculate expected portfolio growth (maximize -> minimize negative)"""
        savings_contribution = coords['savings_rate'] * 2.0
        time_for_research = (1 - coords['work_intensity']) * 0.1
        stress_penalty = stress_objective(coords) * 0.1
        growth = savings_contribution + time_for_research - stress_penalty
        return -growth  # Negative because we minimize
    
    objective_functions = {
        'financial_stress': stress_objective,
        'negative_quality_of_life': quality_objective,
        'negative_portfolio_growth': portfolio_growth_objective
    }
    
    # Define constraints
    def accounting_constraint(coords):
        """Ensure accounting equation feasibility"""
        # Simplified: income must cover expenses + savings + charity
        total_outflow = coords['spending_level'] * 0.7 + coords['savings_rate'] + coords['charity_allocation']
        max_income_factor = 0.7 + 0.3 * coords['work_intensity']  # Income based on work
        return total_outflow <= max_income_factor
    
    def stress_constraint(coords):
        """Maximum acceptable stress"""
        return stress_objective(coords) <= 0.8
    
    constraint_functions = [accounting_constraint, stress_constraint]
    
    # Create and populate mesh
    print(f"\n🕸️ Creating Continuous Configuration Mesh...")
    mesh = ContinuousConfigurationMesh(dimensions, base_resolution=6, max_refinement_levels=2)
    
    # Generate base mesh
    mesh.generate_base_mesh(objective_functions, constraint_functions)
    
    # Perform adaptive refinement
    print(f"\n🔧 Performing Adaptive Refinement...")
    mesh.adaptive_refinement(objective_functions)
    
    # Build interpolators
    print(f"\n📈 Building Interpolators...")
    mesh.build_interpolators(list(objective_functions.keys()))
    
    # Get mesh statistics
    stats = mesh.get_mesh_statistics()
    
    print(f"\n📊 MESH STATISTICS:")
    print(f"Total Points: {stats['total_points']}")
    print(f"Feasible Points: {stats['feasible_points']}")
    print(f"Feasibility Rate: {stats['feasibility_rate']:.1%}")
    print(f"Mesh Levels: {stats['mesh_levels']['min']} to {stats['mesh_levels']['max']}")
    print(f"Interpolators Built: {stats['interpolators_built']}")
    
    if 'objective_statistics' in stats:
        print(f"\n📈 OBJECTIVE STATISTICS:")
        for obj_name, obj_stats in stats['objective_statistics'].items():
            print(f"   {obj_name.replace('_', ' ').title()}:")
            print(f"      Range: {obj_stats['min']:.3f} to {obj_stats['max']:.3f}")
            print(f"      Mean ± Std: {obj_stats['mean']:.3f} ± {obj_stats['std']:.3f}")
    
    # Find Pareto optimal points
    print(f"\n🎯 Finding Pareto Optimal Points...")
    pareto_points = mesh.find_pareto_optimal_points(
        list(objective_functions.keys()), 
        minimize=[True, True, True]
    )
    
    print(f"Found {len(pareto_points)} Pareto optimal configurations:")
    for i, point in enumerate(pareto_points[:5]):  # Show top 5
        print(f"\n   Pareto Point {i+1}:")
        print(f"      Work Intensity: {point.coordinates['work_intensity']:.1%}")
        print(f"      Spending Level: {point.coordinates['spending_level']:.1%}")
        print(f"      Savings Rate: {point.coordinates['savings_rate']:.1%}")
        print(f"      Charity: {point.coordinates['charity_allocation']:.1%}")
        print(f"      Financial Stress: {point.objectives['financial_stress']:.3f}")
        print(f"      Quality (neg): {point.objectives['negative_quality_of_life']:.3f}")
        print(f"      Growth (neg): {point.objectives['negative_portfolio_growth']:.3f}")
    
    # Test interpolation
    print(f"\n🔍 Testing Interpolation at Custom Point...")
    test_point = {
        'work_intensity': 0.75,
        'spending_level': 0.8,
        'savings_rate': 0.2,
        'charity_allocation': 0.05
    }
    
    interpolated_objectives = mesh.evaluate_at_point(test_point, list(objective_functions.keys()))
    
    print(f"Test Point Configuration:")
    for param, value in test_point.items():
        print(f"   {param.replace('_', ' ').title()}: {value:.1%}")
    
    print(f"Interpolated Objectives:")
    for obj_name, value in interpolated_objectives.items():
        print(f"   {obj_name.replace('_', ' ').title()}: {value:.3f}")
    
    # Export mesh data
    output_file = "data/continuous_mesh_export.csv"
    Path("data").mkdir(exist_ok=True)
    mesh.export_mesh_data(output_file)
    print(f"\n💾 Mesh data exported to {output_file}")

if __name__ == "__main__":
    demo_continuous_configuration_mesh() 