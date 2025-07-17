"""
MINTT Interpolation System

This module implements the multiple profile interpolation system with:
1. Multiple profile ingestion from PDFs
2. Congruence triangle matching
3. Dynamic interpolation algorithms
4. Feature-based similarity matching
5. Interpolation quality assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from .mintt_core import MINTTCore, FeatureSelection, CongruenceTriangle
from .trial_people_manager import TrialPeopleManager, TrialPerson


@dataclass
class InterpolationResult:
    """Result of profile interpolation"""
    interpolation_id: str
    source_profiles: List[str]
    target_profile: str
    interpolated_features: Dict[str, Any]
    congruence_score: float
    confidence_score: float
    interpolation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CongruenceMatch:
    """Represents a congruence match between profiles"""
    match_id: str
    profile_1: str
    profile_2: str
    congruence_score: float
    matching_features: List[str]
    triangle_area: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MINTTInterpolation:
    """
    MINTT Interpolation System for multiple profile interpolation with congruence triangle matching
    """
    
    def __init__(self, mintt_core: MINTTCore, trial_manager: TrialPeopleManager):
        self.mintt_core = mintt_core
        self.trial_manager = trial_manager
        self.logger = logging.getLogger(__name__)
        
        # Interpolation storage
        self.interpolation_results: Dict[str, InterpolationResult] = {}
        self.congruence_matches: Dict[str, CongruenceMatch] = {}
        
        # Interpolation methods
        self.interpolation_methods = {
            'linear': self._linear_interpolation,
            'polynomial': self._polynomial_interpolation,
            'spline': self._spline_interpolation,
            'rbf': self._rbf_interpolation,
            'congruence_weighted': self._congruence_weighted_interpolation
        }
    
    def ingest_multiple_profiles(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest multiple profiles from PDF files
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Dictionary containing ingested profiles and analysis
        """
        self.logger.info(f"Ingesting {len(pdf_paths)} profiles from PDFs")
        
        ingested_profiles = {}
        all_features = {}
        
        for i, pdf_path in enumerate(pdf_paths):
            try:
                # Process PDF with feature selection
                pdf_result = self.mintt_core.process_pdf_with_feature_selection(pdf_path)
                
                profile_id = f"profile_{i+1:03d}"
                ingested_profiles[profile_id] = {
                    'pdf_path': pdf_path,
                    'milestones': pdf_result['milestones'],
                    'entities': pdf_result['entities'],
                    'features': pdf_result['selected_features'],
                    'normalized_features': pdf_result['normalized_features'],
                    'context_analysis': pdf_result['context_analysis'],
                    'feature_summary': pdf_result['feature_summary']
                }
                
                all_features[profile_id] = pdf_result['normalized_features']
                
                self.logger.info(f"Successfully ingested profile {profile_id}")
                
            except Exception as e:
                self.logger.error(f"Error ingesting profile from {pdf_path}: {e}")
                continue
        
        # Generate congruence triangles
        congruence_triangles = self._generate_congruence_triangles(ingested_profiles)
        
        # Create interpolation network
        interpolation_network = self._create_interpolation_network(ingested_profiles, congruence_triangles)
        
        return {
            'ingested_profiles': ingested_profiles,
            'all_features': all_features,
            'congruence_triangles': congruence_triangles,
            'interpolation_network': interpolation_network,
            'total_profiles': len(ingested_profiles)
        }
    
    def interpolate_profiles(self, target_profile_id: str, 
                           source_profile_ids: List[str],
                           interpolation_method: str = 'congruence_weighted') -> InterpolationResult:
        """
        Interpolate a target profile using source profiles
        
        Args:
            target_profile_id: ID of the target profile
            source_profile_ids: List of source profile IDs
            interpolation_method: Method to use for interpolation
            
        Returns:
            Interpolation result
        """
        self.logger.info(f"Interpolating profile {target_profile_id} using {len(source_profile_ids)} sources")
        
        # Get source profiles
        source_profiles = {}
        for profile_id in source_profile_ids:
            if profile_id in self.trial_manager.trial_people:
                source_profiles[profile_id] = self.trial_manager.trial_people[profile_id]
        
        if not source_profiles:
            raise ValueError(f"No valid source profiles found for interpolation")
        
        # Calculate congruence scores
        congruence_scores = self._calculate_congruence_scores(target_profile_id, source_profile_ids)
        
        # Perform interpolation
        if interpolation_method in self.interpolation_methods:
            interpolated_features = self.interpolation_methods[interpolation_method](
                target_profile_id, source_profiles, congruence_scores
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Calculate confidence score
        confidence_score = self._calculate_interpolation_confidence(
            source_profiles, congruence_scores, interpolated_features
        )
        
        # Create interpolation result
        result = InterpolationResult(
            interpolation_id=f"interpolation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_profiles=list(source_profile_ids),
            target_profile=target_profile_id,
            interpolated_features=interpolated_features,
            congruence_score=np.mean(list(congruence_scores.values())),
            confidence_score=confidence_score,
            interpolation_method=interpolation_method,
            metadata={
                'source_count': len(source_profile_ids),
                'congruence_scores': congruence_scores,
                'interpolation_timestamp': datetime.now().isoformat()
            }
        )
        
        self.interpolation_results[result.interpolation_id] = result
        return result
    
    def _generate_congruence_triangles(self, profiles: Dict[str, Any]) -> List[CongruenceTriangle]:
        """Generate congruence triangles from profiles"""
        self.logger.info("Generating congruence triangles...")
        
        triangles = []
        profile_ids = list(profiles.keys())
        
        # Generate all possible triangles (3-combinations)
        for i in range(len(profile_ids)):
            for j in range(i + 1, len(profile_ids)):
                for k in range(j + 1, len(profile_ids)):
                    profile_1 = profile_ids[i]
                    profile_2 = profile_ids[j]
                    profile_3 = profile_ids[k]
                    
                    # Calculate congruence scores for triangle edges
                    congruence_12 = self._calculate_profile_congruence(profiles[profile_1], profiles[profile_2])
                    congruence_13 = self._calculate_profile_congruence(profiles[profile_1], profiles[profile_3])
                    congruence_23 = self._calculate_profile_congruence(profiles[profile_2], profiles[profile_3])
                    
                    # Calculate triangle congruence score
                    triangle_congruence = (congruence_12 + congruence_13 + congruence_23) / 3
                    
                    # Calculate triangle area (using feature space)
                    triangle_area = self._calculate_triangle_area(profiles[profile_1], profiles[profile_2], profiles[profile_3])
                    
                    # Calculate centroid
                    centroid = self._calculate_triangle_centroid(profiles[profile_1], profiles[profile_2], profiles[profile_3])
                    
                    triangle = CongruenceTriangle(
                        triangle_id=f"triangle_{len(triangles):03d}",
                        vertices=[profile_1, profile_2, profile_3],
                        congruence_score=triangle_congruence,
                        triangle_area=triangle_area,
                        centroid=centroid,
                        metadata={
                            'edge_congruence_scores': {
                                f"{profile_1}-{profile_2}": congruence_12,
                                f"{profile_1}-{profile_3}": congruence_13,
                                f"{profile_2}-{profile_3}": congruence_23
                            }
                        }
                    )
                    
                    triangles.append(triangle)
        
        self.logger.info(f"Generated {len(triangles)} congruence triangles")
        return triangles
    
    def _calculate_profile_congruence(self, profile_1: Dict, profile_2: Dict) -> float:
        """Calculate congruence between two profiles"""
        features_1 = profile_1.get('normalized_features', {})
        features_2 = profile_2.get('normalized_features', {})
        
        if not features_1 or not features_2:
            return 0.0
        
        # Extract feature values for comparison
        common_features = set(features_1.keys()) & set(features_2.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate similarity for each feature type
        similarities = []
        
        for feature_id in common_features:
            feature_1 = features_1[feature_id]
            feature_2 = features_2[feature_id]
            
            if feature_1.feature_type == feature_2.feature_type:
                similarity = self._calculate_feature_similarity(feature_1, feature_2)
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Weight by feature confidence
        weighted_similarity = np.average(similarities, weights=[
            min(features_1[fid].confidence, features_2[fid].confidence)
            for fid in common_features
        ])
        
        return weighted_similarity
    
    def _calculate_feature_similarity(self, feature_1: FeatureSelection, feature_2: FeatureSelection) -> float:
        """Calculate similarity between two features"""
        if feature_1.feature_type != feature_2.feature_type:
            return 0.0
        
        if feature_1.feature_type == 'financial':
            # Compare financial amounts
            val_1 = feature_1.extracted_value
            val_2 = feature_2.extracted_value
            if val_1 == 0 and val_2 == 0:
                return 1.0
            elif val_1 == 0 or val_2 == 0:
                return 0.0
            else:
                return 1.0 / (1.0 + abs(val_1 - val_2) / max(abs(val_1), abs(val_2)))
        
        elif feature_1.feature_type == 'categorical':
            # Compare categorical values
            return 1.0 if feature_1.extracted_value == feature_2.extracted_value else 0.0
        
        elif feature_1.feature_type == 'temporal':
            # Compare temporal values
            val_1 = feature_1.extracted_value
            val_2 = feature_2.extracted_value
            if isinstance(val_1, datetime) and isinstance(val_2, datetime):
                days_diff = abs((val_1 - val_2).days)
                return 1.0 / (1.0 + days_diff / 365.0)  # Normalize by year
            return 0.0
        
        else:
            # Default similarity
            return 0.5
    
    def _calculate_triangle_area(self, profile_1: Dict, profile_2: Dict, profile_3: Dict) -> float:
        """Calculate area of triangle in feature space"""
        # Extract feature vectors
        features_1 = self._extract_feature_vector(profile_1)
        features_2 = self._extract_feature_vector(profile_2)
        features_3 = self._extract_feature_vector(profile_3)
        
        if len(features_1) != len(features_2) or len(features_2) != len(features_3):
            return 0.0
        
        # Create triangle points
        points = np.array([features_1, features_2, features_3])
        
        # Calculate area using cross product
        if len(points[0]) >= 2:
            # For 2D or higher, use determinant method
            if len(points[0]) == 2:
                # 2D triangle area
                area = 0.5 * abs(np.linalg.det(np.column_stack([points, np.ones(3)])))
            else:
                # Higher dimensional - use PCA to reduce to 2D
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)
                area = 0.5 * abs(np.linalg.det(np.column_stack([points_2d, np.ones(3)])))
        else:
            area = 0.0
        
        return area
    
    def _calculate_triangle_centroid(self, profile_1: Dict, profile_2: Dict, profile_3: Dict) -> np.ndarray:
        """Calculate centroid of triangle in feature space"""
        features_1 = self._extract_feature_vector(profile_1)
        features_2 = self._extract_feature_vector(profile_2)
        features_3 = self._extract_feature_vector(profile_3)
        
        # Ensure all vectors have same length
        max_len = max(len(features_1), len(features_2), len(features_3))
        features_1 = np.pad(features_1, (0, max_len - len(features_1)), 'constant')
        features_2 = np.pad(features_2, (0, max_len - len(features_2)), 'constant')
        features_3 = np.pad(features_3, (0, max_len - len(features_3)), 'constant')
        
        # Calculate centroid
        centroid = (features_1 + features_2 + features_3) / 3
        return centroid
    
    def _extract_feature_vector(self, profile: Dict) -> np.ndarray:
        """Extract feature vector from profile"""
        features = profile.get('normalized_features', {})
        
        if not features:
            return np.array([])
        
        # Convert features to numerical vector
        feature_values = []
        for feature in features.values():
            if isinstance(feature.extracted_value, (int, float)):
                feature_values.append(feature.extracted_value)
            elif isinstance(feature.extracted_value, datetime):
                # Convert datetime to days from epoch
                feature_values.append(feature.extracted_value.timestamp() / 86400)
            else:
                # Convert other types to numerical representation
                feature_values.append(hash(str(feature.extracted_value)) % 1000)
        
        return np.array(feature_values)
    
    def _create_interpolation_network(self, profiles: Dict[str, Any], 
                                    triangles: List[CongruenceTriangle]) -> Dict[str, Any]:
        """Create interpolation network from profiles and triangles"""
        network = {
            'nodes': list(profiles.keys()),
            'edges': [],
            'triangles': [t.triangle_id for t in triangles],
            'congruence_scores': {},
            'interpolation_paths': {}
        }
        
        # Create edges from triangles
        for triangle in triangles:
            vertices = triangle.vertices
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    edge = (vertices[i], vertices[j])
                    network['edges'].append(edge)
                    network['congruence_scores'][edge] = triangle.metadata['edge_congruence_scores'][f"{vertices[i]}-{vertices[j]}"]
        
        # Remove duplicate edges
        network['edges'] = list(set(network['edges']))
        
        return network
    
    def _calculate_congruence_scores(self, target_profile_id: str, 
                                   source_profile_ids: List[str]) -> Dict[str, float]:
        """Calculate congruence scores between target and source profiles"""
        congruence_scores = {}
        
        for source_id in source_profile_ids:
            if source_id in self.trial_manager.trial_people:
                source_person = self.trial_manager.trial_people[source_id]
                
                # Convert to profile format for congruence calculation
                source_profile = self._convert_person_to_profile(source_person)
                
                # For now, use a simple similarity based on age and income
                # In a real implementation, this would use the full feature set
                congruence_score = self._calculate_simple_congruence(source_person)
                congruence_scores[source_id] = congruence_score
        
        return congruence_scores
    
    def _convert_person_to_profile(self, person: TrialPerson) -> Dict[str, Any]:
        """Convert TrialPerson to profile format"""
        return {
            'normalized_features': {
                'age': FeatureSelection(
                    feature_id='age',
                    feature_type='numerical',
                    feature_name='age',
                    extracted_value=person.age,
                    confidence=1.0,
                    source_text=f"Age: {person.age}"
                ),
                'income': FeatureSelection(
                    feature_id='income',
                    feature_type='financial',
                    feature_name='income',
                    extracted_value=person.income,
                    confidence=1.0,
                    source_text=f"Income: {person.income}"
                ),
                'risk_tolerance': FeatureSelection(
                    feature_id='risk_tolerance',
                    feature_type='numerical',
                    feature_name='risk_tolerance',
                    extracted_value=person.risk_tolerance,
                    confidence=1.0,
                    source_text=f"Risk tolerance: {person.risk_tolerance}"
                )
            }
        }
    
    def _calculate_simple_congruence(self, person: TrialPerson) -> float:
        """Calculate simple congruence score based on basic attributes"""
        # This is a simplified version - in practice, would use full feature set
        base_score = 0.5
        
        # Adjust based on age (closer ages = higher congruence)
        age_factor = 1.0 / (1.0 + abs(person.age - 35) / 10.0)  # Assume target age 35
        
        # Adjust based on income (similar income levels = higher congruence)
        income_factor = 1.0 / (1.0 + abs(person.income - 75000) / 25000)  # Assume target income $75k
        
        # Adjust based on risk tolerance
        risk_factor = 1.0 / (1.0 + abs(person.risk_tolerance - 0.5) / 0.2)  # Assume target risk 0.5
        
        congruence_score = base_score * age_factor * income_factor * risk_factor
        return min(1.0, max(0.0, congruence_score))
    
    def _linear_interpolation(self, target_profile_id: str, 
                            source_profiles: Dict[str, TrialPerson],
                            congruence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform linear interpolation"""
        interpolated_features = {}
        
        # Get all feature keys from source profiles
        all_feature_keys = set()
        for person in source_profiles.values():
            if person.mesh_data:
                all_feature_keys.update(person.mesh_data.keys())
        
        for feature_key in all_feature_keys:
            values = []
            weights = []
            
            for profile_id, person in source_profiles.items():
                if person.mesh_data and feature_key in person.mesh_data:
                    feature_value = person.mesh_data[feature_key]
                    
                    # Handle different types of values
                    if isinstance(feature_value, (list, np.ndarray)):
                        # For array-like values, interpolate each element
                        if not values:
                            values = [[] for _ in range(len(feature_value))]
                        for i, val in enumerate(feature_value):
                            if i < len(values):
                                values[i].append(val)
                    else:
                        # For scalar values
                        values.append(feature_value)
                    
                    weights.append(congruence_scores.get(profile_id, 0.5))
            
            if values and weights:
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    
                    # Handle different value types
                    if isinstance(values[0], list):
                        # Array-like values
                        interpolated_array = []
                        for i in range(len(values[0])):
                            array_values = [v[i] if i < len(v) else 0 for v in values]
                            interpolated_element = sum(v * w for v, w in zip(array_values, normalized_weights))
                            interpolated_array.append(interpolated_element)
                        interpolated_features[feature_key] = interpolated_array
                    else:
                        # Scalar values
                        interpolated_value = sum(v * w for v, w in zip(values, normalized_weights))
                        interpolated_features[feature_key] = interpolated_value
        
        return interpolated_features
    
    def _polynomial_interpolation(self, target_profile_id: str,
                                source_profiles: Dict[str, TrialPerson],
                                congruence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform polynomial interpolation"""
        # Simplified polynomial interpolation
        return self._linear_interpolation(target_profile_id, source_profiles, congruence_scores)
    
    def _spline_interpolation(self, target_profile_id: str,
                            source_profiles: Dict[str, TrialPerson],
                            congruence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform spline interpolation"""
        # Simplified spline interpolation
        return self._linear_interpolation(target_profile_id, source_profiles, congruence_scores)
    
    def _rbf_interpolation(self, target_profile_id: str,
                          source_profiles: Dict[str, TrialPerson],
                          congruence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform RBF interpolation"""
        # Simplified RBF interpolation
        return self._linear_interpolation(target_profile_id, source_profiles, congruence_scores)
    
    def _congruence_weighted_interpolation(self, target_profile_id: str,
                                         source_profiles: Dict[str, TrialPerson],
                                         congruence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform congruence-weighted interpolation"""
        interpolated_features = {}
        
        # Get all feature keys from source profiles
        all_feature_keys = set()
        for person in source_profiles.values():
            if person.mesh_data:
                all_feature_keys.update(person.mesh_data.keys())
        
        for feature_key in all_feature_keys:
            values = []
            weights = []
            
            for profile_id, person in source_profiles.items():
                if person.mesh_data and feature_key in person.mesh_data:
                    feature_value = person.mesh_data[feature_key]
                    
                    # Handle different types of values
                    if isinstance(feature_value, (list, np.ndarray)):
                        # For array-like values, interpolate each element
                        if not values:
                            values = [[] for _ in range(len(feature_value))]
                        for i, val in enumerate(feature_value):
                            if i < len(values):
                                values[i].append(val)
                    elif isinstance(feature_value, dict):
                        # For dictionary values, skip for now (complex interpolation)
                        continue
                    else:
                        # For scalar values
                        values.append(feature_value)
                    
                    # Use congruence score as weight
                    congruence_weight = congruence_scores.get(profile_id, 0.5)
                    weights.append(congruence_weight)
            
            if values and weights:
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    
                    # Handle different value types
                    if isinstance(values[0], list):
                        # Array-like values
                        interpolated_array = []
                        for i in range(len(values[0])):
                            array_values = [v[i] if i < len(v) else 0 for v in values]
                            interpolated_element = sum(v * w for v, w in zip(array_values, normalized_weights))
                            interpolated_array.append(interpolated_element)
                        interpolated_features[feature_key] = interpolated_array
                    else:
                        # Scalar values
                        interpolated_value = sum(v * w for v, w in zip(values, normalized_weights))
                        interpolated_features[feature_key] = interpolated_value
        
        return interpolated_features
    
    def _calculate_interpolation_confidence(self, source_profiles: Dict[str, TrialPerson],
                                         congruence_scores: Dict[str, float],
                                         interpolated_features: Dict[str, Any]) -> float:
        """Calculate confidence score for interpolation"""
        if not source_profiles or not congruence_scores:
            return 0.0
        
        # Base confidence on congruence scores
        avg_congruence = np.mean(list(congruence_scores.values()))
        
        # Adjust based on number of source profiles
        profile_factor = min(1.0, len(source_profiles) / 5.0)  # More profiles = higher confidence
        
        # Adjust based on feature coverage
        total_features = sum(len(person.mesh_data or {}) for person in source_profiles.values())
        feature_factor = min(1.0, total_features / 50.0)  # More features = higher confidence
        
        confidence = avg_congruence * profile_factor * feature_factor
        return min(1.0, max(0.0, confidence))
    
    def generate_interpolation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive interpolation report"""
        report = {
            'total_interpolations': len(self.interpolation_results),
            'total_congruence_matches': len(self.congruence_matches),
            'interpolation_methods_used': {},
            'congruence_score_distribution': [],
            'confidence_score_distribution': [],
            'recent_interpolations': []
        }
        
        for result in self.interpolation_results.values():
            # Count methods
            method = result.interpolation_method
            report['interpolation_methods_used'][method] = report['interpolation_methods_used'].get(method, 0) + 1
            
            # Collect scores
            report['congruence_score_distribution'].append(result.congruence_score)
            report['confidence_score_distribution'].append(result.confidence_score)
            
            # Recent interpolations
            if len(report['recent_interpolations']) < 10:
                report['recent_interpolations'].append({
                    'interpolation_id': result.interpolation_id,
                    'target_profile': result.target_profile,
                    'source_count': len(result.source_profiles),
                    'congruence_score': result.congruence_score,
                    'confidence_score': result.confidence_score,
                    'method': result.interpolation_method
                })
        
        # Calculate statistics
        if report['congruence_score_distribution']:
            report['avg_congruence_score'] = np.mean(report['congruence_score_distribution'])
            report['std_congruence_score'] = np.std(report['congruence_score_distribution'])
        
        if report['confidence_score_distribution']:
            report['avg_confidence_score'] = np.mean(report['confidence_score_distribution'])
            report['std_confidence_score'] = np.std(report['confidence_score_distribution'])
        
        return report 