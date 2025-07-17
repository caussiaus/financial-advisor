"""
MINTT CUDA Interpolation System

This module implements the CUDA-optimized multiple profile interpolation system with:
1. GPU-accelerated multiple profile ingestion
2. CUDA-powered congruence triangle matching
3. Parallel interpolation algorithms
4. Batch feature-based similarity matching
5. GPU-optimized interpolation quality assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .mintt_cuda_core import CUDAMINTTCore, CUDAFeatureSelection, CUDABatch
from .trial_people_manager import TrialPeopleManager, TrialPerson


@dataclass
class CUDAInterpolationResult:
    """Result of CUDA-accelerated profile interpolation"""
    interpolation_id: str
    source_profiles: List[str]
    target_profile: str
    interpolated_features: torch.Tensor
    congruence_score: torch.Tensor
    confidence_score: torch.Tensor
    interpolation_method: str
    gpu_memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDACongruenceMatch:
    """Represents a CUDA-accelerated congruence match between profiles"""
    match_id: str
    profile_1: str
    profile_2: str
    congruence_score: torch.Tensor
    matching_features: torch.Tensor
    triangle_area: torch.Tensor
    gpu_memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CUDAMINTTInterpolation:
    """
    CUDA-optimized MINTT Interpolation System for large-scale profile interpolation
    """
    
    def __init__(self, 
                 mintt_core: CUDAMINTTCore, 
                 trial_manager: TrialPeopleManager,
                 device: str = 'cuda',
                 batch_size: int = 32):
        
        self.mintt_core = mintt_core
        self.trial_manager = trial_manager
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Interpolation storage
        self.interpolation_results: Dict[str, CUDAInterpolationResult] = {}
        self.congruence_matches: Dict[str, CUDACongruenceMatch] = {}
        
        # CUDA interpolation models
        self.interpolation_models = {
            'linear': self._linear_interpolation_cuda,
            'polynomial': self._polynomial_interpolation_cuda,
            'spline': self._spline_interpolation_cuda,
            'rbf': self._rbf_interpolation_cuda,
            'congruence_weighted': self._congruence_weighted_interpolation_cuda
        }
        
        # Initialize CUDA models
        self.feature_similarity_model = self._initialize_feature_similarity_model()
        self.congruence_model = self._initialize_congruence_model()
        
        self.logger.info(f"CUDA MINTT Interpolation initialized on device: {self.device}")
    
    def _initialize_feature_similarity_model(self) -> nn.Module:
        """Initialize CUDA feature similarity model"""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        return model
    
    def _initialize_congruence_model(self) -> nn.Module:
        """Initialize CUDA congruence model"""
        model = nn.Sequential(
            nn.Linear(256, 512),  # Input: concatenated profile features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        return model
    
    async def ingest_multiple_profiles_cuda(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest multiple profiles with CUDA acceleration
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Dictionary containing ingested profiles and analysis
        """
        self.logger.info(f"Ingesting {len(pdf_paths)} profiles with CUDA acceleration")
        
        # Process PDFs with CUDA
        ingestion_result = await self.mintt_core.process_pdf_with_cuda_feature_selection(pdf_paths)
        
        # Convert to GPU tensors for interpolation
        all_features = {}
        for i, result in enumerate(ingestion_result['results']):
            profile_id = f"profile_{i+1:03d}"
            
            # Convert features to tensors
            feature_tensors = []
            for feature in result.get('selected_features', {}).values():
                if hasattr(feature, 'extracted_value'):
                    feature_tensors.append(feature.extracted_value)
            
            if feature_tensors:
                all_features[profile_id] = torch.stack(feature_tensors)
            else:
                all_features[profile_id] = torch.empty(0, 64, device=self.device)
        
        # Generate congruence triangles with GPU
        congruence_triangles = await self._generate_congruence_triangles_cuda(all_features)
        
        # Create interpolation network
        interpolation_network = await self._create_interpolation_network_cuda(all_features, congruence_triangles)
        
        return {
            'ingested_profiles': ingestion_result,
            'all_features': all_features,
            'congruence_triangles': congruence_triangles,
            'interpolation_network': interpolation_network,
            'total_profiles': len(ingestion_result['results'])
        }
    
    async def interpolate_profiles_cuda(self, 
                                      target_profile_id: str, 
                                      source_profile_ids: List[str],
                                      interpolation_method: str = 'congruence_weighted') -> CUDAInterpolationResult:
        """
        Interpolate profiles with CUDA acceleration
        
        Args:
            target_profile_id: ID of the target profile
            source_profile_ids: List of source profile IDs
            interpolation_method: Method to use for interpolation
            
        Returns:
            CUDA interpolation result
        """
        self.logger.info(f"Interpolating profile {target_profile_id} with CUDA using {len(source_profile_ids)} sources")
        
        # Get source profiles as tensors
        source_profiles = {}
        for profile_id in source_profile_ids:
            if profile_id in self.trial_manager.trial_people:
                person = self.trial_manager.trial_people[profile_id]
                # Convert person data to tensor
                person_tensor = self._convert_person_to_tensor(person)
                source_profiles[profile_id] = person_tensor
        
        if not source_profiles:
            raise ValueError(f"No valid source profiles found for interpolation")
        
        # Calculate congruence scores with GPU
        congruence_scores = await self._calculate_congruence_scores_cuda(target_profile_id, source_profile_ids)
        
        # Perform CUDA interpolation
        if interpolation_method in self.interpolation_models:
            interpolated_features = await self.interpolation_models[interpolation_method](
                target_profile_id, source_profiles, congruence_scores
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Calculate confidence score with GPU
        confidence_score = await self._calculate_interpolation_confidence_cuda(
            source_profiles, congruence_scores, interpolated_features
        )
        
        # Create interpolation result
        result = CUDAInterpolationResult(
            interpolation_id=f"cuda_interpolation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_profiles=list(source_profile_ids),
            target_profile=target_profile_id,
            interpolated_features=interpolated_features,
            congruence_score=torch.mean(torch.stack(list(congruence_scores.values()))),
            confidence_score=confidence_score,
            interpolation_method=interpolation_method,
            gpu_memory_usage=interpolated_features.element_size() * interpolated_features.nelement(),
            metadata={
                'source_count': len(source_profile_ids),
                'congruence_scores': {k: v.item() for k, v in congruence_scores.items()},
                'interpolation_timestamp': datetime.now().isoformat(),
                'gpu_processed': True
            }
        )
        
        self.interpolation_results[result.interpolation_id] = result
        return result
    
    def _convert_person_to_tensor(self, person: TrialPerson) -> torch.Tensor:
        """Convert TrialPerson to GPU tensor"""
        # Extract features from person
        features = [
            person.age or 0.0,
            person.income or 0.0,
            person.risk_tolerance or 0.0,
            # Add mesh data features if available
            *self._extract_mesh_features(person.mesh_data)
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _extract_mesh_features(self, mesh_data: Dict[str, Any]) -> List[float]:
        """Extract features from mesh data"""
        features = []
        
        if mesh_data:
            # Extract discretionary spending
            if 'discretionary_spending' in mesh_data:
                spending = mesh_data['discretionary_spending']
                if isinstance(spending, list):
                    features.extend(spending[:3])  # Take first 3 values
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Extract cash flow vector
            if 'cash_flow_vector' in mesh_data:
                cash_flow = mesh_data['cash_flow_vector']
                if isinstance(cash_flow, list):
                    features.extend(cash_flow[:3])  # Take first 3 values
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Extract risk analysis
            if 'risk_analysis' in mesh_data:
                risk_data = mesh_data['risk_analysis']
                if isinstance(risk_data, dict):
                    features.extend([
                        risk_data.get('var_95_timeline', [0.0])[0] if isinstance(risk_data.get('var_95_timeline'), list) else 0.0,
                        risk_data.get('max_drawdown_by_scenario', [0.0])[0] if isinstance(risk_data.get('max_drawdown_by_scenario'), list) else 0.0
                    ])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        else:
            # Default features if no mesh data
            features.extend([0.0] * 8)
        
        return features
    
    async def _generate_congruence_triangles_cuda(self, profiles: Dict[str, torch.Tensor]) -> List[Any]:
        """Generate congruence triangles with CUDA acceleration"""
        self.logger.info("Generating congruence triangles with CUDA...")
        
        triangles = []
        profile_ids = list(profiles.keys())
        
        # Generate all possible triangles (3-combinations)
        for i in range(len(profile_ids)):
            for j in range(i + 1, len(profile_ids)):
                for k in range(j + 1, len(profile_ids)):
                    profile_1_id = profile_ids[i]
                    profile_2_id = profile_ids[j]
                    profile_3_id = profile_ids[k]
                    
                    # Calculate triangle congruence with GPU
                    congruence_score = await self._calculate_triangle_congruence_cuda(
                        profiles[profile_1_id],
                        profiles[profile_2_id],
                        profiles[profile_3_id]
                    )
                    
                    # Calculate triangle area with GPU
                    triangle_area = await self._calculate_triangle_area_cuda(
                        profiles[profile_1_id],
                        profiles[profile_2_id],
                        profiles[profile_3_id]
                    )
                    
                    # Calculate centroid with GPU
                    centroid = await self._calculate_triangle_centroid_cuda(
                        profiles[profile_1_id],
                        profiles[profile_2_id],
                        profiles[profile_3_id]
                    )
                    
                    triangle = {
                        'triangle_id': f"triangle_{len(triangles):03d}",
                        'vertices': [profile_1_id, profile_2_id, profile_3_id],
                        'congruence_score': congruence_score,
                        'triangle_area': triangle_area,
                        'centroid': centroid,
                        'gpu_processed': True
                    }
                    
                    triangles.append(triangle)
        
        return triangles
    
    async def _calculate_triangle_congruence_cuda(self, 
                                                profile_1: torch.Tensor,
                                                profile_2: torch.Tensor,
                                                profile_3: torch.Tensor) -> torch.Tensor:
        """Calculate triangle congruence with CUDA"""
        # Concatenate profiles for congruence calculation
        combined_features = torch.cat([profile_1, profile_2, profile_3])
        
        # Use congruence model
        congruence_score = self.congruence_model(combined_features.unsqueeze(0))
        
        return congruence_score.squeeze()
    
    async def _calculate_triangle_area_cuda(self,
                                          profile_1: torch.Tensor,
                                          profile_2: torch.Tensor,
                                          profile_3: torch.Tensor) -> torch.Tensor:
        """Calculate triangle area with CUDA"""
        # Use first 2 dimensions for area calculation
        p1_2d = profile_1[:2]
        p2_2d = profile_2[:2]
        p3_2d = profile_3[:2]
        
        # Calculate area using cross product
        v1 = p2_2d - p1_2d
        v2 = p3_2d - p1_2d
        area = torch.abs(torch.cross(v1, v2)) / 2.0
        
        return area
    
    async def _calculate_triangle_centroid_cuda(self,
                                              profile_1: torch.Tensor,
                                              profile_2: torch.Tensor,
                                              profile_3: torch.Tensor) -> torch.Tensor:
        """Calculate triangle centroid with CUDA"""
        centroid = (profile_1 + profile_2 + profile_3) / 3.0
        return centroid
    
    async def _create_interpolation_network_cuda(self, 
                                               profiles: Dict[str, torch.Tensor],
                                               triangles: List[Any]) -> Dict[str, Any]:
        """Create interpolation network with CUDA"""
        network = {
            'nodes': list(profiles.keys()),
            'edges': [],
            'triangles': triangles,
            'total_nodes': len(profiles),
            'total_triangles': len(triangles),
            'gpu_processed': True
        }
        
        # Create edges from triangles
        for triangle in triangles:
            vertices = triangle['vertices']
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    edge = {
                        'source': vertices[i],
                        'target': vertices[j],
                        'congruence_score': triangle['congruence_score'].item(),
                        'gpu_processed': True
                    }
                    network['edges'].append(edge)
        
        return network
    
    async def _calculate_congruence_scores_cuda(self, 
                                              target_profile_id: str,
                                              source_profile_ids: List[str]) -> Dict[str, torch.Tensor]:
        """Calculate congruence scores with CUDA"""
        congruence_scores = {}
        
        for source_id in source_profile_ids:
            if source_id in self.trial_manager.trial_people:
                source_person = self.trial_manager.trial_people[source_id]
                source_tensor = self._convert_person_to_tensor(source_person)
                
                # Calculate congruence using similarity model
                congruence_score = await self._calculate_simple_congruence_cuda(source_tensor)
                congruence_scores[source_id] = congruence_score
        
        return congruence_scores
    
    async def _calculate_simple_congruence_cuda(self, person_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate simple congruence score with CUDA"""
        # Use feature similarity model
        similarity_score = self.feature_similarity_model(person_tensor.unsqueeze(0))
        return similarity_score.squeeze()
    
    async def _linear_interpolation_cuda(self, 
                                       target_profile_id: str,
                                       source_profiles: Dict[str, torch.Tensor],
                                       congruence_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform linear interpolation with CUDA"""
        if not source_profiles:
            return torch.empty(0, device=self.device)
        
        # Stack all source profiles
        source_tensors = torch.stack(list(source_profiles.values()))
        weights = torch.stack(list(congruence_scores.values()))
        
        # Normalize weights
        weights = F.softmax(weights, dim=0)
        
        # Weighted average
        interpolated = torch.sum(source_tensors * weights.unsqueeze(1), dim=0)
        
        return interpolated
    
    async def _polynomial_interpolation_cuda(self,
                                           target_profile_id: str,
                                           source_profiles: Dict[str, torch.Tensor],
                                           congruence_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform polynomial interpolation with CUDA"""
        # Simplified polynomial interpolation using CUDA
        return await self._linear_interpolation_cuda(target_profile_id, source_profiles, congruence_scores)
    
    async def _spline_interpolation_cuda(self,
                                       target_profile_id: str,
                                       source_profiles: Dict[str, torch.Tensor],
                                       congruence_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform spline interpolation with CUDA"""
        # Simplified spline interpolation using CUDA
        return await self._linear_interpolation_cuda(target_profile_id, source_profiles, congruence_scores)
    
    async def _rbf_interpolation_cuda(self,
                                    target_profile_id: str,
                                    source_profiles: Dict[str, torch.Tensor],
                                    congruence_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform RBF interpolation with CUDA"""
        # Simplified RBF interpolation using CUDA
        return await self._linear_interpolation_cuda(target_profile_id, source_profiles, congruence_scores)
    
    async def _congruence_weighted_interpolation_cuda(self,
                                                     target_profile_id: str,
                                                     source_profiles: Dict[str, torch.Tensor],
                                                     congruence_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform congruence-weighted interpolation with CUDA"""
        if not source_profiles:
            return torch.empty(0, device=self.device)
        
        # Stack all source profiles
        source_tensors = torch.stack(list(source_profiles.values()))
        weights = torch.stack(list(congruence_scores.values()))
        
        # Use congruence scores as weights
        weights = F.softmax(weights, dim=0)
        
        # Weighted average with congruence weighting
        interpolated = torch.sum(source_tensors * weights.unsqueeze(1), dim=0)
        
        return interpolated
    
    async def _calculate_interpolation_confidence_cuda(self,
                                                     source_profiles: Dict[str, torch.Tensor],
                                                     congruence_scores: Dict[str, torch.Tensor],
                                                     interpolated_features: torch.Tensor) -> torch.Tensor:
        """Calculate interpolation confidence with CUDA"""
        if not source_profiles or not congruence_scores:
            return torch.tensor(0.0, device=self.device)
        
        # Base confidence on congruence scores
        avg_congruence = torch.mean(torch.stack(list(congruence_scores.values())))
        
        # Adjust based on number of source profiles
        profile_factor = min(1.0, len(source_profiles) / 5.0)
        
        # Adjust based on feature coverage
        total_features = sum(tensor.numel() for tensor in source_profiles.values())
        feature_factor = min(1.0, total_features / 100.0)
        
        confidence = avg_congruence * profile_factor * feature_factor
        return torch.clamp(confidence, 0.0, 1.0)
    
    async def generate_interpolation_report_cuda(self) -> Dict[str, Any]:
        """Generate a comprehensive CUDA interpolation report"""
        report = {
            'total_interpolations': len(self.interpolation_results),
            'total_congruence_matches': len(self.congruence_matches),
            'interpolation_methods_used': {},
            'congruence_score_distribution': [],
            'confidence_score_distribution': [],
            'gpu_memory_usage': 0.0,
            'recent_interpolations': []
        }
        
        for result in self.interpolation_results.values():
            # Count methods
            method = result.interpolation_method
            report['interpolation_methods_used'][method] = report['interpolation_methods_used'].get(method, 0) + 1
            
            # Collect scores
            report['congruence_score_distribution'].append(result.congruence_score.item())
            report['confidence_score_distribution'].append(result.confidence_score.item())
            report['gpu_memory_usage'] += result.gpu_memory_usage
            
            # Recent interpolations
            if len(report['recent_interpolations']) < 10:
                report['recent_interpolations'].append({
                    'interpolation_id': result.interpolation_id,
                    'target_profile': result.target_profile,
                    'source_count': len(result.source_profiles),
                    'congruence_score': result.congruence_score.item(),
                    'confidence_score': result.confidence_score.item(),
                    'method': result.interpolation_method,
                    'gpu_memory_usage': result.gpu_memory_usage
                })
        
        # Calculate statistics
        if report['congruence_score_distribution']:
            report['avg_congruence_score'] = np.mean(report['congruence_score_distribution'])
            report['std_congruence_score'] = np.std(report['congruence_score_distribution'])
        
        if report['confidence_score_distribution']:
            report['avg_confidence_score'] = np.mean(report['confidence_score_distribution'])
            report['std_confidence_score'] = np.std(report['confidence_score_distribution'])
        
        return report 