"""
MINTT Core System - Multiple INterpolation Trial Triangle

This module implements the core MINTT system for:
1. PDF generation with feature selection
2. Multiple profile ingestion and interpolation
3. Congruence triangle matching of the mesh
4. Dynamic normalization based on detected units
5. Service for number detection and context analysis

Key Features:
- Feature selection from PDF content
- Multiple profile interpolation
- Congruence triangle matching
- Dynamic unit detection and conversion
- Context-aware summarization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px

# Import existing components
from .enhanced_pdf_processor import EnhancedPDFProcessor, FinancialMilestone, FinancialEntity
from .mesh_congruence_engine import MeshCongruenceEngine
from .trial_people_manager import TrialPeopleManager, TrialPerson, InterpolatedSurface


@dataclass
class FeatureSelection:
    """Represents a selected feature from PDF content"""
    feature_id: str
    feature_type: str  # 'financial', 'temporal', 'categorical', 'numerical'
    feature_name: str
    extracted_value: Any
    confidence: float
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileInterpolation:
    """Represents interpolation between multiple profiles"""
    interpolation_id: str
    source_profiles: List[str]
    target_profile: str
    interpolation_type: str  # 'linear', 'polynomial', 'spline', 'congruence'
    interpolation_weights: Dict[str, float]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CongruenceTriangle:
    """Represents a congruence triangle in the mesh"""
    triangle_id: str
    vertices: List[str]  # Profile IDs
    congruence_score: float
    triangle_area: float
    centroid: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitConversion:
    """Represents unit conversion and normalization"""
    unit_type: str  # 'currency', 'time', 'percentage', 'ratio'
    source_unit: str
    target_unit: str
    conversion_factor: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MINTTCore:
    """
    Core MINTT system for multiple profile interpolation with congruence triangle matching
    """
    
    def __init__(self, 
                 pdf_processor: Optional[EnhancedPDFProcessor] = None,
                 trial_manager: Optional[TrialPeopleManager] = None,
                 congruence_engine: Optional[MeshCongruenceEngine] = None):
        
        # Initialize components
        self.pdf_processor = pdf_processor or EnhancedPDFProcessor()
        self.trial_manager = trial_manager or TrialPeopleManager()
        self.congruence_engine = congruence_engine or MeshCongruenceEngine()
        
        # Feature selection storage
        self.selected_features: Dict[str, FeatureSelection] = {}
        self.feature_extractors: Dict[str, Callable] = self._initialize_feature_extractors()
        
        # Profile interpolation storage
        self.interpolations: Dict[str, ProfileInterpolation] = {}
        self.congruence_triangles: Dict[str, CongruenceTriangle] = {}
        
        # Unit conversion tables
        self.unit_conversions: Dict[str, UnitConversion] = {}
        self.conversion_tables = self._initialize_conversion_tables()
        
        # Context analysis
        self.context_analyzer = ContextAnalyzer()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the MINTT core"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_feature_extractors(self) -> Dict[str, Callable]:
        """Initialize feature extractors for different types of content"""
        return {
            'financial_amount': self._extract_financial_amount,
            'temporal_event': self._extract_temporal_event,
            'categorical_profile': self._extract_categorical_profile,
            'numerical_metric': self._extract_numerical_metric,
            'risk_assessment': self._extract_risk_assessment,
            'goal_priority': self._extract_goal_priority
        }
    
    def _initialize_conversion_tables(self) -> Dict[str, Dict[str, float]]:
        """Initialize unit conversion tables"""
        return {
            'currency': {
                'USD': 1.0,
                'EUR': 0.85,
                'GBP': 0.73,
                'CAD': 1.25,
                'AUD': 1.35
            },
            'time': {
                'days': 1.0,
                'weeks': 7.0,
                'months': 30.44,
                'quarters': 91.31,
                'years': 365.25
            },
            'percentage': {
                'decimal': 1.0,
                'percentage': 0.01,
                'basis_points': 0.0001
            }
        }
    
    def process_pdf_with_feature_selection(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF with advanced feature selection
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted features and analysis
        """
        self.logger.info(f"Processing PDF with feature selection: {pdf_path}")
        
        # Step 1: Extract basic milestones and entities
        milestones, entities = self.pdf_processor.process_pdf(pdf_path)
        
        # Step 2: Perform feature selection
        selected_features = self._perform_feature_selection(milestones, entities, pdf_path)
        
        # Step 3: Detect and normalize units
        normalized_features = self._normalize_features_with_units(selected_features)
        
        # Step 4: Generate context analysis
        context_analysis = self.context_analyzer.analyze_context(milestones, entities, selected_features)
        
        # Step 5: Create feature summary
        feature_summary = self._create_feature_summary(normalized_features, context_analysis)
        
        return {
            'milestones': milestones,
            'entities': entities,
            'selected_features': selected_features,
            'normalized_features': normalized_features,
            'context_analysis': context_analysis,
            'feature_summary': feature_summary
        }
    
    def _perform_feature_selection(self, milestones: List[FinancialMilestone], 
                                 entities: List[FinancialEntity], 
                                 pdf_path: str) -> Dict[str, FeatureSelection]:
        """Perform feature selection on extracted content"""
        self.logger.info("Performing feature selection...")
        
        selected_features = {}
        
        # Extract features from milestones
        for i, milestone in enumerate(milestones):
            # Financial amount features
            if milestone.financial_impact is not None:
                feature = self._extract_financial_amount(
                    milestone.financial_impact, 
                    f"milestone_{i}_amount",
                    milestone.description
                )
                selected_features[feature.feature_id] = feature
            
            # Temporal features
            feature = self._extract_temporal_event(
                milestone.timestamp,
                f"milestone_{i}_timing",
                milestone.description
            )
            selected_features[feature.feature_id] = feature
            
            # Categorical features
            feature = self._extract_categorical_profile(
                milestone.event_type,
                f"milestone_{i}_category",
                milestone.description
            )
            selected_features[feature.feature_id] = feature
        
        # Extract features from entities
        for i, entity in enumerate(entities):
            # Numerical metrics
            for account, balance in entity.initial_balances.items():
                feature = self._extract_numerical_metric(
                    balance,
                    f"entity_{i}_{account}_balance",
                    f"Balance for {entity.name} in {account}"
                )
                selected_features[feature.feature_id] = feature
        
        self.logger.info(f"Selected {len(selected_features)} features")
        return selected_features
    
    def _extract_financial_amount(self, amount: float, feature_id: str, context: str) -> FeatureSelection:
        """Extract financial amount feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='financial',
            feature_name='financial_amount',
            extracted_value=amount,
            confidence=0.9 if amount > 0 else 0.7,
            source_text=context,
            metadata={'unit': 'USD', 'type': 'amount'}
        )
    
    def _extract_temporal_event(self, timestamp: datetime, feature_id: str, context: str) -> FeatureSelection:
        """Extract temporal event feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='temporal',
            feature_name='event_timing',
            extracted_value=timestamp,
            confidence=0.8,
            source_text=context,
            metadata={'format': 'datetime', 'type': 'timing'}
        )
    
    def _extract_categorical_profile(self, category: str, feature_id: str, context: str) -> FeatureSelection:
        """Extract categorical profile feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='categorical',
            feature_name='event_category',
            extracted_value=category,
            confidence=0.85,
            source_text=context,
            metadata={'type': 'category', 'domain': 'financial_events'}
        )
    
    def _extract_numerical_metric(self, value: float, feature_id: str, context: str) -> FeatureSelection:
        """Extract numerical metric feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='numerical',
            feature_name='balance_metric',
            extracted_value=value,
            confidence=0.9,
            source_text=context,
            metadata={'type': 'balance', 'unit': 'USD'}
        )
    
    def _extract_risk_assessment(self, risk_data: Dict, feature_id: str, context: str) -> FeatureSelection:
        """Extract risk assessment feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='risk',
            feature_name='risk_assessment',
            extracted_value=risk_data,
            confidence=0.75,
            source_text=context,
            metadata={'type': 'risk', 'assessment_method': 'qualitative'}
        )
    
    def _extract_goal_priority(self, goal_data: Dict, feature_id: str, context: str) -> FeatureSelection:
        """Extract goal priority feature"""
        return FeatureSelection(
            feature_id=feature_id,
            feature_type='goal',
            feature_name='goal_priority',
            extracted_value=goal_data,
            confidence=0.8,
            source_text=context,
            metadata={'type': 'goal', 'priority_method': 'ranking'}
        )
    
    def _normalize_features_with_units(self, features: Dict[str, FeatureSelection]) -> Dict[str, FeatureSelection]:
        """Normalize features using detected units and conversion tables"""
        self.logger.info("Normalizing features with unit detection...")
        
        normalized_features = {}
        
        for feature_id, feature in features.items():
            if feature.feature_type == 'financial':
                # Detect currency unit
                unit_info = self._detect_currency_unit(feature.source_text)
                if unit_info:
                    normalized_value = self._convert_currency(feature.extracted_value, unit_info)
                    feature.extracted_value = normalized_value
                    feature.metadata['normalized_unit'] = 'USD'
                    feature.metadata['original_unit'] = unit_info['unit']
                    feature.confidence *= unit_info['confidence']
            
            elif feature.feature_type == 'temporal':
                # Normalize time units
                unit_info = self._detect_time_unit(feature.source_text)
                if unit_info:
                    normalized_value = self._convert_time(feature.extracted_value, unit_info)
                    feature.extracted_value = normalized_value
                    feature.metadata['normalized_unit'] = 'days'
                    feature.metadata['original_unit'] = unit_info['unit']
            
            normalized_features[feature_id] = feature
        
        return normalized_features
    
    def _detect_currency_unit(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect currency unit from text"""
        currency_patterns = {
            r'\$': {'unit': 'USD', 'confidence': 0.9},
            r'€': {'unit': 'EUR', 'confidence': 0.9},
            r'£': {'unit': 'GBP', 'confidence': 0.9},
            r'CAD': {'unit': 'CAD', 'confidence': 0.8},
            r'AUD': {'unit': 'AUD', 'confidence': 0.8}
        }
        
        for pattern, unit_info in currency_patterns.items():
            if re.search(pattern, text):
                return unit_info
        
        return None
    
    def _detect_time_unit(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect time unit from text"""
        time_patterns = {
            r'\b(days?)\b': {'unit': 'days', 'confidence': 0.9},
            r'\b(weeks?)\b': {'unit': 'weeks', 'confidence': 0.9},
            r'\b(months?)\b': {'unit': 'months', 'confidence': 0.9},
            r'\b(quarters?)\b': {'unit': 'quarters', 'confidence': 0.8},
            r'\b(years?)\b': {'unit': 'years', 'confidence': 0.9}
        }
        
        for pattern, unit_info in time_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return unit_info
        
        return None
    
    def _convert_currency(self, amount: float, unit_info: Dict[str, Any]) -> float:
        """Convert currency to USD"""
        source_unit = unit_info['unit']
        if source_unit == 'USD':
            return amount
        
        conversion_rate = self.conversion_tables['currency'].get(source_unit, 1.0)
        return amount * conversion_rate
    
    def _convert_time(self, time_value: datetime, unit_info: Dict[str, Any]) -> int:
        """Convert time to days"""
        # For datetime objects, calculate days from now
        if isinstance(time_value, datetime):
            days_from_now = (time_value - datetime.now()).days
            return days_from_now
        
        return time_value
    
    def _create_feature_summary(self, features: Dict[str, FeatureSelection], 
                               context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of selected features"""
        summary = {
            'total_features': len(features),
            'feature_types': defaultdict(int),
            'confidence_distribution': [],
            'value_ranges': {},
            'context_insights': context_analysis.get('insights', [])
        }
        
        for feature in features.values():
            summary['feature_types'][feature.feature_type] += 1
            summary['confidence_distribution'].append(feature.confidence)
            
            if feature.feature_type == 'financial':
                if 'financial' not in summary['value_ranges']:
                    summary['value_ranges']['financial'] = []
                summary['value_ranges']['financial'].append(feature.extracted_value)
        
        # Calculate statistics
        if summary['confidence_distribution']:
            summary['avg_confidence'] = np.mean(summary['confidence_distribution'])
            summary['confidence_std'] = np.std(summary['confidence_distribution'])
        
        for feature_type, values in summary['value_ranges'].items():
            if values:
                summary['value_ranges'][feature_type] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return summary


class ContextAnalyzer:
    """Analyzes context from extracted features and content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_context(self, milestones: List[FinancialMilestone], 
                       entities: List[FinancialEntity], 
                       features: Dict[str, FeatureSelection]) -> Dict[str, Any]:
        """Analyze context from extracted content"""
        
        analysis = {
            'insights': [],
            'patterns': {},
            'anomalies': [],
            'recommendations': []
        }
        
        # Analyze milestone patterns
        milestone_analysis = self._analyze_milestone_patterns(milestones)
        analysis['patterns']['milestones'] = milestone_analysis
        
        # Analyze entity relationships
        entity_analysis = self._analyze_entity_relationships(entities)
        analysis['patterns']['entities'] = entity_analysis
        
        # Analyze feature correlations
        feature_analysis = self._analyze_feature_correlations(features)
        analysis['patterns']['features'] = feature_analysis
        
        # Generate insights
        analysis['insights'] = self._generate_insights(milestone_analysis, entity_analysis, feature_analysis)
        
        # Detect anomalies
        analysis['anomalies'] = self._detect_anomalies(milestones, entities, features)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_milestone_patterns(self, milestones: List[FinancialMilestone]) -> Dict[str, Any]:
        """Analyze patterns in milestones"""
        if not milestones:
            return {}
        
        analysis = {
            'total_milestones': len(milestones),
            'event_types': defaultdict(int),
            'time_distribution': {},
            'financial_impact_distribution': []
        }
        
        for milestone in milestones:
            analysis['event_types'][milestone.event_type] += 1
            
            if milestone.financial_impact:
                analysis['financial_impact_distribution'].append(milestone.financial_impact)
        
        # Calculate time distribution
        if milestones:
            dates = [m.timestamp for m in milestones]
            analysis['time_distribution'] = {
                'earliest': min(dates),
                'latest': max(dates),
                'span_days': (max(dates) - min(dates)).days
            }
        
        return analysis
    
    def _analyze_entity_relationships(self, entities: List[FinancialEntity]) -> Dict[str, Any]:
        """Analyze relationships between entities"""
        analysis = {
            'total_entities': len(entities),
            'entity_types': defaultdict(int),
            'total_balances': {},
            'relationships': []
        }
        
        for entity in entities:
            analysis['entity_types'][entity.entity_type] += 1
            
            for account, balance in entity.initial_balances.items():
                if account not in analysis['total_balances']:
                    analysis['total_balances'][account] = 0
                analysis['total_balances'][account] += balance
        
        return analysis
    
    def _analyze_feature_correlations(self, features: Dict[str, FeatureSelection]) -> Dict[str, Any]:
        """Analyze correlations between features"""
        analysis = {
            'feature_count': len(features),
            'type_distribution': defaultdict(int),
            'confidence_stats': {},
            'value_correlations': {}
        }
        
        for feature in features.values():
            analysis['type_distribution'][feature.feature_type] += 1
        
        # Calculate confidence statistics
        confidences = [f.confidence for f in features.values()]
        if confidences:
            analysis['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': min(confidences),
                'max': max(confidences)
            }
        
        return analysis
    
    def _generate_insights(self, milestone_analysis: Dict, entity_analysis: Dict, 
                          feature_analysis: Dict) -> List[str]:
        """Generate insights from analysis"""
        insights = []
        
        # Milestone insights
        if milestone_analysis.get('total_milestones', 0) > 0:
            most_common_event = max(milestone_analysis.get('event_types', {}).items(), 
                                  key=lambda x: x[1], default=(None, 0))
            if most_common_event[0]:
                insights.append(f"Most common event type: {most_common_event[0]} ({most_common_event[1]} occurrences)")
        
        # Entity insights
        if entity_analysis.get('total_entities', 0) > 0:
            insights.append(f"Found {entity_analysis['total_entities']} entities with {len(entity_analysis['total_balances'])} account types")
        
        # Feature insights
        if feature_analysis.get('feature_count', 0) > 0:
            avg_confidence = feature_analysis.get('confidence_stats', {}).get('mean', 0)
            insights.append(f"Extracted {feature_analysis['feature_count']} features with {avg_confidence:.2f} average confidence")
        
        return insights
    
    def _detect_anomalies(self, milestones: List[FinancialMilestone], 
                          entities: List[FinancialEntity], 
                          features: Dict[str, FeatureSelection]) -> List[str]:
        """Detect anomalies in the data"""
        anomalies = []
        
        # Check for unusually large financial impacts
        for milestone in milestones:
            if milestone.financial_impact and milestone.financial_impact > 1000000:  # $1M threshold
                anomalies.append(f"Large financial impact detected: ${milestone.financial_impact:,.0f} for {milestone.event_type}")
        
        # Check for low confidence features
        low_confidence_features = [f for f in features.values() if f.confidence < 0.5]
        if low_confidence_features:
            anomalies.append(f"Found {len(low_confidence_features)} features with low confidence (< 0.5)")
        
        return anomalies
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Recommendations based on insights
        if analysis.get('insights'):
            recommendations.append("Consider reviewing extracted features for accuracy")
        
        # Recommendations based on anomalies
        if analysis.get('anomalies'):
            recommendations.append("Review detected anomalies for data quality issues")
        
        # General recommendations
        recommendations.append("Use extracted features for profile interpolation")
        recommendations.append("Apply congruence triangle matching for similar profiles")
        
        return recommendations 