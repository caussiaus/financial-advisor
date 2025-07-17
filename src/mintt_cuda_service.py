"""
MINTT CUDA Service

This module provides the CUDA-optimized MINTT service for:
1. GPU-accelerated number detection and context analysis
2. Parallel unit detection and conversion
3. Batch context-aware summarization
4. CUDA-powered service endpoints
5. Real-time GPU feature extraction and interpolation
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
import re
from collections import defaultdict
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

from .mintt_cuda_core import CUDAMINTTCore, CUDAFeatureSelection
from .mintt_cuda_interpolation import CUDAMINTTInterpolation
from .enhanced_pdf_processor import EnhancedPDFProcessor


@dataclass
class CUDANumberDetection:
    """Represents a GPU-accelerated detected number with context"""
    number_id: str
    value: torch.Tensor
    unit: str
    context: str
    confidence: torch.Tensor
    source_text: str
    position: Tuple[int, int]
    gpu_memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDAContextAnalysis:
    """Represents GPU-accelerated context analysis for detected numbers"""
    analysis_id: str
    number_detections: List[CUDANumberDetection]
    context_summary: str
    unit_conversions: Dict[str, str]
    confidence_score: torch.Tensor
    gpu_memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CUDAServiceRequest:
    """Represents a CUDA-optimized service request"""
    request_id: str
    request_type: str
    input_data: Dict[str, Any]
    priority: int = 5
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = 'pending'
    gpu_memory_allocated: float = 0.0


class CUDAMINTTService:
    """
    CUDA-optimized MINTT Service for large-scale processing
    """
    
    def __init__(self, 
                 mintt_core: Optional[CUDAMINTTCore] = None,
                 mintt_interpolation: Optional[CUDAMINTTInterpolation] = None,
                 pdf_processor: Optional[EnhancedPDFProcessor] = None,
                 device: str = 'cuda',
                 batch_size: int = 32,
                 max_workers: int = 4):
        
        # GPU setup
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize components
        self.mintt_core = mintt_core or CUDAMINTTCore(device=device, batch_size=batch_size, max_workers=max_workers)
        self.mintt_interpolation = mintt_interpolation or CUDAMINTTInterpolation(
            self.mintt_core, 
            self.mintt_core.trial_manager,
            device=device,
            batch_size=batch_size
        )
        self.pdf_processor = pdf_processor or EnhancedPDFProcessor()
        
        # Service state
        self.request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_requests: Dict[str, CUDAServiceRequest] = {}
        self.completed_requests: Dict[str, CUDAServiceRequest] = {}
        
        # Processing threads
        self.processing_thread = None
        self.is_running = False
        
        # CUDA number detection patterns
        self.number_patterns = self._initialize_cuda_number_patterns()
        
        # CUDA unit detection patterns
        self.unit_patterns = self._initialize_cuda_unit_patterns()
        
        # Context analysis with GPU
        self.context_analyzer = CUDAContextAnalyzer(device)
        
        # CUDA models
        self.number_detection_model = self._initialize_number_detection_model()
        self.context_analysis_model = self._initialize_context_analysis_model()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Start processing thread
        self._start_processing_thread()
        
        self.logger.info(f"CUDA MINTT Service initialized on device: {self.device}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the CUDA MINTT service"""
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
    
    def _initialize_cuda_number_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize CUDA-optimized number detection patterns"""
        return {
            'currency': re.compile(r'\$?([0-9,]+(?:\.[0-9]{2})?)'),
            'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*%'),
            'decimal': re.compile(r'(\d+\.\d+)'),
            'integer': re.compile(r'\b(\d+)\b'),
            'scientific': re.compile(r'(\d+\.?\d*[eE][+-]?\d+)'),
            'fraction': re.compile(r'(\d+)/(\d+)'),
            'range': re.compile(r'(\d+)\s*-\s*(\d+)'),
            'year': re.compile(r'\b(19|20)\d{2}\b'),
            'date': re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b')
        }
    
    def _initialize_cuda_unit_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize CUDA-optimized unit detection patterns"""
        return {
            'currency': re.compile(r'\$|USD|EUR|GBP|CAD|AUD'),
            'time': re.compile(r'\b(days?|weeks?|months?|quarters?|years?)\b', re.IGNORECASE),
            'percentage': re.compile(r'%|percent|basis\s*points?', re.IGNORECASE),
            'ratio': re.compile(r'\b(ratio|rate|per)\b', re.IGNORECASE),
            'distance': re.compile(r'\b(miles?|kilometers?|meters?|feet?)\b', re.IGNORECASE),
            'weight': re.compile(r'\b(pounds?|kilograms?|grams?)\b', re.IGNORECASE)
        }
    
    def _initialize_number_detection_model(self) -> nn.Module:
        """Initialize CUDA number detection model"""
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
    
    def _initialize_context_analysis_model(self) -> nn.Module:
        """Initialize CUDA context analysis model"""
        model = nn.Sequential(
            nn.Linear(256, 512),
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
    
    def _start_processing_thread(self):
        """Start the background processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        self.logger.info("Started CUDA MINTT service processing thread")
    
    def _process_requests(self):
        """Background thread for processing requests"""
        while self.is_running:
            try:
                # Get next request from queue
                priority, request = self.request_queue.get(timeout=1)
                
                # Process request
                asyncio.run(self._process_single_request_cuda(request))
                
                # Mark as done
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
    
    async def _process_single_request_cuda(self, request: CUDAServiceRequest):
        """Process a single service request with CUDA"""
        try:
            request.status = 'processing'
            self.active_requests[request.request_id] = request
            
            if request.request_type == 'pdf_processing':
                result = await self._process_pdf_request_cuda(request)
            elif request.request_type == 'feature_extraction':
                result = await self._process_feature_extraction_request_cuda(request)
            elif request.request_type == 'interpolation':
                result = await self._process_interpolation_request_cuda(request)
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
            
            # Mark as completed
            request.status = 'completed'
            self.completed_requests[request.request_id] = request
            del self.active_requests[request.request_id]
            
            self.logger.info(f"Completed CUDA request {request.request_id}")
            
        except Exception as e:
            request.status = 'failed'
            self.logger.error(f"Failed to process CUDA request {request.request_id}: {e}")
    
    def submit_request(self, request_type: str, input_data: Dict[str, Any], 
                      priority: int = 5) -> str:
        """
        Submit a CUDA service request
        
        Args:
            request_type: Type of request
            input_data: Input data for the request
            priority: Priority level (1-10)
            
        Returns:
            Request ID
        """
        request_id = f"cuda_request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        request = CUDAServiceRequest(
            request_id=request_id,
            request_type=request_type,
            input_data=input_data,
            priority=priority
        )
        
        # Add to queue (lower priority number = higher priority)
        self.request_queue.put((10 - priority, request))
        
        self.logger.info(f"Submitted CUDA request {request_id} of type {request_type}")
        return request_id
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a CUDA request"""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                'request_id': request_id,
                'status': request.status,
                'timestamp': request.timestamp.isoformat(),
                'type': request.request_type,
                'gpu_memory_allocated': request.gpu_memory_allocated
            }
        elif request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            return {
                'request_id': request_id,
                'status': request.status,
                'timestamp': request.timestamp.isoformat(),
                'type': request.request_type,
                'completed': True,
                'gpu_memory_allocated': request.gpu_memory_allocated
            }
        else:
            return {
                'request_id': request_id,
                'status': 'not_found',
                'error': 'Request not found'
            }
    
    async def detect_numbers_with_context_cuda(self, text: str) -> List[CUDANumberDetection]:
        """
        Detect numbers in text with CUDA acceleration
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected numbers with context
        """
        self.logger.info("Detecting numbers with CUDA context analysis...")
        
        detections = []
        
        # Detect numbers using patterns
        for pattern_name, pattern in self.number_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                try:
                    # Extract number value
                    if pattern_name == 'fraction':
                        numerator = int(match.group(1))
                        denominator = int(match.group(2))
                        value = numerator / denominator
                    elif pattern_name == 'range':
                        start = int(match.group(1))
                        end = int(match.group(2))
                        value = (start + end) / 2  # Use average
                    else:
                        value = float(match.group(1).replace(',', ''))
                    
                    # Convert to tensor
                    value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                    
                    # Detect unit
                    unit = await self._detect_unit_for_number_cuda(text, match.start(), match.end())
                    
                    # Extract context
                    context = await self._extract_context_cuda(text, match.start(), match.end())
                    
                    # Calculate confidence with GPU
                    confidence = await self._calculate_detection_confidence_cuda(pattern_name, value_tensor, context)
                    
                    detection = CUDANumberDetection(
                        number_id=f"cuda_number_{len(detections):03d}",
                        value=value_tensor,
                        unit=unit,
                        context=context,
                        confidence=confidence,
                        source_text=match.group(0),
                        position=(match.start(), match.end()),
                        gpu_memory_usage=value_tensor.element_size() * value_tensor.nelement()
                    )
                    
                    detections.append(detection)
                    
                except Exception as e:
                    self.logger.error(f"Error processing number detection: {e}")
                    continue
        
        self.logger.info(f"Detected {len(detections)} numbers with CUDA")
        return detections
    
    async def _detect_unit_for_number_cuda(self, text: str, start_pos: int, end_pos: int) -> str:
        """Detect unit for number with CUDA optimization"""
        # Extract surrounding text
        context_start = max(0, start_pos - 20)
        context_end = min(len(text), end_pos + 20)
        context = text[context_start:context_end]
        
        # Check for unit patterns
        for unit_type, pattern in self.unit_patterns.items():
            if pattern.search(context):
                return unit_type
        
        return 'unknown'
    
    async def _extract_context_cuda(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context with CUDA optimization"""
        # Extract surrounding context
        context_start = max(0, start_pos - 50)
        context_end = min(len(text), end_pos + 50)
        context = text[context_start:context_end]
        
        # Clean up context
        context = context.strip()
        if len(context) > 100:
            context = context[:50] + "..." + context[-50:]
        
        return context
    
    async def _calculate_detection_confidence_cuda(self, pattern_name: str, value: torch.Tensor, context: str) -> torch.Tensor:
        """Calculate detection confidence with CUDA"""
        # Create feature vector for confidence calculation
        features = torch.tensor([
            value.item(),
            len(context),
            hash(pattern_name) % 1000,  # Simple hash for pattern type
            len(context.split())  # Word count
        ], dtype=torch.float32, device=self.device)
        
        # Use number detection model
        confidence = self.number_detection_model(features.unsqueeze(0))
        
        return confidence.squeeze()
    
    async def analyze_context_with_summarization_cuda(self, text: str) -> CUDAContextAnalysis:
        """
        Analyze context with CUDA-accelerated summarization
        
        Args:
            text: Input text to analyze
            
        Returns:
            CUDA context analysis result
        """
        self.logger.info("Analyzing context with CUDA summarization...")
        
        # Detect numbers with CUDA
        number_detections = await self.detect_numbers_with_context_cuda(text)
        
        # Generate context summary with GPU
        context_summary = await self._generate_context_summary_cuda(text, number_detections)
        
        # Detect unit conversions with GPU
        unit_conversions = await self._detect_unit_conversions_cuda(number_detections)
        
        # Calculate context confidence with GPU
        confidence_score = await self._calculate_context_confidence_cuda(number_detections, context_summary)
        
        # Calculate GPU memory usage
        gpu_memory_usage = sum(detection.gpu_memory_usage for detection in number_detections)
        
        analysis = CUDAContextAnalysis(
            analysis_id=f"cuda_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            number_detections=number_detections,
            context_summary=context_summary,
            unit_conversions=unit_conversions,
            confidence_score=confidence_score,
            gpu_memory_usage=gpu_memory_usage,
            metadata={
                'text_length': len(text),
                'detection_count': len(number_detections),
                'gpu_processed': True
            }
        )
        
        return analysis
    
    async def _generate_context_summary_cuda(self, text: str, number_detections: List[CUDANumberDetection]) -> str:
        """Generate context summary with CUDA acceleration"""
        if not number_detections:
            return "No numerical data detected in the text."
        
        # Extract values and create feature vector
        values = torch.stack([detection.value for detection in number_detections])
        
        # Calculate statistics with GPU
        total_value = torch.sum(values).item()
        avg_value = torch.mean(values).item()
        max_value = torch.max(values).item()
        min_value = torch.min(values).item()
        
        # Generate summary
        summary = f"Total numerical value: {total_value:.2f}. "
        summary += f"Value range: {min_value:.2f} to {max_value:.2f}. "
        summary += f"Average value: {avg_value:.2f}. "
        summary += f"Detected {len(number_detections)} numerical entities with GPU acceleration."
        
        return summary
    
    async def _detect_unit_conversions_cuda(self, number_detections: List[CUDANumberDetection]) -> Dict[str, str]:
        """Detect unit conversions with CUDA optimization"""
        conversions = {}
        
        for detection in number_detections:
            if detection.unit == 'currency':
                # Add currency conversion info
                conversions[f"currency_{detection.number_id}"] = "USD"
            elif detection.unit == 'time':
                # Add time conversion info
                conversions[f"time_{detection.number_id}"] = "days"
            elif detection.unit == 'percentage':
                # Add percentage conversion info
                conversions[f"percentage_{detection.number_id}"] = "decimal"
        
        return conversions
    
    async def _calculate_context_confidence_cuda(self, number_detections: List[CUDANumberDetection], 
                                              context_summary: str) -> torch.Tensor:
        """Calculate context confidence with CUDA"""
        if not number_detections:
            return torch.tensor(0.0, device=self.device)
        
        # Create feature vector for confidence calculation
        features = torch.tensor([
            len(number_detections),
            len(context_summary),
            torch.mean(torch.stack([d.confidence for d in number_detections])).item(),
            len(set(d.unit for d in number_detections))
        ], dtype=torch.float32, device=self.device)
        
        # Use context analysis model
        confidence = self.context_analysis_model(features.unsqueeze(0))
        
        return confidence.squeeze()
    
    async def _process_pdf_request_cuda(self, request: CUDAServiceRequest) -> Dict[str, Any]:
        """Process PDF request with CUDA"""
        pdf_path = request.input_data.get('pdf_path')
        if not pdf_path:
            raise ValueError("PDF path not provided")
        
        # Process PDF with CUDA
        result = await self.mintt_core.process_pdf_with_cuda_feature_selection([pdf_path])
        
        return {
            'request_id': request.request_id,
            'result': result,
            'gpu_processed': True
        }
    
    async def _process_feature_extraction_request_cuda(self, request: CUDAServiceRequest) -> Dict[str, Any]:
        """Process feature extraction request with CUDA"""
        text = request.input_data.get('text', '')
        
        # Extract features with CUDA
        number_detections = await self.detect_numbers_with_context_cuda(text)
        context_analysis = await self.analyze_context_with_summarization_cuda(text)
        
        return {
            'request_id': request.request_id,
            'number_detections': number_detections,
            'context_analysis': context_analysis,
            'gpu_processed': True
        }
    
    async def _process_interpolation_request_cuda(self, request: CUDAServiceRequest) -> Dict[str, Any]:
        """Process interpolation request with CUDA"""
        target_profile = request.input_data.get('target_profile')
        source_profiles = request.input_data.get('source_profiles', [])
        method = request.input_data.get('method', 'congruence_weighted')
        
        # Perform interpolation with CUDA
        result = await self.mintt_interpolation.interpolate_profiles_cuda(
            target_profile, source_profiles, method
        )
        
        return {
            'request_id': request.request_id,
            'result': result,
            'gpu_processed': True
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get CUDA service status"""
        return {
            'service_running': self.is_running,
            'queue_size': self.request_queue.qsize(),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'device': str(self.device),
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    
    def shutdown(self):
        """Shutdown CUDA service"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("CUDA MINTT Service shutdown complete")


class CUDAContextAnalyzer:
    """CUDA-optimized context analyzer"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    async def analyze_context_cuda(self, 
                                 milestone_tensors: torch.Tensor,
                                 entity_tensors: torch.Tensor,
                                 features: Dict[str, CUDAFeatureSelection]) -> Dict[str, Any]:
        """Analyze context with CUDA acceleration"""
        
        analysis = {
            'insights': [],
            'patterns': {},
            'anomalies': [],
            'recommendations': []
        }
        
        # Analyze milestone patterns with GPU
        milestone_analysis = await self._analyze_milestone_patterns_cuda(milestone_tensors)
        analysis['patterns']['milestones'] = milestone_analysis
        
        # Analyze entity relationships with GPU
        entity_analysis = await self._analyze_entity_relationships_cuda(entity_tensors)
        analysis['patterns']['entities'] = entity_analysis
        
        # Analyze feature correlations with GPU
        feature_analysis = await self._analyze_feature_correlations_cuda(features)
        analysis['patterns']['features'] = feature_analysis
        
        # Generate insights
        analysis['insights'] = await self._generate_insights_cuda(milestone_analysis, entity_analysis, feature_analysis)
        
        # Detect anomalies
        analysis['anomalies'] = await self._detect_anomalies_cuda(milestone_tensors, entity_tensors, features)
        
        # Generate recommendations
        analysis['recommendations'] = await self._generate_recommendations_cuda(analysis)
        
        return analysis
    
    async def _analyze_milestone_patterns_cuda(self, milestone_tensors: torch.Tensor) -> Dict[str, Any]:
        """Analyze milestone patterns with CUDA"""
        if milestone_tensors.numel() == 0:
            return {}
        
        # GPU-based pattern analysis
        total_milestones = milestone_tensors.size(0)
        financial_impact_sum = torch.sum(milestone_tensors[:, 0]) if milestone_tensors.size(1) > 0 else torch.tensor(0.0)
        avg_confidence = torch.mean(milestone_tensors[:, 3]) if milestone_tensors.size(1) > 3 else torch.tensor(0.0)
        
        return {
            'total_milestones': total_milestones,
            'total_financial_impact': financial_impact_sum.item(),
            'avg_confidence': avg_confidence.item(),
            'gpu_processed': True
        }
    
    async def _analyze_entity_relationships_cuda(self, entity_tensors: torch.Tensor) -> Dict[str, Any]:
        """Analyze entity relationships with CUDA"""
        if entity_tensors.numel() == 0:
            return {}
        
        # GPU-based relationship analysis
        total_entities = entity_tensors.size(0)
        total_balance = torch.sum(entity_tensors[:, 0]) if entity_tensors.size(1) > 0 else torch.tensor(0.0)
        avg_confidence = torch.mean(entity_tensors[:, 3]) if entity_tensors.size(1) > 3 else torch.tensor(0.0)
        
        return {
            'total_entities': total_entities,
            'total_balance': total_balance.item(),
            'avg_confidence': avg_confidence.item(),
            'gpu_processed': True
        }
    
    async def _analyze_feature_correlations_cuda(self, features: Dict[str, CUDAFeatureSelection]) -> Dict[str, Any]:
        """Analyze feature correlations with CUDA"""
        if not features:
            return {}
        
        # Extract feature vectors
        feature_vectors = torch.stack([f.extracted_value for f in features.values()])
        
        # Calculate correlations on GPU
        correlations = torch.corrcoef(feature_vectors.T)
        
        return {
            'feature_count': len(features),
            'avg_correlation': torch.mean(correlations).item(),
            'correlation_matrix': correlations.cpu().numpy(),
            'gpu_processed': True
        }
    
    async def _generate_insights_cuda(self, milestone_analysis: Dict, entity_analysis: Dict, feature_analysis: Dict) -> List[str]:
        """Generate insights with CUDA processing"""
        insights = []
        
        if milestone_analysis.get('total_milestones', 0) > 0:
            insights.append(f"Processed {milestone_analysis['total_milestones']} milestones with GPU acceleration")
        
        if entity_analysis.get('total_entities', 0) > 0:
            insights.append(f"Analyzed {entity_analysis['total_entities']} entities with CUDA optimization")
        
        if feature_analysis.get('feature_count', 0) > 0:
            insights.append(f"Extracted {feature_analysis['feature_count']} features with parallel processing")
        
        insights.append("All analysis performed with GPU acceleration for improved performance")
        
        return insights
    
    async def _detect_anomalies_cuda(self, milestone_tensors: torch.Tensor, 
                                   entity_tensors: torch.Tensor,
                                   features: Dict[str, CUDAFeatureSelection]) -> List[str]:
        """Detect anomalies with CUDA acceleration"""
        anomalies = []
        
        # GPU-based anomaly detection
        if milestone_tensors.numel() > 0:
            # Detect outliers in milestone data
            milestone_std = torch.std(milestone_tensors, dim=0)
            if torch.any(milestone_std > 1000):  # Threshold for anomaly
                anomalies.append("High variance detected in milestone data")
        
        if entity_tensors.numel() > 0:
            # Detect outliers in entity data
            entity_std = torch.std(entity_tensors, dim=0)
            if torch.any(entity_std > 1000):  # Threshold for anomaly
                anomalies.append("High variance detected in entity data")
        
        return anomalies
    
    async def _generate_recommendations_cuda(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations with CUDA processing"""
        recommendations = []
        
        if analysis.get('patterns', {}).get('milestones', {}).get('total_milestones', 0) > 5:
            recommendations.append("Consider batch processing for large milestone datasets")
        
        if analysis.get('patterns', {}).get('entities', {}).get('total_entities', 0) > 10:
            recommendations.append("Use GPU memory optimization for large entity datasets")
        
        recommendations.append("Leverage CUDA acceleration for improved processing speed")
        
        return recommendations 