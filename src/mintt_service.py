"""
MINTT Service

This module provides the MINTT service for:
1. Number detection and context analysis
2. Dynamic unit detection and conversion
3. Context-aware summarization
4. Service endpoints for PDF processing
5. Real-time feature extraction and interpolation
"""

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
from concurrent.futures import ThreadPoolExecutor
import queue

from .mintt_core import MINTTCore, FeatureSelection, ContextAnalyzer
from .mintt_interpolation import MINTTInterpolation
from .enhanced_pdf_processor import EnhancedPDFProcessor


@dataclass
class NumberDetection:
    """Represents a detected number with context"""
    number_id: str
    value: float
    unit: str
    context: str
    confidence: float
    source_text: str
    position: Tuple[int, int]  # (start, end) in source text
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextAnalysis:
    """Represents context analysis for detected numbers"""
    analysis_id: str
    number_detections: List[NumberDetection]
    context_summary: str
    unit_conversions: Dict[str, str]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceRequest:
    """Represents a service request"""
    request_id: str
    request_type: str  # 'pdf_processing', 'feature_extraction', 'interpolation'
    input_data: Dict[str, Any]
    priority: int = 5  # 1-10, 10 being highest
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = 'pending'  # pending, processing, completed, failed


class MINTTService:
    """
    MINTT Service for number detection and context analysis
    """
    
    def __init__(self, 
                 mintt_core: Optional[MINTTCore] = None,
                 mintt_interpolation: Optional[MINTTInterpolation] = None,
                 pdf_processor: Optional[EnhancedPDFProcessor] = None):
        
        # Initialize components
        self.mintt_core = mintt_core or MINTTCore()
        self.mintt_interpolation = mintt_interpolation or MINTTInterpolation(
            self.mintt_core, 
            self.mintt_core.trial_manager
        )
        self.pdf_processor = pdf_processor or EnhancedPDFProcessor()
        
        # Service state
        self.request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_requests: Dict[str, ServiceRequest] = {}
        self.completed_requests: Dict[str, ServiceRequest] = {}
        
        # Processing threads
        self.processing_thread = None
        self.is_running = False
        
        # Number detection patterns
        self.number_patterns = self._initialize_number_patterns()
        
        # Unit detection patterns
        self.unit_patterns = self._initialize_unit_patterns()
        
        # Context analysis
        self.context_analyzer = ContextAnalyzer()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Start processing thread
        self._start_processing_thread()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the MINTT service"""
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
    
    def _initialize_number_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize number detection patterns"""
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
    
    def _initialize_unit_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize unit detection patterns"""
        return {
            'currency': re.compile(r'\$|USD|EUR|GBP|CAD|AUD'),
            'time': re.compile(r'\b(days?|weeks?|months?|quarters?|years?)\b', re.IGNORECASE),
            'percentage': re.compile(r'%|percent|basis\s*points?', re.IGNORECASE),
            'ratio': re.compile(r'\b(ratio|rate|per)\b', re.IGNORECASE),
            'distance': re.compile(r'\b(miles?|kilometers?|meters?|feet?)\b', re.IGNORECASE),
            'weight': re.compile(r'\b(pounds?|kilograms?|grams?)\b', re.IGNORECASE)
        }
    
    def _start_processing_thread(self):
        """Start the background processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        self.logger.info("Started MINTT service processing thread")
    
    def _process_requests(self):
        """Background thread for processing requests"""
        while self.is_running:
            try:
                # Get next request from queue
                priority, request = self.request_queue.get(timeout=1)
                
                # Process request
                self._process_single_request(request)
                
                # Mark as done
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
    
    def _process_single_request(self, request: ServiceRequest):
        """Process a single service request"""
        try:
            request.status = 'processing'
            self.active_requests[request.request_id] = request
            
            if request.request_type == 'pdf_processing':
                result = self._process_pdf_request(request)
            elif request.request_type == 'feature_extraction':
                result = self._process_feature_extraction_request(request)
            elif request.request_type == 'interpolation':
                result = self._process_interpolation_request(request)
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
            
            # Mark as completed
            request.status = 'completed'
            self.completed_requests[request.request_id] = request
            del self.active_requests[request.request_id]
            
            self.logger.info(f"Completed request {request.request_id}")
            
        except Exception as e:
            request.status = 'failed'
            self.logger.error(f"Failed to process request {request.request_id}: {e}")
    
    def submit_request(self, request_type: str, input_data: Dict[str, Any], 
                      priority: int = 5) -> str:
        """
        Submit a service request
        
        Args:
            request_type: Type of request
            input_data: Input data for the request
            priority: Priority level (1-10)
            
        Returns:
            Request ID
        """
        request_id = f"request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        request = ServiceRequest(
            request_id=request_id,
            request_type=request_type,
            input_data=input_data,
            priority=priority
        )
        
        # Add to queue (lower priority number = higher priority)
        self.request_queue.put((10 - priority, request))
        
        self.logger.info(f"Submitted request {request_id} of type {request_type}")
        return request_id
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a request"""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            return {
                'request_id': request_id,
                'status': request.status,
                'timestamp': request.timestamp.isoformat(),
                'type': request.request_type
            }
        elif request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            return {
                'request_id': request_id,
                'status': request.status,
                'timestamp': request.timestamp.isoformat(),
                'type': request.request_type,
                'completed': True
            }
        else:
            return {
                'request_id': request_id,
                'status': 'not_found',
                'error': 'Request not found'
            }
    
    def detect_numbers_with_context(self, text: str) -> List[NumberDetection]:
        """
        Detect numbers in text with context analysis
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected numbers with context
        """
        self.logger.info("Detecting numbers with context...")
        
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
                    
                    # Detect unit
                    unit = self._detect_unit_for_number(text, match.start(), match.end())
                    
                    # Extract context
                    context = self._extract_context(text, match.start(), match.end())
                    
                    # Calculate confidence
                    confidence = self._calculate_detection_confidence(pattern_name, value, context)
                    
                    detection = NumberDetection(
                        number_id=f"number_{len(detections):03d}",
                        value=value,
                        unit=unit,
                        context=context,
                        confidence=confidence,
                        source_text=match.group(0),
                        position=(match.start(), match.end()),
                        metadata={
                            'pattern_type': pattern_name,
                            'raw_match': match.group(0)
                        }
                    )
                    
                    detections.append(detection)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing number match: {e}")
                    continue
        
        self.logger.info(f"Detected {len(detections)} numbers with context")
        return detections
    
    def _detect_unit_for_number(self, text: str, start_pos: int, end_pos: int) -> str:
        """Detect unit for a detected number"""
        # Look for units near the number
        context_start = max(0, start_pos - 50)
        context_end = min(len(text), end_pos + 50)
        context = text[context_start:context_end]
        
        # Check for currency units
        if re.search(r'\$', context):
            return 'USD'
        elif re.search(r'€', context):
            return 'EUR'
        elif re.search(r'£', context):
            return 'GBP'
        
        # Check for time units
        time_match = re.search(r'\b(days?|weeks?|months?|quarters?|years?)\b', context, re.IGNORECASE)
        if time_match:
            return time_match.group(1).lower()
        
        # Check for percentage
        if re.search(r'%', context):
            return 'percentage'
        
        # Default to dimensionless
        return 'dimensionless'
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int) -> str:
        """Extract context around a detected number"""
        # Extract surrounding sentence
        context_start = max(0, start_pos - 200)
        context_end = min(len(text), end_pos + 200)
        context = text[context_start:context_end]
        
        # Find sentence boundaries
        sentence_start = context.rfind('.', 0, start_pos - context_start)
        sentence_end = context.find('.', end_pos - context_start)
        
        if sentence_start == -1:
            sentence_start = 0
        if sentence_end == -1:
            sentence_end = len(context)
        
        sentence = context[sentence_start:sentence_end].strip()
        
        # Clean up sentence
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()
        
        return sentence
    
    def _calculate_detection_confidence(self, pattern_name: str, value: float, context: str) -> float:
        """Calculate confidence for a number detection"""
        base_confidence = 0.8
        
        # Adjust based on pattern type
        pattern_confidence = {
            'currency': 0.9,
            'percentage': 0.85,
            'decimal': 0.8,
            'integer': 0.7,
            'scientific': 0.9,
            'fraction': 0.8,
            'range': 0.7,
            'year': 0.9,
            'date': 0.9
        }
        
        base_confidence *= pattern_confidence.get(pattern_name, 0.7)
        
        # Adjust based on context quality
        if len(context) > 20:
            base_confidence *= 1.1
        elif len(context) < 10:
            base_confidence *= 0.9
        
        # Adjust based on value range
        if 0 < value < 1000000:  # Reasonable range
            base_confidence *= 1.0
        else:
            base_confidence *= 0.8
        
        return min(1.0, max(0.0, base_confidence))
    
    def analyze_context_with_summarization(self, text: str) -> ContextAnalysis:
        """
        Analyze context with summarization
        
        Args:
            text: Input text to analyze
            
        Returns:
            Context analysis with summarization
        """
        self.logger.info("Analyzing context with summarization...")
        
        # Detect numbers
        number_detections = self.detect_numbers_with_context(text)
        
        # Generate context summary
        context_summary = self._generate_context_summary(text, number_detections)
        
        # Detect unit conversions
        unit_conversions = self._detect_unit_conversions(number_detections)
        
        # Calculate overall confidence
        confidence_score = self._calculate_context_confidence(number_detections, context_summary)
        
        analysis = ContextAnalysis(
            analysis_id=f"context_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            number_detections=number_detections,
            context_summary=context_summary,
            unit_conversions=unit_conversions,
            confidence_score=confidence_score,
            metadata={
                'text_length': len(text),
                'detection_count': len(number_detections),
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
        
        return analysis
    
    def _generate_context_summary(self, text: str, number_detections: List[NumberDetection]) -> str:
        """Generate context summary from text and number detections"""
        if not number_detections:
            return "No numerical data detected in the text."
        
        # Group detections by unit
        unit_groups = defaultdict(list)
        for detection in number_detections:
            unit_groups[detection.unit].append(detection)
        
        summary_parts = []
        
        # Summary by unit type
        for unit, detections in unit_groups.items():
            if unit == 'USD':
                total_value = sum(d.value for d in detections)
                summary_parts.append(f"Total financial value: ${total_value:,.2f}")
            elif unit == 'percentage':
                avg_percentage = np.mean([d.value for d in detections])
                summary_parts.append(f"Average percentage: {avg_percentage:.1f}%")
            elif unit in ['days', 'weeks', 'months', 'years']:
                total_time = sum(d.value for d in detections)
                summary_parts.append(f"Total time period: {total_time:.1f} {unit}")
            else:
                count = len(detections)
                summary_parts.append(f"Found {count} {unit} values")
        
        # Add key insights
        if len(number_detections) > 0:
            max_value = max(d.value for d in number_detections)
            min_value = min(d.value for d in number_detections)
            summary_parts.append(f"Value range: {min_value:.2f} to {max_value:.2f}")
        
        return ". ".join(summary_parts)
    
    def _detect_unit_conversions(self, number_detections: List[NumberDetection]) -> Dict[str, str]:
        """Detect unit conversions needed"""
        conversions = {}
        
        # Group by similar units
        unit_groups = defaultdict(list)
        for detection in number_detections:
            unit_groups[detection.unit].append(detection)
        
        # Check for mixed units that need conversion
        if len(unit_groups) > 1:
            # If we have multiple currency units, suggest USD conversion
            currency_units = [unit for unit in unit_groups.keys() if unit in ['USD', 'EUR', 'GBP', 'CAD', 'AUD']]
            if len(currency_units) > 1:
                for unit in currency_units:
                    if unit != 'USD':
                        conversions[unit] = 'USD'
        
        return conversions
    
    def _calculate_context_confidence(self, number_detections: List[NumberDetection], 
                                   context_summary: str) -> float:
        """Calculate confidence score for context analysis"""
        if not number_detections:
            return 0.0
        
        # Base confidence on number of detections
        detection_confidence = min(1.0, len(number_detections) / 10.0)
        
        # Average confidence of detections
        avg_detection_confidence = np.mean([d.confidence for d in number_detections])
        
        # Context summary quality
        summary_confidence = min(1.0, len(context_summary) / 100.0)
        
        # Overall confidence
        overall_confidence = (detection_confidence + avg_detection_confidence + summary_confidence) / 3
        
        return overall_confidence
    
    def _process_pdf_request(self, request: ServiceRequest) -> Dict[str, Any]:
        """Process PDF processing request"""
        pdf_path = request.input_data.get('pdf_path')
        if not pdf_path:
            raise ValueError("PDF path not provided")
        
        # Process PDF with feature selection
        result = self.mintt_core.process_pdf_with_feature_selection(pdf_path)
        
        return {
            'request_id': request.request_id,
            'status': 'completed',
            'result': result
        }
    
    def _process_feature_extraction_request(self, request: ServiceRequest) -> Dict[str, Any]:
        """Process feature extraction request"""
        text = request.input_data.get('text', '')
        
        # Detect numbers and analyze context
        number_detections = self.detect_numbers_with_context(text)
        context_analysis = self.analyze_context_with_summarization(text)
        
        return {
            'request_id': request.request_id,
            'status': 'completed',
            'number_detections': [self._number_detection_to_dict(d) for d in number_detections],
            'context_analysis': self._context_analysis_to_dict(context_analysis)
        }
    
    def _process_interpolation_request(self, request: ServiceRequest) -> Dict[str, Any]:
        """Process interpolation request"""
        target_profile = request.input_data.get('target_profile')
        source_profiles = request.input_data.get('source_profiles', [])
        method = request.input_data.get('method', 'congruence_weighted')
        
        if not target_profile or not source_profiles:
            raise ValueError("Target profile and source profiles required")
        
        # Perform interpolation
        result = self.mintt_interpolation.interpolate_profiles(
            target_profile, source_profiles, method
        )
        
        return {
            'request_id': request.request_id,
            'status': 'completed',
            'interpolation_result': self._interpolation_result_to_dict(result)
        }
    
    def _number_detection_to_dict(self, detection: NumberDetection) -> Dict[str, Any]:
        """Convert NumberDetection to dictionary"""
        return {
            'number_id': detection.number_id,
            'value': detection.value,
            'unit': detection.unit,
            'context': detection.context,
            'confidence': detection.confidence,
            'source_text': detection.source_text,
            'position': detection.position,
            'metadata': detection.metadata
        }
    
    def _context_analysis_to_dict(self, analysis: ContextAnalysis) -> Dict[str, Any]:
        """Convert ContextAnalysis to dictionary"""
        return {
            'analysis_id': analysis.analysis_id,
            'number_detections': [self._number_detection_to_dict(d) for d in analysis.number_detections],
            'context_summary': analysis.context_summary,
            'unit_conversions': analysis.unit_conversions,
            'confidence_score': analysis.confidence_score,
            'metadata': analysis.metadata
        }
    
    def _interpolation_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert InterpolationResult to dictionary"""
        return {
            'interpolation_id': result.interpolation_id,
            'source_profiles': result.source_profiles,
            'target_profile': result.target_profile,
            'interpolated_features': result.interpolated_features,
            'congruence_score': result.congruence_score,
            'confidence_score': result.confidence_score,
            'interpolation_method': result.interpolation_method,
            'metadata': result.metadata
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        return {
            'service_running': self.is_running,
            'queue_size': self.request_queue.qsize(),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'processing_thread_alive': self.processing_thread.is_alive() if self.processing_thread else False
        }
    
    def shutdown(self):
        """Shutdown the service"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("MINTT service shutdown complete") 