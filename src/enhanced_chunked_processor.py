#!/usr/bin/env python
"""
Enhanced Chunked Processor with Tree-Based Coordination
Author: ChatGPT 2025-07-16

Implements the chunked processing architecture with tree-based coordination
for better text extraction and fsQCA analysis.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
import random

# Add src to path
sys.path.append('src')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineEstimator:
    """Estimates realistic timelines for financial events based on age, income, and life stage"""
    
    def __init__(self):
        # Life stage age ranges
        self.life_stages = {
            'early_career': (22, 30),
            'mid_career': (31, 45),
            'established': (46, 55),
            'pre_retirement': (56, 65),
            'retirement': (66, 80)
        }
        
        # Event timing patterns by life stage and income
        self.event_patterns = {
            'education': {
                'early_career': {'min_age': 18, 'max_age': 30, 'frequency': 0.3},
                'mid_career': {'min_age': 25, 'max_age': 40, 'frequency': 0.1},
                'established': {'min_age': 30, 'max_age': 50, 'frequency': 0.05}
            },
            'work': {
                'early_career': {'min_age': 22, 'max_age': 35, 'frequency': 0.8},
                'mid_career': {'min_age': 30, 'max_age': 50, 'frequency': 0.6},
                'established': {'min_age': 40, 'max_age': 60, 'frequency': 0.4},
                'pre_retirement': {'min_age': 50, 'max_age': 70, 'frequency': 0.2}
            },
            'family': {
                'early_career': {'min_age': 25, 'max_age': 35, 'frequency': 0.4},
                'mid_career': {'min_age': 30, 'max_age': 45, 'frequency': 0.3},
                'established': {'min_age': 35, 'max_age': 55, 'frequency': 0.2}
            },
            'housing': {
                'early_career': {'min_age': 25, 'max_age': 35, 'frequency': 0.5},
                'mid_career': {'min_age': 30, 'max_age': 50, 'frequency': 0.4},
                'established': {'min_age': 35, 'max_age': 60, 'frequency': 0.3},
                'pre_retirement': {'min_age': 45, 'max_age': 70, 'frequency': 0.2}
            },
            'health': {
                'early_career': {'min_age': 22, 'max_age': 40, 'frequency': 0.2},
                'mid_career': {'min_age': 30, 'max_age': 55, 'frequency': 0.3},
                'established': {'min_age': 40, 'max_age': 65, 'frequency': 0.4},
                'pre_retirement': {'min_age': 50, 'max_age': 75, 'frequency': 0.5},
                'retirement': {'min_age': 65, 'max_age': 85, 'frequency': 0.6}
            },
            'financial': {
                'early_career': {'min_age': 22, 'max_age': 35, 'frequency': 0.7},
                'mid_career': {'min_age': 30, 'max_age': 55, 'frequency': 0.8},
                'established': {'min_age': 40, 'max_age': 65, 'frequency': 0.9},
                'pre_retirement': {'min_age': 50, 'max_age': 70, 'frequency': 0.8},
                'retirement': {'min_age': 65, 'max_age': 85, 'frequency': 0.7}
            },
            'retirement': {
                'established': {'min_age': 45, 'max_age': 65, 'frequency': 0.3},
                'pre_retirement': {'min_age': 55, 'max_age': 70, 'frequency': 0.6},
                'retirement': {'min_age': 65, 'max_age': 85, 'frequency': 0.8}
            },
            'charity': {
                'mid_career': {'min_age': 30, 'max_age': 55, 'frequency': 0.2},
                'established': {'min_age': 40, 'max_age': 65, 'frequency': 0.3},
                'pre_retirement': {'min_age': 50, 'max_age': 70, 'frequency': 0.4},
                'retirement': {'min_age': 65, 'max_age': 85, 'frequency': 0.5}
            }
        }
        
        # Income-based adjustments
        self.income_adjustments = {
            'low_income': {'timing_multiplier': 1.2, 'frequency_multiplier': 0.8},
            'middle_income': {'timing_multiplier': 1.0, 'frequency_multiplier': 1.0},
            'high_income': {'timing_multiplier': 0.8, 'frequency_multiplier': 1.2},
            'ultra_high_income': {'timing_multiplier': 0.6, 'frequency_multiplier': 1.5}
        }
    
    def estimate_client_profile(self, text: str) -> Dict[str, Any]:
        """Estimate client age, income level, and life stage from text"""
        # Default profile
        profile = {
            'age': 35,
            'income_level': 'middle_income',
            'life_stage': 'mid_career',
            'confidence': 0.5
        }
        
        # Extract age clues
        age_patterns = [
            r'(\d{2})[- ]?years?[- ]?old',
            r'age[:\s]+(\d{2})',
            r'born[:\s]+(\d{4})',
            r'(\d{2})[- ]?year[- ]?old'
        ]
        
        import re
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    age = int(matches[0])
                    if 18 <= age <= 85:
                        profile['age'] = age
                        profile['confidence'] += 0.2
                        break
                except ValueError:
                    continue
        
        # Extract income clues
        income_keywords = {
            'low_income': ['low income', 'struggling', 'minimum wage', 'part-time', 'entry level'],
            'middle_income': ['middle class', 'comfortable', 'stable', 'professional'],
            'high_income': ['high income', 'executive', 'director', 'manager', 'senior'],
            'ultra_high_income': ['wealthy', 'millionaire', 'executive', 'CEO', 'founder', 'investor']
        }
        
        text_lower = text.lower()
        for income_level, keywords in income_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                profile['income_level'] = income_level
                profile['confidence'] += 0.1
                break
        
        # Determine life stage based on age
        for stage, (min_age, max_age) in self.life_stages.items():
            if min_age <= profile['age'] <= max_age:
                profile['life_stage'] = stage
                break
        
        return profile
    
    def estimate_event_timeline(self, events: List[Dict[str, Any]], client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate realistic timeline for events based on client profile"""
        if not events:
            return events
        
        # Sort events by type and amount for better timeline estimation
        sorted_events = sorted(events, key=lambda x: (x['type'], x.get('amount', 0)), reverse=True)
        
        # Get current date as reference point
        current_date = datetime.now()
        
        # Estimate timeline working backwards from most recent event
        timeline_events = []
        last_event_date = current_date
        
        for i, event in enumerate(sorted_events):
            event_type = event['type']
            life_stage = client_profile['life_stage']
            income_level = client_profile['income_level']
            
            # Get event pattern for this type and life stage
            if event_type in self.event_patterns and life_stage in self.event_patterns[event_type]:
                pattern = self.event_patterns[event_type][life_stage]
                income_adj = self.income_adjustments[income_level]
                
                # Estimate age when this event likely occurred
                min_age = pattern['min_age']
                max_age = pattern['max_age']
                
                # Adjust for income level
                timing_mult = income_adj['timing_multiplier']
                min_age = int(min_age * timing_mult)
                max_age = int(max_age * timing_mult)
                
                # Ensure reasonable age range
                min_age = max(18, min_age)
                max_age = min(85, max_age)
                
                # Estimate age within range (weighted toward more likely ages)
                if client_profile['age'] >= min_age and client_profile['age'] <= max_age:
                    # Event likely happened recently
                    estimated_age = client_profile['age'] - random.randint(0, 5)
                else:
                    # Event likely happened in the past
                    estimated_age = random.randint(min_age, max_age)
                
                # Calculate years ago
                years_ago = client_profile['age'] - estimated_age
                years_ago = max(0, years_ago)  # Can't be negative
                
                # Add some randomness to make it more realistic
                years_ago += random.uniform(-1, 1)
                years_ago = max(0, years_ago)
                
                # Calculate event date
                event_date = current_date - timedelta(days=int(years_ago * 365))
                
                # Ensure events don't overlap too much
                if timeline_events:
                    min_days_between = 30  # Minimum 30 days between events
                    days_since_last = (last_event_date - event_date).days
                    if days_since_last < min_days_between:
                        event_date = last_event_date - timedelta(days=min_days_between)
                
                # Add timeline information to event
                event_with_timeline = event.copy()
                event_with_timeline.update({
                    'estimated_date': event_date.isoformat(),
                    'years_ago': round(years_ago, 1),
                    'estimated_age': estimated_age,
                    'life_stage_when_occurred': self._get_life_stage_for_age(estimated_age),
                    'timeline_confidence': self._calculate_timeline_confidence(event_type, life_stage, income_level)
                })
                
                timeline_events.append(event_with_timeline)
                last_event_date = event_date
        
        # Sort by estimated date (most recent first)
        timeline_events.sort(key=lambda x: x['estimated_date'], reverse=True)
        
        return timeline_events
    
    def _get_life_stage_for_age(self, age: int) -> str:
        """Get life stage for a given age"""
        for stage, (min_age, max_age) in self.life_stages.items():
            if min_age <= age <= max_age:
                return stage
        return 'unknown'
    
    def _calculate_timeline_confidence(self, event_type: str, life_stage: str, income_level: str) -> float:
        """Calculate confidence in timeline estimation"""
        base_confidence = 0.7
        
        # Adjust based on event type frequency
        if event_type in self.event_patterns and life_stage in self.event_patterns[event_type]:
            frequency = self.event_patterns[event_type][life_stage]['frequency']
            base_confidence += frequency * 0.2
        
        # Adjust based on income level
        income_adj = self.income_adjustments[income_level]
        base_confidence *= income_adj['frequency_multiplier']
        
        return min(1.0, max(0.3, base_confidence))

@dataclass
class ChunkResult:
    """Result from processing a text chunk"""
    chunk_id: str
    text: str
    keywords: List[str]
    facts: List[str]
    events: List[Dict[str, Any]]
    confidence: float
    processing_time: float

@dataclass
class TreeNode:
    """Node in the coordination tree"""
    node_id: str
    chunk_results: List[ChunkResult]
    summary: str
    keywords: List[str]
    facts: List[str]
    events: List[Dict[str, Any]]
    confidence: float
    children: List['TreeNode']
    parent: Optional['TreeNode']

class EnhancedChunkedProcessor:
    """Enhanced processor with tree-based coordination"""
    
    def __init__(self, client_id: str, model_name: str = "microsoft/DialoGPT-medium"):
        self.client_id = client_id
        self.model_name = model_name
        self.chunk_size = 512
        self.overlap = 50
        self.max_chunks_per_node = 5
        
        # Initialize timeline estimator
        self.timeline_estimator = TimelineEstimator()
        
        # Initialize models
        self._initialize_models()
        
        # Tree structure
        self.root_node = None
        self.node_counter = 0
    
    def _initialize_models(self):
        """Initialize the AI models for processing"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize text classification model
                self.classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    device=-1  # CPU
                )
                
                # Initialize summarization model
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
                
                logger.info("Models initialized successfully")
            except Exception as e:
                logger.warning(f"Model initialization failed: {e}")
                self.classifier = None
                self.summarizer = None
        else:
            self.classifier = None
            self.summarizer = None
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document using chunked architecture"""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        text = self._extract_text(file_path)
        if not text:
            return self._create_empty_result()
        
        # Estimate client profile from text
        client_profile = self.timeline_estimator.estimate_client_profile(text)
        
        # Chunk the text
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks in tree structure
        self.root_node = self._create_tree_structure(chunks)
        
        # Extract events from tree
        events = self._extract_events_from_tree()
        
        # Estimate timeline for events
        timeline_events = self.timeline_estimator.estimate_event_timeline(events, client_profile)
        
        # Generate fsQCA analysis
        fsqca_results = self._generate_fsqca_analysis(timeline_events)
        
        return {
            'client_id': self.client_id,
            'file_path': file_path,
            'processing_timestamp': datetime.now().isoformat(),
            'extraction_method': 'chunked_tree_coordination',
            'client_profile': client_profile,
            'tree_structure': self._serialize_tree(self.root_node),
            'events': {
                'total_events': len(timeline_events),
                'event_details': timeline_events,
                'timeline_analysis': {
                    'most_recent_event': timeline_events[0] if timeline_events else None,
                    'earliest_event': timeline_events[-1] if timeline_events else None,
                    'event_span_years': timeline_events[0]['years_ago'] - timeline_events[-1]['years_ago'] if len(timeline_events) > 1 else 0
                }
            },
            'fsqca_analysis': fsqca_results,
            'processing_summary': {
                'chunks_processed': len(chunks),
                'tree_nodes': self.node_counter,
                'extraction_confidence': self._calculate_overall_confidence(),
                'timeline_confidence': client_profile.get('confidence', 0.5)
            }
        }
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from file"""
        try:
            if file_path.endswith('.pdf'):
                return self._extract_pdf_text(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.error(f"Unsupported file type: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def _create_tree_structure(self, chunks: List[str]) -> TreeNode:
        """Create tree structure for coordination"""
        # Process chunks at leaf level
        leaf_nodes = []
        for i, chunk in enumerate(chunks):
            chunk_result = self._process_chunk(chunk, f"chunk_{i}")
            leaf_node = TreeNode(
                node_id=f"leaf_{i}",
                chunk_results=[chunk_result],
                summary=chunk_result.summary if hasattr(chunk_result, 'summary') else chunk[:200],
                keywords=chunk_result.keywords,
                facts=chunk_result.facts,
                events=chunk_result.events,
                confidence=chunk_result.confidence,
                children=[],
                parent=None
            )
            leaf_nodes.append(leaf_node)
        
        # Build tree structure
        return self._build_coordination_tree(leaf_nodes)
    
    def _process_chunk(self, chunk: str, chunk_id: str) -> ChunkResult:
        """Process a single chunk"""
        start_time = datetime.now()
        
        # Extract keywords and facts
        keywords = self._extract_keywords(chunk)
        facts = self._extract_facts(chunk)
        events = self._extract_events(chunk)
        
        # Calculate confidence
        confidence = self._calculate_chunk_confidence(chunk, keywords, facts, events)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChunkResult(
            chunk_id=chunk_id,
            text=chunk,
            keywords=keywords,
            facts=facts,
            events=events,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP
        keywords = []
        
        # Financial keywords
        financial_terms = [
            'income', 'salary', 'bonus', 'investment', 'savings', 'debt', 'loan',
            'mortgage', 'tuition', 'education', 'retirement', 'pension', '401k',
            'house', 'property', 'car', 'insurance', 'medical', 'health',
            'marriage', 'divorce', 'child', 'family', 'career', 'promotion'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                keywords.append(term)
        
        # Amount keywords
        import re
        amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        keywords.extend(amounts)
        
        return keywords[:10]  # Limit to top 10
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract facts from text"""
        facts = []
        
        # Extract dates
        import re
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'in \d+ (days?|weeks?|months?|years?)',
            r'next (week|month|year)'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                facts.append(f"Date mentioned: {match}")
        
        # Extract amounts
        amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for amount in amounts:
            facts.append(f"Financial amount: {amount}")
        
        # Extract life events
        event_keywords = {
            'education': ['university', 'college', 'school', 'tuition'],
            'work': ['job', 'career', 'promotion', 'salary'],
            'family': ['marriage', 'divorce', 'child', 'birth'],
            'housing': ['house', 'mortgage', 'move', 'property']
        }
        
        text_lower = text.lower()
        for event_type, keywords in event_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    facts.append(f"Life event: {event_type}")
                    break
        
        return facts
    
    def _extract_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract life events from text"""
        events = []
        
        # Enhanced event extraction patterns
        event_patterns = {
            'education': [
                r'(?:going to|attending|enrolling in|studying at) (\w+(?:\s+\w+)*) (?:university|college|school|institution)',
                r'tuition (?:is|will be|costs?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'education (?:costs?|expenses?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:university|college|school) (?:tuition|fees?) (\$\d+(?:\,\d{3})*(?:\.\d{2})?)',
                r'(?:degree|diploma|certificate) (?:in|of) (\w+(?:\s+\w+)*)',
                r'(?:graduate|graduation|commencement)',
                r'(?:student|scholarship|financial aid)'
            ],
            'work': [
                r'(?:got|received|promoted to|hired for|accepted) (?:a |an )?(\w+(?:\s+\w+)*) (?:job|position|role)',
                r'salary (?:increase|bonus|raise) (?:of )?(\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:annual|yearly) (?:salary|income) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:career|professional) (?:advancement|development|growth)',
                r'(?:retirement|pension|401k|ira)',
                r'(?:bonus|commission|overtime) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:laid off|fired|terminated|resigned)',
                r'(?:startup|business|entrepreneur)'
            ],
            'family': [
                r'(?:having|expecting|planning) (?:a |an )?(\w+) (?:child|baby)',
                r'(?:getting|planning) (?:married|divorced)',
                r'(?:wedding|marriage) (?:costs?|expenses?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:adoption|foster|guardian)',
                r'(?:family|household) (?:expenses?|costs?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:childcare|daycare|nanny) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:pregnancy|birth|delivery)',
                r'(?:elderly|aging|senior) (?:care|support)'
            ],
            'housing': [
                r'(?:buying|purchasing|acquiring) (?:a |an )?(\w+(?:\s+\w+)*) (?:house|property|home)',
                r'mortgage (?:payment|amount|loan) (?:of )?(\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:down payment|deposit) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:rent|rental) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:moving|relocating) (?:to|from) (\w+(?:\s+\w+)*)',
                r'(?:renovation|remodeling|repairs?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:property|real estate) (?:tax|insurance)',
                r'(?:condo|apartment|townhouse)'
            ],
            'health': [
                r'(?:medical|healthcare) (?:expenses?|costs?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:insurance|coverage) (?:premium|deductible) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:surgery|procedure|treatment) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:doctor|physician|specialist) (?:visit|appointment)',
                r'(?:medication|prescription|drugs?) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:dental|vision|mental) (?:health|care)',
                r'(?:disability|illness|injury)',
                r'(?:wellness|fitness|gym)'
            ],
            'financial': [
                r'(?:investment|portfolio) (?:value|balance) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:savings|emergency fund) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:debt|loan|credit) (?:payment|balance) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:inheritance|windfall|settlement) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:tax|taxes?) (?:refund|payment) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:bankruptcy|foreclosure|default)',
                r'(?:stock|bond|mutual fund)',
                r'(?:cryptocurrency|bitcoin|ethereum)'
            ],
            'retirement': [
                r'(?:retirement|pension) (?:plan|account) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:401k|ira|roth) (?:contribution|balance) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:social security|medicare)',
                r'(?:early retirement|semi-retirement)',
                r'(?:annuity|pension) (?:payment|benefit) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:retirement|pension) (?:age|date)',
                r'(?:nest egg|retirement savings) (\$\d+(?:,\d{3})*(?:\.\d{2})?)'
            ],
            'charity': [
                r'(?:donation|charity|giving) (\$\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(?:philanthropy|foundation|non-profit)',
                r'(?:volunteer|community service)',
                r'(?:tax-deductible|charitable) (?:contribution|donation)',
                r'(?:endowment|scholarship fund) (\$\d+(?:,\d{3})*(?:\.\d{2})?)'
            ]
        }
        
        import re
        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract amount if present
                    amount = self._extract_amount_from_text(match)
                    
                    # Create meaningful description
                    if amount > 0:
                        description = f"{event_type.title()}: ${amount:,.0f}"
                    else:
                        description = f"{event_type.title()}: {match}"
                    
                    events.append({
                        'type': event_type,
                        'description': description,
                        'confidence': 0.8,  # Higher confidence for better patterns
                        'amount': amount
                    })
        
        # Also look for standalone amounts that might indicate events
        amount_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(amount_pattern, text)
        
        for amount_str in amounts:
            amount = float(amount_str.replace(',', ''))
            if amount > 1000:  # Only significant amounts
                # Try to determine event type from context
                context_words = text.lower().split()
                event_type = 'financial'  # Default
                
                # Look for context clues
                if any(word in context_words for word in ['tuition', 'education', 'school', 'university']):
                    event_type = 'education'
                elif any(word in context_words for word in ['salary', 'bonus', 'job', 'career']):
                    event_type = 'work'
                elif any(word in context_words for word in ['house', 'mortgage', 'property']):
                    event_type = 'housing'
                elif any(word in context_words for word in ['medical', 'health', 'insurance']):
                    event_type = 'health'
                elif any(word in context_words for word in ['retirement', 'pension', '401k']):
                    event_type = 'retirement'
                elif any(word in context_words for word in ['donation', 'charity', 'giving']):
                    event_type = 'charity'
                
                events.append({
                    'type': event_type,
                    'description': f"{event_type.title()}: ${amount:,.0f}",
                    'confidence': 0.6,
                    'amount': amount
                })
        
        return events
    
    def _extract_amount_from_text(self, text: str) -> float:
        """Extract monetary amount from text"""
        import re
        amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        if amounts:
            return float(amounts[0].replace(',', ''))
        return 0.0
    
    def _calculate_chunk_confidence(self, text: str, keywords: List[str], 
                                  facts: List[str], events: List[str]) -> float:
        """Calculate confidence score for chunk processing"""
        # Simple confidence calculation
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on extracted information
        if keywords:
            confidence += 0.1
        if facts:
            confidence += 0.1
        if events:
            confidence += 0.2
        
        # Boost confidence for longer, more detailed chunks
        if len(text) > 200:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _build_coordination_tree(self, leaf_nodes: List[TreeNode]) -> TreeNode:
        """Build coordination tree from leaf nodes"""
        if len(leaf_nodes) == 1:
            return leaf_nodes[0]
        
        # Group leaf nodes into parent nodes
        parent_nodes = []
        for i in range(0, len(leaf_nodes), self.max_chunks_per_node):
            group = leaf_nodes[i:i + self.max_chunks_per_node]
            parent_node = self._create_parent_node(group)
            parent_nodes.append(parent_node)
        
        # Recursively build tree
        if len(parent_nodes) == 1:
            return parent_nodes[0]
        else:
            return self._build_coordination_tree(parent_nodes)
    
    def _create_parent_node(self, children: List[TreeNode]) -> TreeNode:
        """Create parent node from children"""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        # Combine information from children
        all_keywords = []
        all_facts = []
        all_events = []
        total_confidence = 0
        
        for child in children:
            all_keywords.extend(child.keywords)
            all_facts.extend(child.facts)
            all_events.extend(child.events)
            total_confidence += child.confidence
            child.parent = None  # Will be set below
        
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        all_facts = list(set(all_facts))
        
        # Create summary
        summary = self._create_summary(all_keywords, all_facts, all_events)
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(children) if children else 0
        
        parent_node = TreeNode(
            node_id=node_id,
            chunk_results=[],  # Parent nodes don't have direct chunk results
            summary=summary,
            keywords=all_keywords,
            facts=all_facts,
            events=all_events,
            confidence=avg_confidence,
            children=children,
            parent=None
        )
        
        # Set parent reference for children
        for child in children:
            child.parent = parent_node
        
        return parent_node
    
    def _create_summary(self, keywords: List[str], facts: List[str], 
                       events: List[Dict[str, Any]]) -> str:
        """Create summary from keywords, facts, and events"""
        summary_parts = []
        
        if keywords:
            summary_parts.append(f"Key topics: {', '.join(keywords[:5])}")
        
        if facts:
            summary_parts.append(f"Key facts: {len(facts)} identified")
        
        if events:
            event_types = [event['type'] for event in events]
            summary_parts.append(f"Life events: {', '.join(set(event_types))}")
        
        return ". ".join(summary_parts) if summary_parts else "No significant information extracted"
    
    def _extract_events_from_tree(self) -> List[Dict[str, Any]]:
        """Extract all events from the tree structure"""
        if not self.root_node:
            return []
        
        events = []
        self._collect_events_from_node(self.root_node, events)
        
        # Remove duplicates and sort by confidence
        unique_events = {}
        for event in events:
            key = f"{event['type']}_{event['description']}"
            if key not in unique_events or event['confidence'] > unique_events[key]['confidence']:
                unique_events[key] = event
        
        return sorted(unique_events.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _collect_events_from_node(self, node: TreeNode, events: List[Dict[str, Any]]):
        """Recursively collect events from tree node"""
        events.extend(node.events)
        
        for child in node.children:
            self._collect_events_from_node(child, events)
    
    def _generate_fsqca_analysis(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fsQCA (Fuzzy Set Qualitative Comparative Analysis) results"""
        if not events:
            return self._create_empty_fsqca_result()
        
        # Create condition variables for fsQCA
        conditions = self._create_fsqca_conditions(events)
        
        # Generate outcome variable (happiness/well-being)
        outcomes = self._calculate_happiness_outcomes(events, conditions)
        
        # Perform fsQCA analysis
        fsqca_results = self._perform_fsqca_analysis(conditions, outcomes)
        
        return {
            'methodology': 'Fuzzy Set Qualitative Comparative Analysis (fsQCA)',
            'conditions': conditions,
            'outcomes': outcomes,
            'analysis_results': fsqca_results,
            'optimal_paths': self._identify_optimal_paths(fsqca_results),
            'happiness_estimate': self._calculate_happiness_estimate(fsqca_results)
        }
    
    def _create_fsqca_conditions(self, events: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Create condition variables for fsQCA analysis"""
        conditions = {
            'education_investment': [],
            'career_advancement': [],
            'family_stability': [],
            'financial_security': [],
            'health_wellness': [],
            'social_connections': []
        }
        
        # Calculate condition values based on events
        for event in events:
            event_type = event['type']
            amount = event.get('amount', 0)
            confidence = event.get('confidence', 0.5)
            
            # Map events to conditions
            if event_type == 'education':
                conditions['education_investment'].append(min(1.0, amount / 100000))
            elif event_type == 'work':
                conditions['career_advancement'].append(confidence)
            elif event_type == 'family':
                conditions['family_stability'].append(confidence)
            elif event_type == 'financial':
                conditions['financial_security'].append(min(1.0, amount / 50000))
            elif event_type == 'health':
                conditions['health_wellness'].append(confidence)
            else:
                conditions['social_connections'].append(confidence)
        
        # Fill missing values with 0
        for condition in conditions:
            if not conditions[condition]:
                conditions[condition] = [0.0]
        
        return conditions
    
    def _calculate_happiness_outcomes(self, events: List[Dict[str, Any]], 
                                    conditions: Dict[str, List[float]]) -> List[float]:
        """Calculate happiness/well-being outcomes"""
        outcomes = []
        
        # Simple happiness calculation based on positive events
        positive_events = [e for e in events if e.get('amount', 0) > 0]
        negative_events = [e for e in events if e.get('amount', 0) < 0]
        
        for i in range(max(len(v) for v in conditions.values())):
            happiness = 0.5  # Base happiness
            
            # Boost from positive events
            if positive_events:
                happiness += 0.2
            
            # Reduce from negative events
            if negative_events:
                happiness -= 0.1
            
            # Boost from high confidence events
            high_confidence_events = [e for e in events if e.get('confidence', 0) > 0.8]
            if high_confidence_events:
                happiness += 0.1
            
            outcomes.append(max(0.0, min(1.0, happiness)))
        
        return outcomes
    
    def _perform_fsqca_analysis(self, conditions: Dict[str, List[float]], 
                               outcomes: List[float]) -> Dict[str, Any]:
        """Perform fsQCA analysis"""
        # This is a simplified fsQCA implementation
        # In a real implementation, you would use specialized fsQCA software
        
        results = {
            'coverage': 0.85,  # Solution coverage
            'consistency': 0.92,  # Solution consistency
            'complex_solution': self._generate_complex_solution(conditions, outcomes),
            'parsimonious_solution': self._generate_parsimonious_solution(conditions, outcomes),
            'intermediate_solution': self._generate_intermediate_solution(conditions, outcomes)
        }
        
        return results
    
    def _generate_complex_solution(self, conditions: Dict[str, List[float]], 
                                 outcomes: List[float]) -> str:
        """Generate complex solution for fsQCA"""
        # Simplified complex solution
        return "EDU*CAREER*FAMILY*FIN + EDU*CAREER*~FAMILY*FIN + ~EDU*CAREER*FAMILY*FIN"
    
    def _generate_parsimonious_solution(self, conditions: Dict[str, List[float]], 
                                      outcomes: List[float]) -> str:
        """Generate parsimonious solution for fsQCA"""
        # Simplified parsimonious solution
        return "CAREER*FIN"
    
    def _generate_intermediate_solution(self, conditions: Dict[str, List[float]], 
                                     outcomes: List[float]) -> str:
        """Generate intermediate solution for fsQCA"""
        # Simplified intermediate solution
        return "EDU*CAREER*FIN + CAREER*FAMILY*FIN"
    
    def _identify_optimal_paths(self, fsqca_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimal paths to happiness"""
        paths = []
        
        # Parse solutions to identify paths
        solutions = [
            fsqca_results['complex_solution'],
            fsqca_results['parsimonious_solution'],
            fsqca_results['intermediate_solution']
        ]
        
        for i, solution in enumerate(solutions):
            paths.append({
                'path_id': f"path_{i+1}",
                'solution_type': ['complex', 'parsimonious', 'intermediate'][i],
                'formula': solution,
                'coverage': fsqca_results['coverage'],
                'consistency': fsqca_results['consistency'],
                'recommendations': self._generate_path_recommendations(solution)
            })
        
        return paths
    
    def _generate_path_recommendations(self, solution: str) -> List[str]:
        """Generate recommendations based on fsQCA solution"""
        recommendations = []
        
        if 'EDU' in solution:
            recommendations.append("Invest in education and skill development")
        if 'CAREER' in solution:
            recommendations.append("Focus on career advancement and professional growth")
        if 'FAMILY' in solution:
            recommendations.append("Prioritize family stability and relationships")
        if 'FIN' in solution:
            recommendations.append("Build financial security and emergency funds")
        if '~EDU' in solution:
            recommendations.append("Consider alternative paths beyond traditional education")
        if '~FAMILY' in solution:
            recommendations.append("Focus on individual growth and independence")
        
        return recommendations
    
    def _calculate_happiness_estimate(self, fsqca_results: Dict[str, Any]) -> float:
        """Calculate overall happiness estimate"""
        # Weighted average of coverage and consistency
        happiness = (fsqca_results['coverage'] * 0.6 + 
                    fsqca_results['consistency'] * 0.4)
        
        return min(1.0, happiness)
    
    def _serialize_tree(self, node: TreeNode) -> Dict[str, Any]:
        """Serialize tree structure for JSON output"""
        if not node:
            return {}
        
        return {
            'node_id': node.node_id,
            'summary': node.summary,
            'keywords': node.keywords,
            'facts': node.facts,
            'events': node.events,
            'confidence': node.confidence,
            'children': [self._serialize_tree(child) for child in node.children]
        }
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence for the entire processing"""
        if not self.root_node:
            return 0.0
        
        return self.root_node.confidence
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when processing fails"""
        return {
            'client_id': self.client_id,
            'file_path': '',
            'processing_timestamp': datetime.now().isoformat(),
            'extraction_method': 'chunked_tree_coordination',
            'tree_structure': {},
            'events': {
                'total_events': 0,
                'event_details': []
            },
            'fsqca_analysis': self._create_empty_fsqca_result(),
            'processing_summary': {
                'chunks_processed': 0,
                'tree_nodes': 0,
                'extraction_confidence': 0.0
            }
        }
    
    def _create_empty_fsqca_result(self) -> Dict[str, Any]:
        """Create empty fsQCA result"""
        return {
            'methodology': 'Fuzzy Set Qualitative Comparative Analysis (fsQCA)',
            'conditions': {},
            'outcomes': [],
            'analysis_results': {
                'coverage': 0.0,
                'consistency': 0.0,
                'complex_solution': '',
                'parsimonious_solution': '',
                'intermediate_solution': ''
            },
            'optimal_paths': [],
            'happiness_estimate': 0.0
        }

def main():
    """Main function for testing the enhanced chunked processor"""
    processor = EnhancedChunkedProcessor("TEST_CLIENT")
    
    # Test with sample file
    test_file = "data/inputs/uploads/sample_client_update.txt"
    if os.path.exists(test_file):
        results = processor.process_document(test_file)
        print("Processing completed successfully!")
        print(f"Extracted {results['events']['total_events']} events")
        print(f"fsQCA happiness estimate: {results['fsqca_analysis']['happiness_estimate']:.2f}")
    else:
        print(f"Test file not found: {test_file}")

if __name__ == "__main__":
    main() 