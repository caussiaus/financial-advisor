"""
PDF Processor Layer

Responsible for:
- Document processing and text extraction
- Financial milestone identification
- Entity recognition and tracking
- Data validation and sanitization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Protocol
import os
import re
import traceback
from pathlib import Path

import pdfplumber
import spacy
from collections import defaultdict


@dataclass
class FinancialEntity:
    """Represents a person or entity with financial tracking"""
    name: str
    entity_type: str  # 'person', 'child', 'spouse', 'organization', 'fund'
    initial_balances: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FinancialMilestone:
    """Represents a financial milestone or event"""
    timestamp: datetime
    event_type: str
    description: str
    financial_impact: Optional[float] = None
    probability: float = 0.6
    dependencies: List[str] = field(default_factory=list)
    payment_flexibility: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    entity: Optional[str] = None


class DocumentProcessor(Protocol):
    """Protocol for document processing capabilities"""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from document"""
        ...
    
    def validate_document(self, file_path: str) -> bool:
        """Validate document format and content"""
        ...


class MilestoneExtractor(Protocol):
    """Protocol for milestone extraction capabilities"""
    
    def extract_milestones(self, text: str) -> List[FinancialMilestone]:
        """Extract financial milestones from text"""
        ...


class EntityRecognizer(Protocol):
    """Protocol for entity recognition capabilities"""
    
    def extract_entities(self, text: str) -> List[FinancialEntity]:
        """Extract financial entities from text"""
        ...


class PDFProcessorLayer:
    """
    PDF Processor Layer - Clean API for document processing
    
    Responsibilities:
    - Document validation and text extraction
    - Financial milestone identification
    - Entity recognition and tracking
    - Data validation and sanitization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._nlp_model = self._initialize_nlp()
        self._event_types = self._initialize_event_types()
        self._patterns = self._initialize_patterns()
        
    def _initialize_nlp(self) -> spacy.language.Language:
        """Initialize NLP model with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model for NLP processing")
            return nlp
        except OSError:
            print("⚠️ spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")
    
    def _initialize_event_types(self) -> Dict[str, List[str]]:
        """Initialize event type mappings"""
        return {
            'education': ['education', 'tuition', 'school', 'university', 'college', 'graduate', 'degree'],
            'housing': ['house', 'home', 'mortgage', 'rent', 'property', 'buy', 'purchase'],
            'investment': ['investment', 'portfolio', 'stock', 'bond', 'fund', 'ent'],
            'career': ['career', 'job', 'salary', 'promotion', 'retirement', 'work'],
            'family': ['family', 'wedding', 'child', 'children', 'parent', 'marriage'],
            'health': ['health', 'medical', 'insurance', 'hospital', 'doctor']
        }
    
    def _initialize_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns"""
        return {
            'amount': r'\$?([0-9,]+(?:\.[0-9]{2})?)',
            'date': r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            'probability': r'(\d+(?:\.\d+)?)\s*%'
        }
    
    def process_document(self, file_path: str) -> Tuple[List[FinancialMilestone], List[FinancialEntity]]:
        """
        Main API method to process a document
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Tuple of (milestones, entities)
        """
        if not self._validate_document(file_path):
            raise ValueError(f"Invalid document: {file_path}")
        
        text = self._extract_text(file_path)
        chunks = self._create_chunks(text)
        
        milestones = []
        entities = []
        
        for chunk in chunks:
            chunk_milestones = self._extract_milestones_from_chunk(chunk)
            chunk_entities = self._extract_entities_from_chunk(chunk)
            
            milestones.extend(chunk_milestones)
            entities.extend(chunk_entities)
        
        # Deduplicate and link entities to milestones
        unique_entities = self._deduplicate_entities(entities)
        linked_milestones = self._link_entities_to_milestones(milestones, unique_entities)
        
        return linked_milestones, list(unique_entities.values())
    
    def _validate_document(self, file_path: str) -> bool:
        """Validate document format and accessibility"""
        if not os.path.exists(file_path):
            return False
        
        if not file_path.lower().endswith('.pdf'):
            return False
        
        return True
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF document"""
        text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {file_path}: {e}")
        
        return text
    
    def _create_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Create manageable chunks for processing"""
        doc = self._nlp_model(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_milestones_from_chunk(self, chunk_text: str) -> List[FinancialMilestone]:
        """Extract financial milestones from a text chunk"""
        milestones = []
        doc = self._nlp_model(chunk_text)
        
        for sent in doc.sents:
            milestone = self._extract_milestone_from_sentence(sent)
            if milestone:
                milestones.append(milestone)
        
        return milestones
    
    def _extract_milestone_from_sentence(self, sentence) -> Optional[FinancialMilestone]:
        """Extract a single milestone from a sentence"""
        text = sentence.text.lower()
        
        # Determine event type
        event_type = self._determine_event_type(text)
        if not event_type:
            return None
        
        # Extract amount
        amount = self._extract_amount_from_text(sentence.text)
        
        # Extract date
        date = self._extract_date_from_text(sentence.text)
        if not date:
            date = datetime.now()  # Default to current date
        
        # Extract probability
        probability = self._extract_probability_from_text(sentence.text)
        
        # Extract entity reference
        entity = self._extract_entity_reference(sentence)
        
        return FinancialMilestone(
            timestamp=date,
            event_type=event_type,
            description=sentence.text.strip(),
            financial_impact=amount,
            probability=probability,
            entity=entity
        )
    
    def _determine_event_type(self, text: str) -> Optional[str]:
        """Determine the type of financial event"""
        for event_type, keywords in self._event_types.items():
            if any(keyword in text for keyword in keywords):
                return event_type
        return None
    
    def _extract_amount_from_text(self, text: str) -> Optional[float]:
        """Extract monetary amount from text"""
        match = re.search(self._patterns['amount'], text)
        if match:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        return None
    
    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date from text"""
        match = re.search(self._patterns['date'], text)
        if match:
            date_str = match.group(0)
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%Y/%m/%d')
                except ValueError:
                    return None
        return None
    
    def _extract_probability_from_text(self, text: str) -> float:
        """Extract probability from text"""
        match = re.search(self._patterns['probability'], text)
        if match:
            return float(match.group(1)) / 100.0
        return 0.6  # Default probability
    
    def _extract_entity_reference(self, sentence) -> Optional[str]:
        """Extract entity reference from sentence"""
        for ent in sentence.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        return None
    
    def _extract_entities_from_chunk(self, chunk_text: str) -> List[FinancialEntity]:
        """Extract financial entities from a text chunk"""
        entities = []
        doc = self._nlp_model(chunk_text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entity_type = self._determine_entity_type(ent.text, chunk_text)
                entity = FinancialEntity(
                    name=ent.text.strip(),
                    entity_type=entity_type,
                    metadata={'source': 'pdf_processor'}
                )
                entities.append(entity)
        
        return entities
    
    def _determine_entity_type(self, entity_name: str, context: str) -> str:
        """Determine the type of entity based on context"""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['child', 'children', 'son', 'daughter']):
            return 'child'
        elif any(word in context_lower for word in ['spouse', 'wife', 'husband', 'partner']):
            return 'spouse'
        elif any(word in context_lower for word in ['fund', 'trust', 'foundation']):
            return 'fund'
        elif any(word in context_lower for word in ['company', 'corporation', 'inc', 'llc']):
            return 'organization'
        else:
            return 'person'
    
    def _deduplicate_entities(self, entities: List[FinancialEntity]) -> Dict[str, FinancialEntity]:
        """Deduplicate entities by name"""
        unique_entities = {}
        for entity in entities:
            if entity.name not in unique_entities:
                unique_entities[entity.name] = entity
        return unique_entities
    
    def _link_entities_to_milestones(self, milestones: List[FinancialMilestone], 
                                   entities: Dict[str, FinancialEntity]) -> List[FinancialMilestone]:
        """Link entities to milestones where appropriate"""
        for milestone in milestones:
            if milestone.entity and milestone.entity in entities:
                # Update entity metadata with milestone reference
                entities[milestone.entity].metadata['milestones'] = entities[milestone.entity].metadata.get('milestones', [])
                entities[milestone.entity].metadata['milestones'].append(milestone.description)
        
        return milestones 