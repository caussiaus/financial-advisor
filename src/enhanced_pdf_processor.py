"""
Enhanced PDF processor for financial milestone extraction with NLP
"""
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pdfplumber
import json
import spacy
from collections import defaultdict

@dataclass
class FinancialEntity:
    """Represents a person or entity with financial tracking"""
    name: str
    entity_type: str  # 'person', 'child', 'spouse', 'organization', 'fund'
    initial_balances: Dict[str, float]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class FinancialMilestone:
    """Represents a financial milestone or event"""
    timestamp: datetime
    event_type: str
    description: str
    financial_impact: Optional[float] = None
    probability: float = 0.6  # Default probability if not specified
    dependencies: List[str] = None
    payment_flexibility: Dict = None
    metadata: Dict = None
    entity: Optional[str] = None  # Link to FinancialEntity

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.payment_flexibility is None:
            self.payment_flexibility = {'structure_type': 'flexible'}
        if self.metadata is None:
            self.metadata = {}

class EnhancedPDFProcessor:
    """
    Enhanced PDF processor that extracts financial milestones and entities
    using advanced NLP techniques with chunking and coordination
    """
    
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy model for NLP processing")
        except OSError:
            print("⚠️ spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.event_types = {
            'education': ['education', 'tuition', 'school', 'university', 'college', 'graduate', 'degree'],
            'housing': ['house', 'home', 'mortgage', 'rent', 'property', 'buy', 'purchase'],
            'investment': ['investment', 'portfolio', 'stock', 'bond', 'fund', 'ent'],
            'career': ['career', 'job', 'salary', 'promotion', 'retirement', 'work'],
            'family': ['family', 'wedding', 'child', 'children', 'parent', 'marriage'],
            'health': ['health', 'medical', 'insurance', 'hospital', 'doctor']
        }
        
        self.amount_pattern = r'\$?([0-9,]+(?:\.[0-9]{2})?)'
        self.date_pattern = r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        self.probability_pattern = r'(\d+(?:\.\d+)?)\s*%'
        # Entity tracking
        self.entities = {}
        self.entity_balances = defaultdict(dict)
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[FinancialMilestone], List[FinancialEntity]]:
        """Process PDF and extract financial milestones and entities"""
        if not os.path.exists(pdf_path):
            print(f"Error: File not found - {pdf_path}")
            return [], []
            
        milestones = []
        entities = []
        current_time = datetime.now()
        
        try:
            print(f"Opening PDF: {pdf_path}")
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            print(f"Extracted text from page {i+1}")
                    except Exception as e:
                        print(f"Error extracting text from page {i+1}: {e}")
                        print(traceback.format_exc())
                
                # Process text using chunking and coordination architecture
                milestones, entities = self._process_text_with_nlp(text, pdf_path)
                
                # Initialize accounting balances for entities
                self._initialize_entity_balances(entities, current_time)
        
        except Exception as e:
            print(f"Error processing PDF: {e}")
            print(traceback.format_exc())
            return [], []
        
        # Sort milestones by date
        milestones.sort(key=lambda x: x.timestamp)
        print(f"\nExtracted {len(milestones)} milestones and {len(entities)} entities")
        
        # Save results for debugging
        try:
            self._save_results(milestones, entities, pdf_path)
            print("Saved results to JSON")
        except Exception as e:
            print(f"Error saving results: {e}")
            print(traceback.format_exc())
        
        return milestones, entities
    
    def _process_text_with_nlp(self, text: str, pdf_path: str) -> Tuple[List[FinancialMilestone], List[FinancialEntity]]:
        """Process text using NLP with chunking and coordination"""
        print("🔄 Processing text with NLP chunking and coordination...")
        
        # Step 1: Chunk the text into manageable segments
        chunks = self._create_chunks(text)
        print(f"Created {len(chunks)} chunks for processing")
        
        # Step 2: Process each chunk for entities and events
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_entities, chunk_milestones = self._process_chunk(chunk, i, pdf_path)
            chunk_results.append({
                'chunk_id': i,
                'entities': chunk_entities,
                'milestones': chunk_milestones
            })
        
        # Step 3: Coordinate and merge results (tree-style aggregation)
        all_entities = self._coordinate_entities(chunk_results)
        all_milestones = self._coordinate_milestones(chunk_results)
        
        # Step 4: Link entities to milestones
        linked_milestones = self._link_entities_to_milestones(all_milestones, all_entities)
        
        return linked_milestones, list(all_entities.values())
    
    def _create_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Create chunks of text for processing"""
        # Split by sentences first
        doc = self.nlp(text)
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
    
    def _process_chunk(self, chunk_text: str, chunk_id: int, pdf_path: str) -> Tuple[Dict[str, FinancialEntity], List[FinancialMilestone]]:
        """Process a single chunk for entities and milestones"""
        doc = self.nlp(chunk_text)
        
        # Extract entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entity_name = ent.text.strip()
                entity_type = self._determine_entity_type(entity_name, chunk_text)
                
                if entity_name not in entities:
                    entities[entity_name] = FinancialEntity(
                        name=entity_name,
                        entity_type=entity_type,
                        initial_balances={},
                        metadata={'chunk_id': chunk_id, 'source': pdf_path}
                    )
        
        # Extract milestones
        milestones = []
        sentences = list(doc.sents)
        
        for sent in sentences:
            milestone = self._extract_milestone_from_sentence(sent, chunk_id)
            if milestone:
                milestones.append(milestone)
        
        return entities, milestones
    
    def _determine_entity_type(self, entity_name: str, context: str) -> str:
        """Determine the type of entity based on context"""
        context_lower = context.lower()
        entity_lower = entity_name.lower()
        
        # Check for family relationships
        if any(word in context_lower for word in ["child", "children", "son", "daughter"]):
            return 'child'
        elif any(word in context_lower for word in ["spouse", "wife", "husband", "partner"]):
            return 'spouse'
        elif any(word in context_lower for word in ["university", "college", "school"]):
            return 'organization'
        elif any(word in context_lower for word in ["fund", "account", "savings"]):
            return 'fund'
        else:
            return 'person'
    
    def _extract_milestone_from_sentence(self, sentence, chunk_id: int) -> Optional[FinancialMilestone]:
        """Extract milestone from a single sentence"""
        sent_text = sentence.text.lower()
        
        # Check for event indicators
        event_type = None
        for event_category, keywords in self.event_types.items():
            if any(keyword in sent_text for keyword in keywords):
                event_type = event_category
                break
        
        if not event_type:
            return None
        
        # Extract financial amount
        financial_impact = self._extract_amount_from_sentence(sentence.text)
        
        # Extract date
        timestamp = self._extract_date_from_sentence(sentence.text)
        if not timestamp:
            timestamp = datetime.now() + timedelta(days=365)
        
        # Extract probability
        probability = self._extract_probability_from_sentence(sentence.text)
        
        # Extract entity reference
        entity_ref = self._extract_entity_reference(sentence)
        
        return FinancialMilestone(
            timestamp=timestamp,
            event_type=event_type,
            description=sentence.text.strip()[:500],
            financial_impact=financial_impact,
            probability=probability,
            entity=entity_ref,
            metadata={'chunk_id': chunk_id}
        )
    
    def _extract_amount_from_sentence(self, text: str) -> Optional[float]:
        """Extract financial amount from sentence"""
        matches = re.findall(self.amount_pattern, text)
        if matches:
            try:
                amount_str = matches[0].replace(',', '')
                return float(amount_str)
            except (ValueError, IndexError):
                return None
        return None
    
    def _extract_date_from_sentence(self, text: str) -> Optional[datetime]:
        """Extract date from sentence"""
        matches = re.findall(self.date_pattern, text)
        if matches:
            try:
                date_str = matches[0]
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    try:
                        return datetime.strptime(date_str, '%Y/%m/%d')
                    except ValueError:
                        return None
            except (ValueError, IndexError):
                return None
        return None
    
    def _extract_probability_from_sentence(self, text: str) -> float:
        """Extract probability from sentence"""
        matches = re.findall(self.probability_pattern, text)
        if matches:
            try:
                prob = float(matches[0]) / 100.0
                return min(1.0, max(0.0, prob))  # Clamp between 0 and 1
            except (ValueError, IndexError):
                return 0.7  # Default probability
        return 0.7  # Default probability
    
    def _extract_entity_reference(self, sentence) -> Optional[str]:
        """Extract entity reference from sentence"""
        for ent in sentence.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        return None
    
    def _coordinate_entities(self, chunk_results: List[Dict]) -> Dict[str, FinancialEntity]:
        """Coordinate and merge entities from all chunks"""
        all_entities = {}
        
        for chunk_result in chunk_results:
            for entity_name, entity in chunk_result['entities'].items():
                if entity_name not in all_entities:
                    all_entities[entity_name] = entity
                else:
                    # Merge metadata from different chunks
                    all_entities[entity_name].metadata.update(entity.metadata)
        
        return all_entities
    
    def _coordinate_milestones(self, chunk_results: List[Dict]) -> List[FinancialMilestone]:
        """Coordinate and merge milestones from all chunks"""
        all_milestones = []
        
        for chunk_result in chunk_results:
            all_milestones.extend(chunk_result['milestones'])
        
        return all_milestones
    
    def _link_entities_to_milestones(self, milestones: List[FinancialMilestone], entities: Dict[str, FinancialEntity]) -> List[FinancialMilestone]:
        """Ties to milestones and update entity balances"""
        for milestone in milestones:
            if milestone.entity and milestone.entity in entities:
                # Update entity's initial balances based on milestone
                entity = entities[milestone.entity]
                if milestone.financial_impact:
                    if milestone.event_type not in entity.initial_balances:
                        entity.initial_balances[milestone.event_type] = 0
                    entity.initial_balances[milestone.event_type] += milestone.financial_impact
        
        return milestones
    
    def _initialize_entity_balances(self, entities: List[FinancialEntity], current_time: datetime):
        """Initialize accounting balances for all entities"""
        print("💰 Initializing entity balances for accounting framework...")
        
        for entity in entities:
            # Set default balances if none exist
            if not entity.initial_balances:
                if entity.entity_type == 'person':
                    entity.initial_balances = {
                        'salary': 0,
                        'savings': 0,
                        'investments': 0
                    }
                elif entity.entity_type == 'child':
                    entity.initial_balances = {
                        'education_fund': 0,
                        'allowance': 0
                    }
                elif entity.entity_type == 'fund':
                    entity.initial_balances = {
                        'balance': 0
                    }
            
            print(f"  {entity.name} ({entity.entity_type}): {entity.initial_balances}")
    
    def _save_results(self, milestones: List[FinancialMilestone], entities: List[FinancialEntity], pdf_path: str):
        """Save extraction results for debugging"""
        results = {
            'source_file': os.path.basename(pdf_path),
            'extraction_time': datetime.now().isoformat(),
            'entities': [
                {
                    'name': e.name,
                    'type': e.entity_type,
                    'initial_balances': e.initial_balances,
                    'metadata': e.metadata
                }
                for e in entities
            ],
            'milestones': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'event_type': m.event_type,
                    'description': m.description,
                    'financial_impact': m.financial_impact,
                    'probability': m.probability,
                    'entity': m.entity
                }
                for m in milestones
            ]
        }
        
        # Save to JSON file
        output_path = pdf_path + '_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)