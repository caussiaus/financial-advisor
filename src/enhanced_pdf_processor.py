import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pdfplumber
import pandas as pd
from dataclasses import dataclass
import numpy as np
from dateutil.parser import parse as parse_date


@dataclass
class FinancialMilestone:
    """Represents a financial milestone or life event extracted from PDF"""
    timestamp: datetime
    event_type: str
    description: str
    financial_impact: Optional[float]
    probability: float
    dependencies: List[str]
    payment_flexibility: Dict[str, any]
    metadata: Dict[str, any]


class EnhancedPDFProcessor:
    """
    Advanced PDF processor that extracts life milestones and financial events
    with sophisticated pattern recognition for various payment structures
    """
    
    def __init__(self):
        self.milestone_patterns = {
            'education': [
                r'college|university|graduate|degree|tuition|student loan',
                r'MBA|PhD|masters|bachelor',
                r'education|academic|school'
            ],
            'career': [
                r'promotion|salary|bonus|raise|career|job',
                r'retirement|401k|pension|social security',
                r'unemployment|layoff|career change'
            ],
            'family': [
                r'marriage|wedding|spouse|partner',
                r'children|baby|pregnancy|family',
                r'divorce|separation|custody'
            ],
            'housing': [
                r'house|home|mortgage|rent|property',
                r'down payment|closing costs|refinance',
                r'moving|relocation'
            ],
            'health': [
                r'medical|health|insurance|hospital',
                r'disability|illness|surgery',
                r'life insurance|health insurance'
            ],
            'investment': [
                r'investment|portfolio|stocks|bonds',
                r'mutual funds|ETF|real estate',
                r'savings|emergency fund|retirement'
            ]
        }
        
        self.payment_structure_patterns = {
            'lump_sum': r'lump sum|one time|single payment|pay in full',
            'installments': r'installments|monthly|quarterly|annual|payments',
            'percentage_based': r'percentage|%|percent|portion|fraction',
            'milestone_based': r'upon|when|after|completion|achievement',
            'flexible': r'flexible|variable|custom|as needed|any amount'
        }
        
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\w+\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\bin\s+\d{4}\b',
            r'\bby\s+\w+\s+\d{4}\b'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def identify_milestones(self, text: str) -> List[FinancialMilestone]:
        """Identify financial milestones and life events from text"""
        milestones = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Check for milestone patterns
            for category, patterns in self.milestone_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        milestone = self._create_milestone_from_sentence(
                            sentence, category, text
                        )
                        if milestone:
                            milestones.append(milestone)
                        break
        
        return self._deduplicate_milestones(milestones)

    def _create_milestone_from_sentence(self, sentence: str, category: str, full_text: str) -> Optional[FinancialMilestone]:
        """Create a milestone from a sentence with extracted metadata"""
        try:
            # Extract timestamp
            timestamp = self._extract_timestamp(sentence, full_text)
            
            # Extract financial impact
            financial_impact = self._extract_financial_amount(sentence)
            
            # Determine payment flexibility
            payment_flexibility = self._analyze_payment_flexibility(sentence, full_text)
            
            # Calculate probability based on language certainty
            probability = self._calculate_probability(sentence)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(sentence, full_text)
            
            return FinancialMilestone(
                timestamp=timestamp,
                event_type=category,
                description=sentence.strip(),
                financial_impact=financial_impact,
                probability=probability,
                dependencies=dependencies,
                payment_flexibility=payment_flexibility,
                metadata={
                    'source_sentence': sentence,
                    'extraction_confidence': self._calculate_extraction_confidence(sentence)
                }
            )
        except Exception as e:
            print(f"Error creating milestone: {e}")
            return None

    def _extract_timestamp(self, sentence: str, full_text: str) -> datetime:
        """Extract timestamp from sentence or infer from context"""
        # Look for explicit dates in sentence
        for pattern in self.date_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                try:
                    return parse_date(match.group())
                except:
                    continue
        
        # Look for relative time indicators
        relative_patterns = {
            r'next year': timedelta(days=365),
            r'in \d+ years?': lambda m: timedelta(days=365 * int(re.search(r'\d+', m.group()).group())),
            r'retirement': timedelta(days=365 * 30),  # Assume 30 years to retirement
            r'college': timedelta(days=365 * 18),  # Assume 18 years for college
        }
        
        base_date = datetime.now()
        for pattern, delta in relative_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                if callable(delta):
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    delta = delta(match)
                return base_date + delta
        
        # Default to current date if no timestamp found
        return base_date

    def _extract_financial_amount(self, sentence: str) -> Optional[float]:
        """Extract financial amounts from sentence"""
        # Look for currency amounts
        currency_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*dollars?',
            r'[\d,]+\.?\d*\s*USD'
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                amount_str = re.sub(r'[^\d.]', '', match.group())
                try:
                    return float(amount_str)
                except:
                    continue
        
        return None

    def _analyze_payment_flexibility(self, sentence: str, full_text: str) -> Dict[str, any]:
        """Analyze payment structure flexibility from text"""
        flexibility = {
            'structure_type': 'flexible',
            'min_payment': None,
            'max_payment': None,
            'frequency_options': [],
            'custom_dates_allowed': True,
            'percentage_based': False
        }
        
        # Check for payment structure patterns
        for structure_type, pattern in self.payment_structure_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                flexibility['structure_type'] = structure_type
                break
        
        # Look for percentage indicators
        if re.search(r'\d+%|\d+\s*percent', sentence, re.IGNORECASE):
            flexibility['percentage_based'] = True
        
        # Look for frequency indicators
        frequency_patterns = {
            'daily': r'daily|every day',
            'weekly': r'weekly|every week',
            'monthly': r'monthly|every month',
            'quarterly': r'quarterly|every quarter',
            'annually': r'annually|yearly|every year'
        }
        
        for freq, pattern in frequency_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                flexibility['frequency_options'].append(freq)
        
        return flexibility

    def _calculate_probability(self, sentence: str) -> float:
        """Calculate probability based on language certainty indicators"""
        certainty_indicators = {
            'high': ['will', 'must', 'definitely', 'certainly', 'plan to'],
            'medium': ['should', 'likely', 'probably', 'expect to', 'intend to'],
            'low': ['might', 'may', 'could', 'possibly', 'consider']
        }
        
        for certainty, words in certainty_indicators.items():
            for word in words:
                if word in sentence.lower():
                    if certainty == 'high':
                        return 0.9
                    elif certainty == 'medium':
                        return 0.7
                    else:
                        return 0.4
        
        return 0.6  # Default probability

    def _extract_dependencies(self, sentence: str, full_text: str) -> List[str]:
        """Extract dependencies between milestones"""
        dependency_patterns = [
            r'after|once|when|upon|following|depends on',
            r'requires|needs|must have|conditional on'
        ]
        
        dependencies = []
        for pattern in dependency_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # This is a simplified implementation
                # In practice, would use NLP to identify the dependency
                dependencies.append('conditional_event')
        
        return dependencies

    def _calculate_extraction_confidence(self, sentence: str) -> float:
        """Calculate confidence in the extraction quality"""
        confidence = 0.5
        
        # Boost confidence for specific financial terms
        financial_terms = ['dollar', 'payment', 'cost', 'price', 'amount']
        for term in financial_terms:
            if term in sentence.lower():
                confidence += 0.1
        
        # Boost confidence for specific dates
        for pattern in self.date_patterns:
            if re.search(pattern, sentence):
                confidence += 0.2
                break
        
        return min(confidence, 1.0)

    def _deduplicate_milestones(self, milestones: List[FinancialMilestone]) -> List[FinancialMilestone]:
        """Remove duplicate milestones based on similarity"""
        if not milestones:
            return []
        
        unique_milestones = []
        for milestone in milestones:
            is_duplicate = False
            for existing in unique_milestones:
                # Simple similarity check
                if (milestone.event_type == existing.event_type and
                    abs((milestone.timestamp - existing.timestamp).days) < 30 and
                    milestone.description[:50] == existing.description[:50]):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_milestones.append(milestone)
        
        return unique_milestones

    def process_pdf(self, pdf_path: str) -> List[FinancialMilestone]:
        """Main method to process PDF and extract milestones"""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        milestones = self.identify_milestones(text)
        
        # Sort by timestamp
        milestones.sort(key=lambda m: m.timestamp)
        
        return milestones

    def export_milestones_to_json(self, milestones: List[FinancialMilestone], output_path: str):
        """Export milestones to JSON format"""
        data = []
        for milestone in milestones:
            data.append({
                'timestamp': milestone.timestamp.isoformat(),
                'event_type': milestone.event_type,
                'description': milestone.description,
                'financial_impact': milestone.financial_impact,
                'probability': milestone.probability,
                'dependencies': milestone.dependencies,
                'payment_flexibility': milestone.payment_flexibility,
                'metadata': milestone.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    processor = EnhancedPDFProcessor()
    
    # Example usage
    pdf_path = "data/uploads/Case_1_IPS_Individual.pdf"
    milestones = processor.process_pdf(pdf_path)
    
    print(f"Extracted {len(milestones)} milestones:")
    for milestone in milestones:
        print(f"- {milestone.event_type}: {milestone.description[:100]}...")
        print(f"  Timestamp: {milestone.timestamp}")
        print(f"  Financial Impact: ${milestone.financial_impact}")
        print(f"  Payment Flexibility: {milestone.payment_flexibility['structure_type']}")
        print()