#!/usr/bin/env python
"""
Client Input Processor - Email/PDF to Life Event Pipeline
Author: ChatGPT 2025-01-27

Processes client emails and PDFs to extract life events, timestamp them,
and create configuration spaces for optimal decision making.

Features:
- Email/PDF ingestion using KOR libraries
- AI-powered event extraction and timestamping
- Balance tracking across life scenarios
- Stress scenario modeling
- Configuration space generation
"""

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
from dataclasses import dataclass
import logging

# For PDF processing (you'll need to install these)
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("PDF libraries not available. Install with: pip install PyPDF2 pdfplumber")

# For email processing
try:
    import email
    from email import policy
    import imaplib
    import smtplib
except ImportError:
    print("Email libraries not available")

# For AI processing (you'll need to install these)
try:
    import openai
    from transformers import pipeline
except ImportError:
    print("AI libraries not available. Install with: pip install openai transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LifeEvent:
    """Represents a life event extracted from client input"""
    event_type: str
    description: str
    planned_date: datetime
    confidence: float
    cash_flow_impact: str  # 'positive', 'negative', 'neutral'
    impact_amount: float
    source_text: str
    extracted_date: datetime
    status: str = 'extracted'  # 'extracted', 'confirmed', 'cancelled'
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'description': self.description,
            'planned_date': self.planned_date.isoformat(),
            'confidence': self.confidence,
            'cash_flow_impact': self.cash_flow_impact,
            'impact_amount': self.impact_amount,
            'source_text': self.source_text,
            'extracted_date': self.extracted_date.isoformat(),
            'status': self.status
        }

@dataclass
class BalanceEntry:
    """Represents a financial balance entry"""
    account_type: str
    balance: float
    date: datetime
    description: str
    source_event: str
    
    def to_dict(self) -> Dict:
        return {
            'account_type': self.account_type,
            'balance': self.balance,
            'date': self.date.isoformat(),
            'description': self.description,
            'source_event': self.source_event
        }

class ClientInputProcessor:
    """Main class for processing client inputs and extracting life events"""
    
    def __init__(self, client_id: str, config_path: str = "config/ips_config.json"):
        self.client_id = client_id
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize storage
        self.extracted_events: List[LifeEvent] = []
        self.balance_tracker = BalanceTracker(client_id)
        self.scenario_manager = ScenarioManager(client_id, self.config)
        
        # Threading for parallel processing
        self.event_queue = queue.Queue()
        self.processing_threads = []
        self.max_threads = 4
        
        # AI model for event extraction
        self.ai_extractor = AIEventExtractor()
        
        logger.info(f"Initialized ClientInputProcessor for client {client_id}")
    
    def _load_config(self) -> Dict:
        """Load the IPS configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def process_email(self, email_content: str, email_date: datetime = None) -> List[LifeEvent]:
        """Process email content and extract life events"""
        logger.info(f"Processing email for client {self.client_id}")
        
        # Clean and preprocess email content
        cleaned_content = self._preprocess_text(email_content)
        
        # Extract events using AI
        events = self.ai_extractor.extract_events(cleaned_content, email_date)
        
        # Add to extracted events
        for event in events:
            event.client_id = self.client_id
            self.extracted_events.append(event)
        
        # Queue for balance recalculation
        self._queue_balance_recalculation(events)
        
        logger.info(f"Extracted {len(events)} events from email")
        return events
    
    def process_pdf(self, pdf_path: str) -> List[LifeEvent]:
        """Process PDF document and extract life events"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        pdf_text = self._extract_pdf_text(pdf_path)
        
        # Process the extracted text
        return self.process_email(pdf_text)
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"PDF extraction failed: {e2}")
                return ""
        
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better AI extraction"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove email headers and signatures
        lines = text.split('\n')
        cleaned_lines = []
        in_body = False
        
        for line in lines:
            if line.strip().startswith('From:') or line.strip().startswith('To:'):
                continue
            if line.strip().startswith('Subject:'):
                continue
            if line.strip().startswith('Date:'):
                continue
            if line.strip().startswith('--') or line.strip().startswith('Sent from'):
                break
            if line.strip():
                in_body = True
            if in_body:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _queue_balance_recalculation(self, events: List[LifeEvent]):
        """Queue events for balance recalculation in separate threads"""
        for event in events:
            self.event_queue.put(event)
        
        # Start processing threads if not already running
        if len(self.processing_threads) < self.max_threads:
            thread = threading.Thread(target=self._balance_worker, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
    
    def _balance_worker(self):
        """Worker thread for processing balance recalculations"""
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                self._recalculate_balances_for_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                break
    
    def _recalculate_balances_for_event(self, event: LifeEvent):
        """Recalculate balances for all scenarios when a new event is added"""
        logger.info(f"Recalculating balances for event: {event.event_type}")
        
        # Get all possible scenarios
        scenarios = self.scenario_manager.generate_scenarios()
        
        for scenario in scenarios:
            # Create a thread for each scenario calculation
            thread = threading.Thread(
                target=self._calculate_scenario_balance,
                args=(scenario, event),
                daemon=True
            )
            thread.start()
    
    def _calculate_scenario_balance(self, scenario: Dict, new_event: LifeEvent):
        """Calculate balance for a specific scenario with the new event"""
        try:
            # Add the new event to the scenario
            scenario_with_event = scenario.copy()
            scenario_with_event['events'] = scenario.get('events', []) + [new_event]
            
            # Calculate cash flows
            cash_flows = self._calculate_scenario_cashflows(scenario_with_event)
            
            # Update balance tracker
            self.balance_tracker.update_scenario_balance(
                scenario['scenario_id'],
                cash_flows,
                new_event
            )
            
            # Check for stress scenarios
            stress_level = self._calculate_stress_level(cash_flows)
            if stress_level > 0.3:  # High stress threshold
                self._flag_stress_scenario(scenario['scenario_id'], stress_level)
                
        except Exception as e:
            logger.error(f"Error calculating scenario balance: {e}")
    
    def _calculate_scenario_cashflows(self, scenario: Dict) -> List[Dict]:
        """Calculate cash flows for a given scenario"""
        # This would integrate with your existing cash flow calculation logic
        # For now, return a simplified version
        cash_flows = []
        
        for year in range(self.config.get('YEARS', 40)):
            cash_flow = {
                'year': year,
                'income': 0,
                'expenses': 0,
                'net_cash_flow': 0
            }
            
            # Calculate based on scenario parameters
            # This would use your existing cash flow logic from ips_model.py
            
            cash_flows.append(cash_flow)
        
        return cash_flows
    
    def _calculate_stress_level(self, cash_flows: List[Dict]) -> float:
        """Calculate stress level based on cash flows"""
        # Simple stress calculation - could be enhanced
        total_negative = sum(cf['net_cash_flow'] for cf in cash_flows if cf['net_cash_flow'] < 0)
        total_positive = sum(cf['net_cash_flow'] for cf in cash_flows if cf['net_cash_flow'] > 0)
        
        if total_positive == 0:
            return 1.0
        
        stress_ratio = abs(total_negative) / total_positive
        return min(1.0, stress_ratio)
    
    def _flag_stress_scenario(self, scenario_id: str, stress_level: float):
        """Flag a scenario as stressful and potentially rule it out"""
        logger.warning(f"High stress scenario detected: {scenario_id} (level: {stress_level:.2f})")
        
        # Mark scenario as stressful
        self.scenario_manager.flag_stress_scenario(scenario_id, stress_level)
        
        # Check if scenario should be ruled out
        if stress_level > 0.5:  # Very high stress
            self.scenario_manager.rule_out_scenario(scenario_id, "Excessive stress level")
    
    def get_optimal_decisions(self) -> Dict:
        """Get optimal decisions based on current events and scenarios"""
        # Get all feasible scenarios
        feasible_scenarios = self.scenario_manager.get_feasible_scenarios()
        
        # Calculate optimal decisions for each scenario
        optimal_decisions = {}
        
        for scenario in feasible_scenarios:
            decisions = self._calculate_optimal_decisions_for_scenario(scenario)
            optimal_decisions[scenario['scenario_id']] = decisions
        
        return optimal_decisions
    
    def _calculate_optimal_decisions_for_scenario(self, scenario: Dict) -> Dict:
        """Calculate optimal decisions for a specific scenario"""
        # This would implement your decision optimization logic
        # For now, return a simplified structure
        return {
            'portfolio_allocation': scenario.get('RISK_BAND', 2),
            'education_path': scenario.get('ED_PATH', 'McGill'),
            'work_arrangement': scenario.get('HEL_WORK', 'Full-time'),
            'charitable_giving': scenario.get('DON_STYLE', 0),
            'confidence_score': 0.85
        }
    
    def export_configuration_space(self, output_path: str = None) -> str:
        """Export the current configuration space to JSON format"""
        if output_path is None:
            output_path = f"ips_output/{self.client_id}_config_space.json"
        
        config_space = {
            'client_id': self.client_id,
            'extracted_events': [event.to_dict() for event in self.extracted_events],
            'scenarios': self.scenario_manager.get_all_scenarios(),
            'balances': self.balance_tracker.get_all_balances(),
            'optimal_decisions': self.get_optimal_decisions(),
            'export_date': datetime.now().isoformat()
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config_space, f, indent=2)
        
        logger.info(f"Configuration space exported to: {output_path}")
        return output_path

class AIEventExtractor:
    """AI-powered event extraction from natural language"""
    
    def __init__(self):
        # Initialize AI models
        self.event_classifier = None
        self.date_extractor = None
        self.amount_extractor = None
        
        # Event type mappings
        self.event_types = {
            'education': ['school', 'university', 'college', 'tuition', 'education'],
            'work': ['job', 'career', 'promotion', 'salary', 'bonus', 'work'],
            'family': ['marriage', 'divorce', 'child', 'birth', 'adoption'],
            'health': ['medical', 'health', 'surgery', 'treatment', 'insurance'],
            'housing': ['house', 'mortgage', 'rent', 'move', 'property'],
            'financial': ['investment', 'savings', 'debt', 'loan', 'credit'],
            'retirement': ['retirement', 'pension', '401k', 'ira'],
            'charity': ['donation', 'charity', 'giving', 'philanthropy']
        }
    
    def extract_events(self, text: str, reference_date: datetime = None) -> List[LifeEvent]:
        """Extract life events from text using AI"""
        events = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            event = self._extract_event_from_sentence(sentence, reference_date)
            if event:
                events.append(event)
        
        return events
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_event_from_sentence(self, sentence: str, reference_date: datetime) -> Optional[LifeEvent]:
        """Extract a single event from a sentence"""
        # Determine event type
        event_type = self._classify_event_type(sentence)
        if not event_type:
            return None
        
        # Extract date
        event_date = self._extract_date(sentence, reference_date)
        if not event_date:
            return None
        
        # Extract amount
        amount = self._extract_amount(sentence)
        
        # Determine cash flow impact
        cash_flow_impact = self._determine_cash_flow_impact(event_type, amount)
        
        # Calculate confidence
        confidence = self._calculate_confidence(sentence, event_type, event_date, amount)
        
        return LifeEvent(
            event_type=event_type,
            description=sentence,
            planned_date=event_date,
            confidence=confidence,
            cash_flow_impact=cash_flow_impact,
            impact_amount=amount,
            source_text=sentence,
            extracted_date=datetime.now()
        )
    
    def _classify_event_type(self, sentence: str) -> Optional[str]:
        """Classify the type of event in a sentence"""
        sentence_lower = sentence.lower()
        
        for event_type, keywords in self.event_types.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return event_type
        
        return None
    
    def _extract_date(self, sentence: str, reference_date: datetime) -> Optional[datetime]:
        """Extract date from sentence"""
        # Simple date extraction - could be enhanced with NLP
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'in (\d+) (days?|weeks?|months?|years?)',  # Relative dates
            r'next (week|month|year)',  # Relative dates
            r'this (week|month|year)'  # Relative dates
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if 'in' in pattern or 'next' in pattern or 'this' in pattern:
                    # Handle relative dates
                    return self._parse_relative_date(sentence, reference_date)
                else:
                    # Handle absolute dates
                    return self._parse_absolute_date(match)
        
        return None
    
    def _parse_relative_date(self, sentence: str, reference_date: datetime) -> Optional[datetime]:
        """Parse relative dates like 'in 3 months'"""
        if not reference_date:
            reference_date = datetime.now()
        
        # Simple relative date parsing
        if 'next year' in sentence.lower():
            return reference_date.replace(year=reference_date.year + 1)
        elif 'next month' in sentence.lower():
            return reference_date.replace(month=reference_date.month + 1)
        elif 'next week' in sentence.lower():
            return reference_date + timedelta(weeks=1)
        
        # Extract number and unit
        match = re.search(r'in (\d+) (days?|weeks?|months?|years?)', sentence.lower())
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if 'year' in unit:
                return reference_date.replace(year=reference_date.year + number)
            elif 'month' in unit:
                return reference_date.replace(month=reference_date.month + number)
            elif 'week' in unit:
                return reference_date + timedelta(weeks=number)
            elif 'day' in unit:
                return reference_date + timedelta(days=number)
        
        return None
    
    def _parse_absolute_date(self, match) -> Optional[datetime]:
        """Parse absolute dates"""
        try:
            if len(match.groups()) == 3:
                if match.group(1).isdigit() and len(match.group(1)) == 4:
                    # YYYY-MM-DD format
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                else:
                    # MM/DD/YYYY format
                    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                
                return datetime(year, month, day)
        except (ValueError, IndexError):
            return None
        
        return None
    
    def _extract_amount(self, sentence: str) -> float:
        """Extract monetary amount from sentence"""
        # Simple amount extraction
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        match = re.search(amount_pattern, sentence)
        
        if match:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        
        return 0.0
    
    def _determine_cash_flow_impact(self, event_type: str, amount: float) -> str:
        """Determine if event has positive, negative, or neutral cash flow impact"""
        if amount == 0:
            return 'neutral'
        
        # Define impact by event type
        negative_events = ['education', 'health', 'housing', 'charity']
        positive_events = ['work', 'retirement']
        
        if event_type in negative_events:
            return 'negative'
        elif event_type in positive_events:
            return 'positive'
        else:
            return 'negative' if amount < 0 else 'positive'
    
    def _calculate_confidence(self, sentence: str, event_type: str, 
                            event_date: datetime, amount: float) -> float:
        """Calculate confidence score for extracted event"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear event types
        if event_type:
            confidence += 0.2
        
        # Boost confidence for clear dates
        if event_date:
            confidence += 0.2
        
        # Boost confidence for clear amounts
        if amount != 0:
            confidence += 0.1
        
        # Boost confidence for longer, more detailed sentences
        if len(sentence.split()) > 5:
            confidence += 0.1
        
        return min(1.0, confidence)

class BalanceTracker:
    """Track financial balances across different scenarios"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.scenario_balances: Dict[str, List[BalanceEntry]] = {}
        self.account_types = ['checking', 'savings', 'investment', 'retirement', 'debt']
    
    def update_scenario_balance(self, scenario_id: str, cash_flows: List[Dict], 
                              new_event: LifeEvent):
        """Update balance for a specific scenario"""
        if scenario_id not in self.scenario_balances:
            self.scenario_balances[scenario_id] = []
        
        # Calculate new balance based on cash flows and event
        current_balance = self._get_current_balance(scenario_id)
        
        # Apply cash flows
        for cash_flow in cash_flows:
            current_balance += cash_flow.get('net_cash_flow', 0)
        
        # Apply event impact
        if new_event.cash_flow_impact == 'negative':
            current_balance -= new_event.impact_amount
        elif new_event.cash_flow_impact == 'positive':
            current_balance += new_event.impact_amount
        
        # Create new balance entry
        balance_entry = BalanceEntry(
            account_type='total',
            balance=current_balance,
            date=datetime.now(),
            description=f"Updated after {new_event.event_type}",
            source_event=new_event.event_type
        )
        
        self.scenario_balances[scenario_id].append(balance_entry)
    
    def _get_current_balance(self, scenario_id: str) -> float:
        """Get current balance for a scenario"""
        if scenario_id not in self.scenario_balances:
            return 0.0
        
        balances = self.scenario_balances[scenario_id]
        if not balances:
            return 0.0
        
        return balances[-1].balance
    
    def get_all_balances(self) -> Dict[str, List[Dict]]:
        """Get all balances for all scenarios"""
        return {
            scenario_id: [entry.to_dict() for entry in balances]
            for scenario_id, balances in self.scenario_balances.items()
        }

class ScenarioManager:
    """Manage different life scenarios and their feasibility"""
    
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        self.scenarios: Dict[str, Dict] = {}
        self.stress_scenarios: Dict[str, float] = {}
        self.ruled_out_scenarios: Dict[str, str] = {}  # scenario_id -> reason
    
    def generate_scenarios(self) -> List[Dict]:
        """Generate all possible scenarios based on configuration"""
        factor_space = self.config.get('FACTOR_SPACE', {})
        
        # Generate all combinations
        keys = list(factor_space.keys())
        values = list(factor_space.values())
        
        scenarios = []
        for i, combination in enumerate(itertools.product(*values)):
            scenario = dict(zip(keys, combination))
            scenario['scenario_id'] = f"SCENARIO_{i:04d}"
            scenario['events'] = []
            scenario['status'] = 'active'
            
            scenarios.append(scenario)
            self.scenarios[scenario['scenario_id']] = scenario
        
        return scenarios
    
    def flag_stress_scenario(self, scenario_id: str, stress_level: float):
        """Flag a scenario as stressful"""
        self.stress_scenarios[scenario_id] = stress_level
        
        if scenario_id in self.scenarios:
            self.scenarios[scenario_id]['stress_level'] = stress_level
    
    def rule_out_scenario(self, scenario_id: str, reason: str):
        """Rule out a scenario as infeasible"""
        self.ruled_out_scenarios[scenario_id] = reason
        
        if scenario_id in self.scenarios:
            self.scenarios[scenario_id]['status'] = 'ruled_out'
            self.scenarios[scenario_id]['ruled_out_reason'] = reason
    
    def get_feasible_scenarios(self) -> List[Dict]:
        """Get all feasible scenarios (not ruled out)"""
        return [
            scenario for scenario in self.scenarios.values()
            if scenario['status'] == 'active'
        ]
    
    def get_all_scenarios(self) -> Dict[str, Dict]:
        """Get all scenarios"""
        return self.scenarios

# Import itertools for scenario generation
import itertools

def demo_client_input_processing():
    """Demo the client input processing pipeline"""
    print("=== Client Input Processing Demo ===")
    
    # Initialize processor
    processor = ClientInputProcessor("DEMO_CLIENT_001")
    
    # Sample email content
    sample_email = """
    Hi,
    
    I wanted to update you on some changes in our situation. 
    We've decided to send our child to Johns Hopkins University starting next year, 
    which will cost about $110,000 per year. Also, my wife is planning to work 
    part-time starting in 3 months to help with childcare costs.
    
    We're also considering buying a new house in 2 years for around $2.5 million.
    
    Thanks,
    Client
    """
    
    # Process the email
    events = processor.process_email(sample_email)
    
    print(f"\nExtracted {len(events)} events:")
    for event in events:
        print(f"- {event.event_type}: {event.description}")
        print(f"  Date: {event.planned_date.strftime('%Y-%m-%d')}")
        print(f"  Impact: {event.cash_flow_impact} ${event.impact_amount:,.2f}")
        print(f"  Confidence: {event.confidence:.2f}")
        print()
    
    # Get optimal decisions
    optimal_decisions = processor.get_optimal_decisions()
    print(f"\nOptimal decisions for {len(optimal_decisions)} scenarios")
    
    # Export configuration space
    output_path = processor.export_configuration_space()
    print(f"\nConfiguration space exported to: {output_path}")

if __name__ == "__main__":
    demo_client_input_processing() 