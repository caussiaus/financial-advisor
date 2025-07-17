"""
Real People Integrator

This module integrates real people data from the provided JSON format into the trial people manager
for vector database processing and similarity matching.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from .trial_people_manager import TrialPeopleManager, TrialPerson
from .json_to_vector_converter import LifeStage, EventCategory


@dataclass
class RealPersonData:
    """Represents real person data from the JSON format"""
    name: str
    entity_type: str
    income: float
    savings: float
    events: List[Dict[str, Any]]
    modulators: List[Dict[str, Any]]
    relationships: List[str]
    source_text: str


class RealPeopleIntegrator:
    """
    Integrates real people data into the trial people manager system
    """
    
    def __init__(self, trial_manager: TrialPeopleManager):
        self.trial_manager = trial_manager
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the integrator"""
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
    
    def parse_real_people_json(self, json_data: List[Dict[str, Any]]) -> List[RealPersonData]:
        """Parse the real people JSON format into structured data"""
        real_people = []
        
        for person_data in json_data:
            # Extract entity information
            entities = person_data.get('entities', [])
            if not entities:
                continue
                
            entity = entities[0]  # Take the first entity
            
            # Extract basic information
            name = entity.get('name', 'Unknown')
            entity_type = entity.get('entity_type', 'person')
            initial_balances = entity.get('initial_balances', {})
            relationships = entity.get('relationships', [])
            source_text = entity.get('source_text', '')
            
            # Extract income and savings
            income = 0
            savings = 0
            
            for account, amount in initial_balances.items():
                if 'income' in account.lower() or 'salary' in account.lower():
                    income = max(income, amount)
                elif 'savings' in account.lower():
                    savings = max(savings, amount)
            
            # Extract events and modulators
            events = person_data.get('events', [])
            modulators = person_data.get('modulators', [])
            
            real_person = RealPersonData(
                name=name,
                entity_type=entity_type,
                income=income,
                savings=savings,
                events=events,
                modulators=modulators,
                relationships=relationships,
                source_text=source_text
            )
            
            real_people.append(real_person)
        
        return real_people
    
    def estimate_age_from_income_and_events(self, income: float, events: List[Dict], modulators: List[Dict]) -> int:
        """Estimate age based on income level and life events"""
        # Base age estimation on income level
        if income < 50000:
            base_age = 25  # Early career
        elif income < 80000:
            base_age = 32  # Mid career
        elif income < 120000:
            base_age = 38  # Established
        elif income < 150000:
            base_age = 45  # Senior level
        else:
            base_age = 50  # Executive level
        
        # Adjust based on events and modulators
        for event in events:
            if 'retirement' in event.get('description', '').lower():
                base_age = max(base_age, 55)
            elif 'education' in event.get('description', '').lower() or 'mba' in event.get('description', '').lower():
                base_age = min(base_age, 35)
        
        for modulator in modulators:
            if 'retirement' in modulator.get('modulator_type', '').lower():
                base_age = max(base_age, 55)
            elif 'marriage' in modulator.get('modulator_type', '').lower():
                base_age = min(base_age, 40)
            elif 'childcare' in modulator.get('description', '').lower():
                base_age = min(base_age, 35)
        
        return base_age
    
    def determine_life_stage(self, age: int, income: float, events: List[Dict]) -> LifeStage:
        """Determine life stage based on age, income, and events"""
        if age < 30:
            return LifeStage.EARLY_CAREER
        elif age < 45:
            return LifeStage.MID_CAREER
        elif age < 60:
            return LifeStage.ESTABLISHED
        elif age < 68:
            return LifeStage.PRE_RETIREMENT
        else:
            return LifeStage.RETIREMENT
    
    def estimate_risk_tolerance(self, income: float, savings: float, events: List[Dict]) -> float:
        """Estimate risk tolerance based on financial profile"""
        # Base risk tolerance on income level
        if income < 50000:
            base_risk = 0.3  # Conservative
        elif income < 80000:
            base_risk = 0.5  # Moderate
        elif income < 120000:
            base_risk = 0.7  # Moderate-aggressive
        else:
            base_risk = 0.8  # Aggressive
        
        # Adjust based on savings rate
        savings_rate = savings / income if income > 0 else 0
        if savings_rate > 0.5:
            base_risk += 0.1  # Higher savings = higher risk tolerance
        elif savings_rate < 0.1:
            base_risk -= 0.1  # Lower savings = lower risk tolerance
        
        # Adjust based on events
        for event in events:
            if 'business' in event.get('description', '').lower():
                base_risk += 0.2  # Business owners tend to be more risk-tolerant
            elif 'freelance' in event.get('description', '').lower():
                base_risk += 0.1  # Freelancers have some risk tolerance
        
        return max(0.1, min(1.0, base_risk))
    
    def convert_events_to_lifestyle_events(self, events: List[Dict], modulators: List[Dict]) -> List[Dict]:
        """Convert real events to lifestyle events format"""
        lifestyle_events = []
        
        # Process income events
        for event in events:
            if event.get('event_type') == 'income':
                lifestyle_events.append({
                    'event_type': 'career',
                    'expected_age': self.estimate_age_from_income_and_events(
                        event.get('amount', 0), events, modulators
                    ),
                    'estimated_cost': event.get('amount', 0),
                    'probability': event.get('probability', 1.0)
                })
        
        # Process modulators as events
        for modulator in modulators:
            modulator_type = modulator.get('modulator_type', '')
            
            if 'education' in modulator_type.lower() or 'mba' in modulator.get('description', '').lower():
                lifestyle_events.append({
                    'event_type': 'education',
                    'expected_age': self.estimate_age_from_income_and_events(0, events, modulators),
                    'estimated_cost': -50000,  # Cost of education
                    'probability': 0.8
                })
            elif 'retirement' in modulator_type.lower():
                lifestyle_events.append({
                    'event_type': 'retirement',
                    'expected_age': 60,
                    'estimated_cost': 0,
                    'probability': 0.9
                })
            elif 'marriage' in modulator_type.lower():
                lifestyle_events.append({
                    'event_type': 'family',
                    'expected_age': self.estimate_age_from_income_and_events(0, events, modulators),
                    'estimated_cost': 25000,  # Wedding costs
                    'probability': 0.9
                })
            elif 'asset_purchase' in modulator_type.lower():
                lifestyle_events.append({
                    'event_type': 'housing',
                    'expected_age': self.estimate_age_from_income_and_events(0, events, modulators),
                    'estimated_cost': 300000,  # Asset purchase
                    'probability': 0.7
                })
            elif 'expense' in modulator_type.lower():
                lifestyle_events.append({
                    'event_type': 'family',
                    'expected_age': self.estimate_age_from_income_and_events(0, events, modulators),
                    'estimated_cost': 15000,  # Childcare expense
                    'probability': 1.0
                })
        
        return lifestyle_events
    
    def create_trial_person_files(self, real_people: List[RealPersonData]) -> List[str]:
        """Create trial person files for the real people"""
        created_folders = []
        
        for person in real_people:
            # Estimate age and life stage
            age = self.estimate_age_from_income_and_events(person.income, person.events, person.modulators)
            life_stage = self.determine_life_stage(age, person.income, person.events)
            risk_tolerance = self.estimate_risk_tolerance(person.income, person.savings, person.events)
            
            # Create folder name
            folder_name = person.name.lower().replace(' ', '_').replace('-', '_')
            
            # Create personal info
            personal_info = {
                'name': person.name,
                'age': age,
                'income': person.income,
                'net_worth': person.savings,
                'risk_tolerance': risk_tolerance,
                'life_stage': life_stage.value
            }
            
            # Create lifestyle events
            lifestyle_events = {
                'events': self.convert_events_to_lifestyle_events(person.events, person.modulators)
            }
            
            # Create financial profile
            monthly_income = person.income / 12
            monthly_expenses = monthly_income * 0.7  # Estimate 70% of income as expenses
            savings_rate = (monthly_income - monthly_expenses) / monthly_income if monthly_income > 0 else 0
            debt_to_income_ratio = 0.2  # Conservative estimate
            
            financial_profile = {
                'monthly_income': monthly_income,
                'monthly_expenses': monthly_expenses,
                'savings_rate': savings_rate,
                'debt_to_income_ratio': debt_to_income_ratio,
                'investment_portfolio': {
                    'stocks': 0.6,
                    'bonds': 0.3,
                    'cash': 0.1
                }
            }
            
            # Create goals based on life stage
            if life_stage == LifeStage.EARLY_CAREER:
                goals = {
                    'short_term_goals': ['emergency_fund', 'debt_payoff'],
                    'medium_term_goals': ['career_advancement', 'education_fund'],
                    'long_term_goals': ['retirement_savings', 'investment_portfolio']
                }
            elif life_stage == LifeStage.MID_CAREER:
                goals = {
                    'short_term_goals': ['family_expenses', 'house_purchase'],
                    'medium_term_goals': ['education_fund', 'career_growth'],
                    'long_term_goals': ['retirement_savings', 'estate_planning']
                }
            elif life_stage == LifeStage.ESTABLISHED:
                goals = {
                    'short_term_goals': ['tax_optimization', 'investment_rebalancing'],
                    'medium_term_goals': ['education_funding', 'business_development'],
                    'long_term_goals': ['retirement_planning', 'wealth_preservation']
                }
            elif life_stage == LifeStage.PRE_RETIREMENT:
                goals = {
                    'short_term_goals': ['health_planning', 'tax_efficiency'],
                    'medium_term_goals': ['retirement_transition', 'legacy_planning'],
                    'long_term_goals': ['retirement_income', 'estate_planning']
                }
            else:  # Retirement
                goals = {
                    'short_term_goals': ['income_optimization', 'health_care'],
                    'medium_term_goals': ['legacy_planning', 'family_support'],
                    'long_term_goals': ['estate_planning', 'wealth_transfer']
                }
            
            # Create folder and files
            folder_path = self.trial_manager.upload_dir / folder_name
            folder_path.mkdir(exist_ok=True)
            
            # Write files
            with open(folder_path / 'PERSONAL_INFO.json', 'w') as f:
                json.dump(personal_info, f, indent=2)
            
            with open(folder_path / 'LIFESTYLE_EVENTS.json', 'w') as f:
                json.dump(lifestyle_events, f, indent=2)
            
            with open(folder_path / 'FINANCIAL_PROFILE.json', 'w') as f:
                json.dump(financial_profile, f, indent=2)
            
            with open(folder_path / 'GOALS.json', 'w') as f:
                json.dump(goals, f, indent=2)
            
            created_folders.append(folder_name)
            self.logger.info(f"Created trial person files for {person.name} ({folder_name})")
        
        return created_folders
    
    def integrate_real_people(self, json_data: List[Dict[str, Any]]) -> List[str]:
        """Integrate real people data into the trial people manager"""
        self.logger.info(f"Integrating {len(json_data)} real people into trial system")
        
        # Parse real people data
        real_people = self.parse_real_people_json(json_data)
        self.logger.info(f"Parsed {len(real_people)} real people")
        
        # Create trial person files
        created_folders = self.create_trial_person_files(real_people)
        self.logger.info(f"Created {len(created_folders)} trial person folders")
        
        return created_folders
    
    def process_real_people_with_analysis(self, json_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process real people with complete analysis"""
        self.logger.info("Starting real people integration and analysis")
        
        # Integrate real people
        created_folders = self.integrate_real_people(json_data)
        
        # Scan for uploaded people
        self.logger.info("Scanning for uploaded trial people...")
        people_folders = self.trial_manager.scan_upload_directory()
        
        if people_folders:
            self.logger.info(f"Found {len(people_folders)} trial people")
            
            # Ingest trial people
            self.logger.info("Ingesting trial people...")
            for folder in people_folders:
                person = self.trial_manager.ingest_trial_person(folder)
                self.logger.info(f"Ingested {person.name} ({person.person_id})")
            
            # Process with mesh engine
            self.logger.info("Processing with mesh engine...")
            for person in self.trial_manager.trial_people.values():
                person = self.trial_manager.process_trial_person_with_mesh(person)
                self.logger.info(f"Processed {person.name} with mesh engine")
            
            # Interpolate surfaces
            self.logger.info("Interpolating surfaces...")
            surfaces = self.trial_manager.interpolate_surfaces_across_group()
            self.logger.info(f"Generated {len(surfaces)} interpolated surfaces")
            
            # Schedule tasks
            self.logger.info("Scheduling tasks...")
            tasks = self.trial_manager.schedule_tasks()
            self.logger.info(f"Scheduled {len(tasks)} tasks")
            
            # Identify less dense sections
            self.logger.info("Identifying less dense sections...")
            density_analysis = self.trial_manager.identify_less_dense_sections()
            if density_analysis:
                self.logger.info(f"Found {density_analysis['outliers_found']} outliers")
                self.logger.info(f"Found {len(density_analysis['low_density_people'])} low-density people")
            
            # Create visualizations
            self.logger.info("Creating topology visualizations...")
            viz_files = self.trial_manager.visualize_high_dimensional_topology()
            self.logger.info(f"Created {len(viz_files)} visualizations")
            
            # Save results
            self.logger.info("Saving analysis results...")
            results_file = self.trial_manager.save_analysis_results()
            self.logger.info(f"Saved results to {results_file}")
            
            return {
                'created_folders': created_folders,
                'surfaces': surfaces,
                'density_analysis': density_analysis,
                'viz_files': viz_files,
                'results_file': results_file
            }
        else:
            self.logger.warning("No trial people found after integration")
            return {
                'created_folders': created_folders,
                'surfaces': {},
                'density_analysis': {},
                'viz_files': {},
                'results_file': None
            }


def create_real_people_demo():
    """Create a demo with real people data"""
    print("🚀 Real People Integration Demo")
    print("=" * 50)
    
    # Create trial people manager
    trial_manager = TrialPeopleManager()
    
    # Create integrator
    integrator = RealPeopleIntegrator(trial_manager)
    
    # Sample real people data (you can replace this with your actual data)
    real_people_data = [
        {
            "events": [
                {
                    "event_id": "evt_101",
                    "description": "Annual salary for software engineer",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 120000,
                    "amount_type": "annual",
                    "account_affected": "salary_income",
                    "tax_implications": {"federal": 0.20, "state": 0.06},
                    "probability": 1.0,
                    "source_text": "Currently earning an annual salary of $120,000 as a software engineer."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_101",
                    "description": "Plans to pursue MBA",
                    "modulator_type": "education",
                    "date": "2026-09-01",
                    "profile_change": "Temporary income loss during MBA studies",
                    "accounts_impacted": ["salary_income", "education_expenses"],
                    "source_text": "Considering leaving current job to pursue an MBA in September 2026."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_101",
                    "name": "Alex",
                    "entity_type": "person",
                    "initial_balances": {"salary": 120000, "savings": 30000},
                    "relationships": [],
                    "source_text": "Alex, a software engineer, has an annual income of $120,000."
                }
            ]
        },
        {
            "events": [
                {
                    "event_id": "evt_102",
                    "description": "Annual freelance graphic design income",
                    "event_type": "income",
                    "date": "2024-01-01",
                    "amount": 85000,
                    "amount_type": "annual",
                    "account_affected": "freelance_income",
                    "tax_implications": {"federal": 0.18, "state": 0.05},
                    "probability": 1.0,
                    "source_text": "Freelance graphic designer earning $85,000 annually."
                }
            ],
            "modulators": [
                {
                    "modulator_id": "mod_102",
                    "description": "Planning to buy a new studio",
                    "modulator_type": "asset_purchase",
                    "date": "2025-05-01",
                    "profile_change": "One-time expense of $300,000",
                    "accounts_impacted": ["real_estate_asset", "savings"],
                    "source_text": "Plans to purchase a studio space for $300,000 in May 2025."
                }
            ],
            "entities": [
                {
                    "entity_id": "ent_102",
                    "name": "Jordan",
                    "entity_type": "person",
                    "initial_balances": {"freelance_income": 85000, "savings": 45000},
                    "relationships": [],
                    "source_text": "Jordan is a freelance graphic designer with annual earnings of $85,000."
                }
            ]
        }
    ]
    
    # Process real people
    results = integrator.process_real_people_with_analysis(real_people_data)
    
    print(f"\n📊 Integration Results:")
    print(f"   Created {len(results['created_folders'])} trial person folders")
    print(f"   Generated {len(results['surfaces'])} interpolated surfaces")
    print(f"   Created {len(results['viz_files'])} visualizations")
    
    if results['density_analysis']:
        print(f"   Found {results['density_analysis']['outliers_found']} outliers")
        print(f"   Found {len(results['density_analysis']['low_density_people'])} low-density people")
    
    print(f"   Results saved to {results['results_file']}")
    
    return integrator, results


if __name__ == "__main__":
    integrator, results = create_real_people_demo()
    print("\n✅ Real People Integration Demo completed!") 