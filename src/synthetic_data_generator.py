import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PersonProfile:
    """Profile of a synthetic person for financial modeling"""
    name: str
    age: int
    occupation: str
    base_income: float
    family_status: str
    location: str
    risk_tolerance: str
    financial_goals: List[str]
    current_assets: Dict[str, float]
    debts: Dict[str, float]


class SyntheticFinancialDataGenerator:
    """
    Generates realistic synthetic natural text describing people's financial situations
    for testing the Omega mesh system
    """
    
    def __init__(self):
        self.occupations = [
            "Software Engineer", "Doctor", "Teacher", "Marketing Manager", "Nurse",
            "Accountant", "Lawyer", "Sales Representative", "Consultant", "Architect",
            "Chef", "Real Estate Agent", "Financial Advisor", "Engineer", "Designer",
            "Project Manager", "Pharmacist", "Therapist", "Writer", "Entrepreneur"
        ]
        
        self.income_ranges = {
            "Software Engineer": (80000, 150000),
            "Doctor": (120000, 300000),
            "Teacher": (40000, 70000),
            "Marketing Manager": (60000, 120000),
            "Nurse": (50000, 80000),
            "Accountant": (45000, 85000),
            "Lawyer": (70000, 200000),
            "Sales Representative": (40000, 100000),
            "Consultant": (70000, 150000),
            "Architect": (55000, 110000),
            "Chef": (35000, 80000),
            "Real Estate Agent": (40000, 120000),
            "Financial Advisor": (50000, 130000),
            "Engineer": (65000, 120000),
            "Designer": (45000, 90000),
            "Project Manager": (70000, 130000),
            "Pharmacist": (90000, 140000),
            "Therapist": (50000, 90000),
            "Writer": (30000, 80000),
            "Entrepreneur": (40000, 200000)
        }
        
        self.family_statuses = [
            "Single", "Married", "Married with 1 child", "Married with 2 children",
            "Married with 3+ children", "Single parent", "Divorced", "Widowed"
        ]
        
        self.locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
            "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
            "Fort Worth, TX", "Columbus, OH", "Charlotte, NC", "Seattle, WA",
            "Denver, CO", "Boston, MA", "Nashville, TN", "Baltimore, MD"
        ]
        
        self.risk_tolerances = ["Conservative", "Moderate", "Aggressive", "Very Aggressive"]
        
        self.financial_goals = [
            "retirement planning", "children's education", "home purchase", "debt payoff",
            "emergency fund", "vacation savings", "investment growth", "business startup",
            "wedding expenses", "healthcare costs", "home renovation", "car purchase",
            "financial independence", "real estate investment", "stock portfolio growth"
        ]
        
        self.life_events = [
            "getting married", "having a baby", "buying a house", "changing jobs",
            "getting promoted", "starting a business", "retiring", "moving cities",
            "going back to school", "paying off student loans", "receiving inheritance",
            "dealing with medical expenses", "getting divorced", "caring for aging parents"
        ]
        
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna"
        ]

    def generate_person_profile(self) -> PersonProfile:
        """Generate a random person profile"""
        name = random.choice(self.first_names)
        age = random.randint(25, 65)
        occupation = random.choice(self.occupations)
        
        # Generate income based on occupation and some randomness
        income_range = self.income_ranges[occupation]
        base_income = random.randint(income_range[0], income_range[1])
        
        family_status = random.choice(self.family_statuses)
        location = random.choice(self.locations)
        risk_tolerance = random.choice(self.risk_tolerances)
        
        # Generate 2-4 financial goals
        goals = random.sample(self.financial_goals, random.randint(2, 4))
        
        # Generate current assets based on age and income
        assets = self._generate_assets(age, base_income)
        debts = self._generate_debts(age, base_income, family_status)
        
        return PersonProfile(
            name=name,
            age=age,
            occupation=occupation,
            base_income=base_income,
            family_status=family_status,
            location=location,
            risk_tolerance=risk_tolerance,
            financial_goals=goals,
            current_assets=assets,
            debts=debts
        )

    def _generate_assets(self, age: int, income: float) -> Dict[str, float]:
        """Generate realistic asset distribution"""
        # Assets generally increase with age and income
        savings_multiplier = (age - 25) / 40 * (income / 60000)
        
        checking = max(1000, income * random.uniform(0.05, 0.15))
        savings = max(5000, income * savings_multiplier * random.uniform(0.2, 0.8))
        investments = max(0, income * savings_multiplier * random.uniform(0.1, 1.5))
        retirement = max(0, income * (age - 25) / 40 * random.uniform(0.5, 2.0))
        
        # Chance of real estate ownership increases with age and income
        real_estate = 0
        if age > 30 and income > 50000 and random.random() < 0.6:
            real_estate = income * random.uniform(2.0, 5.0)
        
        return {
            'checking': round(checking, 2),
            'savings': round(savings, 2),
            'investments': round(investments, 2),
            'retirement': round(retirement, 2),
            'real_estate': round(real_estate, 2)
        }

    def _generate_debts(self, age: int, income: float, family_status: str) -> Dict[str, float]:
        """Generate realistic debt distribution"""
        debts = {}
        
        # Credit card debt (most people have some)
        if random.random() < 0.7:
            debts['credit_cards'] = random.uniform(1000, min(25000, income * 0.3))
        
        # Student loans (more likely for younger people)
        if age < 40 and random.random() < 0.6:
            debts['student_loans'] = random.uniform(10000, min(100000, income * 1.5))
        
        # Mortgage (more likely for families and higher income)
        if 'child' in family_status or income > 60000:
            if random.random() < 0.5:
                debts['mortgage'] = random.uniform(100000, min(500000, income * 6))
        
        # Auto loans
        if random.random() < 0.4:
            debts['auto_loans'] = random.uniform(5000, 40000)
        
        return {k: round(v, 2) for k, v in debts.items()}

    def generate_financial_narrative(self, profile: PersonProfile) -> str:
        """Generate natural text describing someone's financial situation"""
        
        # Calculate net worth
        total_assets = sum(profile.current_assets.values())
        total_debts = sum(profile.debts.values())
        net_worth = total_assets - total_debts
        
        # Generate narrative
        narrative_parts = []
        
        # Introduction
        intro = f"My name is {profile.name}, I'm {profile.age} years old and work as a {profile.occupation} in {profile.location}. "
        intro += f"I'm {profile.family_status.lower()} and currently earn ${profile.base_income:,.0f} per year. "
        narrative_parts.append(intro)
        
        # Current financial situation
        financial_status = f"My current financial situation includes ${profile.current_assets['checking']:,.0f} in checking, "
        financial_status += f"${profile.current_assets['savings']:,.0f} in savings, and "
        financial_status += f"${profile.current_assets['investments']:,.0f} in investments. "
        
        if profile.current_assets['retirement'] > 0:
            financial_status += f"I have ${profile.current_assets['retirement']:,.0f} saved for retirement. "
        
        if profile.current_assets['real_estate'] > 0:
            financial_status += f"I own real estate worth approximately ${profile.current_assets['real_estate']:,.0f}. "
        
        narrative_parts.append(financial_status)
        
        # Debts
        if profile.debts:
            debt_text = "On the debt side, "
            debt_items = []
            for debt_type, amount in profile.debts.items():
                debt_items.append(f"${amount:,.0f} in {debt_type.replace('_', ' ')}")
            debt_text += ", ".join(debt_items) + ". "
            narrative_parts.append(debt_text)
        
        # Risk tolerance and investment philosophy
        risk_text = f"I consider myself a {profile.risk_tolerance.lower()} investor. "
        narrative_parts.append(risk_text)
        
        # Financial goals with timelines
        goals_text = "My main financial goals include: "
        goal_details = []
        
        for goal in profile.financial_goals:
            timeline = self._generate_goal_timeline(goal, profile.age)
            amount = self._generate_goal_amount(goal, profile.base_income)
            goal_details.append(f"{goal} (${amount:,.0f} needed by {timeline})")
        
        goals_text += "; ".join(goal_details) + ". "
        narrative_parts.append(goals_text)
        
        # Future plans and concerns
        future_events = random.sample(self.life_events, random.randint(1, 3))
        future_text = "Looking ahead, I'm planning for " + " and ".join(future_events) + ". "
        narrative_parts.append(future_text)
        
        # Monthly cash flow
        monthly_income = profile.base_income / 12
        monthly_expenses = self._estimate_monthly_expenses(profile)
        monthly_surplus = monthly_income - monthly_expenses
        
        cash_flow_text = f"Each month I bring home about ${monthly_income:,.0f} after taxes and "
        cash_flow_text += f"my typical expenses are around ${monthly_expenses:,.0f}, "
        
        if monthly_surplus > 0:
            cash_flow_text += f"leaving me with approximately ${monthly_surplus:,.0f} that I can save or invest. "
        else:
            cash_flow_text += f"which means I'm currently spending about ${abs(monthly_surplus):,.0f} more than I earn each month. "
        
        narrative_parts.append(cash_flow_text)
        
        # Investment preferences and concerns
        investment_text = self._generate_investment_preferences(profile)
        narrative_parts.append(investment_text)
        
        return "".join(narrative_parts)

    def _generate_goal_timeline(self, goal: str, age: int) -> str:
        """Generate realistic timeline for financial goals"""
        timelines = {
            "retirement planning": f"{65 - age} years",
            "children's education": "10-18 years",
            "home purchase": "2-5 years",
            "debt payoff": "3-7 years",
            "emergency fund": "1-2 years",
            "vacation savings": "1 year",
            "investment growth": "10-20 years",
            "business startup": "2-3 years",
            "wedding expenses": "1-2 years",
            "healthcare costs": "ongoing",
            "home renovation": "2-4 years",
            "car purchase": "1-3 years",
            "financial independence": f"{55 - age} years",
            "real estate investment": "5-10 years",
            "stock portfolio growth": "10-15 years"
        }
        return timelines.get(goal, "5-10 years")

    def _generate_goal_amount(self, goal: str, income: float) -> float:
        """Generate realistic amounts for financial goals"""
        amounts = {
            "retirement planning": income * random.uniform(8, 15),
            "children's education": random.uniform(50000, 200000),
            "home purchase": income * random.uniform(3, 6),
            "debt payoff": income * random.uniform(0.5, 2),
            "emergency fund": income * random.uniform(0.25, 0.75),
            "vacation savings": random.uniform(3000, 15000),
            "investment growth": income * random.uniform(2, 8),
            "business startup": random.uniform(25000, 100000),
            "wedding expenses": random.uniform(15000, 50000),
            "healthcare costs": random.uniform(10000, 50000),
            "home renovation": random.uniform(20000, 100000),
            "car purchase": random.uniform(15000, 60000),
            "financial independence": income * random.uniform(10, 25),
            "real estate investment": income * random.uniform(2, 8),
            "stock portfolio growth": income * random.uniform(1, 5)
        }
        return amounts.get(goal, income)

    def _estimate_monthly_expenses(self, profile: PersonProfile) -> float:
        """Estimate realistic monthly expenses"""
        base_expenses = profile.base_income * 0.6 / 12  # 60% of income as base
        
        # Adjust for family status
        if "child" in profile.family_status:
            child_count = 1
            if "2 children" in profile.family_status:
                child_count = 2
            elif "3+" in profile.family_status:
                child_count = 3
            base_expenses *= (1 + child_count * 0.3)
        
        # Adjust for location (rough cost of living)
        high_cost_cities = ["New York", "Los Angeles", "San Jose", "San Diego", "Boston", "Seattle"]
        if any(city in profile.location for city in high_cost_cities):
            base_expenses *= 1.3
        
        # Add debt payments
        for debt_type, amount in profile.debts.items():
            if debt_type == "mortgage":
                base_expenses += amount * 0.005  # ~6% annual rate
            elif debt_type == "student_loans":
                base_expenses += amount * 0.008  # ~10% annual rate
            elif debt_type == "auto_loans":
                base_expenses += amount * 0.01   # ~12% annual rate
            elif debt_type == "credit_cards":
                base_expenses += amount * 0.015  # ~18% annual rate
        
        return base_expenses

    def _generate_investment_preferences(self, profile: PersonProfile) -> str:
        """Generate text about investment preferences"""
        preferences = {
            "Conservative": "I prefer stable investments like bonds and CDs, and I'm uncomfortable with significant market volatility. ",
            "Moderate": "I'm willing to accept some risk for better returns, typically investing in a mix of stocks and bonds. ",
            "Aggressive": "I'm comfortable with higher risk investments and focus primarily on growth stocks and equity funds. ",
            "Very Aggressive": "I actively seek high-risk, high-reward investments including individual stocks, options, and emerging markets. "
        }
        
        base_text = preferences[profile.risk_tolerance]
        
        # Add specific concerns or preferences
        concerns = [
            "I'm concerned about inflation affecting my purchasing power",
            "I want to ensure I have enough liquidity for emergencies",
            "I'm interested in tax-advantaged investment accounts",
            "I'm considering real estate as part of my portfolio",
            "I want to diversify across different asset classes",
            "I'm worried about market timing and volatility"
        ]
        
        selected_concern = random.choice(concerns)
        return base_text + selected_concern + ". "

    def generate_multiple_profiles(self, count: int) -> List[Tuple[PersonProfile, str]]:
        """Generate multiple synthetic profiles with narratives"""
        profiles = []
        for _ in range(count):
            profile = self.generate_person_profile()
            narrative = self.generate_financial_narrative(profile)
            profiles.append((profile, narrative))
        return profiles

    def save_synthetic_data(self, profiles: List[Tuple[PersonProfile, str]], filename: str):
        """Save synthetic data to file"""
        data = []
        for profile, narrative in profiles:
            data.append({
                'profile': {
                    'name': profile.name,
                    'age': profile.age,
                    'occupation': profile.occupation,
                    'base_income': profile.base_income,
                    'family_status': profile.family_status,
                    'location': profile.location,
                    'risk_tolerance': profile.risk_tolerance,
                    'financial_goals': profile.financial_goals,
                    'current_assets': profile.current_assets,
                    'debts': profile.debts
                },
                'narrative': narrative
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Generate sample data
    generator = SyntheticFinancialDataGenerator()
    
    print("🔄 Generating synthetic financial profiles...")
    profiles = generator.generate_multiple_profiles(10)
    
    print(f"✅ Generated {len(profiles)} profiles")
    
    # Show a sample
    sample_profile, sample_narrative = profiles[0]
    print("\n📄 Sample Profile:")
    print(f"Name: {sample_profile.name}")
    print(f"Age: {sample_profile.age}")
    print(f"Occupation: {sample_profile.occupation}")
    print(f"Income: ${sample_profile.base_income:,}")
    print(f"Net Worth: ${sum(sample_profile.current_assets.values()) - sum(sample_profile.debts.values()):,.0f}")
    
    print("\n📝 Sample Narrative:")
    print(sample_narrative)
    
    # Save data
    generator.save_synthetic_data(profiles, "data/synthetic_financial_profiles.json")
    print(f"\n💾 Saved synthetic data to data/synthetic_financial_profiles.json")