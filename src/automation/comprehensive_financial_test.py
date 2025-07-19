"""
Comprehensive Financial Test

Generate a wide range of financial profiles with realistic income, expense,
and savings rate distributions to thoroughly test the financial advisor system.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from faker import Faker
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()


class ComprehensiveFinancialTest:
    """Generate comprehensive financial profiles with realistic distributions"""
    
    def __init__(self):
        self.test_dir = Path("data/test_comprehensive")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Define realistic financial ranges
        self.income_ranges = {
            "low": (2000, 5000),      # $2K-$5K monthly
            "medium": (5000, 15000),   # $5K-$15K monthly
            "high": (15000, 50000),    # $15K-$50K monthly
            "very_high": (50000, 100000)  # $50K-$100K monthly
        }
        
        self.savings_rate_ranges = {
            "struggling": (0, 10),     # 0-10% savings
            "minimal": (10, 20),       # 10-20% savings
            "moderate": (20, 40),      # 20-40% savings
            "good": (40, 60),          # 40-60% savings
            "excellent": (60, 80),     # 60-80% savings
            "exceptional": (80, 95)    # 80-95% savings
        }
        
        self.expense_categories = {
            "Housing": (0.25, 0.50),      # 25-50% of income
            "Transportation": (0.10, 0.20), # 10-20% of income
            "Food": (0.08, 0.15),         # 8-15% of income
            "Healthcare": (0.05, 0.12),    # 5-12% of income
            "Utilities": (0.05, 0.10),    # 5-10% of income
            "Entertainment": (0.03, 0.08), # 3-8% of income
            "Shopping": (0.05, 0.12),     # 5-12% of income
            "Education": (0.02, 0.08),    # 2-8% of income
            "Insurance": (0.03, 0.08),    # 3-8% of income
            "Miscellaneous": (0.02, 0.05) # 2-5% of income
        }
    
    def generate_realistic_income(self, income_level: str) -> float:
        """Generate realistic income based on level"""
        min_income, max_income = self.income_ranges[income_level]
        return round(random.uniform(min_income, max_income), 2)
    
    def generate_expense_distribution(self, monthly_income: float, target_savings_rate: float) -> Dict[str, float]:
        """Generate realistic expense distribution"""
        available_for_expenses = monthly_income * (1 - target_savings_rate / 100)
        
        # Start with base allocations
        expenses = {}
        total_allocated = 0
        
        # Allocate to major categories first
        for category, (min_pct, max_pct) in self.expense_categories.items():
            if total_allocated >= available_for_expenses:
                break
                
            # Calculate remaining budget
            remaining = available_for_expenses - total_allocated
            remaining_categories = len(self.expense_categories) - len(expenses)
            
            # Adjust allocation based on remaining budget
            if remaining_categories > 0:
                max_allocation = min(remaining, available_for_expenses * max_pct)
                min_allocation = min(remaining / remaining_categories, available_for_expenses * min_pct)
                
                allocation = random.uniform(min_allocation, max_allocation)
                expenses[category] = round(allocation, 2)
                total_allocated += allocation
            else:
                # Distribute remaining amount
                expenses[category] = round(remaining, 2)
                total_allocated += remaining
        
        return expenses
    
    def generate_transaction_history(self, monthly_income: float, expenses: Dict[str, float], 
                                   num_transactions: int = 50) -> List[Dict]:
        """Generate realistic transaction history"""
        transactions = []
        
        # Add income transactions
        income_transactions = random.randint(1, 3)  # 1-3 income sources
        for i in range(income_transactions):
            income_amount = monthly_income / income_transactions
            transactions.append({
                "date": fake.date_between(start_date='-30d', end_date='today').isoformat(),
                "amount": round(income_amount, 2),
                "category": "Income",
                "description": fake.job(),
                "type": "income"
            })
        
        # Add expense transactions
        remaining_transactions = num_transactions - income_transactions
        
        for category, amount in expenses.items():
            if remaining_transactions <= 0:
                break
                
            # Split category amount into multiple transactions
            num_category_transactions = max(1, min(remaining_transactions // len(expenses), 5))
            
            for _ in range(num_category_transactions):
                transaction_amount = amount / num_category_transactions
                if transaction_amount > 0:
                    transactions.append({
                        "date": fake.date_between(start_date='-30d', end_date='today').isoformat(),
                        "amount": -round(transaction_amount, 2),  # Negative for expenses
                        "category": category,
                        "description": fake.company(),
                        "type": "expense"
                    })
                    remaining_transactions -= 1
        
        # Sort by date
        transactions.sort(key=lambda x: x["date"])
        return transactions
    
    def create_financial_profile(self, profile_id: str, income_level: str, 
                               savings_level: str) -> Dict:
        """Create a comprehensive financial profile"""
        
        # Generate income
        monthly_income = self.generate_realistic_income(income_level)
        
        # Get target savings rate
        min_savings, max_savings = self.savings_rate_ranges[savings_level]
        target_savings_rate = random.uniform(min_savings, max_savings)
        
        # Generate expenses
        expenses = self.generate_expense_distribution(monthly_income, target_savings_rate)
        total_expenses = sum(expenses.values())
        
        # Calculate actual savings rate
        actual_savings_rate = ((monthly_income - total_expenses) / monthly_income * 100) if monthly_income > 0 else 0
        
        # Generate transaction history
        transactions = self.generate_transaction_history(monthly_income, expenses)
        
        # Determine risk tolerance based on spending patterns
        expense_volatility = np.std([abs(t["amount"]) for t in transactions if t["amount"] < 0])
        if expense_volatility > 1000:
            risk_tolerance = "high"
        elif expense_volatility > 500:
            risk_tolerance = "medium"
        else:
            risk_tolerance = "low"
        
        profile = {
            "profile_id": profile_id,
            "income_level": income_level,
            "savings_level": savings_level,
            "monthly_income": monthly_income,
            "monthly_expenses": total_expenses,
            "savings_rate": actual_savings_rate,
            "risk_tolerance": risk_tolerance,
            "expense_categories": expenses,
            "transactions": transactions,
            "financial_health": self._assess_financial_health(actual_savings_rate, total_expenses, monthly_income)
        }
        
        return profile
    
    def _assess_financial_health(self, savings_rate: float, expenses: float, income: float) -> str:
        """Assess overall financial health"""
        if savings_rate >= 50:
            return "excellent"
        elif savings_rate >= 30:
            return "good"
        elif savings_rate >= 15:
            return "moderate"
        elif savings_rate >= 5:
            return "struggling"
        else:
            return "critical"
    
    def generate_comprehensive_test_set(self, num_profiles: int = 100) -> List[Dict]:
        """Generate a comprehensive test set with diverse financial scenarios"""
        
        profiles = []
        
        # Define test scenarios
        scenarios = [
            # Income levels
            ("low", "struggling"),
            ("low", "minimal"),
            ("low", "moderate"),
            ("medium", "struggling"),
            ("medium", "minimal"),
            ("medium", "moderate"),
            ("medium", "good"),
            ("high", "minimal"),
            ("high", "moderate"),
            ("high", "good"),
            ("high", "excellent"),
            ("very_high", "moderate"),
            ("very_high", "good"),
            ("very_high", "excellent"),
            ("very_high", "exceptional")
        ]
        
        # Generate profiles for each scenario
        profiles_per_scenario = max(1, num_profiles // len(scenarios))
        
        for i, (income_level, savings_level) in enumerate(scenarios):
            for j in range(profiles_per_scenario):
                profile_id = f"comprehensive_{income_level}_{savings_level}_{i}_{j:03d}"
                profile = self.create_financial_profile(profile_id, income_level, savings_level)
                profiles.append(profile)
        
        # Add some random variations
        remaining_profiles = num_profiles - len(profiles)
        for i in range(remaining_profiles):
            income_level = random.choice(list(self.income_ranges.keys()))
            savings_level = random.choice(list(self.savings_rate_ranges.keys()))
            profile_id = f"comprehensive_random_{i:03d}"
            profile = self.create_financial_profile(profile_id, income_level, savings_level)
            profiles.append(profile)
        
        return profiles
    
    def save_test_data(self, profiles: List[Dict]):
        """Save comprehensive test data"""
        
        # Save individual profiles
        profiles_dir = self.test_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        for profile in profiles:
            profile_file = profiles_dir / f"{profile['profile_id']}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile, f, indent=2)
        
        # Save summary statistics
        summary = {
            "test_date": datetime.now().isoformat(),
            "total_profiles": len(profiles),
            "income_distribution": {},
            "savings_distribution": {},
            "financial_health_distribution": {},
            "statistics": {
                "average_income": np.mean([p["monthly_income"] for p in profiles]),
                "average_expenses": np.mean([p["monthly_expenses"] for p in profiles]),
                "average_savings_rate": np.mean([p["savings_rate"] for p in profiles]),
                "min_savings_rate": min([p["savings_rate"] for p in profiles]),
                "max_savings_rate": max([p["savings_rate"] for p in profiles]),
                "income_range": {
                    "min": min([p["monthly_income"] for p in profiles]),
                    "max": max([p["monthly_income"] for p in profiles])
                }
            }
        }
        
        # Calculate distributions
        for profile in profiles:
            income_level = profile["income_level"]
            savings_level = profile["savings_level"]
            financial_health = profile["financial_health"]
            
            summary["income_distribution"][income_level] = summary["income_distribution"].get(income_level, 0) + 1
            summary["savings_distribution"][savings_level] = summary["savings_distribution"].get(savings_level, 0) + 1
            summary["financial_health_distribution"][financial_health] = summary["financial_health_distribution"].get(financial_health, 0) + 1
        
        # Save summary
        summary_file = self.test_dir / "comprehensive_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(profiles)} comprehensive test profiles")
        logger.info(f"Summary saved to {summary_file}")
        
        return summary
    
    def run_comprehensive_test(self, num_profiles: int = 100):
        """Run the comprehensive financial test"""
        
        logger.info(f"Starting comprehensive financial test with {num_profiles} profiles...")
        
        # Generate comprehensive test set
        profiles = self.generate_comprehensive_test_set(num_profiles)
        
        # Save test data
        summary = self.save_test_data(profiles)
        
        # Print results
        print(f"\n{'='*80}")
        print(f" COMPREHENSIVE FINANCIAL TEST RESULTS")
        print(f"{'='*80}")
        print(f"Total profiles generated: {len(profiles)}")
        print(f"Average monthly income: ${summary['statistics']['average_income']:.2f}")
        print(f"Average monthly expenses: ${summary['statistics']['average_expenses']:.2f}")
        print(f"Average savings rate: {summary['statistics']['average_savings_rate']:.1f}%")
        print(f"Savings rate range: {summary['statistics']['min_savings_rate']:.1f}% - {summary['statistics']['max_savings_rate']:.1f}%")
        
        print(f"\nIncome Distribution:")
        for level, count in summary['income_distribution'].items():
            print(f"  {level}: {count} profiles")
        
        print(f"\nSavings Rate Distribution:")
        for level, count in summary['savings_distribution'].items():
            print(f"  {level}: {count} profiles")
        
        print(f"\nFinancial Health Distribution:")
        for health, count in summary['financial_health_distribution'].items():
            print(f"  {health}: {count} profiles")
        
        print(f"\nTest data saved to: {self.test_dir}")
        
        return profiles, summary


def main():
    """Run comprehensive financial test"""
    test = ComprehensiveFinancialTest()
    
    # Run test with 100 profiles for comprehensive coverage
    profiles, summary = test.run_comprehensive_test(num_profiles=100)
    
    print(f"\n🎉 Comprehensive financial test completed!")
    print(f"📊 Generated {len(profiles)} diverse financial profiles")
    print(f"📊 Savings rates range from {summary['statistics']['min_savings_rate']:.1f}% to {summary['statistics']['max_savings_rate']:.1f}%")
    print(f"📊 Income ranges from ${summary['statistics']['income_range']['min']:.2f} to ${summary['statistics']['income_range']['max']:.2f}")


if __name__ == "__main__":
    main() 