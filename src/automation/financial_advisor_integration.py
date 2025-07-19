"""
Financial Advisor Integration

Integrates the training data with the main financial advisor system
for real financial simulations and analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinancialProfile:
    """Financial profile for advisor analysis"""
    profile_id: str
    monthly_income: float
    monthly_expenses: float
    savings_rate: float
    risk_tolerance: str  # 'low', 'medium', 'high'
    investment_horizon: int  # years
    categories: Dict[str, float]
    transaction_history: List[Dict]


class FinancialAdvisorIntegration:
    """Integrates training data with financial advisor system"""
    
    def __init__(self):
        self.cache_dir = Path("data/cache/statements")
        self.analysis_dir = Path("data/analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_profiles(self) -> List[FinancialProfile]:
        """Load and convert training profiles to financial profiles"""
        
        # Load vectorized profiles
        vectorized_files = list(self.cache_dir.glob("*_vectorized.json"))
        profiles = []
        
        for file_path in vectorized_files:
            try:
                with open(file_path, 'r') as f:
                    vectorized_data = json.load(f)
                
                # Calculate financial metrics
                amounts = np.array(vectorized_data["amounts"])
                categories = vectorized_data["categories"]
                
                # Separate income and expenses
                income_mask = amounts > 0
                expense_mask = amounts < 0
                
                monthly_income = np.sum(amounts[income_mask]) if np.any(income_mask) else 0
                monthly_expenses = abs(np.sum(amounts[expense_mask])) if np.any(expense_mask) else 0
                
                # Calculate savings rate
                savings_rate = ((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0
                
                # Determine risk tolerance based on spending patterns
                expense_volatility = np.std(amounts[expense_mask]) if np.any(expense_mask) else 0
                if expense_volatility > 500:
                    risk_tolerance = "high"
                elif expense_volatility > 200:
                    risk_tolerance = "medium"
                else:
                    risk_tolerance = "low"
                
                # Calculate category spending
                category_spending = {}
                for i, category in enumerate(categories):
                    if i < len(amounts):
                        amount = amounts[i]
                        if amount < 0:  # Only expenses
                            category_spending[category] = category_spending.get(category, 0) + abs(amount)
                
                # Create transaction history
                transaction_history = []
                for i, amount in enumerate(amounts):
                    if i < len(vectorized_data["dates"]) and i < len(categories):
                        transaction_history.append({
                            "date": vectorized_data["dates"][i],
                            "amount": amount,
                            "category": categories[i],
                            "type": "income" if amount > 0 else "expense"
                        })
                
                profile = FinancialProfile(
                    profile_id=vectorized_data["filename"],
                    monthly_income=monthly_income,
                    monthly_expenses=monthly_expenses,
                    savings_rate=savings_rate,
                    risk_tolerance=risk_tolerance,
                    investment_horizon=30,  # Default 30 years
                    categories=category_spending,
                    transaction_history=transaction_history
                )
                
                profiles.append(profile)
                logger.info(f"Created financial profile: {profile.profile_id}")
                logger.info(f"  Income: ${monthly_income:.2f}, Expenses: ${monthly_expenses:.2f}")
                logger.info(f"  Savings Rate: {savings_rate:.1f}%, Risk: {risk_tolerance}")
                
            except Exception as e:
                logger.error(f"Failed to create financial profile for {file_path}: {e}")
        
        return profiles
    
    def run_financial_analysis(self, profiles: List[FinancialProfile]) -> Dict:
        """Run comprehensive financial analysis on profiles"""
        
        analysis_results = {
            "analysis_date": datetime.now().isoformat(),
            "total_profiles": len(profiles),
            "financial_summary": {},
            "risk_analysis": {},
            "investment_recommendations": {},
            "profile_analyses": []
        }
        
        # Aggregate financial metrics
        all_incomes = [p.monthly_income for p in profiles]
        all_expenses = [p.monthly_expenses for p in profiles]
        all_savings_rates = [p.savings_rate for p in profiles]
        
        analysis_results["financial_summary"] = {
            "average_monthly_income": np.mean(all_incomes),
            "average_monthly_expenses": np.mean(all_expenses),
            "average_savings_rate": np.mean(all_savings_rates),
            "total_annual_income": np.sum(all_incomes) * 12,
            "total_annual_expenses": np.sum(all_expenses) * 12,
            "total_annual_savings": np.sum(all_incomes) * 12 - np.sum(all_expenses) * 12
        }
        
        # Risk analysis
        risk_distribution = {}
        for profile in profiles:
            risk = profile.risk_tolerance
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        analysis_results["risk_analysis"] = {
            "risk_distribution": risk_distribution,
            "high_risk_profiles": len([p for p in profiles if p.risk_tolerance == "high"]),
            "medium_risk_profiles": len([p for p in profiles if p.risk_tolerance == "medium"]),
            "low_risk_profiles": len([p for p in profiles if p.risk_tolerance == "low"])
        }
        
        # Generate investment recommendations
        for profile in profiles:
            recommendation = self._generate_investment_recommendation(profile)
            analysis_results["investment_recommendations"][profile.profile_id] = recommendation
            
            # Individual profile analysis
            profile_analysis = {
                "profile_id": profile.profile_id,
                "monthly_income": profile.monthly_income,
                "monthly_expenses": profile.monthly_expenses,
                "savings_rate": profile.savings_rate,
                "risk_tolerance": profile.risk_tolerance,
                "top_categories": sorted(profile.categories.items(), key=lambda x: x[1], reverse=True)[:5],
                "recommendation": recommendation
            }
            analysis_results["profile_analyses"].append(profile_analysis)
        
        return analysis_results
    
    def _generate_investment_recommendation(self, profile: FinancialProfile) -> Dict:
        """Generate personalized investment recommendation"""
        
        # Base allocation based on risk tolerance
        if profile.risk_tolerance == "low":
            stock_allocation = 0.30
            bond_allocation = 0.60
            cash_allocation = 0.10
        elif profile.risk_tolerance == "medium":
            stock_allocation = 0.60
            bond_allocation = 0.30
            cash_allocation = 0.10
        else:  # high risk
            stock_allocation = 0.80
            bond_allocation = 0.15
            cash_allocation = 0.05
        
        # Adjust based on savings rate
        if profile.savings_rate < 10:
            recommendation = "Increase savings rate to at least 15%"
            priority = "high"
        elif profile.savings_rate < 20:
            recommendation = "Good savings rate, focus on investment allocation"
            priority = "medium"
        else:
            recommendation = "Excellent savings rate, maximize investment returns"
            priority = "low"
        
        # Calculate monthly investment amount
        monthly_investment = (profile.monthly_income - profile.monthly_expenses) * 0.8  # 80% of savings
        
        return {
            "stock_allocation": stock_allocation,
            "bond_allocation": bond_allocation,
            "cash_allocation": cash_allocation,
            "monthly_investment": monthly_investment,
            "recommendation": recommendation,
            "priority": priority,
            "expected_return": stock_allocation * 0.08 + bond_allocation * 0.04 + cash_allocation * 0.02
        }
    
    def run_portfolio_simulations(self, profiles: List[FinancialProfile], 
                                years: int = 30, simulations: int = 100) -> Dict:
        """Run portfolio simulations for each profile"""
        
        simulation_results = {
            "simulation_date": datetime.now().isoformat(),
            "years_simulated": years,
            "simulations_per_profile": simulations,
            "portfolio_results": []
        }
        
        for profile in profiles:
            logger.info(f"Running portfolio simulation for {profile.profile_id}")
            
            # Get investment recommendation
            recommendation = self._generate_investment_recommendation(profile)
            monthly_investment = recommendation["monthly_investment"]
            
            # Run Monte Carlo simulation
            portfolio_paths = self._simulate_portfolio(
                monthly_investment=monthly_investment,
                stock_allocation=recommendation["stock_allocation"],
                bond_allocation=recommendation["bond_allocation"],
                years=years,
                simulations=simulations
            )
            
            # Calculate statistics
            final_values = portfolio_paths[:, -1]
            
            portfolio_result = {
                "profile_id": profile.profile_id,
                "monthly_investment": monthly_investment,
                "allocation": {
                    "stocks": recommendation["stock_allocation"],
                    "bonds": recommendation["bond_allocation"],
                    "cash": recommendation["cash_allocation"]
                },
                "simulation_stats": {
                    "mean_final_value": np.mean(final_values),
                    "std_final_value": np.std(final_values),
                    "min_final_value": np.min(final_values),
                    "max_final_value": np.max(final_values),
                    "var_95": np.percentile(final_values, 5),
                    "var_99": np.percentile(final_values, 1),
                    "median_final_value": np.median(final_values)
                },
                "recommendation": recommendation
            }
            
            simulation_results["portfolio_results"].append(portfolio_result)
            logger.info(f"Completed simulation for {profile.profile_id}")
        
        return simulation_results
    
    def _simulate_portfolio(self, monthly_investment: float, stock_allocation: float,
                          bond_allocation: float, years: int, simulations: int) -> np.ndarray:
        """Simulate portfolio growth using Monte Carlo"""
        
        np.random.seed(42)  # For reproducibility
        
        # Parameters
        monthly_stock_return = 0.08 / 12  # 8% annual return
        monthly_bond_return = 0.04 / 12   # 4% annual return
        monthly_stock_vol = 0.15 / np.sqrt(12)  # 15% annual volatility
        monthly_bond_vol = 0.05 / np.sqrt(12)   # 5% annual volatility
        
        # Initialize portfolio paths
        months = years * 12
        paths = np.zeros((simulations, months + 1))
        
        for sim in range(simulations):
            portfolio_value = 0
            
            for month in range(months):
                # Add monthly investment
                portfolio_value += monthly_investment
                
                # Generate returns
                stock_return = np.random.normal(monthly_stock_return, monthly_stock_vol)
                bond_return = np.random.normal(monthly_bond_return, monthly_bond_vol)
                
                # Calculate total return
                total_return = (stock_allocation * stock_return + 
                              bond_allocation * bond_return)
                
                # Update portfolio value
                portfolio_value *= (1 + total_return)
                paths[sim, month + 1] = portfolio_value
        
        return paths
    
    def save_analysis_results(self, analysis_results: Dict, simulation_results: Dict):
        """Save comprehensive analysis results"""
        
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to JSON serializable types
        analysis_results_serializable = convert_numpy_types(analysis_results)
        simulation_results_serializable = convert_numpy_types(simulation_results)
        
        # Save financial analysis
        analysis_path = self.analysis_dir / "financial_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results_serializable, f, indent=2)
        
        # Save portfolio simulations
        simulation_path = self.analysis_dir / "portfolio_simulations.json"
        with open(simulation_path, 'w') as f:
            json.dump(simulation_results_serializable, f, indent=2)
        
        # Save summary report
        summary_path = self.analysis_dir / "advisor_summary.json"
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_profiles": analysis_results["total_profiles"],
            "financial_summary": convert_numpy_types(analysis_results["financial_summary"]),
            "risk_analysis": analysis_results["risk_analysis"],
            "simulation_summary": {
                "years_simulated": simulation_results["years_simulated"],
                "simulations_per_profile": simulation_results["simulations_per_profile"],
                "total_portfolio_simulations": len(simulation_results["portfolio_results"])
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis results saved to {self.analysis_dir}")
    
    def run_complete_advisor_pipeline(self):
        """Run complete financial advisor pipeline"""
        
        logger.info("Starting complete financial advisor pipeline...")
        
        # Step 1: Load and convert training profiles
        logger.info("Loading training profiles...")
        profiles = self.load_training_profiles()
        
        if not profiles:
            logger.error("No profiles found for analysis")
            return
        
        logger.info(f"Loaded {len(profiles)} financial profiles")
        
        # Step 2: Run financial analysis
        logger.info("Running financial analysis...")
        analysis_results = self.run_financial_analysis(profiles)
        
        # Step 3: Run portfolio simulations
        logger.info("Running portfolio simulations...")
        simulation_results = self.run_portfolio_simulations(profiles, years=30, simulations=100)
        
        # Step 4: Save results
        logger.info("Saving analysis results...")
        self.save_analysis_results(analysis_results, simulation_results)
        
        # Step 5: Print summary
        logger.info("Financial advisor pipeline complete!")
        logger.info(f"Profiles analyzed: {len(profiles)}")
        logger.info(f"Portfolio simulations: {len(simulation_results['portfolio_results'])}")
        
        return {
            "analysis_results": analysis_results,
            "simulation_results": simulation_results
        }


def main():
    """Run the complete financial advisor pipeline"""
    advisor = FinancialAdvisorIntegration()
    
    # Run complete pipeline
    results = advisor.run_complete_advisor_pipeline()
    
    if results:
        print("Financial advisor pipeline completed successfully!")
        print(f"Profiles analyzed: {len(results['analysis_results']['profile_analyses'])}")
        print(f"Portfolio simulations: {len(results['simulation_results']['portfolio_results'])}")
        
        # Print key metrics
        financial_summary = results['analysis_results']['financial_summary']
        print(f"Average monthly income: ${financial_summary['average_monthly_income']:.2f}")
        print(f"Average monthly expenses: ${financial_summary['average_monthly_expenses']:.2f}")
        print(f"Average savings rate: {financial_summary['average_savings_rate']:.1f}%")
    else:
        print("Financial advisor pipeline failed!")


if __name__ == "__main__":
    main() 