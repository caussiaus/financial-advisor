"""
Comprehensive Test Integration

Process comprehensive test profiles through the financial advisor system
to test with realistic financial values and diverse scenarios.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime
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


class ComprehensiveTestIntegration:
    """Integrate comprehensive test profiles with financial advisor system"""
    
    def __init__(self):
        self.test_dir = Path("data/test_comprehensive")
        self.results_dir = Path("data/comprehensive_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_comprehensive_profiles(self) -> List[Dict]:
        """Load comprehensive test profiles"""
        profiles = []
        profiles_dir = self.test_dir / "profiles"
        
        if not profiles_dir.exists():
            logger.error(f"Test profiles directory not found: {profiles_dir}")
            return profiles
        
        profile_files = list(profiles_dir.glob("*.json"))
        logger.info(f"Found {len(profile_files)} comprehensive test profiles")
        
        for profile_file in profile_files:
            try:
                with open(profile_file, 'r') as f:
                    profile = json.load(f)
                profiles.append(profile)
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")
        
        return profiles
    
    def convert_to_financial_profiles(self, comprehensive_profiles: List[Dict]) -> List[FinancialProfile]:
        """Convert comprehensive test profiles to financial advisor profiles"""
        financial_profiles = []
        
        for profile in comprehensive_profiles:
            try:
                # Extract transaction data
                transactions = profile.get("transactions", [])
                
                # Calculate amounts and categories
                amounts = []
                categories = []
                dates = []
                
                for transaction in transactions:
                    amounts.append(transaction["amount"])
                    categories.append(transaction["category"])
                    dates.append(transaction["date"])
                
                # Create category spending dictionary
                category_spending = {}
                for i, amount in enumerate(amounts):
                    if amount < 0:  # Only expenses
                        category = categories[i]
                        category_spending[category] = category_spending.get(category, 0) + abs(amount)
                
                # Create transaction history
                transaction_history = []
                for i, amount in enumerate(amounts):
                    if i < len(dates) and i < len(categories):
                        transaction_history.append({
                            "date": dates[i],
                            "amount": amount,
                            "category": categories[i],
                            "type": "income" if amount > 0 else "expense"
                        })
                
                # Create financial profile
                financial_profile = FinancialProfile(
                    profile_id=profile["profile_id"],
                    monthly_income=profile["monthly_income"],
                    monthly_expenses=profile["monthly_expenses"],
                    savings_rate=profile["savings_rate"],
                    risk_tolerance=profile["risk_tolerance"],
                    investment_horizon=30,
                    categories=category_spending,
                    transaction_history=transaction_history
                )
                
                financial_profiles.append(financial_profile)
                logger.info(f"Converted profile: {profile['profile_id']}")
                logger.info(f"  Income: ${profile['monthly_income']:.2f}, Expenses: ${profile['monthly_expenses']:.2f}")
                logger.info(f"  Savings Rate: {profile['savings_rate']:.1f}%, Risk: {profile['risk_tolerance']}")
                
            except Exception as e:
                logger.error(f"Failed to convert profile {profile.get('profile_id', 'unknown')}: {e}")
        
        return financial_profiles
    
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
    
    def run_comprehensive_analysis(self, financial_profiles: List[FinancialProfile]) -> Dict:
        """Run comprehensive financial analysis"""
        
        logger.info(f"Running comprehensive analysis on {len(financial_profiles)} profiles...")
        
        # Run financial analysis
        analysis_results = self.run_financial_analysis(financial_profiles)
        
        # Run portfolio simulations
        simulation_results = self.run_portfolio_simulations(
            financial_profiles, years=30, simulations=100
        )
        
        # Add comprehensive test metadata
        analysis_results["comprehensive_test_metadata"] = {
            "test_date": datetime.now().isoformat(),
            "total_profiles": len(financial_profiles),
            "income_levels": {},
            "savings_levels": {},
            "financial_health_levels": {}
        }
        
        # Calculate distributions
        for profile in financial_profiles:
            # Extract original comprehensive profile data
            profile_id = profile.profile_id
            if profile_id.startswith("comprehensive_"):
                parts = profile_id.split("_")
                if len(parts) >= 3:
                    income_level = parts[1]
                    savings_level = parts[2]
                    
                    analysis_results["comprehensive_test_metadata"]["income_levels"][income_level] = \
                        analysis_results["comprehensive_test_metadata"]["income_levels"].get(income_level, 0) + 1
                    
                    analysis_results["comprehensive_test_metadata"]["savings_levels"][savings_level] = \
                        analysis_results["comprehensive_test_metadata"]["savings_levels"].get(savings_level, 0) + 1
        
        return analysis_results, simulation_results
    
    def save_comprehensive_results(self, analysis_results: Dict, simulation_results: Dict):
        """Save comprehensive test results"""
        
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
        
        # Save analysis results
        analysis_path = self.results_dir / "comprehensive_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results_serializable, f, indent=2)
        
        # Save simulation results
        simulation_path = self.results_dir / "comprehensive_simulations.json"
        with open(simulation_path, 'w') as f:
            json.dump(simulation_results_serializable, f, indent=2)
        
        # Save summary report
        summary_path = self.results_dir / "comprehensive_summary.json"
        summary = {
            "test_date": datetime.now().isoformat(),
            "total_profiles": analysis_results["total_profiles"],
            "financial_summary": convert_numpy_types(analysis_results["financial_summary"]),
            "risk_analysis": analysis_results["risk_analysis"],
            "comprehensive_metadata": analysis_results.get("comprehensive_test_metadata", {}),
            "simulation_summary": {
                "years_simulated": simulation_results["years_simulated"],
                "simulations_per_profile": simulation_results["simulations_per_profile"],
                "total_portfolio_simulations": len(simulation_results["portfolio_results"])
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Comprehensive results saved to {self.results_dir}")
    
    def print_comprehensive_results(self, analysis_results: Dict, simulation_results: Dict):
        """Print comprehensive test results"""
        
        print(f"\n{'='*80}")
        print(f" COMPREHENSIVE FINANCIAL ADVISOR TEST RESULTS")
        print(f"{'='*80}")
        
        # Basic statistics
        financial_summary = analysis_results["financial_summary"]
        print(f"📊 Total profiles analyzed: {analysis_results['total_profiles']}")
        print(f"💰 Average monthly income: ${financial_summary['average_monthly_income']:.2f}")
        print(f"💰 Average monthly expenses: ${financial_summary['average_monthly_expenses']:.2f}")
        print(f"💰 Average savings rate: {financial_summary['average_savings_rate']:.1f}%")
        print(f"💰 Total annual income: ${financial_summary['total_annual_income']:.2f}")
        print(f"💰 Total annual savings: ${financial_summary['total_annual_savings']:.2f}")
        
        # Risk analysis
        risk_analysis = analysis_results["risk_analysis"]
        print(f"\n📈 Risk Analysis:")
        print(f"   Low risk profiles: {risk_analysis['low_risk_profiles']}")
        print(f"   Medium risk profiles: {risk_analysis['medium_risk_profiles']}")
        print(f"   High risk profiles: {risk_analysis['high_risk_profiles']}")
        
        # Comprehensive test metadata
        if "comprehensive_test_metadata" in analysis_results:
            metadata = analysis_results["comprehensive_test_metadata"]
            print(f"\n🎯 Test Distribution:")
            print(f"   Income levels: {metadata.get('income_levels', {})}")
            print(f"   Savings levels: {metadata.get('savings_levels', {})}")
        
        # Simulation results
        simulation_summary = simulation_results
        print(f"\n🎯 Portfolio Simulations:")
        print(f"   Years simulated: {simulation_summary['years_simulated']}")
        print(f"   Simulations per profile: {simulation_summary['simulations_per_profile']}")
        print(f"   Total portfolio simulations: {len(simulation_summary['portfolio_results'])}")
        
        # Compare with original test data
        print(f"\n📊 Comparison with Original Test Data:")
        print(f"   This test provides much more realistic financial scenarios")
        print(f"   Savings rates range from struggling (0-10%) to exceptional (80-95%)")
        print(f"   Income levels range from low ($2K-$5K) to very high ($50K-$100K)")
        print(f"   Risk tolerance varies based on actual spending patterns")
    
    def run_comprehensive_test_integration(self):
        """Run complete comprehensive test integration"""
        
        logger.info("Starting comprehensive test integration...")
        
        # Step 1: Load comprehensive test profiles
        logger.info("Loading comprehensive test profiles...")
        comprehensive_profiles = self.load_comprehensive_profiles()
        
        if not comprehensive_profiles:
            logger.error("No comprehensive test profiles found")
            return
        
        logger.info(f"Loaded {len(comprehensive_profiles)} comprehensive test profiles")
        
        # Step 2: Convert to financial profiles
        logger.info("Converting to financial profiles...")
        financial_profiles = self.convert_to_financial_profiles(comprehensive_profiles)
        
        if not financial_profiles:
            logger.error("No financial profiles created")
            return
        
        logger.info(f"Created {len(financial_profiles)} financial profiles")
        
        # Step 3: Run comprehensive analysis
        logger.info("Running comprehensive analysis...")
        analysis_results, simulation_results = self.run_comprehensive_analysis(financial_profiles)
        
        # Step 4: Save results
        logger.info("Saving comprehensive results...")
        self.save_comprehensive_results(analysis_results, simulation_results)
        
        # Step 5: Print results
        logger.info("Printing comprehensive results...")
        self.print_comprehensive_results(analysis_results, simulation_results)
        
        logger.info("Comprehensive test integration complete!")
        
        return analysis_results, simulation_results


def main():
    """Run comprehensive test integration"""
    integration = ComprehensiveTestIntegration()
    
    # Run comprehensive test integration
    results = integration.run_comprehensive_test_integration()
    
    if results:
        analysis_results, simulation_results = results
        print(f"\n🎉 Comprehensive test integration completed successfully!")
        print(f"📊 Analyzed {analysis_results['total_profiles']} diverse financial profiles")
        print(f"📊 Ran {len(simulation_results['portfolio_results'])} portfolio simulations")
        print(f"📊 Results saved to data/comprehensive_results/")
    else:
        print("Comprehensive test integration failed!")


if __name__ == "__main__":
    main() 