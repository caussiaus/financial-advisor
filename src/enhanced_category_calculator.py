
Enhanced Category Calculator for Omega Mesh System

This module leverages the system's computational power to calculate detailed amounts
for each category based on income percentages and actual dollar ranges. It supports
both balance and spending items for the accounting engine.

The system processes case content to define categories and calculates amounts that
go into each category, providing both percentage-based and absolute dollar ranges.import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal
import json
from enum import Enum
import logging


class CategoryType(Enum):
    """Types of financial categories  INCOME = "income"
    EXPENSE =expense"
    ASSET = "asset"
    LIABILITY = "liability    INVESTMENT = investment    SAVINGS = savings  DEBT = "debt"


class CalculationMethod(Enum):
ds for calculating category amounts
    PERCENTAGE_OF_INCOME = "percentage_of_income"
    ABSOLUTE_DOLLAR_RANGE = "absolute_dollar_range    FIXED_AMOUNT = fixed_amount   DYNAMIC_SCALING = dynamic_scaling"
    MILESTONE_BASED = "milestone_based"


@dataclass
class CategoryDefinition:
    ion of a financial category with calculation parameters"  category_id: str
    name: str
    category_type: CategoryType
    calculation_method: CalculationMethod
    base_percentage: float =00rcentage of income
    min_amount: float = 00 max_amount: float = float('inf')
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryCalculation:
    """Result of category calculation"  category_id: str
    calculated_amount: float
    percentage_of_income: float
    dollar_range: Tuple[float, float]
    confidence_score: float
    calculation_method: CalculationMethod
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedCategoryCalculator:
    ""Enhanced category calculator that leverages computational power to calculate
    detailed amounts for each category based on case content and financial profiles."   
    def __init__(self):
        self.categories =[object Object]}
        self.calculation_history = []
        self.logger = self._setup_logging()
        
        # Initialize standard categories based on financial planning best practices
        self._initialize_standard_categories()
        
        # Computational parameters for vectorized calculations
        self.vector_size =1000Number of parallel calculations
        self.precision = 0.01  # Dollar precision for calculations
        
    def _setup_logging(self) -> logging.Logger:
      Setup logging for category calculations  logger = logging.getLogger(enhanced_category_calculator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
               %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_standard_categories(self):
   nitialize standard financial categories with best practice percentages        standard_categories = [
            # Income Categories
            CategoryDefinition(
                category_id="salary_income,              name="Salary Income,          category_type=CategoryType.INCOME,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=1.0,  # 100% of income
                priority=1
            ),
            CategoryDefinition(
                category_id=investment_income,              name=Investment Income,          category_type=CategoryType.INCOME,
                calculation_method=CalculationMethod.DYNAMIC_SCALING,
                base_percentage=0.05,  # 5% of portfolio value
                priority=2
            ),
            
            # Essential Expense Categories (50/30/20 Rule)
            CategoryDefinition(
                category_id=housing_expenses,              name=HousingExpenses,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.30,  # 30% of income
                min_amount=1000               max_amount=10000          priority=1
            ),
            CategoryDefinition(
                category_id="utilities_expenses,              name="Utilities,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.05,  # 5% of income
                min_amount=200               max_amount=800          priority=1
            ),
            CategoryDefinition(
                category_id=food_groceries,              name=Food & Groceries,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.10,  # 10% of income
                min_amount=400               max_amount=2000          priority=1
            ),
            CategoryDefinition(
                category_id=transportation,              name=Transportation,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.10,  # 10% of income
                min_amount=300               max_amount=1500          priority=1
            ),
            CategoryDefinition(
                category_id="healthcare_insurance,              name="Healthcare & Insurance,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.08,  # 8% of income
                min_amount=200               max_amount=1200          priority=1
            ),
            
            # Discretionary Expense Categories
            CategoryDefinition(
                category_id="entertainment,              name="Entertainment,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.05,  # 5% of income
                min_amount=100               max_amount=500          priority=2
            ),
            CategoryDefinition(
                category_id=shopping_personal,              name="Shopping & Personal,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.05,  # 5% of income
                min_amount=100               max_amount=800          priority=2
            ),
            CategoryDefinition(
                category_id="vacation_travel,              name=Vacation & Travel,          category_type=CategoryType.EXPENSE,
                calculation_method=CalculationMethod.ABSOLUTE_DOLLAR_RANGE,
                min_amount=1000               max_amount=8000          priority=3
            ),
            
            # Savings & Investment Categories
            CategoryDefinition(
                category_id=emergency_fund,              name=Emergency Fund,          category_type=CategoryType.SAVINGS,
                calculation_method=CalculationMethod.MILESTONE_BASED,
                base_percentage=0.10,  # 10% of income
                min_amount=500               max_amount=50000          priority=1
            ),
            CategoryDefinition(
                category_id="retirement_savings,              name="Retirement Savings,          category_type=CategoryType.INVESTMENT,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.15,  # 15% of income
                min_amount=200               max_amount=5000          priority=1
            ),
            CategoryDefinition(
                category_id="investment_portfolio,              name="Investment Portfolio,          category_type=CategoryType.INVESTMENT,
                calculation_method=CalculationMethod.DYNAMIC_SCALING,
                base_percentage=0.20,  # 20% of surplus
                priority=2
            ),
            
            # Debt Categories
            CategoryDefinition(
                category_id="debt_payments,              name="Debt Payments,          category_type=CategoryType.DEBT,
                calculation_method=CalculationMethod.PERCENTAGE_OF_INCOME,
                base_percentage=0.20,  # 20% of income
                min_amount=200               max_amount=3000          priority=1
            ),
            
            # Asset Categories
            CategoryDefinition(
                category_id="real_estate,              name="Real Estate,          category_type=CategoryType.ASSET,
                calculation_method=CalculationMethod.ABSOLUTE_DOLLAR_RANGE,
                min_amount=50               max_amount=1000000          priority=2
            ),
            CategoryDefinition(
                category_id=vehicle_assets,              name=Vehicle Assets,          category_type=CategoryType.ASSET,
                calculation_method=CalculationMethod.ABSOLUTE_DOLLAR_RANGE,
                min_amount=5000               max_amount=100000          priority=3
            )
        ]
        
        for category in standard_categories:
            self.categories[category.category_id] = category
    
    def add_category(self, category: CategoryDefinition) -> bool:
      a custom category to the calculator"""
        if category.category_id in self.categories:
            self.logger.warning(f"Category {category.category_id} already exists")
            return False
        
        self.categories[category.category_id] = category
        self.logger.info(fAdded category: {category.name} ({category.category_id}))        returntrue
    def calculate_all_categories(self, 
                               income_data: Dict[str, float],
                               profile_data: Dict[str, Any],
                               milestones: List[Dict] = None) -> Dict[str, CategoryCalculation]:
 
        Calculate amounts for all categories using vectorized operations
        
        Args:
            income_data: Dictionary with income sources and amounts
            profile_data: Client profile information (age, risk tolerance, etc.)
            milestones: List of financial milestones that may affect calculations
            
        Returns:
            Dictionary mapping category_id to CategoryCalculation
         if milestones is None:
            milestones = []
        
        # Extract key financial data
        base_income = income_data.get(base_income, 6000      total_income = sum(income_data.values())
        age = profile_data.get(age, 30    risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        family_size = profile_data.get(family_size', 1)
        
        # Create vectorized income array for parallel processing
        income_vector = np.array([total_income] * self.vector_size)
        
        # Calculate all categories in parallel
        calculations = {}
        
        for category_id, category in self.categories.items():
            try:
                calculation = self._calculate_single_category(
                    category, income_vector, profile_data, milestones
                )
                calculations[category_id] = calculation
                
            except Exception as e:
                self.logger.error(fError calculating category[object Object]category_id}: {e})          continue
        
        # Store calculation history
        self.calculation_history.append({
      timestamp': datetime.now(),
        income_data': income_data,
         profile_data': profile_data,
         calculations': calculations
        })
        
        return calculations
    
    def _calculate_single_category(self,
                                 category: CategoryDefinition,
                                 income_vector: np.ndarray,
                                 profile_data: Dict[str, Any],
                                 milestones: List[Dict]) -> CategoryCalculation:
  lculate amount for a single category using vectorized operations""       
        if category.calculation_method == CalculationMethod.PERCENTAGE_OF_INCOME:
            return self._calculate_percentage_based(category, income_vector, profile_data)
        
        elif category.calculation_method == CalculationMethod.ABSOLUTE_DOLLAR_RANGE:
            return self._calculate_dollar_range(category, income_vector, profile_data)
        
        elif category.calculation_method == CalculationMethod.FIXED_AMOUNT:
            return self._calculate_fixed_amount(category, income_vector, profile_data)
        
        elif category.calculation_method == CalculationMethod.DYNAMIC_SCALING:
            return self._calculate_dynamic_scaling(category, income_vector, profile_data)
        
        elif category.calculation_method == CalculationMethod.MILESTONE_BASED:
            return self._calculate_milestone_based(category, income_vector, profile_data, milestones)
        
        else:
            raise ValueError(f"Unknown calculation method: {category.calculation_method}")
    
    def _calculate_percentage_based(self,
                                  category: CategoryDefinition,
                                  income_vector: np.ndarray,
                                  profile_data: Dict[str, Any]) -> CategoryCalculation:
  lculate percentage-based category amounts        
        # Apply percentage to income vector
        percentage_vector = np.full(self.vector_size, category.base_percentage)
        calculated_amounts = income_vector * percentage_vector
        
        # Apply min/max constraints
        if category.min_amount > 0:
            calculated_amounts = np.maximum(calculated_amounts, category.min_amount)
        if category.max_amount < float('inf'):
            calculated_amounts = np.minimum(calculated_amounts, category.max_amount)
        
        # Calculate statistics
        mean_amount = np.mean(calculated_amounts)
        std_amount = np.std(calculated_amounts)
        min_amount = np.min(calculated_amounts)
        max_amount = np.max(calculated_amounts)
        
        # Calculate confidence score based on variance
        confidence_score = max(00.1(std_amount / mean_amount)) if mean_amount > 0 else 0.1   
        return CategoryCalculation(
            category_id=category.category_id,
            calculated_amount=float(mean_amount),
            percentage_of_income=category.base_percentage,
            dollar_range=(float(min_amount), float(max_amount)),
            confidence_score=confidence_score,
            calculation_method=category.calculation_method,
            timestamp=datetime.now(),
            metadata={
              std_deviation': float(std_amount),
            vector_size': self.vector_size,
              constraints_applied': {
               min_amount': category.min_amount,
               max_amount': category.max_amount
                }
            }
        )
    
    def _calculate_dollar_range(self,
                              category: CategoryDefinition,
                              income_vector: np.ndarray,
                              profile_data: Dict[str, Any]) -> CategoryCalculation:
  Calculate absolute dollar range category amounts        
        # Generate range of possible amounts
        range_size = min(100, self.vector_size)
        amounts = np.linspace(category.min_amount, category.max_amount, range_size)
        
        # Apply profile-based adjustments
        age = profile_data.get(age, 30    risk_tolerance = profile_data.get('risk_tolerance', 'Moderate')
        
        # Age-based adjustments
        if age < 30:
            amounts *= 0.8  # Younger people spend less
        elif age > 50:
            amounts *= 10.2# Older people may spend more
        
        # Risk tolerance adjustments
        if risk_tolerance == 'Conservative':
            amounts *= 0.9
        elif risk_tolerance == 'Aggressive':
            amounts *= 10.1        
        # Calculate statistics
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        min_amount = np.min(amounts)
        max_amount = np.max(amounts)
        
        # Calculate percentage of income
        avg_income = np.mean(income_vector)
        percentage_of_income = (mean_amount / avg_income) if avg_income > 00        confidence_score = 0.8  # High confidence for range-based calculations
        
        return CategoryCalculation(
            category_id=category.category_id,
            calculated_amount=float(mean_amount),
            percentage_of_income=percentage_of_income,
            dollar_range=(float(min_amount), float(max_amount)),
            confidence_score=confidence_score,
            calculation_method=category.calculation_method,
            timestamp=datetime.now(),
            metadata={
              std_deviation': float(std_amount),
           range_size': range_size,
             profile_adjustments': {
                   age_factor': age,
                   risk_tolerance: risk_tolerance
                }
            }
        )
    
    def _calculate_fixed_amount(self,
                              category: CategoryDefinition,
                              income_vector: np.ndarray,
                              profile_data: Dict[str, Any]) -> CategoryCalculation:
        ulate fixed amount category        fixed_amount = category.min_amount if category.min_amount > 0 else 1000   
        return CategoryCalculation(
            category_id=category.category_id,
            calculated_amount=fixed_amount,
            percentage_of_income=(fixed_amount / np.mean(income_vector)) if np.mean(income_vector) > 0 else 0,
            dollar_range=(fixed_amount, fixed_amount),
            confidence_score=1.0,  # Perfect confidence for fixed amounts
            calculation_method=category.calculation_method,
            timestamp=datetime.now(),
            metadata={'fixed_amount': fixed_amount}
        )
    
    def _calculate_dynamic_scaling(self,
                                 category: CategoryDefinition,
                                 income_vector: np.ndarray,
                                 profile_data: Dict[str, Any]) -> CategoryCalculation:
  culate dynamically scaling category amounts        
        # Base calculation on income volatility
        income_std = np.std(income_vector)
        income_mean = np.mean(income_vector)
        
        # Dynamic scaling factor based on income stability
        scaling_factor =10ncome_std / income_mean) * 0.5
        base_amount = income_mean * category.base_percentage * scaling_factor
        
        # Apply constraints
        if category.min_amount > 0:
            base_amount = max(base_amount, category.min_amount)
        if category.max_amount < float('inf'):
            base_amount = min(base_amount, category.max_amount)
        
        # Calculate range based on income volatility
        range_factor =10ncome_std / income_mean)
        min_amount = base_amount / range_factor
        max_amount = base_amount * range_factor
        
        confidence_score = max(00.110ncome_std / income_mean))
        
        return CategoryCalculation(
            category_id=category.category_id,
            calculated_amount=float(base_amount),
            percentage_of_income=category.base_percentage * scaling_factor,
            dollar_range=(float(min_amount), float(max_amount)),
            confidence_score=confidence_score,
            calculation_method=category.calculation_method,
            timestamp=datetime.now(),
            metadata={
               scaling_factor': scaling_factor,
                income_volatility': float(income_std / income_mean),
             range_factor': range_factor
            }
        )
    
    def _calculate_milestone_based(self,
                                 category: CategoryDefinition,
                                 income_vector: np.ndarray,
                                 profile_data: Dict[str, Any],
                                 milestones: List[Dict]) -> CategoryCalculation:
  alculate milestone-based category amounts        
        # Find relevant milestones for this category
        relevant_milestones = [
            m for m in milestones 
            if m.get('category', '').lower() in category.name.lower()
            or category.name.lower() in m.get(description', ).lower()
        ]
        
        if relevant_milestones:
            # Calculate based on milestone requirements
            total_milestone_amount = sum(m.get('financial_impact', 0) for m in relevant_milestones)
            time_horizon = max(1, len(relevant_milestones))  # At least 1 year
            
            monthly_amount = total_milestone_amount / (time_horizon *12          # Ensure it fits within income constraints
            avg_income = np.mean(income_vector)
            max_monthly_amount = avg_income *0.3 Max 30% of income
            
            monthly_amount = min(monthly_amount, max_monthly_amount)
            
            confidence_score =00.9elevant_milestones else 0.5 else:
            # Fall back to percentage-based calculation
            monthly_amount = np.mean(income_vector) * category.base_percentage
            confidence_score = 00.6        
        # Apply constraints
        if category.min_amount > 0:
            monthly_amount = max(monthly_amount, category.min_amount)
        if category.max_amount < float('inf'):
            monthly_amount = min(monthly_amount, category.max_amount)
        
        return CategoryCalculation(
            category_id=category.category_id,
            calculated_amount=float(monthly_amount),
            percentage_of_income=(monthly_amount / np.mean(income_vector)) if np.mean(income_vector) > 0 else 0,
            dollar_range=(float(monthly_amount *00.8oat(monthly_amount * 1.2        confidence_score=confidence_score,
            calculation_method=category.calculation_method,
            timestamp=datetime.now(),
            metadata={
               milestones_count': len(relevant_milestones),
                milestone_amounts': [m.get('financial_impact', 0) for m in relevant_milestones]
            }
        )
    
    def generate_category_report(self, calculations: Dict[str, CategoryCalculation]) -> Dict[str, Any]:
 rate comprehensive report of all category calculations        total_income_percentage = sum(c.percentage_of_income for c in calculations.values())
        total_calculated_amount = sum(c.calculated_amount for c in calculations.values())
        
        # Group by category type
        by_type = {}
        for category_id, calculation in calculations.items():
            category = self.categories[category_id]
            cat_type = category.category_type.value
            
            if cat_type not in by_type:
                by_type[cat_type] = []
            
            by_type[cat_type].append({
            category_id': category_id,
              name: category.name,
       amount': calculation.calculated_amount,
            percentage': calculation.percentage_of_income,
       range': calculation.dollar_range,
            confidence': calculation.confidence_score
             })
        
        # Calculate summary statistics
        summary = [object Object]           total_categories': len(calculations),
           total_income_percentage': total_income_percentage,
           total_calculated_amount': total_calculated_amount,
   average_confidence: np.mean([c.confidence_score for c in calculations.values()]),
        categories_by_type': by_type,
            recommendations': self._generate_recommendations(calculations)
        }
        
        return summary
    
    def _generate_recommendations(self, calculations: Dict[str, CategoryCalculation]) -> List[str]:
 te recommendations based on category calculations"""
        recommendations = []
        
        total_income_percentage = sum(c.percentage_of_income for c in calculations.values())
        
        # Check if total exceeds 100%
        if total_income_percentage > 1.0:
            recommendations.append(
                f"Warning: Total allocations ({total_income_percentage:00.1%.
                fConsider reducing discretionary spending categories."
            )
        
        # Check for low confidence categories
        low_confidence = [c for c in calculations.values() if c.confidence_score < 0.6 if low_confidence:
            recommendations.append(
                f"Consider reviewing {len(low_confidence)} categories with low confidence scores
                f"(average: {np.mean([c.confidence_score for c in low_confidence]):0.2})"
            )
        
        # Check for missing essential categories
        essential_categories = ['housing_expenses,food_groceries', 'healthcare_insurance]
        missing_essential = [cat for cat in essential_categories if cat not in calculations]
        if missing_essential:
            recommendations.append(
                fMissing essential categories: {missing_essential}.
                f"These should be prioritized in financial planning."
            )
        
        return recommendations
    
    def export_calculations(self, calculations: Dict[str, CategoryCalculation], 
                          output_path: str):
        egory calculations to JSON file        export_data = [object Object]timestamp:datetime.now().isoformat(),
            calculations[object Object]          category_id: {
                    calculated_amount: calc.calculated_amount,
                 percentage_of_income:calc.percentage_of_income,
                 dollar_range': calc.dollar_range,
                   confidence_score: calc.confidence_score,
              calculation_method': calc.calculation_method.value,
                    metadata': calc.metadata
                }
                for category_id, calc in calculations.items()
            },
        report': self.generate_category_report(calculations)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported calculations to {output_path}")
    
    def get_category_definitions(self) -> Dict[str, CategoryDefinition]:
     tegory definitions       return self.categories.copy()
    
    def get_calculation_history(self) -> List[Dict]:
        alculation history       return self.calculation_history.copy() 