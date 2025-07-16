"""
Spending Surface Integration
Integrates spending surface analysis with existing timeline bias engine and IPS system
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import asyncio

# Import existing systems
try:
    from .timeline_bias_engine import TimelineBiasEngine
    from .enhanced_chunked_processor import EnhancedChunkedProcessor
    from .ips_model import IPSModel
except ImportError:
    # Fallback for demo purposes
    TimelineBiasEngine = None
    EnhancedChunkedProcessor = None
    IPSModel = None

# Import spending surface components
from .spending_pattern_scraper import SpendingDataScraper
from .spending_vector_database import SpendingPatternVectorDB
from .spending_surface_modeler import SpendingSurfaceModeler
from .discretionary_spending_classifier import DiscretionarySpendingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTimelineBiasEngine:
    """Enhanced timeline bias engine with spending surface analysis"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # Initialize original timeline bias engine if available
        if TimelineBiasEngine:
            self.original_engine = TimelineBiasEngine()
        else:
            self.original_engine = None
            logger.warning("Original TimelineBiasEngine not available")
        
        # Initialize spending surface components
        self.spending_scraper = SpendingDataScraper(
            db_path=str(self.data_dir / "spending_patterns.db")
        )
        self.vector_db = SpendingPatternVectorDB(
            db_path=str(self.data_dir / "spending_vectors"),
            spending_db_path=str(self.data_dir / "spending_patterns.db")
        )
        self.surface_modeler = SpendingSurfaceModeler(self.vector_db)
        self.classifier = DiscretionarySpendingClassifier()
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if self.is_initialized:
            return
        
        logger.info("🚀 Initializing Enhanced Timeline Bias Engine...")
        
        # Set up spending surface system
        await self.spending_scraper.scrape_all_sources()
        self.spending_scraper.generate_milestone_patterns()
        
        self.vector_db.vectorize_and_store_patterns()
        
        # Train classifier
        training_df, training_labels = self.classifier.generate_training_data(3000)
        self.classifier.train_classifier(training_df, training_labels)
        
        # Create surfaces for key milestones
        for milestone in ['home_purchase', 'marriage', 'first_child']:
            try:
                surface_points = self.surface_modeler.extract_surface_data(milestone)
                if surface_points:
                    self.surface_modeler.create_gaussian_process_surface(milestone, surface_points)
                    logger.info(f"✓ Created surface for {milestone}")
            except Exception as e:
                logger.warning(f"Could not create surface for {milestone}: {e}")
        
        self.is_initialized = True
        logger.info("✅ Enhanced Timeline Bias Engine initialized")
    
    def estimate_milestone_timing_enhanced(self, 
                                         client_profile: Dict,
                                         milestone: str,
                                         use_spending_surface: bool = True) -> Dict:
        """Enhanced milestone timing estimation using both demographic and spending analysis"""
        
        if not self.is_initialized:
            raise ValueError("Engine must be initialized before use")
        
        results = {
            'milestone': milestone,
            'client_profile': client_profile,
            'predictions': {},
            'confidence_factors': {}
        }
        
        # Original demographic-based prediction
        if self.original_engine:
            try:
                original_prediction = self.original_engine.estimate_timeline_bias(
                    client_profile, milestone
                )
                results['predictions']['demographic'] = original_prediction
                results['confidence_factors']['demographic'] = 0.7
            except Exception as e:
                logger.warning(f"Original engine prediction failed: {e}")
                results['predictions']['demographic'] = None
        
        # Spending surface-based prediction
        if use_spending_surface and milestone in self.surface_modeler.surfaces:
            try:
                # Extract spending pattern from client profile
                spending_pattern = self.extract_spending_pattern(client_profile)
                
                surface_prediction = self.surface_modeler.predict_milestone_timing_surface(
                    milestone,
                    spending_pattern['income'],
                    spending_pattern['age'],
                    spending_pattern['discretionary_ratio'],
                    return_uncertainty=True
                )
                
                results['predictions']['spending_surface'] = surface_prediction
                results['confidence_factors']['spending_surface'] = surface_prediction.get('confidence', 0.5)
                
            except Exception as e:
                logger.warning(f"Spending surface prediction failed: {e}")
                results['predictions']['spending_surface'] = None
        
        # Vector database similarity prediction
        try:
            similar_patterns = self.vector_db.find_similar_patterns(
                client_profile, n_results=10
            )
            
            vector_prediction = self.vector_db.predict_milestone_timing(
                client_profile, milestone
            )
            
            results['predictions']['vector_similarity'] = vector_prediction
            results['confidence_factors']['vector_similarity'] = vector_prediction.get('confidence', 0.5)
            
        except Exception as e:
            logger.warning(f"Vector similarity prediction failed: {e}")
            results['predictions']['vector_similarity'] = None
        
        # Combine predictions using weighted average
        combined_prediction = self.combine_predictions(results['predictions'], results['confidence_factors'])
        results['combined_prediction'] = combined_prediction
        
        return results
    
    def extract_spending_pattern(self, client_profile: Dict) -> Dict:
        """Extract spending pattern from client profile"""
        
        # Default spending pattern if not provided
        default_pattern = {
            'age': client_profile.get('age', 30),
            'income': client_profile.get('income', client_profile.get('annual_income', 60000)),
            'discretionary_ratio': 0.15,  # Default 15%
            'education_level': client_profile.get('education_level', 'Bachelor'),
            'location': client_profile.get('location', 'Suburban'),
            'marital_status': client_profile.get('marital_status', 'Single'),
            'household_size': client_profile.get('household_size', 1)
        }
        
        # If spending data is provided, analyze it
        if 'expenses' in client_profile:
            expenses = client_profile['expenses']
            if expenses:
                spending_analysis = self.classifier.analyze_spending_pattern(expenses)
                default_pattern['discretionary_ratio'] = spending_analysis['discretionary_ratio']
        
        # Use provided discretionary ratio if available
        if 'discretionary_ratio' in client_profile:
            default_pattern['discretionary_ratio'] = client_profile['discretionary_ratio']
        
        return default_pattern
    
    def combine_predictions(self, predictions: Dict, confidence_factors: Dict) -> Dict:
        """Combine multiple predictions using weighted average"""
        
        valid_predictions = {}
        total_weight = 0
        
        # Collect valid predictions and their weights
        for method, prediction in predictions.items():
            if prediction and 'predicted_age' in prediction and prediction['predicted_age']:
                confidence = confidence_factors.get(method, 0.5)
                valid_predictions[method] = {
                    'age': prediction['predicted_age'],
                    'weight': confidence
                }
                total_weight += confidence
        
        if not valid_predictions:
            return {
                'predicted_age': None,
                'confidence': 0.0,
                'methods_used': [],
                'error': 'No valid predictions available'
            }
        
        # Calculate weighted average
        weighted_sum = sum(pred['age'] * pred['weight'] for pred in valid_predictions.values())
        combined_age = weighted_sum / total_weight if total_weight > 0 else None
        
        # Calculate combined confidence
        combined_confidence = total_weight / len(predictions)
        
        # Calculate prediction variance for uncertainty
        if len(valid_predictions) > 1:
            variance = sum(
                pred['weight'] * (pred['age'] - combined_age) ** 2 
                for pred in valid_predictions.values()
            ) / total_weight
            uncertainty = np.sqrt(variance)
        else:
            uncertainty = 0.0
        
        return {
            'predicted_age': combined_age,
            'confidence': combined_confidence,
            'uncertainty': uncertainty,
            'methods_used': list(valid_predictions.keys()),
            'method_details': valid_predictions,
            'combination_method': 'weighted_average'
        }
    
    def optimize_spending_for_timeline(self, 
                                     client_profile: Dict,
                                     milestone: str,
                                     target_age: float) -> Dict:
        """Optimize spending pattern to achieve milestone by target age"""
        
        if milestone not in self.surface_modeler.surfaces:
            return {
                'success': False,
                'error': f'No surface model available for {milestone}'
            }
        
        spending_pattern = self.extract_spending_pattern(client_profile)
        
        optimization = self.surface_modeler.optimize_spending_for_milestone(
            milestone,
            spending_pattern['age'],
            spending_pattern['income'],
            target_age
        )
        
        if optimization.get('optimization_success', False):
            # Calculate the changes needed
            current_ratio = spending_pattern['discretionary_ratio']
            optimal_ratio = optimization['optimal_discretionary_ratio']
            ratio_change = optimal_ratio - current_ratio
            
            monthly_change = (spending_pattern['income'] * ratio_change) / 12
            
            optimization['recommendations'] = {
                'current_discretionary_ratio': current_ratio,
                'recommended_ratio': optimal_ratio,
                'ratio_change': ratio_change,
                'monthly_spending_change': monthly_change,
                'direction': 'increase' if ratio_change > 0 else 'decrease',
                'magnitude': abs(ratio_change)
            }
        
        return optimization
    
    def analyze_milestone_feasibility(self, 
                                    client_profile: Dict,
                                    milestone: str,
                                    target_timeline: float) -> Dict:
        """Analyze feasibility of achieving milestone within target timeline"""
        
        # Get enhanced prediction
        prediction = self.estimate_milestone_timing_enhanced(client_profile, milestone)
        
        if not prediction['combined_prediction']['predicted_age']:
            return {
                'feasible': False,
                'reason': 'Unable to predict milestone timing'
            }
        
        predicted_age = prediction['combined_prediction']['predicted_age']
        current_age = client_profile.get('age', 30)
        predicted_timeline = predicted_age - current_age
        
        feasibility = {
            'feasible': predicted_timeline <= target_timeline,
            'predicted_timeline': predicted_timeline,
            'target_timeline': target_timeline,
            'timeline_gap': target_timeline - predicted_timeline,
            'confidence': prediction['combined_prediction']['confidence']
        }
        
        # If not feasible, suggest optimizations
        if not feasibility['feasible']:
            target_age = current_age + target_timeline
            optimization = self.optimize_spending_for_timeline(
                client_profile, milestone, target_age
            )
            feasibility['optimization_suggestions'] = optimization
        
        return feasibility

class SpendingSurfaceIPSIntegration:
    """Integration with IPS system for enhanced financial planning"""
    
    def __init__(self, enhanced_engine: EnhancedTimelineBiasEngine):
        self.enhanced_engine = enhanced_engine
        
    def enhance_ips_analysis(self, ips_data: Dict) -> Dict:
        """Enhance IPS analysis with spending surface insights"""
        
        enhanced_ips = ips_data.copy()
        
        # Extract client profile from IPS data
        client_profile = {
            'age': ips_data.get('age', 30),
            'income': ips_data.get('annual_income', 60000),
            'discretionary_ratio': ips_data.get('discretionary_spending_ratio', 0.15),
            'education_level': ips_data.get('education', 'Bachelor'),
            'location': ips_data.get('location', 'Suburban'),
            'marital_status': ips_data.get('marital_status', 'Single')
        }
        
        # Analyze key life milestones
        milestone_analysis = {}
        
        for milestone in ['home_purchase', 'marriage', 'first_child']:
            if milestone not in enhanced_ips.get('life_events', {}):
                continue
            
            # Get enhanced timeline prediction
            prediction = self.enhanced_engine.estimate_milestone_timing_enhanced(
                client_profile, milestone
            )
            
            # Analyze feasibility if target date is provided
            target_date = enhanced_ips['life_events'].get(milestone, {}).get('target_date')
            if target_date:
                current_age = client_profile['age']
                target_age = current_age + (target_date - 2025)  # Assuming current year 2025
                
                feasibility = self.enhanced_engine.analyze_milestone_feasibility(
                    client_profile, milestone, target_age - current_age
                )
                
                milestone_analysis[milestone] = {
                    'prediction': prediction,
                    'feasibility': feasibility
                }
            else:
                milestone_analysis[milestone] = {
                    'prediction': prediction
                }
        
        enhanced_ips['spending_surface_analysis'] = milestone_analysis
        
        # Add spending optimization recommendations
        optimization_recommendations = self.generate_spending_recommendations(
            client_profile, milestone_analysis
        )
        enhanced_ips['spending_recommendations'] = optimization_recommendations
        
        return enhanced_ips
    
    def generate_spending_recommendations(self, 
                                        client_profile: Dict,
                                        milestone_analysis: Dict) -> List[Dict]:
        """Generate spending optimization recommendations"""
        
        recommendations = []
        
        for milestone, analysis in milestone_analysis.items():
            if 'feasibility' in analysis and not analysis['feasibility']['feasible']:
                # Milestone not feasible with current spending - suggest optimization
                
                optimization = analysis['feasibility'].get('optimization_suggestions', {})
                
                if optimization.get('optimization_success', False):
                    rec = optimization['recommendations']
                    
                    recommendation = {
                        'milestone': milestone,
                        'type': 'spending_optimization',
                        'priority': 'high' if abs(rec['ratio_change']) > 0.05 else 'medium',
                        'action': f"{rec['direction'].title()} discretionary spending",
                        'details': {
                            'current_ratio': f"{rec['current_discretionary_ratio']:.1%}",
                            'recommended_ratio': f"{rec['recommended_ratio']:.1%}",
                            'monthly_change': f"${abs(rec['monthly_spending_change']):.0f}",
                            'direction': rec['direction']
                        },
                        'impact': f"Achieve {milestone.replace('_', ' ')} on target timeline"
                    }
                    
                    recommendations.append(recommendation)
        
        return recommendations

# Usage example and demo
async def demo_integration():
    """Demonstrate the integrated spending surface system"""
    
    print("🔗 Spending Surface Integration Demo")
    print("=" * 50)
    
    # Initialize enhanced engine
    enhanced_engine = EnhancedTimelineBiasEngine()
    await enhanced_engine.initialize()
    
    # Sample client profile
    client_profile = {
        'age': 28,
        'income': 75000,
        'discretionary_ratio': 0.12,
        'education_level': 'Bachelor',
        'location': 'Urban',
        'marital_status': 'Single',
        'expenses': [
            {'amount': 1800, 'description': 'rent', 'category': 'housing'},
            {'amount': 400, 'description': 'groceries', 'category': 'food'},
            {'amount': 300, 'description': 'dining out', 'category': 'entertainment'},
            {'amount': 200, 'description': 'car insurance', 'category': 'insurance'}
        ]
    }
    
    # Test enhanced predictions
    print("\n🎯 Enhanced Milestone Predictions:")
    for milestone in ['home_purchase', 'marriage', 'first_child']:
        prediction = enhanced_engine.estimate_milestone_timing_enhanced(
            client_profile, milestone
        )
        
        combined = prediction['combined_prediction']
        if combined['predicted_age']:
            print(f"  📈 {milestone.replace('_', ' ').title()}:")
            print(f"    - Predicted Age: {combined['predicted_age']:.1f}")
            print(f"    - Confidence: {combined['confidence']:.2f}")
            print(f"    - Methods Used: {', '.join(combined['methods_used'])}")
        else:
            print(f"  ❌ {milestone.replace('_', ' ').title()}: Prediction failed")
    
    # Test spending optimization
    print("\n⚡ Spending Optimization:")
    target_age = 32  # Want to buy house by age 32
    optimization = enhanced_engine.optimize_spending_for_timeline(
        client_profile, 'home_purchase', target_age
    )
    
    if optimization.get('optimization_success', False):
        rec = optimization['recommendations']
        print(f"  🎯 To achieve home purchase by age {target_age}:")
        print(f"    - Current discretionary ratio: {rec['current_discretionary_ratio']:.1%}")
        print(f"    - Recommended ratio: {rec['recommended_ratio']:.1%}")
        print(f"    - Monthly change: {rec['direction']} ${abs(rec['monthly_spending_change']):.0f}")
    else:
        print(f"  ❌ Optimization failed: {optimization.get('error', 'Unknown error')}")
    
    # Test IPS integration
    print("\n📊 IPS Integration:")
    ips_integration = SpendingSurfaceIPSIntegration(enhanced_engine)
    
    sample_ips = {
        'age': 28,
        'annual_income': 75000,
        'discretionary_spending_ratio': 0.12,
        'life_events': {
            'home_purchase': {'target_date': 2028},
            'marriage': {'target_date': 2027}
        }
    }
    
    enhanced_ips = ips_integration.enhance_ips_analysis(sample_ips)
    
    if 'spending_recommendations' in enhanced_ips:
        print(f"  📋 Generated {len(enhanced_ips['spending_recommendations'])} recommendations")
        for rec in enhanced_ips['spending_recommendations']:
            print(f"    - {rec['action']}: {rec['impact']}")

if __name__ == "__main__":
    asyncio.run(demo_integration()) 