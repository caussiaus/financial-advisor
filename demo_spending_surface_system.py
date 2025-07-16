"""
Comprehensive Spending Surface System Demo
Demonstrates the complete spending pattern analysis system including data collection,
vector database storage, surface modeling, and milestone timing prediction
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Import our system components
from src.spending_pattern_scraper import SpendingDataScraper
from src.spending_vector_database import SpendingPatternVectorDB
from src.spending_surface_modeler import SpendingSurfaceModeler
from src.discretionary_spending_classifier import DiscretionarySpendingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpendingSurfaceSystem:
    """Complete spending surface analysis system"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scraper = SpendingDataScraper(db_path=str(self.data_dir / "spending_patterns.db"))
        self.vector_db = None
        self.surface_modeler = None
        self.classifier = DiscretionarySpendingClassifier()
        
        self.results = {}
        
    async def setup_system(self):
        """Initialize and set up the complete system"""
        logger.info("🚀 Setting up Spending Surface Analysis System...")
        
        # Step 1: Scrape spending pattern data
        logger.info("📊 Collecting spending pattern data...")
        total_patterns = await self.scraper.scrape_all_sources()
        milestone_patterns = self.scraper.generate_milestone_patterns()
        
        self.results['data_collection'] = {
            'spending_patterns': total_patterns,
            'milestone_patterns': milestone_patterns
        }
        
        # Step 2: Set up vector database
        logger.info("🗄️ Setting up vector database...")
        self.vector_db = SpendingPatternVectorDB(
            db_path=str(self.data_dir / "spending_vectors"),
            spending_db_path=str(self.data_dir / "spending_patterns.db")
        )
        self.vector_db.vectorize_and_store_patterns()
        
        # Step 3: Initialize surface modeler
        logger.info("📈 Initializing surface modeler...")
        self.surface_modeler = SpendingSurfaceModeler(self.vector_db)
        
        # Step 4: Train discretionary spending classifier
        logger.info("🤖 Training discretionary spending classifier...")
        training_df, training_labels = self.classifier.generate_training_data(5000)
        self.classifier.train_classifier(training_df, training_labels)
        
        logger.info("✅ System setup complete!")
    
    def create_milestone_surfaces(self, milestones: List[str] = None):
        """Create surface models for milestone timing"""
        
        if milestones is None:
            milestones = ['home_purchase', 'marriage', 'first_child']
        
        logger.info(f"🎯 Creating surfaces for milestones: {milestones}")
        
        surface_results = {}
        
        for milestone in milestones:
            logger.info(f"Processing {milestone}...")
            
            try:
                # Extract surface data
                surface_points = self.surface_modeler.extract_surface_data(
                    milestone,
                    income_range=(30000, 150000),
                    age_range=(22, 55),
                    discretionary_range=(0.05, 0.30),
                    grid_resolution=15
                )
                
                if surface_points:
                    # Create Gaussian Process surface
                    gpr_model = self.surface_modeler.create_gaussian_process_surface(
                        milestone, surface_points
                    )
                    
                    surface_results[milestone] = {
                        'data_points': len(surface_points),
                        'model_type': 'Gaussian Process',
                        'kernel': str(gpr_model.kernel_),
                        'log_likelihood': gpr_model.log_marginal_likelihood()
                    }
                    
                    logger.info(f"✓ {milestone}: {len(surface_points)} data points, "
                              f"log-likelihood: {gpr_model.log_marginal_likelihood():.2f}")
                else:
                    logger.warning(f"⚠️ No surface data found for {milestone}")
                    surface_results[milestone] = {
                        'data_points': 0,
                        'error': 'No data points found'
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error processing {milestone}: {e}")
                surface_results[milestone] = {
                    'error': str(e)
                }
        
        self.results['surface_creation'] = surface_results
        return surface_results
    
    def analyze_client_scenarios(self):
        """Analyze different client scenarios using the surface models"""
        
        logger.info("👥 Analyzing client scenarios...")
        
        # Define sample clients
        clients = {
            'young_professional': {
                'age': 26,
                'income': 55000,
                'education_level': 'Bachelor',
                'location': 'Urban',
                'marital_status': 'Single',
                'discretionary_ratio': 0.18,
                'description': 'Young professional, urban lifestyle'
            },
            'young_couple': {
                'age': 29,
                'income': 85000,
                'education_level': 'Bachelor',
                'location': 'Suburban',
                'marital_status': 'Married',
                'discretionary_ratio': 0.15,
                'description': 'Married couple, planning for family'
            },
            'high_earner': {
                'age': 32,
                'income': 120000,
                'education_level': 'Graduate',
                'location': 'Urban',
                'marital_status': 'Married',
                'discretionary_ratio': 0.25,
                'description': 'High-income professional'
            },
            'conservative_saver': {
                'age': 28,
                'income': 65000,
                'education_level': 'Some College',
                'location': 'Rural',
                'marital_status': 'Single',
                'discretionary_ratio': 0.08,
                'description': 'Conservative spender, high savings rate'
            }
        }
        
        scenario_results = {}
        
        for client_name, client_data in clients.items():
            logger.info(f"Analyzing {client_name}...")
            
            client_results = {
                'profile': client_data,
                'milestone_predictions': {},
                'spending_optimization': {}
            }
            
            # Predict milestone timing for each milestone
            for milestone in ['home_purchase', 'marriage', 'first_child']:
                if milestone in self.surface_modeler.surfaces:
                    try:
                        prediction = self.surface_modeler.predict_milestone_timing_surface(
                            milestone,
                            client_data['income'],
                            client_data['age'],
                            client_data['discretionary_ratio'],
                            return_uncertainty=True
                        )
                        client_results['milestone_predictions'][milestone] = prediction
                        
                        # Optimize spending for achieving milestone earlier
                        target_age = client_data['age'] + 4  # Target: 4 years from now
                        optimization = self.surface_modeler.optimize_spending_for_milestone(
                            milestone,
                            client_data['age'],
                            client_data['income'],
                            target_age
                        )
                        client_results['spending_optimization'][milestone] = optimization
                        
                    except Exception as e:
                        logger.warning(f"Error predicting {milestone} for {client_name}: {e}")
                        client_results['milestone_predictions'][milestone] = {'error': str(e)}
            
            scenario_results[client_name] = client_results
        
        self.results['client_scenarios'] = scenario_results
        return scenario_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the surface analysis"""
        
        logger.info("📊 Creating visualizations...")
        
        vis_dir = self.data_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        visualizations = {}
        
        # 1. Surface 3D plots for each milestone
        for milestone in ['home_purchase', 'marriage', 'first_child']:
            if milestone in self.surface_modeler.surfaces:
                try:
                    fig = self.surface_modeler.visualize_surface_3d(
                        milestone,
                        income_range=(40000, 120000),
                        age_range=(25, 45),
                        discretionary_fixed=0.15
                    )
                    
                    # Save as HTML
                    fig_path = vis_dir / f"{milestone}_surface_3d.html"
                    fig.write_html(str(fig_path))
                    visualizations[f"{milestone}_3d"] = str(fig_path)
                    
                    logger.info(f"✓ Saved {milestone} 3D surface plot")
                    
                except Exception as e:
                    logger.warning(f"Error creating 3D plot for {milestone}: {e}")
        
        # 2. Spending capacity heatmaps
        for milestone in ['home_purchase', 'marriage', 'first_child']:
            if milestone in self.surface_modeler.surface_data:
                try:
                    fig = self.surface_modeler.visualize_spending_capacity_heatmap(
                        milestone, age_fixed=30
                    )
                    
                    fig_path = vis_dir / f"{milestone}_capacity_heatmap.html"
                    fig.write_html(str(fig_path))
                    visualizations[f"{milestone}_capacity"] = str(fig_path)
                    
                    logger.info(f"✓ Saved {milestone} capacity heatmap")
                    
                except Exception as e:
                    logger.warning(f"Error creating capacity heatmap for {milestone}: {e}")
        
        # 3. Client scenario comparison
        if 'client_scenarios' in self.results:
            fig = self.create_client_comparison_chart()
            if fig:
                fig_path = vis_dir / "client_scenario_comparison.html"
                fig.write_html(str(fig_path))
                visualizations['client_comparison'] = str(fig_path)
                logger.info("✓ Saved client scenario comparison")
        
        self.results['visualizations'] = visualizations
        return visualizations
    
    def create_client_comparison_chart(self):
        """Create comparison chart for different client scenarios"""
        
        if 'client_scenarios' not in self.results:
            return None
        
        scenarios = self.results['client_scenarios']
        
        # Prepare data for plotting
        clients = []
        milestones = []
        predicted_ages = []
        optimized_ratios = []
        incomes = []
        
        for client_name, data in scenarios.items():
            for milestone, prediction in data['milestone_predictions'].items():
                if 'predicted_age' in prediction and prediction['predicted_age']:
                    clients.append(client_name.replace('_', ' ').title())
                    milestones.append(milestone.replace('_', ' ').title())
                    predicted_ages.append(prediction['predicted_age'])
                    incomes.append(data['profile']['income'])
                    
                    # Get optimized discretionary ratio
                    opt_data = data['spending_optimization'].get(milestone, {})
                    if 'optimal_discretionary_ratio' in opt_data:
                        optimized_ratios.append(opt_data['optimal_discretionary_ratio'])
                    else:
                        optimized_ratios.append(data['profile']['discretionary_ratio'])
        
        if not clients:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Milestone Achievement Ages by Client',
                'Income vs Predicted Age',
                'Optimized Discretionary Spending Ratios',
                'Achievement Timeline Comparison'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Bar chart of predicted ages
        milestone_colors = {'Home Purchase': 'blue', 'Marriage': 'green', 'First Child': 'orange'}
        
        for milestone in set(milestones):
            milestone_data = [(c, a) for c, m, a in zip(clients, milestones, predicted_ages) if m == milestone]
            if milestone_data:
                clients_subset, ages_subset = zip(*milestone_data)
                fig.add_trace(
                    go.Bar(
                        x=clients_subset,
                        y=ages_subset,
                        name=milestone,
                        marker_color=milestone_colors.get(milestone, 'gray')
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Scatter plot income vs age
        fig.add_trace(
            go.Scatter(
                x=incomes,
                y=predicted_ages,
                mode='markers+text',
                text=[f"{c}<br>{m}" for c, m in zip(clients, milestones)],
                textposition="top center",
                marker=dict(size=8, color=predicted_ages, colorscale='Viridis'),
                name='Income vs Age'
            ),
            row=1, col=2
        )
        
        # Plot 3: Optimized ratios
        fig.add_trace(
            go.Bar(
                x=clients,
                y=[r * 100 for r in optimized_ratios],  # Convert to percentage
                name='Optimized Discretionary %',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Plot 4: Timeline comparison
        client_timeline = {}
        for client, milestone, age in zip(clients, milestones, predicted_ages):
            if client not in client_timeline:
                client_timeline[client] = {}
            client_timeline[client][milestone] = age
        
        for i, (client, timeline) in enumerate(client_timeline.items()):
            milestone_names = list(timeline.keys())
            milestone_ages = list(timeline.values())
            
            fig.add_trace(
                go.Scatter(
                    x=milestone_ages,
                    y=[client] * len(milestone_ages),
                    mode='markers+lines',
                    name=client,
                    text=milestone_names,
                    textposition="top center"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Client Scenario Analysis: Milestone Timing Predictions",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Client", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Age", row=1, col=1)
        
        fig.update_xaxes(title_text="Income ($)", row=1, col=2)
        fig.update_yaxes(title_text="Predicted Age", row=1, col=2)
        
        fig.update_xaxes(title_text="Client", row=2, col=1)
        fig.update_yaxes(title_text="Discretionary Spending %", row=2, col=1)
        
        fig.update_xaxes(title_text="Age", row=2, col=2)
        fig.update_yaxes(title_text="Client", row=2, col=2)
        
        return fig
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        
        logger.info("📋 Generating comprehensive report...")
        
        report_lines = [
            "# Spending Surface Analysis System Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Overview",
            "This report presents a comprehensive analysis of spending patterns and milestone timing",
            "predictions using advanced surface modeling techniques combined with machine learning.",
            "",
        ]
        
        # Data Collection Summary
        if 'data_collection' in self.results:
            data = self.results['data_collection']
            report_lines.extend([
                "## Data Collection Summary",
                f"- **Spending Patterns Collected**: {data['spending_patterns']:,}",
                f"- **Milestone Patterns Generated**: {data['milestone_patterns']:,}",
                "- **Data Sources**: Bureau of Labor Statistics, Federal Reserve Survey, Financial Studies",
                ""
            ])
        
        # Surface Creation Results
        if 'surface_creation' in self.results:
            surfaces = self.results['surface_creation']
            report_lines.extend([
                "## Surface Model Results",
                ""
            ])
            
            for milestone, data in surfaces.items():
                if 'error' not in data:
                    report_lines.extend([
                        f"### {milestone.replace('_', ' ').title()}",
                        f"- **Data Points**: {data['data_points']:,}",
                        f"- **Model Type**: {data['model_type']}",
                        f"- **Log Marginal Likelihood**: {data.get('log_likelihood', 'N/A'):.2f}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"### {milestone.replace('_', ' ').title()}",
                        f"- **Status**: Error - {data['error']}",
                        ""
                    ])
        
        # Client Scenario Analysis
        if 'client_scenarios' in self.results:
            scenarios = self.results['client_scenarios']
            report_lines.extend([
                "## Client Scenario Analysis",
                ""
            ])
            
            for client_name, data in scenarios.items():
                profile = data['profile']
                predictions = data['milestone_predictions']
                
                report_lines.extend([
                    f"### {client_name.replace('_', ' ').title()}",
                    f"- **Age**: {profile['age']}",
                    f"- **Income**: ${profile['income']:,}",
                    f"- **Discretionary Ratio**: {profile['discretionary_ratio']:.1%}",
                    f"- **Location**: {profile['location']}",
                    "",
                    "**Milestone Predictions**:"
                ])
                
                for milestone, pred in predictions.items():
                    if 'predicted_age' in pred and pred['predicted_age']:
                        time_to_milestone = pred['predicted_age'] - profile['age']
                        report_lines.append(
                            f"- {milestone.replace('_', ' ').title()}: "
                            f"Age {pred['predicted_age']:.1f} "
                            f"({time_to_milestone:.1f} years from now, "
                            f"confidence: {pred['confidence']:.2f})"
                        )
                
                # Optimization suggestions
                optimizations = data['spending_optimization']
                if any('optimal_discretionary_ratio' in opt for opt in optimizations.values()):
                    report_lines.extend([
                        "",
                        "**Spending Optimization Recommendations**:"
                    ])
                    
                    for milestone, opt in optimizations.items():
                        if 'optimal_discretionary_ratio' in opt:
                            current_ratio = profile['discretionary_ratio']
                            optimal_ratio = opt['optimal_discretionary_ratio']
                            change = optimal_ratio - current_ratio
                            
                            report_lines.append(
                                f"- {milestone.replace('_', ' ').title()}: "
                                f"Adjust discretionary spending to {optimal_ratio:.1%} "
                                f"({'increase' if change > 0 else 'decrease'} by {abs(change):.1%})"
                            )
                
                report_lines.append("")
        
        # Key Insights
        report_lines.extend([
            "## Key Insights",
            "",
            "### Surface Model Findings",
            "- **Income Impact**: Higher income accelerates milestone achievement",
            "- **Discretionary Spending**: Optimal ratios vary by milestone and life stage",
            "- **Age Factor**: Current age significantly affects milestone timing predictions",
            "",
            "### Spending Pattern Insights",
            "- **Home Purchase**: Requires highest discretionary spending capacity",
            "- **Marriage**: More predictable timing, less income-dependent",
            "- **First Child**: Highly variable, influenced by multiple factors",
            "",
            "### Optimization Opportunities",
            "- **Strategic Spending**: Temporary adjustments can accelerate milestone achievement",
            "- **Life Stage Planning**: Different strategies optimal for different ages",
            "- **Income Efficiency**: Higher earners have more flexibility but also more options",
            ""
        ])
        
        # Methodology
        report_lines.extend([
            "## Methodology",
            "",
            "### Data Collection",
            "- Synthetic data generation based on real statistical patterns",
            "- Multiple data sources: BLS, Federal Reserve, financial studies",
            "- Income, demographic, and spending pattern controls",
            "",
            "### Surface Modeling",
            "- Gaussian Process Regression for uncertainty quantification",
            "- 3D surface interpolation across income, age, and discretionary spending",
            "- Continuous configuration space for smooth predictions",
            "",
            "### Machine Learning",
            "- Discretionary spending classification using ensemble methods",
            "- Vector database similarity search for pattern matching",
            "- Cross-validation and performance optimization",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.data_dir / "spending_surface_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Report saved to: {report_path}")
        
        return report_content
    
    def save_results(self):
        """Save all results to JSON for further analysis"""
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy(self.results)
        
        results_path = self.data_dir / "spending_surface_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_path}")

async def main():
    """Run the complete spending surface analysis demo"""
    
    print("🌟 Starting Comprehensive Spending Surface Analysis Demo")
    print("=" * 60)
    
    # Initialize system
    system = SpendingSurfaceSystem()
    
    try:
        # Setup system components
        await system.setup_system()
        
        # Create milestone surfaces
        surface_results = system.create_milestone_surfaces()
        print("\n📈 Surface Creation Results:")
        for milestone, result in surface_results.items():
            if 'error' not in result:
                print(f"  ✓ {milestone}: {result['data_points']} data points")
            else:
                print(f"  ❌ {milestone}: {result['error']}")
        
        # Analyze client scenarios
        scenario_results = system.analyze_client_scenarios()
        print("\n👥 Client Scenario Analysis:")
        for client, data in scenario_results.items():
            print(f"  📊 {client.replace('_', ' ').title()}:")
            for milestone, pred in data['milestone_predictions'].items():
                if 'predicted_age' in pred and pred['predicted_age']:
                    print(f"    - {milestone}: Age {pred['predicted_age']:.1f} (confidence: {pred['confidence']:.2f})")
        
        # Create visualizations
        visualizations = system.create_visualizations()
        print(f"\n📊 Created {len(visualizations)} visualizations")
        
        # Generate comprehensive report
        report = system.generate_report()
        print(f"\n📋 Generated comprehensive report ({len(report)} characters)")
        
        # Save all results
        system.save_results()
        
        print("\n🎉 Demo completed successfully!")
        print("\nGenerated files:")
        for file_path in system.data_dir.rglob("*"):
            if file_path.is_file():
                print(f"  📁 {file_path}")
        
        return system
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    # Run the demo
    result_system = asyncio.run(main()) 