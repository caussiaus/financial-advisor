#!/usr/bin/env python
# Enhanced Dashboard with Life Choice Optimization
# Features toggle for optimization mode and life choice interface
# Author: ChatGPT 2025-01-16

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from dynamic_portfolio_engine import DynamicPortfolioEngine
from life_choice_optimizer import LifeChoiceOptimizer

class EnhancedDashboardWithOptimization:
    """Enhanced dashboard with life choice optimization toggle and interface"""
    
    def __init__(self, portfolio_engine=None):
        """Initialize dashboard with portfolio engine"""
        if portfolio_engine is None:
            # Create default portfolio engine
            client_config = {
                'income': 250000,
                'disposable_cash': 8000,
                'allowable_var': 0.15,
                'age': 42,
                'risk_profile': 3,
                'portfolio_value': 1500000,
                'target_allocation': {'equity': 0.58, 'bonds': 0.32, 'cash': 0.10}
            }
            self.portfolio_engine = DynamicPortfolioEngine(client_config)
        else:
            self.portfolio_engine = portfolio_engine
        
        self.optimizer = LifeChoiceOptimizer(self.portfolio_engine)
        self.optimization_mode = False
        
    def create_enhanced_dashboard(self):
        """Create the main enhanced dashboard with optimization toggle"""
        # Create main dashboard with optimization panel
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Portfolio Allocation', 'Comfort Score', 'Optimization Toggle',
                'Life Choices Timeline', 'Portfolio Evolution', 'Choice Impacts',
                'Risk Metrics', 'Market Stress', 'Category Distribution',
                'Life Choice Interface', 'Optimization Results', 'Recommendations'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Get portfolio data
        portfolio_data = self.portfolio_engine.export_data()
        
        if portfolio_data['portfolio_snapshots']:
            dates = [s['date'] for s in portfolio_data['portfolio_snapshots']]
            equity_alloc = [s['portfolio']['equity'] for s in portfolio_data['portfolio_snapshots']]
            bonds_alloc = [s['portfolio']['bonds'] for s in portfolio_data['portfolio_snapshots']]
            cash_alloc = [s['portfolio']['cash'] for s in portfolio_data['portfolio_snapshots']]
            comfort_scores = [s['comfort_metrics']['comfort_score'] for s in portfolio_data['portfolio_snapshots']]
            market_stress = [s['market_data']['market_stress'] for s in portfolio_data['portfolio_snapshots']]
            
            # 1. Portfolio Allocation (top left)
            fig.add_trace(
                go.Scatter(x=dates, y=equity_alloc, name='Equity', fill='tonexty',
                          line=dict(color='#1f77b4')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=bonds_alloc, name='Bonds', fill='tonexty',
                          line=dict(color='#ff7f0e')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=cash_alloc, name='Cash', fill='tonexty',
                          line=dict(color='#2ca02c')),
                row=1, col=1
            )
            
            # 2. Comfort Score (top middle)
            fig.add_trace(
                go.Scatter(x=dates, y=comfort_scores, name='Comfort Score',
                          line=dict(color='#d62728', width=3)),
                row=1, col=2
            )
            
            # 3. Market Stress (top right)
            fig.add_trace(
                go.Scatter(x=dates, y=market_stress, name='Market Stress',
                          line=dict(color='#9467bd', width=3)),
                row=1, col=3
            )
            
            # 4. Life Choices Timeline (second row left)
            if self.optimizer.life_choices:
                choice_dates = [choice['date'] for choice in self.optimizer.life_choices]
                choice_labels = [f"{choice['category']}: {choice['choice']}" for choice in self.optimizer.life_choices]
                
                fig.add_trace(
                    go.Scatter(x=choice_dates, y=list(range(len(choice_dates))), 
                              mode='markers+text',
                              marker=dict(size=15, color='red'),
                              text=choice_labels, textposition='top center',
                              name='Life Choices'),
                    row=2, col=1
                )
            
            # 5. Portfolio Evolution (second row middle)
            if self.optimizer.life_choices:
                choice_equity = [choice['portfolio_after']['equity'] for choice in self.optimizer.life_choices]
                choice_bonds = [choice['portfolio_after']['bonds'] for choice in self.optimizer.life_choices]
                choice_cash = [choice['portfolio_after']['cash'] for choice in self.optimizer.life_choices]
                choice_dates = [choice['date'] for choice in self.optimizer.life_choices]
                
                fig.add_trace(
                    go.Scatter(x=choice_dates, y=choice_equity, name='Equity',
                              line=dict(color='#1f77b4')),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=choice_dates, y=choice_bonds, name='Bonds',
                              line=dict(color='#ff7f0e')),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=choice_dates, y=choice_cash, name='Cash',
                              line=dict(color='#2ca02c')),
                    row=2, col=2
                )
            
            # 6. Choice Impacts (second row right)
            if self.optimizer.life_choices:
                financial_impacts = [choice['impacts'].get('income_boost', 0) for choice in self.optimizer.life_choices]
                choice_dates = [choice['date'] for choice in self.optimizer.life_choices]
                
                fig.add_trace(
                    go.Bar(x=choice_dates, y=financial_impacts, name='Income Impact',
                          marker_color='green'),
                    row=2, col=3
                )
            
            # 7. Risk Metrics (third row left)
            volatilities = [s['comfort_metrics']['volatility'] for s in portfolio_data['portfolio_snapshots']]
            drawdowns = [s['comfort_metrics']['potential_drawdown'] for s in portfolio_data['portfolio_snapshots']]
            
            fig.add_trace(
                go.Scatter(x=dates, y=volatilities, name='Portfolio Volatility',
                          line=dict(color='#e377c2')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, name='Potential Drawdown',
                          line=dict(color='#8c564b')),
                row=3, col=1
            )
            
            # 8. Market Stress (third row middle)
            fig.add_trace(
                go.Scatter(x=dates, y=market_stress, name='Market Stress',
                          line=dict(color='#9467bd')),
                row=3, col=2
            )
            
            # 9. Category Distribution (third row right)
            if self.optimizer.life_choices:
                category_counts = {}
                for choice in self.optimizer.life_choices:
                    cat = choice['category']
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                if category_counts:
                    fig.add_trace(
                        go.Pie(labels=list(category_counts.keys()), 
                              values=list(category_counts.values()),
                              name='Category Distribution'),
                        row=3, col=3
                    )
        
        # 10. Life Choice Interface (fourth row left)
        # Show available life choices
        available_choices = []
        for category, choices in self.optimizer.choice_categories.items():
            for choice in choices:
                available_choices.append(f"{category}: {choice}")
        
        fig.add_trace(
            go.Bar(x=available_choices[:10], y=[1] * min(10, len(available_choices)),
                  name='Available Choices', marker_color='lightblue'),
            row=4, col=1
        )
        
        # 11. Optimization Results (fourth row middle)
        if self.optimizer.life_choices:
            objectives = list(self.optimizer.optimization_objectives.keys())
            best_scores = []
            
            for objective in objectives:
                result = self.optimizer.optimize_next_choice(objective)
                if result['best_choice']:
                    best_scores.append(result['best_choice']['total_score'])
                else:
                    best_scores.append(0)
            
            fig.add_trace(
                go.Bar(x=objectives, y=best_scores, name='Best Choice Score',
                      marker_color='purple'),
                row=4, col=2
            )
        
        # 12. Recommendations (fourth row right)
        if self.optimizer.life_choices:
            result = self.optimizer.optimize_next_choice('financial_growth')
            if result['recommendations']:
                top_choices = [rec['choice'] for rec in result['recommendations'][:5]]
                top_scores = [rec['total_score'] for rec in result['recommendations'][:5]]
                
                fig.add_trace(
                    go.Bar(x=top_choices, y=top_scores, name='Top Recommendations',
                          marker_color='orange'),
                    row=4, col=3
                )
        
        # Update layout
        fig.update_layout(
            title="Enhanced Portfolio Dashboard with Life Choice Optimization",
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Allocation %", row=1, col=1)
        fig.update_yaxes(title_text="Comfort Score", row=1, col=2)
        fig.update_yaxes(title_text="Market Stress", row=1, col=3)
        fig.update_yaxes(title_text="Life Choices", row=2, col=1)
        fig.update_yaxes(title_text="Allocation %", row=2, col=2)
        fig.update_yaxes(title_text="Income Impact", row=2, col=3)
        fig.update_yaxes(title_text="Risk Metrics", row=3, col=1)
        fig.update_yaxes(title_text="Market Stress", row=3, col=2)
        fig.update_yaxes(title_text="Available Choices", row=4, col=1)
        fig.update_yaxes(title_text="Optimization Score", row=4, col=2)
        fig.update_yaxes(title_text="Recommendation Score", row=4, col=3)
        
        return fig
    
    def generate_interactive_html(self):
        """Generate interactive HTML with optimization toggle and life choice interface"""
        import json
        from datetime import datetime
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Portfolio Dashboard with Life Choice Optimization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .header {{ background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .dashboard-container {{ display: flex; gap: 20px; }}
                .main-panel {{ flex: 2; background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .optimization-panel {{ flex: 1; background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .toggle-switch {{ position: relative; display: inline-block; width: 60px; height: 34px; }}
                .toggle-switch input {{ opacity: 0; width: 0; height: 0; }}
                .slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }}
                .slider:before {{ position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }}
                input:checked + .slider {{ background-color: #2196F3; }}
                input:checked + .slider:before {{ transform: translateX(26px); }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                .life-choice {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; }}
                .positive {{ border-left-color: #28a745; }}
                .negative {{ border-left-color: #dc3545; }}
                select, input, button {{ padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }}
                button {{ background-color: #007bff; color: white; cursor: pointer; }}
                button:hover {{ background-color: #0056b3; }}
                .recommendation {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #28a745; }}
                .hidden {{ display: none; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Enhanced Portfolio Dashboard with Life Choice Optimization</h1>
                <p>Analyze your life path and optimize your next financial decisions</p>
            </div>
            
            <div class="dashboard-container">
                <div class="main-panel">
                    <div class="section">
                        <h2>📊 Portfolio Summary</h2>
                        <div class="metric">
                            <strong>Portfolio Value:</strong> ${self.portfolio_engine.client_config.get('portfolio_value', 0):,}
                        </div>
                        <div class="metric">
                            <strong>Client Age:</strong> {self.portfolio_engine.client_config.get('age', 0)}
                        </div>
                        <div class="metric">
                            <strong>Risk Profile:</strong> {self.portfolio_engine.client_config.get('risk_profile', 0)}/5
                        </div>
                        <div class="metric">
                            <strong>Life Choices Made:</strong> {len(self.optimizer.life_choices)}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📈 Portfolio Evolution</h2>
                        <div id="portfolioChart"></div>
                    </div>
                    
                    <div class="section">
                        <h2>📅 Life Choices Timeline</h2>
                        <div id="lifeChoicesTimeline"></div>
                    </div>
                </div>
                
                <div class="optimization-panel">
                    <div class="section">
                        <h2>⚙️ Optimization Mode</h2>
                        <label class="toggle-switch">
                            <input type="checkbox" id="optimizationToggle" onchange="toggleOptimization()">
                            <span class="slider"></span>
                        </label>
                        <span id="toggleLabel">Optimization: OFF</span>
                    </div>
                    
                    <div class="section" id="lifeChoiceInterface">
                        <h2>🎯 Life Choice Interface</h2>
                        <p>Enter your life choices to see optimization recommendations:</p>
                        
                        <div>
                            <label>Category:</label>
                            <select id="choiceCategory">
                                <option value="">Select category...</option>
                                <option value="career">Career</option>
                                <option value="family">Family</option>
                                <option value="lifestyle">Lifestyle</option>
                                <option value="education">Education</option>
                                <option value="health">Health</option>
                            </select>
                        </div>
                        
                        <div>
                            <label>Choice:</label>
                            <select id="choiceType" disabled>
                                <option value="">Select choice...</option>
                            </select>
                        </div>
                        
                        <div>
                            <label>Date:</label>
                            <input type="date" id="choiceDate" value="{datetime.now().strftime('%Y-%m-%d')}">
                        </div>
                        
                        <button onclick="addLifeChoice()">Add Life Choice</button>
                        <button onclick="optimizeNextChoice()">Optimize Next Choice</button>
                    </div>
                    
                    <div class="section" id="optimizationResults" style="display: none;">
                        <h2>🎯 Optimization Results</h2>
                        <div id="optimizationContent"></div>
                    </div>
                    
                    <div class="section">
                        <h2>📋 Current Recommendations</h2>
                        <div id="recommendationsContent"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // Life choice categories and options
                const choiceOptions = {json.dumps(self.optimizer.choice_categories)};
                
                // Update choice options when category is selected
                document.getElementById('choiceCategory').addEventListener('change', function() {{
                    const category = this.value;
                    const choiceSelect = document.getElementById('choiceType');
                    choiceSelect.innerHTML = '<option value="">Select choice...</option>';
                    
                    if (category && choiceOptions[category]) {{
                        Object.keys(choiceOptions[category]).forEach(choice => {{
                            const option = document.createElement('option');
                            option.value = choice;
                            option.textContent = choice.replace('_', ' ');
                            choiceSelect.appendChild(option);
                        }});
                        choiceSelect.disabled = false;
                    }} else {{
                        choiceSelect.disabled = true;
                    }}
                }});
                
                // Toggle optimization mode
                function toggleOptimization() {{
                    const toggle = document.getElementById('optimizationToggle');
                    const label = document.getElementById('toggleLabel');
                    const results = document.getElementById('optimizationResults');
                    
                    if (toggle.checked) {{
                        label.textContent = 'Optimization: ON';
                        results.style.display = 'block';
                        optimizeNextChoice();
                    }} else {{
                        label.textContent = 'Optimization: OFF';
                        results.style.display = 'none';
                    }}
                }}
                
                // Add life choice
                function addLifeChoice() {{
                    const category = document.getElementById('choiceCategory').value;
                    const choice = document.getElementById('choiceType').value;
                    const date = document.getElementById('choiceDate').value;
                    
                    if (!category || !choice || !date) {{
                        alert('Please fill in all fields');
                        return;
                    }}
                    
                    // Here you would typically send this to the backend
                    console.log('Adding life choice:', {{category, choice, date}});
                    
                    // For demo purposes, show a success message
                    alert(`Life choice added: ${{category}} - ${{choice}} on ${{date}}`);
                    
                    // Clear form
                    document.getElementById('choiceCategory').value = '';
                    document.getElementById('choiceType').value = '';
                    document.getElementById('choiceType').disabled = true;
                }}
                
                // Optimize next choice
                function optimizeNextChoice() {{
                    const content = document.getElementById('optimizationContent');
                    
                    // Simulate optimization results
                    const objectives = ['financial_growth', 'comfort_stability', 'risk_management', 'lifestyle_quality'];
                    const recommendations = [
                        {{choice: 'promotion', category: 'career', score: 0.85, impact: '+15% income'}},
                        {{choice: 'advanced_degree', category: 'education', score: 0.72, impact: '+20% income'}},
                        {{choice: 'buy_house', category: 'lifestyle', score: 0.68, impact: '+40% stability'}},
                        {{choice: 'health_improvement', category: 'health', score: 0.65, impact: '+30% comfort'}}
                    ];
                    
                    let html = '<h3>Top Recommendations:</h3>';
                    recommendations.forEach((rec, i) => {{
                        html += `
                            <div class="recommendation">
                                <strong>${{i+1}}. ${{rec.choice.replace('_', ' ').toUpperCase()}}</strong> (${{rec.category}})
                                <br>Score: ${{rec.score}} | Impact: ${{rec.impact}}
                            </div>
                        `;
                    }});
                    
                    content.innerHTML = html;
                }}
                
                // Initialize
                document.addEventListener('DOMContentLoaded', function() {{
                    optimizeNextChoice();
                }});
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def create_optimization_demo(self):
        """Create a demonstration of the optimization system"""
        print("🎯 Life Choice Optimization Demo")
        print("=" * 50)
        
        # Add some sample life choices
        sample_choices = [
            ('career', 'promotion', '2023-01-15'),
            ('family', 'marriage', '2023-06-20'),
            ('lifestyle', 'buy_house', '2024-03-10'),
            ('education', 'certification', '2024-09-05')
        ]
        
        print("📝 Adding sample life choices...")
        for category, choice, date in sample_choices:
            result = self.optimizer.add_life_choice(category, choice, date)
            print(f"   ✅ {category}: {choice} on {date}")
            print(f"      Comfort Score: {result['comfort_score']:.2f}")
            print(f"      Portfolio: Equity {result['portfolio_after']['equity']:.1%}, "
                  f"Bonds {result['portfolio_after']['bonds']:.1%}, "
                  f"Cash {result['portfolio_after']['cash']:.1%}")
        
        # Run optimization for different objectives
        print(f"\n🎯 Running optimization for different objectives...")
        objectives = ['financial_growth', 'comfort_stability', 'risk_management', 'lifestyle_quality']
        
        for objective in objectives:
            result = self.optimizer.optimize_next_choice(objective)
            print(f"\n📊 {objective.replace('_', ' ').title()}:")
            if result['best_choice']:
                best = result['best_choice']
                print(f"   🏆 Best Choice: {best['choice'].replace('_', ' ').title()} ({best['category']})")
                print(f"      Score: {best['total_score']:.3f}")
                print(f"      Financial: {best['financial_score']:+.3f}")
                print(f"      Comfort: {best['comfort_score']:+.3f}")
                print(f"      Risk: {best['risk_score']:+.3f}")
                print(f"      Lifestyle: {best['lifestyle_score']:+.3f}")
        
        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        
        # Main dashboard
        main_dashboard = self.create_enhanced_dashboard()
        main_dashboard.write_html("enhanced_dashboard_with_optimization.html")
        
        # Optimization dashboard
        opt_dashboard = self.optimizer.create_optimization_dashboard()
        opt_dashboard.write_html("life_choice_optimization_dashboard.html")
        
        # Interactive HTML
        interactive_html = self.generate_interactive_html()
        with open("interactive_optimization_dashboard.html", "w") as f:
            f.write(interactive_html)
        
        # Generate report
        report = self.optimizer.generate_optimization_report('financial_growth')
        with open("optimization_report.md", "w") as f:
            f.write(report)
        
        print("✅ Demo completed successfully!")
        print("   - enhanced_dashboard_with_optimization.html (main dashboard)")
        print("   - life_choice_optimization_dashboard.html (optimization dashboard)")
        print("   - interactive_optimization_dashboard.html (interactive interface)")
        print("   - optimization_report.md (detailed report)")
        
        return {
            'main_dashboard': main_dashboard,
            'optimization_dashboard': opt_dashboard,
            'interactive_html': interactive_html,
            'report': report
        }

def create_optimization_demo():
    """Create and run the optimization demonstration"""
    dashboard = EnhancedDashboardWithOptimization()
    return dashboard.create_optimization_demo()

if __name__ == "__main__":
    create_optimization_demo() 