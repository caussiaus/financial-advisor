#!/usr/bin/env python
"""
Visual Timeline Builder for IPS Configurations
Author: ChatGPT 2025-07-16

Generates visual timelines showing major life events, cash flows, and stress patterns
for each financial configuration. Designed for advisor-client quarterly reviews.

Usage:
    python visual_timeline_builder.py

Features:
- Timeline visualization of major life events
- Cash flow patterns over 40 years
- Stress level progression
- Configuration comparison charts
- Client-ready visual reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import configuration data
from ips_model import (
    PARAM, YEARS, cashflow_row, FX_SCENARIOS, 
    calculate_financial_stress_metrics, StressMonitor
)

class TimelineVisualizer:
    """Generate visual timelines for financial configurations"""
    
    def __init__(self, output_dir="visual_timelines"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define event colors and styles
        self.event_colors = {
            'education': '#FF6B6B',      # Red for education costs
            'career': '#4ECDC4',         # Teal for career changes
            'housing': '#45B7D1',        # Blue for housing
            'charity': '#96CEB4',        # Green for charitable giving
            'family': '#FFEAA7',         # Yellow for family events
            'financial': '#DDA0DD',      # Purple for financial milestones
            'stress_critical': '#FF4757', # Bright red for critical stress
            'stress_warning': '#FFA502',  # Orange for warning stress
            'stress_normal': '#2ED573'    # Green for normal stress
        }
        
        # Major life event definitions
        self.life_events = {
            'daycare_start': {'year': 0, 'description': 'Daycare begins', 'category': 'family'},
            'daycare_end': {'year': 5, 'description': 'Daycare ends', 'category': 'family'},
            'education_start': {'year': 5, 'description': 'Higher education begins', 'category': 'education'},
            'education_end': {'year': 10, 'description': 'Education completed', 'category': 'education'},
            'mortgage_complete': {'year': 30, 'description': 'Mortgage paid off', 'category': 'housing'},
            'retirement_start': {'year': 35, 'description': 'Retirement begins', 'category': 'career'}
        }
    
    def create_configuration_timeline(self, cfg, cfg_id, stress_metrics=None):
        """Create comprehensive timeline for a single configuration"""
        
        # Generate cash flow data
        cashflows = pd.DataFrame([cashflow_row(y, cfg) for y in range(YEARS)])
        cashflows["NetCF"] = cashflows.sum(axis=1, numeric_only=True)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Financial Timeline: {cfg_id}\n'
                    f'{cfg["ED_PATH"]} • {cfg["HEL_WORK"]} • {cfg["BONUS_PCT"]:.0%} Bonus • '
                    f'{["Regular", "Deferred", "Immediate"][cfg["DON_STYLE"]]} Giving', 
                    fontsize=16, fontweight='bold')
        
        # 1. Major Life Events Timeline
        self._plot_life_events_timeline(ax1, cfg, cashflows)
        
        # 2. Cash Flow Waterfall
        self._plot_cash_flow_components(ax2, cashflows)
        
        # 3. Net Cash Flow Over Time
        self._plot_net_cash_flow(ax3, cashflows, stress_metrics)
        
        # 4. Financial Stress Progression (if available)
        if stress_metrics:
            self._plot_stress_progression(ax4, stress_metrics, cfg)
        else:
            ax4.text(0.5, 0.5, 'Stress metrics not available\nRun Monte Carlo analysis first', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Financial Stress Progression')
        
        plt.tight_layout()
        
        # Save the timeline
        timeline_path = self.output_dir / f"timeline_{cfg_id}.png"
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return timeline_path
    
    def _plot_life_events_timeline(self, ax, cfg, cashflows):
        """Plot major life events on a timeline"""
        ax.set_title('Major Life Events & Financial Milestones', fontweight='bold', pad=20)
        
        # Create timeline base
        years = range(YEARS)
        ax.plot(years, [1]*YEARS, 'k-', linewidth=2, alpha=0.3)
        
        # Plot standard life events
        event_y_positions = {}
        y_level = 1
        
        for event_key, event_info in self.life_events.items():
            if event_key in ['daycare_start', 'daycare_end'] and cfg["HEL_WORK"] != "Full-time":
                continue  # Skip daycare events for part-time work
            
            year = event_info['year']
            color = self.event_colors[event_info['category']]
            
            # Adjust y position to avoid overlap
            while y_level in event_y_positions.values():
                y_level += 0.1
            event_y_positions[event_key] = y_level
            
            ax.scatter(year, y_level, s=200, c=color, alpha=0.8, edgecolors='black', linewidth=2)
            ax.annotate(event_info['description'], (year, y_level), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                       fontsize=9, ha='left')
        
        # Add configuration-specific events
        self._add_config_specific_events(ax, cfg, cashflows)
        
        ax.set_xlim(-2, YEARS+2)
        ax.set_ylim(0.8, 2.5)
        ax.set_xlabel('Years from Today', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    def _add_config_specific_events(self, ax, cfg, cashflows):
        """Add configuration-specific events to timeline"""
        
        # Charity events
        charity_years = cashflows[cashflows['Charity'] < 0]['Year'].tolist()
        for year in charity_years:
            ax.scatter(year, 1.3, s=150, c=self.event_colors['charity'], 
                      marker='D', alpha=0.8, edgecolors='black')
            if len(charity_years) == 1:  # Lump sum
                label = f'${abs(cashflows.loc[year, "Charity"]/1000):.0f}K Donation'
            else:  # Annual
                label = f'${abs(cashflows.loc[year, "Charity"]/1000):.0f}K Annual'
            
            ax.annotate(label, (year, 1.3), xytext=(0, 15), 
                       textcoords='offset points', ha='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=self.event_colors['charity'], alpha=0.7))
        
        # Education costs
        education_years = cashflows[cashflows['Tuition'] < 0]['Year'].tolist()
        if education_years:
            start_year, end_year = min(education_years), max(education_years)
            total_tuition = abs(cashflows['Tuition'].sum())
            
            # Draw education period as a span
            ax.axvspan(start_year, end_year, alpha=0.2, color=self.event_colors['education'])
            ax.text((start_year + end_year) / 2, 1.7, 
                   f'{cfg["ED_PATH"]}\n${total_tuition/1000:.0f}K Total',
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=self.event_colors['education'], alpha=0.8))
    
    def _plot_cash_flow_components(self, ax, cashflows):
        """Plot cash flow components as stacked areas"""
        ax.set_title('Cash Flow Components Over Time', fontweight='bold', pad=20)
        
        years = cashflows['Year']
        
        # Positive cash flows (income)
        income_components = ['H_SAL', 'H_BONUS', 'HEL_SAL']
        income_data = cashflows[income_components].values.T
        
        ax.stackplot(years, *income_data, 
                    labels=['H Salary', 'H Bonus', 'Hel Salary'],
                    colors=['#2ECC71', '#27AE60', '#16A085'], alpha=0.8)
        
        # Negative cash flows (expenses) - plot as negative
        expense_components = ['Daycare', 'Tuition', 'Mortgage', 'Ppty', 'Charity']
        expense_data = -cashflows[expense_components].values.T  # Make positive for stacking
        
        ax.stackplot(years, *expense_data, 
                    labels=['Daycare', 'Tuition', 'Mortgage', 'Property', 'Charity'],
                    colors=['#E74C3C', '#C0392B', '#8E44AD', '#2980B9', '#F39C12'], alpha=0.8)
        
        # Add net cash flow line
        ax.plot(years, cashflows['NetCF'], 'k-', linewidth=3, label='Net Cash Flow', alpha=0.9)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Years from Today', fontweight='bold')
        ax.set_ylabel('Annual Cash Flow ($)', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_net_cash_flow(self, ax, cashflows, stress_metrics):
        """Plot net cash flow with stress indicators"""
        ax.set_title('Net Cash Flow & Financial Health', fontweight='bold', pad=20)
        
        years = cashflows['Year']
        net_cf = cashflows['NetCF']
        
        # Color code based on cash flow
        colors = ['red' if cf < 0 else 'green' if cf > 50000 else 'orange' for cf in net_cf]
        
        # Plot bars
        bars = ax.bar(years, net_cf, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add cumulative line
        cumulative_cf = net_cf.cumsum()
        ax2 = ax.twinx()
        ax2.plot(years, cumulative_cf, 'b-', linewidth=3, label='Cumulative Cash Flow', alpha=0.8)
        ax2.set_ylabel('Cumulative Cash Flow ($)', fontweight='bold', color='blue')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Highlight stress periods if available
        if stress_metrics and 'prob_shortfall_10yr' in stress_metrics:
            stress_level = stress_metrics['prob_shortfall_10yr']
            if stress_level > 0.3:
                ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.1, color='red')
                ax.text(0.02, 0.98, f'HIGH STRESS\n{stress_level:.1%} shortfall risk', 
                       transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                       fontweight='bold', color='white')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Years from Today', fontweight='bold')
        ax.set_ylabel('Annual Net Cash Flow ($)', fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.grid(True, alpha=0.3)
    
    def _plot_stress_progression(self, ax, stress_metrics, cfg):
        """Plot financial stress progression over time"""
        ax.set_title('Financial Stress Indicators', fontweight='bold', pad=20)
        
        # Simulate stress progression (in real implementation, this would come from Monte Carlo)
        years = list(range(YEARS))
        
        # Base stress level from metrics
        base_stress = stress_metrics.get('prob_shortfall_10yr', 0.2)
        
        # Simulate varying stress over time
        stress_progression = []
        for year in years:
            # Higher stress during education years and early career
            if 5 <= year <= 10:  # Education period
                year_stress = base_stress * 1.5
            elif year <= 15:  # Early career
                year_stress = base_stress * 1.2
            elif year >= 30:  # Later career/retirement
                year_stress = base_stress * 0.7
            else:
                year_stress = base_stress
            
            # Add some variability
            year_stress += np.random.normal(0, 0.02)
            stress_progression.append(max(0, min(1, year_stress)))
        
        # Plot stress line with color coding
        for i in range(len(years)-1):
            stress = stress_progression[i]
            if stress > 0.4:
                color = self.event_colors['stress_critical']
            elif stress > 0.25:
                color = self.event_colors['stress_warning']
            else:
                color = self.event_colors['stress_normal']
            
            ax.plot([years[i], years[i+1]], [stress, stress_progression[i+1]], 
                   color=color, linewidth=3, alpha=0.8)
        
        # Add stress threshold lines
        ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        # Highlight critical periods
        critical_periods = [i for i, s in enumerate(stress_progression) if s > 0.4]
        if critical_periods:
            ax.fill_between(years, 0, stress_progression, 
                           where=[s > 0.4 for s in stress_progression],
                           alpha=0.2, color='red', label='Critical Stress Periods')
        
        ax.set_xlabel('Years from Today', fontweight='bold')
        ax.set_ylabel('Financial Stress Level', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def create_configuration_comparison(self, configs_data, top_n=6):
        """Create comparison chart of top configurations"""
        
        # Select top configurations by different criteria
        sorted_by_stress = sorted(configs_data, key=lambda x: x.get('Financial_Stress_Rank', 0))
        sorted_by_qol = sorted(configs_data, key=lambda x: x.get('QoL_Score', 0), reverse=True)
        
        # Take mix of best and worst for comparison
        comparison_configs = (sorted_by_stress[:3] + sorted_by_stress[-3:])[:top_n]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Configuration Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Stress vs QoL scatter plot
        self._plot_stress_qol_comparison(ax1, configs_data)
        
        # 2. Financial metrics comparison
        self._plot_financial_metrics_comparison(ax2, comparison_configs)
        
        # 3. Configuration summary table
        self._plot_configuration_summary(ax3, comparison_configs)
        
        # 4. Decision matrix
        self._plot_decision_matrix(ax4, comparison_configs)
        
        plt.tight_layout()
        
        comparison_path = self.output_dir / "configuration_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return comparison_path
    
    def _plot_stress_qol_comparison(self, ax, configs_data):
        """Plot stress vs quality of life scatter"""
        ax.set_title('Financial Stress vs Quality of Life', fontweight='bold', pad=20)
        
        stress_scores = [cfg.get('Financial_Stress_Rank', 0) for cfg in configs_data]
        qol_scores = [cfg.get('QoL_Score', 0) for cfg in configs_data]
        
        # Color by education path
        colors = ['red' if cfg.get('ED_PATH') == 'JohnsHopkins' else 'blue' for cfg in configs_data]
        
        scatter = ax.scatter(stress_scores, qol_scores, c=colors, alpha=0.6, s=50)
        
        # Add quadrant lines
        ax.axvline(x=np.median(stress_scores), color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=np.median(qol_scores), color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        ax.text(0.02, 0.98, 'Low Stress\nHigh QoL\n(OPTIMAL)', transform=ax.transAxes, 
               va='top', ha='left', fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        ax.text(0.98, 0.02, 'High Stress\nLow QoL\n(AVOID)', transform=ax.transAxes, 
               va='bottom', ha='right', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlabel('Financial Stress Score', fontweight='bold')
        ax.set_ylabel('Quality of Life Score', fontweight='bold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=8, label='Johns Hopkins'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                 markersize=8, label='McGill')]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_financial_metrics_comparison(self, ax, configs):
        """Plot key financial metrics comparison"""
        ax.set_title('Key Financial Metrics Comparison', fontweight='bold', pad=20)
        
        cfg_ids = [cfg['cfg_id'] for cfg in configs]
        
        metrics = ['Financial_Stress_Rank', 'QoL_Score', 'prob_shortfall_10yr']
        metric_labels = ['Stress Score', 'QoL Score', 'Shortfall Risk']
        
        x = np.arange(len(cfg_ids))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [cfg.get(metric, 0) for cfg in configs]
            
            # Normalize values for comparison
            if metric == 'QoL_Score':
                normalized_values = values  # Already 0-1
            else:
                max_val = max(values) if values else 1
                normalized_values = [v / max_val for v in values]
            
            ax.bar(x + i*width, normalized_values, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Configuration ID', fontweight='bold')
        ax.set_ylabel('Normalized Score (0-1)', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(cfg_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_configuration_summary(self, ax, configs):
        """Plot configuration summary table"""
        ax.set_title('Configuration Summary', fontweight='bold', pad=20)
        ax.axis('off')
        
        # Create summary table data
        table_data = []
        headers = ['Config ID', 'Education', 'Work Style', 'Bonus', 'Stress', 'QoL', 'Recommendation']
        
        for cfg in configs:
            stress = cfg.get('Financial_Stress_Rank', 0)
            qol = cfg.get('QoL_Score', 0)
            
            # Generate recommendation
            if stress < 0.2 and qol > 0.6:
                recommendation = "✅ OPTIMAL"
            elif stress < 0.3:
                recommendation = "⚠️ ACCEPTABLE"
            else:
                recommendation = "❌ HIGH RISK"
            
            row = [
                cfg['cfg_id'],
                cfg.get('ED_PATH', 'N/A')[:8],  # Truncate for space
                cfg.get('HEL_WORK', 'N/A')[:8],
                f"{cfg.get('BONUS_PCT', 0):.0%}",
                f"{stress:.1%}",
                f"{qol:.2f}",
                recommendation
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold')
        
        # Color code recommendations
        for i, row in enumerate(table_data):
            recommendation = row[-1]
            recommendation_col = len(headers) - 1  # Last column index
            if "OPTIMAL" in recommendation:
                table[(i+1, recommendation_col)].set_facecolor('#90EE90')
            elif "ACCEPTABLE" in recommendation:
                table[(i+1, recommendation_col)].set_facecolor('#FFE4B5')
            elif "HIGH RISK" in recommendation:
                table[(i+1, recommendation_col)].set_facecolor('#FFB6C1')
    
    def _plot_decision_matrix(self, ax, configs):
        """Plot decision matrix heatmap"""
        ax.set_title('Decision Matrix Heatmap', fontweight='bold', pad=20)
        
        # Create decision matrix
        cfg_ids = [cfg['cfg_id'] for cfg in configs]
        criteria = ['Low Stress', 'High QoL', 'Low Shortfall Risk', 'Overall Score']
        
        matrix_data = []
        for cfg in configs:
            stress_score = 1 - cfg.get('Financial_Stress_Rank', 0)  # Invert stress (higher is better)
            qol_score = cfg.get('QoL_Score', 0)
            shortfall_score = 1 - cfg.get('prob_shortfall_10yr', 0)  # Invert shortfall risk
            overall_score = (stress_score + qol_score + shortfall_score) / 3
            
            matrix_data.append([stress_score, qol_score, shortfall_score, overall_score])
        
        matrix_df = pd.DataFrame(matrix_data, index=cfg_ids, columns=criteria)
        
        # Create heatmap
        sns.heatmap(matrix_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'Score (0-1, higher is better)'})
        
        ax.set_ylabel('Configuration ID', fontweight='bold')
        ax.set_xlabel('Decision Criteria', fontweight='bold')

def load_enhanced_configurations():
    """Load enhanced configuration data from CSV"""
    try:
        configs_df = pd.read_csv('ips_output/configurations_enhanced.csv')
        return configs_df.to_dict('records')
    except FileNotFoundError:
        print("Enhanced configurations not found. Please run ips_model.py first.")
        return []

def generate_all_timelines():
    """Generate timeline visualizations for all configurations"""
    
    print("🎨 VISUAL TIMELINE BUILDER")
    print("=" * 50)
    
    # Load configuration data
    configs_data = load_enhanced_configurations()
    if not configs_data:
        return
    
    # Initialize visualizer
    visualizer = TimelineVisualizer()
    
    print(f"📊 Generating timelines for {len(configs_data)} configurations...")
    
    # Generate timelines for top configurations (to avoid overwhelming output)
    sorted_configs = sorted(configs_data, key=lambda x: x.get('Financial_Stress_Rank', 0))
    top_configs = sorted_configs[:10]  # Top 10 lowest stress
    high_stress_configs = sorted_configs[-5:]  # Top 5 highest stress
    
    timeline_paths = []
    
    # Generate individual timelines
    for i, cfg in enumerate(top_configs + high_stress_configs):
        print(f"   📈 Creating timeline {i+1}/15: {cfg['cfg_id']}")
        
        # Extract stress metrics
        stress_metrics = {
            'prob_shortfall_10yr': cfg.get('prob_shortfall_10yr', 0),
            'cashflow_volatility_10yr': cfg.get('cashflow_volatility_10yr', 0),
            'Financial_Stress_Rank': cfg.get('Financial_Stress_Rank', 0)
        }
        
        timeline_path = visualizer.create_configuration_timeline(cfg, cfg['cfg_id'], stress_metrics)
        timeline_paths.append(timeline_path)
    
    # Generate comparison dashboard
    print("   📊 Creating configuration comparison dashboard...")
    comparison_path = visualizer.create_configuration_comparison(configs_data)
    
    print(f"\n✅ Timeline generation complete!")
    print(f"   📁 Individual timelines: {len(timeline_paths)} files")
    print(f"   📊 Comparison dashboard: {comparison_path}")
    print(f"   📂 Output directory: {visualizer.output_dir}")
    
    # Generate summary report
    print(f"\n📋 TIMELINE SUMMARY:")
    print(f"   🎯 Best configurations (lowest stress): {len(top_configs)}")
    print(f"   ⚠️  High-risk configurations: {len(high_stress_configs)}")
    print(f"   📈 Visual files ready for client presentations")
    
    return timeline_paths, comparison_path

if __name__ == "__main__":
    generate_all_timelines() 