#!/usr/bin/env python
"""
Age-Based Lifecycle Modeling Demo - Streamlined Version

Demonstrates key lifecycle features:
- Risk aversion evolution with age
- 100-minus-age asset allocation
- Lifecycle stage transitions
- Age-adjusted cash flows

Usage: python age_based_lifecycle_demo.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, sys

# Import from main model
sys.path.append(str(Path(__file__).parent.parent / 'src'))
try:
    from ips_model import (get_age, interpolate_risk_aversion, calculate_allocation, 
                          get_lifecycle_stage, get_dynamic_profile, calculate_cashflow, MODEL_CONFIG)
    data = json.loads(Path("config/ips_config.json").read_text())
    YEARS = data["YEARS"]
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    sys.exit(1)

def demonstrate_lifecycle_evolution():
    """Show complete lifecycle evolution over 40 years"""
    print("🎪 LIFECYCLE EVOLUTION DEMONSTRATION")
    print("=" * 50)
    
    # Sample ages and calculations
    sample_years = [0, 10, 20, 30, 39]
    print(f"{'Year':<4} {'Age':<3} {'Stage':<15} {'Risk':<5} {'Equity':<6} {'Bonds':<6} {'Cash':<5}")
    print("-" * 55)
    
    evolution_data = []
    for year in sample_years:
        profile = get_dynamic_profile(year, base_risk_band=2)  # Moderate base risk
        allocation = profile['allocation']
        
        print(f"{year:<4} {profile['age']:<3} {profile['lifecycle_stage'].replace('_', ' ').title():<15} "
              f"{profile['risk_aversion']:<5.0%} {allocation['equity']:<6.0%} "
              f"{allocation['bonds']:<6.0%} {allocation['cash']:<5.0%}")
        
        evolution_data.append({
            'Year': year, 'Age': profile['age'], 'Stage': profile['lifecycle_stage'],
            'Risk_Aversion': profile['risk_aversion'], **allocation
        })
    
    return pd.DataFrame(evolution_data)

def demonstrate_cashflow_evolution():
    """Show how cash flows change with age"""
    print("\n💰 CASH FLOW EVOLUTION WITH AGE")
    print("=" * 50)
    
    # Sample configuration
    cfg = {'ED_PATH': 'McGill', 'HEL_WORK': 'Full-time', 'BONUS_PCT': 0.20, 
           'DON_STYLE': 0, 'RISK_BAND': 2, 'FX_SCENARIO': 'Base'}
    
    print(f"Configuration: {cfg['ED_PATH']}, {cfg['HEL_WORK']}, {cfg['BONUS_PCT']:.0%} bonus")
    print(f"\n{'Year':<4} {'Age':<3} {'H_Salary':<10} {'Hel_Salary':<11} {'Healthcare':<10} {'Net_CF':<8}")
    print("-" * 55)
    
    cashflow_data = []
    for year in [0, 10, 20, 30, 39]:
        cf = calculate_cashflow(year, cfg)
        net_cf = sum(v for k, v in cf.items() if isinstance(v, (int, float)) and k not in ['Year', 'Age'])
        
        print(f"{year:<4} {cf['Age']:<3} ${cf['H_SAL']:<9,.0f} ${cf['HEL_SAL']:<10,.0f} "
              f"${abs(cf.get('Healthcare', 0)):<9,.0f} ${net_cf:<7,.0f}")
        
        cashflow_data.append({'Year': year, 'Age': cf['Age'], 'Net_CF': net_cf, **cf})
    
    return pd.DataFrame(cashflow_data)

def create_lifecycle_visualization():
    """Create streamlined lifecycle visualization"""
    print("\n📈 CREATING LIFECYCLE VISUALIZATION")
    print("=" * 50)
    
    try:
        # Generate data
        years = list(range(0, YEARS, 2))  # Every 2 years for clarity
        ages = [get_age(y) for y in years]
        
        # Get profiles for each year
        profiles = [get_dynamic_profile(y, base_risk_band=2) for y in years]
        risk_aversion = [p['risk_aversion'] for p in profiles]
        equity_pct = [p['allocation']['equity'] for p in profiles]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Age-Based Lifecycle Evolution (40 Years)', fontweight='bold')
        
        # Risk aversion evolution
        ax1.plot(ages, risk_aversion, 'r-o', linewidth=2, markersize=4)
        ax1.set_title('Risk Aversion by Age')
        ax1.set_xlabel('Age'), ax1.set_ylabel('Risk Aversion')
        ax1.grid(True, alpha=0.3), ax1.set_ylim(0, 1)
        
        # Equity allocation evolution
        ax2.plot(ages, equity_pct, 'g-s', linewidth=2, markersize=4)
        ax2.set_title('Equity Allocation (100-minus-age)')
        ax2.set_xlabel('Age'), ax2.set_ylabel('Equity %')
        ax2.grid(True, alpha=0.3), ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save
        output_path = Path("ips_output") / "lifecycle_evolution_demo.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Visualization saved: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")

def generate_demo_report():
    """Generate comprehensive demo dataset"""
    print("\n📋 GENERATING DEMO REPORT")
    print("=" * 50)
    
    # High-stress configuration for demonstration
    demo_cfg = {'ED_PATH': 'JohnsHopkins', 'HEL_WORK': 'Full-time', 'BONUS_PCT': 0.30,
                'DON_STYLE': 2, 'RISK_BAND': 3, 'FX_SCENARIO': 'Stress'}
    
    print(f"Demo Configuration: High-stress scenario")
    print(f"   {demo_cfg['ED_PATH']}, {demo_cfg['HEL_WORK']}, {demo_cfg['BONUS_PCT']:.0%} bonus")
    
    # Generate full lifecycle data
    lifecycle_data = []
    for year in range(YEARS):
        profile = get_dynamic_profile(year, demo_cfg['RISK_BAND'])
        cf = calculate_cashflow(year, demo_cfg)
        
        lifecycle_data.append({
            'year': year, 'age': profile['age'], 'lifecycle_stage': profile['lifecycle_stage'],
            'risk_aversion': profile['risk_aversion'], 'dynamic_risk_band': profile['dynamic_risk_band'],
            **profile['allocation'], 'total_income': cf['H_SAL'] + cf['HEL_SAL'] + cf['H_BONUS'],
            'net_cashflow': sum(v for k, v in cf.items() if isinstance(v, (int, float)) and k not in ['Year', 'Age'])
        })
    
    df = pd.DataFrame(lifecycle_data)
    
    # Save report
    output_dir = Path("ips_output")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "lifecycle_demo_report.csv", index=False)
    
    # Key insights
    print(f"\n📊 Key Insights:")
    print(f"   Risk aversion range: {df['risk_aversion'].min():.0%} - {df['risk_aversion'].max():.0%}")
    print(f"   Equity allocation range: {df['equity'].min():.0%} - {df['equity'].max():.0%}")
    print(f"   Peak income year: {df['total_income'].idxmax()}")
    print(f"   Lifecycle stages: {df['lifecycle_stage'].unique()}")
    print(f"   Report saved: {output_dir / 'lifecycle_demo_report.csv'}")
    
    return df

def main():
    """Run streamlined lifecycle demonstration"""
    print("🎉 AGE-BASED LIFECYCLE MODELING DEMO")
    print("=" * 60)
    print(f"📅 Modeling {YEARS} years starting at age {MODEL_CONFIG['lifecycle']['starting_age']}")
    print(f"📊 100-minus-age rule: {'Enabled' if MODEL_CONFIG['lifecycle']['use_100_minus_age_rule'] else 'Disabled'}")
    
    # Run demonstrations
    evolution_df = demonstrate_lifecycle_evolution()
    cashflow_df = demonstrate_cashflow_evolution()
    create_lifecycle_visualization()
    demo_df = generate_demo_report()
    
    # Summary
    print(f"\n🎯 DEMONSTRATION COMPLETE!")
    print(f"   📊 Lifecycle evolution: {len(evolution_df)} key milestones")
    print(f"   💰 Cash flow analysis: {len(cashflow_df)} sample years")
    print(f"   📈 Full demo dataset: {len(demo_df)} years")
    print(f"   📁 Files saved to: ips_output/")
    
    print(f"\n💡 Next Steps:")
    print(f"   🔄 Run: python ips_model.py (enhanced analysis)")
    print(f"   📊 Check: ips_output/ (generated files)")
    print(f"   🎨 View: lifecycle_evolution_demo.png (visual)")

if __name__ == "__main__":
    main() 