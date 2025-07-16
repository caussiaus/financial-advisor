#!/usr/bin/env python
"""
Financial Stress Pattern Analyzer
Analyzes the Monte Carlo simulation results to identify key stress drivers
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_stress_patterns():
    """Analyze financial stress patterns from Monte Carlo results"""
    
    # Load the enhanced configuration results
    results_path = Path("ips_output/configurations_enhanced.csv")
    if not results_path.exists():
        print("❌ Enhanced results not found. Run ips_model.py first.")
        return
    
    df = pd.read_csv(results_path)
    
    print("🔍 FINANCIAL STRESS PATTERN ANALYSIS")
    print("=" * 50)
    
    # 1. Factor Impact Analysis
    print("\n📊 STRESS FACTORS ANALYSIS:")
    
    factors = ['ED_PATH', 'HEL_WORK', 'BONUS_PCT', 'DON_STYLE', 'FX_SCENARIO']
    
    for factor in factors:
        if factor in df.columns:
            stress_by_factor = df.groupby(factor)['Financial_Stress_Rank'].agg(['mean', 'std', 'count'])
            print(f"\n{factor}:")
            for idx, row in stress_by_factor.iterrows():
                print(f"  {idx}: μ={row['mean']:.3f}, σ={row['std']:.3f}, n={row['count']}")
    
    # 2. High Stress Configuration Analysis
    print(f"\n🚨 HIGH STRESS CONFIGURATIONS (>75th percentile):")
    high_stress_threshold = df['Financial_Stress_Rank'].quantile(0.75)
    high_stress = df[df['Financial_Stress_Rank'] > high_stress_threshold]
    
    print(f"Threshold: {high_stress_threshold:.3f}")
    print(f"Count: {len(high_stress)} out of {len(df)} ({len(high_stress)/len(df)*100:.1f}%)")
    
    # Analyze patterns in high stress configs
    print(f"\nPatterns in High Stress Configurations:")
    for factor in factors:
        if factor in high_stress.columns:
            pattern = high_stress[factor].value_counts(normalize=True)
            print(f"{factor}: {dict(pattern)}")
    
    # 3. Quality of Life Analysis
    print(f"\n🎯 QUALITY OF LIFE BREAKDOWN:")
    qol_columns = [col for col in df.columns if col.startswith('QoL_')]
    
    # Compare high stress vs low stress QoL
    low_stress_threshold = df['Financial_Stress_Rank'].quantile(0.25)
    low_stress = df[df['Financial_Stress_Rank'] < low_stress_threshold]
    
    print(f"\nQoL Comparison (High Stress vs Low Stress):")
    for col in qol_columns:
        if col in df.columns:
            high_mean = high_stress[col].mean()
            low_mean = low_stress[col].mean()
            diff = high_mean - low_mean
            print(f"  {col.replace('QoL_', '')}: High={high_mean:.3f}, Low={low_mean:.3f}, Δ={diff:.3f}")
    
    # 4. Risk Metric Analysis
    print(f"\n📈 RISK METRICS SUMMARY:")
    risk_metrics = ['prob_shortfall_10yr', 'cashflow_volatility_10yr', 'prob_insolvency_10yr']
    
    for metric in risk_metrics:
        if metric in df.columns:
            print(f"{metric}:")
            print(f"  Mean: {df[metric].mean():.3f}")
            print(f"  Std:  {df[metric].std():.3f}")
            print(f"  Max:  {df[metric].max():.3f}")
            print(f"  >20%: {len(df[df[metric] > 0.2])} configs")
    
    # 5. Generate Decision Matrix
    print(f"\n🎯 DECISION MATRIX - TOP 10 LEAST STRESSFUL:")
    
    least_stress = df.nsmallest(10, 'Financial_Stress_Rank')[
        ['cfg_id', 'ED_PATH', 'HEL_WORK', 'BONUS_PCT', 'QoL_Score', 'Financial_Stress_Rank']
    ]
    
    for _, row in least_stress.iterrows():
        print(f"  {row['cfg_id']}: {row['ED_PATH']}, {row['HEL_WORK']}, "
              f"Bonus: {row['BONUS_PCT']:.0%}, QoL: {row['QoL_Score']:.2f}, "
              f"Stress: {row['Financial_Stress_Rank']:.3f}")
    
    # 6. FSQCA Insights
    fsqca_path = Path("ips_output/fsqca_analysis_ready.csv")
    if fsqca_path.exists():
        print(f"\n🔬 FSQCA CAUSAL ANALYSIS:")
        fsqca_df = pd.read_csv(fsqca_path)
        
        # Find configurations with high financial stress
        high_fs = fsqca_df[fsqca_df['financial_stress'] > 0.3]
        
        if len(high_fs) > 0:
            print(f"Conditions present in {len(high_fs)} high-stress configurations:")
            conditions = ['high_education', 'full_time_work', 'high_bonus', 'immediate_giving', 'fx_stress']
            for condition in conditions:
                if condition in high_fs.columns:
                    avg_presence = high_fs[condition].mean()
                    print(f"  {condition}: {avg_presence:.2f} (present in {avg_presence*100:.0f}% of cases)")
    
    # 7. Recommendations Summary
    print(f"\n💡 KEY RECOMMENDATIONS:")
    
    # Find the safest education path
    ed_stress = df.groupby('ED_PATH')['Financial_Stress_Rank'].mean()
    safest_ed = ed_stress.idxmin()
    riskiest_ed = ed_stress.idxmax()
    
    # Find the safest work arrangement
    work_stress = df.groupby('HEL_WORK')['Financial_Stress_Rank'].mean()
    safest_work = work_stress.idxmin()
    
    # Find the safest bonus level
    bonus_stress = df.groupby('BONUS_PCT')['Financial_Stress_Rank'].mean()
    safest_bonus = bonus_stress.idxmin()
    
    print(f"  1. Education: Choose {safest_ed} over {riskiest_ed}")
    print(f"     (Avg stress: {ed_stress[safest_ed]:.3f} vs {ed_stress[riskiest_ed]:.3f})")
    
    print(f"  2. Work: {safest_work} work provides most stability")
    print(f"     (Avg stress: {work_stress[safest_work]:.3f})")
    
    print(f"  3. Bonus: Conservative {safest_bonus:.0%} expectation is safest")
    print(f"     (Avg stress: {bonus_stress[safest_bonus]:.3f})")
    
    # Risk tolerance analysis
    risk_stress = df.groupby('RISK_BAND')['Financial_Stress_Rank'].mean()
    print(f"  4. Risk Band Impact: {dict(risk_stress)}")
    
    print(f"\n✅ Analysis complete. Review configurations_enhanced.csv for full details.")

if __name__ == "__main__":
    analyze_stress_patterns() 