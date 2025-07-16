#!/usr/bin/env python
# IPS Config‑Space Generator with Age-Based Lifecycle Modeling
# Enhanced Monte Carlo and Financial Stress Analysis
# Author: ChatGPT 2025‑07‑16

import json, itertools, string, math, random
from pathlib import Path
import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy import stats

# ------------------------------------------------------------------#
# CONSOLIDATED CONFIGURATION PARAMETERS                             #
# ------------------------------------------------------------------#

# Load base configuration
def load_config(config_path="config/ips_config.json"):
    """Loads configuration from a JSON file."""
    data = json.loads(Path(config_path).read_text())
    YEARS, PARAM, FACTOR_SPACE = data["YEARS"], data["PARAM"], data["FACTOR_SPACE"]
    RISK_SPLITS = {int(k): v for k, v in data["RISK_SPLITS"].items()}
    FX_SCENARIOS = data.get("FX_SCENARIOS", {"Base": 1.0, "Stress": 0.9, "Growth": 1.1})
    return YEARS, PARAM, FACTOR_SPACE, RISK_SPLITS, FX_SCENARIOS

YEARS, PARAM, FACTOR_SPACE, RISK_SPLITS, FX_SCENARIOS = load_config()

# Consolidated Model Parameters
MODEL_CONFIG = {
    # Monte Carlo & Economic Scenarios
    'mc_iterations': 100,
    'random_seed': 42,
    'equity_return': {'mean': 0.08, 'std': 0.16},
    'bond_return': {'mean': 0.04, 'std': 0.08},
    'economic_shocks': {
        'recession_prob': 0.15, 'recession_income_drop': 0.15, 'recession_expense_increase': 0.05,
        'inflation_spike_prob': 0.10, 'fx_volatility': 0.05,
        'income_volatility': 0.10, 'expense_volatility': 0.08
    },
    
    # Age-Based Lifecycle Parameters
    'lifecycle': {
        'starting_age': 30, 'retirement_age': 65, 'life_expectancy': 90,
        'use_100_minus_age_rule': True, 'min_equity': 0.20, 'max_equity': 0.90,
        'age_risk_decay_rate': 0.012, 'income_peak_age': 50, 'income_decline_age': 60,
        'healthcare_cost_age': 55, 'mortgage_completion_age': 55
    },
    
    # Age-Risk Aversion Mapping (interpolated for all ages)
    'age_risk_aversion': {25: 0.15, 30: 0.20, 35: 0.25, 40: 0.35, 45: 0.45, 50: 0.55, 
                         55: 0.65, 60: 0.75, 65: 0.85, 70: 0.90, 75: 0.95},
    
    # Scoring Weights
    'weights': {
        'qol': {'financial_security': 0.35, 'income_stability': 0.25, 'lifestyle_quality': 0.20, 
                'generosity_fulfillment': 0.10, 'cushion_comfort': 0.10},
        'stress': {'shortfall': 0.4, 'volatility': 0.3, 'insolvency': 0.3}
    },
    
    # Intervention Parameters
    'interventions': {
        'stress_thresholds': {'alert': 0.25, 'critical': 0.40},
        'impacts': {
            'reduce_education_cost': 0.15, 'switch_to_part_time': 0.10, 'reduce_bonus_dependency': 0.08,
            'defer_charitable_giving': 0.12, 'increase_portfolio_conservatism': 0.06, 'build_emergency_fund': 0.20
        }
    }
}

# Initialize random seeds
np.random.seed(MODEL_CONFIG['random_seed'])
random.seed(MODEL_CONFIG['random_seed'])

# ID generator
def id_sequence():
    for n in itertools.count(1):
        for s in itertools.product(string.ascii_uppercase, repeat=n):
            yield "".join(s)
id_iter = id_sequence()

# ------------------------------------------------------------------#
# CORE LIFECYCLE FUNCTIONS                                          #
# ------------------------------------------------------------------#

def get_age(year, starting_age=None):
    """Calculate age for any year in the plan"""
    return (starting_age or MODEL_CONFIG['lifecycle']['starting_age']) + year

def interpolate_risk_aversion(age):
    """Interpolate risk aversion for any age"""
    risk_table = MODEL_CONFIG['age_risk_aversion']
    ages = sorted(risk_table.keys())
    
    if age <= ages[0]: return risk_table[ages[0]]
    if age >= ages[-1]: return risk_table[ages[-1]]
    
    lower = max(a for a in ages if a <= age)
    upper = min(a for a in ages if a >= age)
    if lower == upper: return risk_table[lower]
    
    weight = (age - lower) / (upper - lower)
    return risk_table[lower] + weight * (risk_table[upper] - risk_table[lower])

def get_lifecycle_stage(age):
    """Determine lifecycle stage"""
    if age < 35: return 'young_family'
    elif age < 50: return 'mid_career'
    elif age < MODEL_CONFIG['lifecycle']['retirement_age']: return 'pre_retirement'
    elif age < 75: return 'retirement'
    else: return 'legacy'

def calculate_allocation(age):
    """Calculate asset allocation using 100-minus-age rule with constraints"""
    if not MODEL_CONFIG['lifecycle']['use_100_minus_age_rule']:
        return None
    
    equity = max(MODEL_CONFIG['lifecycle']['min_equity'], 
                min(MODEL_CONFIG['lifecycle']['max_equity'], (100 - age) / 100))
    
    # Cash increases after age 55
    cash = MODEL_CONFIG['lifecycle']['starting_age'] / 1000 + (max(0, age - 55) * 0.001)
    cash = max(0.05, min(0.15, cash))
    
    bonds = 1.0 - equity - cash
    total = equity + bonds + cash
    
    return {'equity': equity/total, 'bonds': bonds/total, 'cash': cash/total}

def get_dynamic_profile(year, base_risk_band, starting_age=None):
    """Calculate complete dynamic profile for a given year"""
    age = get_age(year, starting_age)
    risk_aversion = interpolate_risk_aversion(age)
    lifecycle_stage = get_lifecycle_stage(age)
    allocation = calculate_allocation(age)
    
    # Risk capacity factors by stage
    capacity_factors = {'young_family': 1.2, 'mid_career': 1.0, 'pre_retirement': 0.7, 'retirement': 0.4, 'legacy': 0.3}
    base_tolerance = {1: 0.2, 2: 0.5, 3: 0.8}[base_risk_band]
    adjusted_tolerance = base_tolerance * (1 - risk_aversion) * capacity_factors[lifecycle_stage]
    
    dynamic_risk_band = 1 if adjusted_tolerance <= 0.3 else (2 if adjusted_tolerance <= 0.6 else 3)
    
    return {
        'age': age, 'lifecycle_stage': lifecycle_stage, 'risk_aversion': risk_aversion,
        'dynamic_risk_band': dynamic_risk_band, 'allocation': allocation,
        'risk_tolerance': adjusted_tolerance
    }

# ------------------------------------------------------------------#
# CASH FLOW CALCULATIONS                                            #
# ------------------------------------------------------------------#

def mortgage_pmt(rate, years, pv):
    """Calculate annual mortgage payment"""
    return -npf.pmt(rate/12, years*12, pv) * 12

def calculate_cashflow(year, cfg, scenario=None, starting_age=None):
    """Calculate cash flows with age adjustments and optional stochastic shocks"""
    age = get_age(year, starting_age)
    lc = MODEL_CONFIG['lifecycle']
    
    # Base salaries with age adjustments
    hel_salary = PARAM["SALARY_HEL_FT"] if cfg["HEL_WORK"] == "Full‑time" else PARAM["SALARY_HEL_PT"]
    
    # Income evolution with age
    if age >= lc['income_decline_age']:
        decline_factor = max(0.5, 1 - (age - lc['income_decline_age']) * 0.02)
        h_salary = PARAM["SALARY_H"] * decline_factor
        hel_salary *= decline_factor
        bonus_factor = decline_factor * 0.8
    elif age <= lc['income_peak_age']:
        growth_factor = min(1.3, 1 + (lc['income_peak_age'] - age) * 0.006)
        h_salary = PARAM["SALARY_H"] * growth_factor
        hel_salary *= growth_factor
        bonus_factor = 1.0
    else:
        h_salary = PARAM["SALARY_H"]
        bonus_factor = 1.0
    
    # Core cash flows
    cashflow = {
        'Year': year, 'Age': age,
        'H_SAL': h_salary,
        'H_BONUS': h_salary * cfg["BONUS_PCT"] * bonus_factor,
        'HEL_SAL': hel_salary,
        'Daycare': -PARAM["DAYCARE"] if cfg["HEL_WORK"] == "Full‑time" and year < 5 else 0,
        'Tuition': -(PARAM["TUITION_JHU_USD"] * FX_SCENARIOS[cfg["FX_SCENARIO"]] 
                    if cfg["ED_PATH"] == "JohnsHopkins" else PARAM["TUITION_MCG"]) if 5 <= year < 10 else 0,
        'Mortgage': -mortgage_pmt(PARAM["MORT_RATE"], PARAM["MORT_TERM_YRS"], PARAM["HOUSE_PRICE"] * 0.75) 
                   if age < lc['mortgage_completion_age'] else 0,
        'Ppty': -PARAM["PPTY_COST"],
        'Healthcare': -(5000 + (age - lc['healthcare_cost_age']) * 500) if age >= lc['healthcare_cost_age'] else 0
    }
    
    # Charity based on style
    if cfg["DON_STYLE"] == 0:  # Regular
        cashflow['Charity'] = -PARAM["CHARITY_TARGET"]/10 if year < 10 else 0
    elif cfg["DON_STYLE"] == 1:  # Lump later
        cashflow['Charity'] = -PARAM["CHARITY_TARGET"] if year == 10 else 0
    else:  # Lump now
        cashflow['Charity'] = -PARAM["CHARITY_TARGET"] if year == 0 else 0
    
    # Apply stochastic shocks if scenario provided
    if scenario:
        shocks = MODEL_CONFIG['economic_shocks']
        income_shock = scenario.get('income_volatility', [1])[min(year, len(scenario.get('income_volatility', [1]))-1)]
        expense_shock = scenario.get('expense_shocks', [1])[min(year, len(scenario.get('expense_shocks', [1]))-1)]
        
        if scenario.get('recession_years', [0])[min(year, len(scenario.get('recession_years', [0]))-1)]:
            income_shock *= (1 - shocks['recession_income_drop'])
            expense_shock *= (1 + shocks['recession_expense_increase'])
        
        # Apply shocks
        for key in ['H_SAL', 'H_BONUS', 'HEL_SAL']:
            cashflow[key] *= income_shock
        for key in ['Daycare', 'Tuition', 'Ppty']:
            if cashflow[key] < 0:
                cashflow[key] *= expense_shock
    
    return cashflow

# ------------------------------------------------------------------#
# MONTE CARLO AND ANALYSIS                                          #
# ------------------------------------------------------------------#

def generate_scenarios(iterations=None):
    """Generate economic scenarios for Monte Carlo analysis"""
    iterations = iterations or MODEL_CONFIG['mc_iterations']
    eq_ret, bond_ret = MODEL_CONFIG['equity_return'], MODEL_CONFIG['bond_return']
    shocks = MODEL_CONFIG['economic_shocks']
    
    scenarios = []
    for i in range(iterations):
        scenarios.append({
            'equity_returns': np.random.normal(eq_ret['mean'], eq_ret['std'], YEARS),
            'bond_returns': np.random.normal(bond_ret['mean'], bond_ret['std'], YEARS),
            'recession_years': np.random.binomial(1, shocks['recession_prob'], YEARS),
            'income_volatility': np.random.normal(1.0, shocks['income_volatility'], YEARS),
            'expense_shocks': np.random.normal(1.0, shocks['expense_volatility'], YEARS)
        })
    return scenarios

def calculate_stress_metrics(cashflows_mc):
    """Calculate financial stress metrics from Monte Carlo results"""
    net_cfs = np.array([[sum(cf.values()) - cf['Year'] - cf['Age'] for cf in iteration] for iteration in cashflows_mc])
    
    shortfall_prob = np.mean(net_cfs < 0, axis=0)
    cf_volatility = np.mean([np.std(cf) / (abs(np.mean(cf)) + 1e-6) for cf in net_cfs.T])
    insolvency_prob = np.mean(np.cumsum(net_cfs, axis=1)[:, min(9, net_cfs.shape[1]-1)] < 0)
    
    return {
        'prob_shortfall_10yr': np.mean(shortfall_prob[:10]),
        'cashflow_volatility_10yr': cf_volatility,
        'prob_insolvency_10yr': insolvency_prob,
        'var_95_10yr': np.mean(np.percentile(net_cfs[:, :10], 5, axis=0)),
        'min_cushion_median': np.median(np.min(net_cfs, axis=1))
    }

def calculate_qol_score(cfg, stress_metrics):
    """Calculate Quality of Life score"""
    weights = MODEL_CONFIG['weights']['qol']
    
    components = {
        'financial_security': max(0, 1 - stress_metrics['prob_shortfall_10yr']),
        'income_stability': max(0, 1 - stress_metrics['cashflow_volatility_10yr']),
        'lifestyle_quality': 0.7 if cfg["HEL_WORK"] == "Full‑time" else 0.5,
        'generosity_fulfillment': 0.8 if cfg["DON_STYLE"] == 0 else 0.6,
        'cushion_comfort': max(0.1, min(1.0, stress_metrics['min_cushion_median'] / 50000))
    }
    
    return sum(components[k] * weights[k] for k in components), components

def is_feasible(cfg):
    """Check if configuration is feasible"""
    return not (cfg["DON_STYLE"] == 2 and cfg["MOM_GIFT_USE"] == 1)

# ------------------------------------------------------------------#
# MAIN ANALYSIS PIPELINE                                            #
# ------------------------------------------------------------------#

def analyze_configuration(cfg, scenarios):
    """Complete analysis pipeline for a single configuration"""
    cfg_id = f"CFG_{next(id_iter)}"
    cfg['cfg_id'] = cfg_id
    
    # Generate deterministic and stochastic cash flows
    cf_deterministic = []
    cf_monte_carlo = []
    
    for year in range(YEARS):
        # Deterministic cash flow with lifecycle
        cf_det = calculate_cashflow(year, cfg)
        profile = get_dynamic_profile(year, cfg['RISK_BAND'])
        
        cf_det.update({
            'Lifecycle_Stage': profile['lifecycle_stage'],
            'Risk_Aversion': profile['risk_aversion'],
            **{f"{asset.title()}_Allocation": pct for asset, pct in profile['allocation'].items()}
        })
        cf_deterministic.append(cf_det)
        
        # Monte Carlo scenarios
        if not cf_monte_carlo:
            cf_monte_carlo = [[] for _ in range(len(scenarios))]
        
        for i, scenario in enumerate(scenarios):
            cf_stoch = calculate_cashflow(year, cfg, scenario)
            cf_monte_carlo[i].append(cf_stoch)
    
    # Convert to DataFrame
    cf_df = pd.DataFrame(cf_deterministic)
    cf_df['NetCF'] = cf_df.select_dtypes(include=[np.number]).sum(axis=1) - cf_df['Year'] - cf_df['Age']
    
    # Calculate metrics
    stress_metrics = calculate_stress_metrics(cf_monte_carlo)
    qol_score, qol_components = calculate_qol_score(cfg, stress_metrics)
    
    # Compile results
    weights = MODEL_CONFIG['weights']['stress']
    stress_rank = (stress_metrics['prob_shortfall_10yr'] * weights['shortfall'] + 
                  stress_metrics['cashflow_volatility_10yr'] * weights['volatility'] + 
                  stress_metrics['prob_insolvency_10yr'] * weights['insolvency'])
    
    result = {
        **cfg,
        'PV_10yr_Surplus': cf_df.loc[:9, 'NetCF'].sum(),
        'Mean_Annual_CF': cf_df['NetCF'].mean(),
        'QoL_Score': qol_score,
        'Financial_Stress_Rank': stress_rank,
        'Avg_Equity_Allocation': cf_df['Equity_Allocation'].mean(),
        'Risk_Evolution_Range': cf_df['Risk_Aversion'].max() - cf_df['Risk_Aversion'].min(),
        **stress_metrics,
        **{f"QoL_{k}": v for k, v in qol_components.items()}
    }
    
    return result, cf_df

def main():
    """Main analysis pipeline"""
    print("🎲 Initializing IPS Analysis with Age-Based Lifecycle Modeling...")
    print(f"   📊 {MODEL_CONFIG['mc_iterations']} Monte Carlo iterations")
    print(f"   👥 Age range: {MODEL_CONFIG['lifecycle']['starting_age']}-{MODEL_CONFIG['lifecycle']['life_expectancy']}")
    
    # Setup
    scenarios = generate_scenarios()
    out_dir = Path.cwd() / "ips_output"
    out_dir.mkdir(exist_ok=True)
    
    results = []
    lifecycle_data = []
    
    # Process all configurations
    for combo in itertools.product(*FACTOR_SPACE.values()):
        cfg = dict(zip(FACTOR_SPACE.keys(), combo))
        cfg['BONUS_PCT'] = float(cfg['BONUS_PCT'])
        
        if not is_feasible(cfg):
            continue
        
        print(f"🔄 Processing {cfg.get('cfg_id', 'CFG_?')}...")
        
        result, cf_df = analyze_configuration(cfg, scenarios)
        results.append(result)
        
        # Save individual files
        cf_df.to_csv(out_dir / f"cashflows_{result['cfg_id']}.csv", index=False)
        
        # Collect lifecycle data
        lifecycle_df = cf_df[['Year', 'Age', 'Lifecycle_Stage', 'Risk_Aversion', 
                             'Equity_Allocation', 'Bonds_Allocation', 'Cash_Allocation']].copy()
        lifecycle_df['cfg_id'] = result['cfg_id']
        lifecycle_data.append(lifecycle_df)
    
    # Save consolidated results
    results_df = pd.DataFrame(results).sort_values('Financial_Stress_Rank', ascending=False)
    results_df.to_csv(out_dir / "configurations_enhanced.csv", index=False)
    
    if lifecycle_data:
        pd.concat(lifecycle_data, ignore_index=True).to_csv(out_dir / "comprehensive_lifecycle_analysis.csv", index=False)
    
    # Summary report
    print(f"\n🎉 Analysis Complete! {len(results)} configurations processed")
    print(f"   📊 Average equity allocation: {results_df['Avg_Equity_Allocation'].mean():.1%}")
    print(f"   🎯 High stress configs (>20% shortfall): {len(results_df[results_df['prob_shortfall_10yr'] > 0.2])}")
    print(f"   📁 Results saved to: {out_dir}")
    
    # Show top stress configurations
    print(f"\n🚨 Top 3 Highest Stress Configurations:")
    for _, row in results_df.head(3).iterrows():
        print(f"   {row['cfg_id']}: {row['ED_PATH']}, {row['HEL_WORK']}, "
              f"{row['BONUS_PCT']:.0%} bonus → {row['Financial_Stress_Rank']:.1%} stress")

# --- Compatibility Stubs for Demos ---

def cashflow_row(year, cfg):
    """Alias for calculate_cashflow for compatibility with demos."""
    return calculate_cashflow(year, cfg)

class StressMonitor:
    """Stub for compatibility. Add real logic as needed."""
    def __init__(self, config, baseline_metrics):
        self.config = config
        self.baseline_metrics = baseline_metrics

    def update_stress(self, current_metrics):
        # Return a dummy response for now
        return {
            'stress_level': current_metrics.get('Financial_Stress_Rank', 0.2),
            'stress_change': current_metrics.get('Financial_Stress_Rank', 0.2) - self.baseline_metrics.get('Financial_Stress_Rank', 0.2),
            'alerts': [],
            'recommendations': [],
            'auto_interventions': []
        }

def simulate_intervention_impact(*args, **kwargs):
    """Stub for compatibility. Add real logic as needed."""
    return {}

if __name__ == "__main__":
    main()
