#!/usr/bin/env python3
"""
Financial Fingerprint Color Mapping System
Maps individual financial characteristics to color wheel dimensions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Any
import colorsys

class FinancialFingerprintMapper:
    """Maps financial characteristics to color wheel dimensions"""
    
    def __init__(self):
        # Define color mappings for different financial aspects
        self.aspect_mappings = {
            'risk_tolerance': {
                'hue': (0, 0.3),  # Green to Blue (conservative to aggressive)
                'description': 'Risk tolerance affects hue'
            },
            'income_stability': {
                'saturation': (0.3, 1.0),  # Low to high saturation
                'description': 'Income stability affects saturation'
            },
            'debt_ratio': {
                'value': (0.7, 1.0),  # Brightness
                'description': 'Debt ratio affects brightness'
            },
            'savings_rate': {
                'hue_offset': (0, 0.2),  # Additional hue variation
                'description': 'Savings rate adds hue variation'
            },
            'investment_aggressiveness': {
                'saturation_boost': (0, 0.3),  # Extra saturation
                'description': 'Investment style affects saturation'
            }
        }
    
    def extract_financial_characteristics(self, person_data: Dict) -> Dict[str, float]:
        """Extract key financial characteristics from person data"""
        characteristics = {}
        
        # Calculate total assets and liabilities
        total_assets = 0
        total_liabilities = 0
        annual_income = 0
        monthly_savings = 0
        
        if 'financial_state' in person_data:
            fin_state = person_data['financial_state']
            
            # Calculate total assets
            if 'assets' in fin_state:
                assets = fin_state['assets']
                for asset_type, value in assets.items():
                    if isinstance(value, (int, float)):
                        total_assets += value
            
            # Calculate total liabilities
            if 'liabilities' in fin_state:
                liabilities = fin_state['liabilities']
                for liability_type, value in liabilities.items():
                    if isinstance(value, (int, float)):
                        total_liabilities += value
            
            # Calculate annual income
            if 'income' in fin_state:
                income = fin_state['income']
                for income_type, value in income.items():
                    if isinstance(value, (int, float)):
                        annual_income += value
            
            # Calculate monthly savings
            if 'expenses' in fin_state:
                expenses = fin_state['expenses']
                if 'monthly_savings' in expenses:
                    monthly_savings = expenses['monthly_savings']
        
        # 1. Risk tolerance based on investment allocation
        risk_tolerance = 0.5  # Default moderate
        if total_assets > 0:
            # Higher proportion of investments vs cash suggests higher risk tolerance
            if 'assets' in person_data.get('financial_state', {}):
                assets = person_data['financial_state']['assets']
                investments = assets.get('investments', 0)
                cash = assets.get('cash', 0)
                if investments + cash > 0:
                    investment_ratio = investments / (investments + cash)
                    risk_tolerance = 0.3 + (investment_ratio * 0.4)  # 0.3 to 0.7 range
        
        characteristics['risk_tolerance'] = risk_tolerance
        
        # 2. Income stability based on income diversity and amount
        income_stability = 0.5
        if annual_income > 0:
            # Higher income generally means more stability
            income_stability = min(1.0, annual_income / 150000)  # Cap at 150k
        
        characteristics['income_stability'] = income_stability
        
        # 3. Debt ratio
        debt_ratio = 0.3
        if total_assets > 0:
            debt_ratio = min(1.0, total_liabilities / total_assets)
        
        characteristics['debt_ratio'] = debt_ratio
        
        # 4. Savings rate
        savings_rate = 0.2
        if annual_income > 0:
            annual_savings = monthly_savings * 12
            savings_rate = min(1.0, annual_savings / annual_income)
        
        characteristics['savings_rate'] = savings_rate
        
        # 5. Investment aggressiveness based on asset allocation
        investment_aggressiveness = 0.5
        if total_assets > 0:
            if 'assets' in person_data.get('financial_state', {}):
                assets = person_data['financial_state']['assets']
                investments = assets.get('investments', 0)
                real_estate = assets.get('real_estate', 0)
                retirement = assets.get('retirement_accounts', 0)
                
                # Higher proportion of investments vs real estate suggests more aggressive
                liquid_assets = investments + retirement
                illiquid_assets = real_estate
                
                if liquid_assets + illiquid_assets > 0:
                    liquid_ratio = liquid_assets / (liquid_assets + illiquid_assets)
                    investment_aggressiveness = 0.3 + (liquid_ratio * 0.4)
        
        characteristics['investment_aggressiveness'] = investment_aggressiveness
        
        return characteristics
    
    def map_to_color_wheel(self, characteristics: Dict[str, float]) -> Tuple[float, float, float]:
        """Map financial characteristics to HSV color space"""
        
        # Base hue from risk tolerance
        base_hue = characteristics['risk_tolerance']
        
        # Add savings rate variation
        hue_offset = characteristics['savings_rate'] * 0.2
        final_hue = (base_hue + hue_offset) % 1.0
        
        # Saturation from income stability
        base_saturation = 0.3 + (characteristics['income_stability'] * 0.4)
        
        # Add investment aggressiveness boost
        saturation_boost = characteristics['investment_aggressiveness'] * 0.3
        final_saturation = min(1.0, base_saturation + saturation_boost)
        
        # Value (brightness) from debt ratio (inverted)
        final_value = 0.7 + ((1.0 - characteristics['debt_ratio']) * 0.3)
        
        return (final_hue, final_saturation, final_value)
    
    def hsv_to_rgb(self, hsv: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        return colorsys.hsv_to_rgb(*hsv)
    
    def create_fingerprint_summary(self, person_name: str, characteristics: Dict[str, float], 
                                 color: Tuple[float, float, float]) -> Dict:
        """Create a summary of the financial fingerprint"""
        return {
            'person_name': person_name,
            'characteristics': characteristics,
            'color_hsv': color,
            'color_rgb': self.hsv_to_rgb(color),
            'fingerprint_description': self.describe_fingerprint(characteristics)
        }
    
    def describe_fingerprint(self, characteristics: Dict[str, float]) -> str:
        """Generate a human-readable description of the financial fingerprint"""
        risk_level = "Conservative" if characteristics['risk_tolerance'] < 0.4 else \
                    "Moderate" if characteristics['risk_tolerance'] < 0.7 else "Aggressive"
        
        stability = "Low" if characteristics['income_stability'] < 0.4 else \
                   "Moderate" if characteristics['income_stability'] < 0.7 else "High"
        
        debt_status = "High" if characteristics['debt_ratio'] > 0.6 else \
                     "Moderate" if characteristics['debt_ratio'] > 0.3 else "Low"
        
        savings_style = "Low" if characteristics['savings_rate'] < 0.2 else \
                       "Moderate" if characteristics['savings_rate'] < 0.4 else "High"
        
        investment_style = "Conservative" if characteristics['investment_aggressiveness'] < 0.4 else \
                          "Moderate" if characteristics['investment_aggressiveness'] < 0.7 else "Aggressive"
        
        return f"{risk_level} risk tolerance, {stability} income stability, {debt_status} debt ratio, {savings_style} savings rate, {investment_style} investment style"

def load_people_data() -> Dict[str, Dict]:
    """Load all people data from current and archived directories"""
    people_data = {}
    
    # Load current people
    current_dir = Path("data/inputs/people/current")
    if current_dir.exists():
        for person_dir in current_dir.iterdir():
            if person_dir.is_dir():
                person_name = f"current_{person_dir.name}"
                person_data = {}
                for json_file in person_dir.glob("*.json"):
                    data_type = json_file.stem
                    try:
                        with open(json_file, 'r') as f:
                            person_data[data_type] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                
                if person_data:
                    people_data[person_name] = person_data
    
    # Load archived people
    archived_dir = Path("data/inputs/people/archived")
    if archived_dir.exists():
        for person_dir in archived_dir.iterdir():
            if person_dir.is_dir():
                person_name = f"archived_{person_dir.name}"
                person_data = {}
                for json_file in person_dir.glob("*.json"):
                    data_type = json_file.stem
                    try:
                        with open(json_file, 'r') as f:
                            person_data[data_type] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                
                if person_data:
                    people_data[person_name] = person_data
    
    return people_data

def create_fingerprint_visualization(fingerprints: List[Dict]):
    """Create visualizations of the financial fingerprints"""
    
    # Create color wheel visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Financial Fingerprints: Color Wheel Mapping', fontsize=16, fontweight='bold')
    
    # 1. Color wheel with all fingerprints
    ax1 = axes[0, 0]
    ax1.set_aspect('equal')
    
    # Create color wheel
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 50)
    theta_grid, r_grid = np.meshgrid(theta, r)
    
    # Create HSV color wheel
    hsv_wheel = np.zeros((50, 100, 3))
    for i in range(50):
        for j in range(100):
            hsv_wheel[i, j] = [theta_grid[i, j]/(2*np.pi), r_grid[i, j], 1.0]
    
    rgb_wheel = hsv_to_rgb(hsv_wheel)
    ax1.imshow(rgb_wheel, extent=[0, 2*np.pi, 0, 1], aspect='auto')
    
    # Plot fingerprint points
    for fp in fingerprints:
        h, s, v = fp['color_hsv']
        angle = h * 2 * np.pi
        radius = s
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Convert HSV to RGB for point color
        point_color = colorsys.hsv_to_rgb(h, s, v)
        
        ax1.plot(x, y, 'o', markersize=10, color=point_color, 
                markeredgecolor='black', markeredgewidth=2)
        ax1.annotate(fp['person_name'], (x, y), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax1.set_title('Financial Fingerprints on Color Wheel')
    ax1.set_xlabel('Hue (Risk Tolerance)')
    ax1.set_ylabel('Saturation (Income Stability)')
    
    # 2. Characteristics radar chart
    ax2 = axes[0, 1]
    
    # Select a few representative fingerprints
    sample_fps = fingerprints[:min(5, len(fingerprints))]
    
    categories = ['Risk Tolerance', 'Income Stability', 'Debt Ratio', 'Savings Rate', 'Investment Aggressiveness']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for i, fp in enumerate(sample_fps):
        values = [
            fp['characteristics']['risk_tolerance'],
            fp['characteristics']['income_stability'],
            fp['characteristics']['debt_ratio'],
            fp['characteristics']['savings_rate'],
            fp['characteristics']['investment_aggressiveness']
        ]
        values += values[:1]  # Close the loop
        
        color = colorsys.hsv_to_rgb(*fp['color_hsv'])
        ax2.plot(angles, values, 'o-', linewidth=2, label=fp['person_name'], color=color)
        ax2.fill(angles, values, alpha=0.1, color=color)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Financial Characteristics Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Color distribution
    ax3 = axes[1, 0]
    
    # Extract RGB values
    rgb_values = [fp['color_rgb'] for fp in fingerprints]
    rgb_array = np.array(rgb_values)
    
    # Create scatter plot of RGB values
    ax3.scatter(rgb_array[:, 0], rgb_array[:, 1], c=rgb_values, s=100, alpha=0.7)
    ax3.set_xlabel('Red Component')
    ax3.set_ylabel('Green Component')
    ax3.set_title('RGB Color Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add person labels
    for i, fp in enumerate(fingerprints):
        ax3.annotate(fp['person_name'], (rgb_array[i, 0], rgb_array[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Fingerprint summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "Financial Fingerprint Summary:\n\n"
    for fp in fingerprints:
        summary_text += f"• {fp['person_name']}: {fp['fingerprint_description']}\n"
        summary_text += f"  Color: HSV{fp['color_hsv']} → RGB{fp['color_rgb']}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('financial_fingerprints_color_wheel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to create financial fingerprints"""
    print("Creating Financial Fingerprints...")
    
    # Load people data
    people_data = load_people_data()
    print(f"Loaded data for {len(people_data)} people")
    
    # Create fingerprint mapper
    mapper = FinancialFingerprintMapper()
    
    # Generate fingerprints for each person
    fingerprints = []
    
    for person_name, person_data in people_data.items():
        print(f"\nProcessing {person_name}...")
        
        # Extract characteristics
        characteristics = mapper.extract_financial_characteristics(person_data)
        print(f"  Characteristics: {characteristics}")
        
        # Map to color wheel
        color_hsv = mapper.map_to_color_wheel(characteristics)
        print(f"  Color HSV: {color_hsv}")
        
        # Create fingerprint summary
        fingerprint = mapper.create_fingerprint_summary(person_name, characteristics, color_hsv)
        fingerprints.append(fingerprint)
        
        print(f"  Description: {fingerprint['fingerprint_description']}")
    
    # Save fingerprints to JSON
    with open('financial_fingerprints.json', 'w') as f:
        json.dump(fingerprints, f, indent=2)
    
    print(f"\nSaved {len(fingerprints)} fingerprints to financial_fingerprints.json")
    
    # Create visualization
    if fingerprints:
        create_fingerprint_visualization(fingerprints)
        print("Created financial fingerprint visualization")
    
    # Create summary DataFrame
    df_data = []
    for fp in fingerprints:
        df_data.append({
            'Person': fp['person_name'],
            'Risk_Tolerance': fp['characteristics']['risk_tolerance'],
            'Income_Stability': fp['characteristics']['income_stability'],
            'Debt_Ratio': fp['characteristics']['debt_ratio'],
            'Savings_Rate': fp['characteristics']['savings_rate'],
            'Investment_Aggressiveness': fp['characteristics']['investment_aggressiveness'],
            'Hue': fp['color_hsv'][0],
            'Saturation': fp['color_hsv'][1],
            'Value': fp['color_hsv'][2],
            'RGB': str(fp['color_rgb'])
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv('financial_fingerprints_summary.csv', index=False)
    print("Saved summary to financial_fingerprints_summary.csv")
    
    return fingerprints

if __name__ == "__main__":
    fingerprints = main() 