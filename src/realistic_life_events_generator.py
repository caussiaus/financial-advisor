#!/usr/bin/env python
"""
Realistic Life Events Generator
Author: ChatGPT 2025-07-16

Creates realistic life events anchored in time to simulate looking back
at portfolio performance over years. Fills in gaps where extraction is ambiguous.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class RealisticLifeEvent:
    """Represents a realistic life event with timing"""
    event_type: str
    description: str
    actual_date: datetime
    planned_date: datetime
    cash_flow_impact: str  # 'positive', 'negative', 'neutral'
    impact_amount: float
    confidence: float
    trigger_reason: str
    portfolio_adjustment: Dict[str, Any]
    life_stage: str

class RealisticLifeEventsGenerator:
    """Generate realistic life events for portfolio analysis"""
    
    def __init__(self, client_id: str, starting_year: int = 2020):
        self.client_id = client_id
        self.starting_year = starting_year
        self.current_year = 2025  # Assume we're looking back from 2025
        self.events = []
        
    def generate_complete_life_journey(self, ips_document_content: str = None) -> List[RealisticLifeEvent]:
        """Generate a complete life journey with realistic events"""
        
        # Define the journey phases
        events = []
        
        # Year 0 (2020): Starting family phase
        events.append(RealisticLifeEvent(
            event_type="family_planning",
            description="First child born - increased need for financial stability",
            actual_date=datetime(2020, 3, 15),
            planned_date=datetime(2020, 6, 1),
            cash_flow_impact="negative",
            impact_amount=-25000,  # Daycare costs
            confidence=1.0,
            trigger_reason="Family expansion",
            portfolio_adjustment={"risk_reduction": 0.15, "cash_increase": 0.10},
            life_stage="young_family"
        ))
        
        # Year 1 (2021): Career advancement
        events.append(RealisticLifeEvent(
            event_type="career_advancement",
            description="Promotion to senior role - increased income and bonus",
            actual_date=datetime(2021, 1, 15),
            planned_date=datetime(2021, 3, 1),
            cash_flow_impact="positive",
            impact_amount=30000,  # Salary increase + bonus
            confidence=0.95,
            trigger_reason="Strong performance review",
            portfolio_adjustment={"equity_increase": 0.10, "risk_band": 3},
            life_stage="career_growth"
        ))
        
        # Year 2 (2022): Market volatility response
        events.append(RealisticLifeEvent(
            event_type="portfolio_rebalancing",
            description="COVID market recovery - increased risk tolerance",
            actual_date=datetime(2022, 6, 1),
            planned_date=datetime(2022, 1, 1),
            cash_flow_impact="neutral",
            impact_amount=0,
            confidence=0.85,
            trigger_reason="Market conditions improved",
            portfolio_adjustment={"equity_increase": 0.20, "bonds_decrease": 0.15},
            life_stage="accumulation"
        ))
        
        # Year 3 (2023): Education planning begins
        events.append(RealisticLifeEvent(
            event_type="education_planning",
            description="Starting education fund - considering Johns Hopkins vs McGill",
            actual_date=datetime(2023, 9, 1),
            planned_date=datetime(2023, 6, 1),
            cash_flow_impact="negative",
            impact_amount=-15000,  # Annual education savings
            confidence=0.90,
            trigger_reason="Child approaching school age",
            portfolio_adjustment={"education_fund": 0.12, "risk_band": 2},
            life_stage="education_planning"
        ))
        
        # Year 4 (2024): Major decision point
        events.append(RealisticLifeEvent(
            event_type="education_decision",
            description="Chose McGill over Johns Hopkins - cost concerns",
            actual_date=datetime(2024, 2, 15),
            planned_date=datetime(2024, 6, 1),
            cash_flow_impact="positive",
            impact_amount=75000,  # Savings from choosing McGill
            confidence=1.0,
            trigger_reason="Financial stress analysis showed Johns Hopkins too risky",
            portfolio_adjustment={"stress_reduction": 0.25, "cash_freed": 75000},
            life_stage="education_commitment"
        ))
        
        # Year 5 (2025): Current year - work-life balance
        events.append(RealisticLifeEvent(
            event_type="work_arrangement",
            description="Switched to part-time work for better work-life balance",
            actual_date=datetime(2025, 1, 1),
            planned_date=datetime(2025, 3, 1),
            cash_flow_impact="negative",
            impact_amount=-40000,  # Income reduction
            confidence=0.95,
            trigger_reason="Quality of life priority after portfolio stress analysis",
            portfolio_adjustment={"conservative_shift": 0.20, "risk_band": 1},
            life_stage="lifestyle_optimization"
        ))
        
        return events
    
    def generate_portfolio_performance_timeline(self, events: List[RealisticLifeEvent]) -> pd.DataFrame:
        """Generate portfolio performance timeline showing how decisions played out"""
        
        timeline_data = []
        
        # Starting portfolio (2020)
        portfolio_value = 500000
        equity_allocation = 0.60
        bonds_allocation = 0.35
        cash_allocation = 0.05
        annual_return = 0.08  # Base return assumption
        
        for year in range(2020, 2026):
            # Find events for this year
            year_events = [e for e in events if e.actual_date.year == year]
            
            # Calculate market returns (simplified)
            if year == 2020:
                market_return = -0.05  # COVID impact
            elif year == 2021:
                market_return = 0.28   # Recovery
            elif year == 2022:
                market_return = -0.18  # Inflation concerns
            elif year == 2023:
                market_return = 0.24   # AI rally
            elif year == 2024:
                market_return = 0.18   # Steady growth
            else:  # 2025
                market_return = 0.12   # Current year
            
            # Apply portfolio adjustments from events
            for event in year_events:
                if "equity_increase" in event.portfolio_adjustment:
                    equity_allocation += event.portfolio_adjustment["equity_increase"]
                if "risk_reduction" in event.portfolio_adjustment:
                    equity_allocation -= event.portfolio_adjustment["risk_reduction"]
                if "conservative_shift" in event.portfolio_adjustment:
                    equity_allocation -= event.portfolio_adjustment["conservative_shift"]
                    bonds_allocation += event.portfolio_adjustment["conservative_shift"]
            
            # Normalize allocations
            total_allocation = equity_allocation + bonds_allocation + cash_allocation
            equity_allocation /= total_allocation
            bonds_allocation /= total_allocation  
            cash_allocation /= total_allocation
            
            # Calculate portfolio return
            portfolio_return = (equity_allocation * market_return + 
                              bonds_allocation * 0.04 + 
                              cash_allocation * 0.02)
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            
            # Add cash flows from events
            for event in year_events:
                portfolio_value += event.impact_amount
            
            timeline_data.append({
                'year': year,
                'portfolio_value': portfolio_value,
                'equity_allocation': equity_allocation,
                'bonds_allocation': bonds_allocation,
                'cash_allocation': cash_allocation,
                'annual_return': portfolio_return,
                'market_return': market_return,
                'major_events': [e.description for e in year_events],
                'stress_level': self._calculate_stress_level(year, events),
                'life_stage': year_events[0].life_stage if year_events else 'stable'
            })
        
        return pd.DataFrame(timeline_data)
    
    def _calculate_stress_level(self, year: int, events: List[RealisticLifeEvent]) -> float:
        """Calculate financial stress level for a given year"""
        base_stress = 0.15  # Baseline stress
        
        year_events = [e for e in events if e.actual_date.year == year]
        
        for event in year_events:
            if event.cash_flow_impact == "negative":
                base_stress += 0.10
            elif event.cash_flow_impact == "positive":
                base_stress -= 0.05
                
        return min(0.80, max(0.05, base_stress))
    
    def generate_analysis_report(self, events: List[RealisticLifeEvent], 
                               timeline: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        return {
            'analysis_period': f"{timeline['year'].min()}-{timeline['year'].max()}",
            'total_events': len(events),
            'portfolio_performance': {
                'starting_value': timeline.iloc[0]['portfolio_value'],
                'ending_value': timeline.iloc[-1]['portfolio_value'],
                'total_return': (timeline.iloc[-1]['portfolio_value'] / timeline.iloc[0]['portfolio_value'] - 1),
                'average_annual_return': timeline['annual_return'].mean(),
                'volatility': timeline['annual_return'].std(),
                'best_year': timeline.loc[timeline['annual_return'].idxmax(), 'year'],
                'worst_year': timeline.loc[timeline['annual_return'].idxmin(), 'year']
            },
            'life_events_impact': {
                'positive_events': len([e for e in events if e.cash_flow_impact == "positive"]),
                'negative_events': len([e for e in events if e.cash_flow_impact == "negative"]),
                'total_cash_impact': sum([e.impact_amount for e in events]),
                'biggest_positive_impact': max([e.impact_amount for e in events if e.impact_amount > 0], default=0),
                'biggest_negative_impact': min([e.impact_amount for e in events if e.impact_amount < 0], default=0)
            },
            'stress_analysis': {
                'average_stress': timeline['stress_level'].mean(),
                'peak_stress_year': timeline.loc[timeline['stress_level'].idxmax(), 'year'],
                'peak_stress_level': timeline['stress_level'].max(),
                'stress_trend': 'improving' if timeline['stress_level'].iloc[-1] < timeline['stress_level'].iloc[0] else 'stable'
            },
            'portfolio_evolution': {
                'equity_trend': 'decreasing' if timeline['equity_allocation'].iloc[-1] < timeline['equity_allocation'].iloc[0] else 'stable',
                'final_allocation': {
                    'equity': timeline.iloc[-1]['equity_allocation'],
                    'bonds': timeline.iloc[-1]['bonds_allocation'], 
                    'cash': timeline.iloc[-1]['cash_allocation']
                }
            },
            'key_decisions': [
                {
                    'year': event.actual_date.year,
                    'decision': event.description,
                    'impact': event.impact_amount,
                    'rationale': event.trigger_reason
                }
                for event in events if abs(event.impact_amount) > 20000
            ]
        }

def main():
    """Generate realistic life events and analysis"""
    
    print("🎯 GENERATING REALISTIC LIFE EVENTS & PORTFOLIO ANALYSIS")
    print("=" * 65)
    
    # Create generator
    generator = RealisticLifeEventsGenerator("CLIENT_Case_1_IPS_Individual")
    
    # Generate life journey
    events = generator.generate_complete_life_journey()
    
    print(f"📅 Generated {len(events)} realistic life events:")
    for i, event in enumerate(events, 1):
        print(f"   {i}. {event.actual_date.year}: {event.description}")
        print(f"      💰 Impact: ${event.impact_amount:,} ({event.cash_flow_impact})")
        print(f"      📊 Portfolio: {event.portfolio_adjustment}")
        print()
    
    # Generate portfolio timeline
    timeline = generator.generate_portfolio_performance_timeline(events)
    
    print("📈 PORTFOLIO PERFORMANCE TIMELINE:")
    print(timeline[['year', 'portfolio_value', 'annual_return', 'equity_allocation', 'stress_level', 'life_stage']].round(3))
    
    # Generate analysis report
    report = generator.generate_analysis_report(events, timeline)
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   💼 Portfolio Growth: ${report['portfolio_performance']['starting_value']:,.0f} → ${report['portfolio_performance']['ending_value']:,.0f}")
    print(f"   📈 Total Return: {report['portfolio_performance']['total_return']:.2%}")
    print(f"   🎯 Average Annual Return: {report['portfolio_performance']['average_annual_return']:.2%}")
    print(f"   ⚡ Volatility: {report['portfolio_performance']['volatility']:.2%}")
    print(f"   💰 Net Cash Flow Impact: ${report['life_events_impact']['total_cash_impact']:,}")
    print(f"   📉 Average Stress Level: {report['stress_analysis']['average_stress']:.2%}")
    print(f"   🔄 Stress Trend: {report['stress_analysis']['stress_trend']}")
    
    print(f"\n🎯 KEY DECISIONS THAT SHAPED THE JOURNEY:")
    for decision in report['key_decisions']:
        print(f"   {decision['year']}: {decision['decision']}")
        print(f"      Impact: ${decision['impact']:,} - {decision['rationale']}")
    
    # Save results
    timeline.to_csv('realistic_portfolio_timeline.csv', index=False)
    
    with open('realistic_life_events_analysis.json', 'w') as f:
        json.dump({
            'events': [
                {
                    'event_type': e.event_type,
                    'description': e.description,
                    'actual_date': e.actual_date.isoformat(),
                    'planned_date': e.planned_date.isoformat(),
                    'cash_flow_impact': e.cash_flow_impact,
                    'impact_amount': e.impact_amount,
                    'confidence': e.confidence,
                    'trigger_reason': e.trigger_reason,
                    'portfolio_adjustment': e.portfolio_adjustment,
                    'life_stage': e.life_stage
                }
                for e in events
            ],
            'analysis_report': report
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to:")
    print(f"   📊 realistic_portfolio_timeline.csv")
    print(f"   📋 realistic_life_events_analysis.json")

if __name__ == "__main__":
    main() 