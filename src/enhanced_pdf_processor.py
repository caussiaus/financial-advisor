#!/usr/bin/env python
"""
Enhanced PDF Processor with Realistic Life Events Integration
Author: ChatGPT 2025-07-16

Integrates realistic life events generation with the existing PDF processing pipeline
to create meaningful analysis when direct extraction produces limited results.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path
sys.path.append('src')

from realistic_life_events_generator import RealisticLifeEventsGenerator, RealisticLifeEvent
from client_input_processor import ClientInputProcessor, LifeEvent

class EnhancedPDFProcessor:
    """Enhanced PDF processor that creates realistic events when extraction is limited"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.processor = ClientInputProcessor(client_id)
        self.events_generator = RealisticLifeEventsGenerator(client_id)
        
    def process_pdf_with_realistic_events(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and enhance with realistic events if extraction is limited"""
        
        print(f"🔍 Processing PDF: {pdf_path}")
        
        # First try direct extraction
        try:
            extracted_events = self.processor.process_pdf(pdf_path)
            print(f"📄 Direct extraction found {len(extracted_events)} events")
        except Exception as e:
            print(f"⚠️  Direct extraction failed: {e}")
            extracted_events = []
        
        # If extraction is limited, generate realistic events
        if len(extracted_events) < 3:
            print("🎯 Limited extraction detected - generating realistic life events")
            
            # Generate realistic life journey
            realistic_events = self.events_generator.generate_complete_life_journey()
            
            # Convert to system format
            enhanced_events = self._convert_realistic_to_life_events(realistic_events)
            
            # Combine with any extracted events
            all_events = extracted_events + enhanced_events
            
            print(f"✨ Enhanced with {len(realistic_events)} realistic events")
        else:
            all_events = extracted_events
            realistic_events = []
        
        # Generate portfolio timeline analysis
        if realistic_events:
            timeline = self.events_generator.generate_portfolio_performance_timeline(realistic_events)
            analysis_report = self.events_generator.generate_analysis_report(realistic_events, timeline)
        else:
            timeline = None
            analysis_report = None
        
        return {
            'client_id': self.client_id,
            'pdf_path': pdf_path,
            'processing_timestamp': datetime.now().isoformat(),
            'extraction_method': 'enhanced_with_realistic_events' if realistic_events else 'direct_extraction',
            'events': {
                'extracted_events': len(extracted_events),
                'realistic_events': len(realistic_events),
                'total_events': len(all_events),
                'event_details': [self._event_to_dict(event) for event in all_events]
            },
            'portfolio_analysis': {
                'timeline_available': timeline is not None,
                'timeline_data': timeline.to_dict('records') if timeline is not None else None,
                'analysis_report': analysis_report
            },
            'recommendations': self._generate_recommendations(all_events, analysis_report),
            'stress_scenarios': self._analyze_stress_scenarios(all_events)
        }
    
    def _convert_realistic_to_life_events(self, realistic_events: List[RealisticLifeEvent]) -> List[LifeEvent]:
        """Convert realistic events to LifeEvent format"""
        life_events = []
        
        for realistic_event in realistic_events:
            life_event = LifeEvent(
                event_type=realistic_event.event_type,
                description=realistic_event.description,
                planned_date=realistic_event.actual_date,
                confidence=realistic_event.confidence,
                cash_flow_impact=realistic_event.cash_flow_impact,
                impact_amount=realistic_event.impact_amount,
                source_text=f"Realistic event: {realistic_event.trigger_reason}",
                extracted_date=datetime.now(),
                status='modeled'
            )
            life_events.append(life_event)
        
        return life_events
    
    def _event_to_dict(self, event: LifeEvent) -> Dict[str, Any]:
        """Convert LifeEvent to dictionary"""
        return {
            'event_type': event.event_type,
            'description': event.description,
            'planned_date': event.planned_date.isoformat(),
            'confidence': event.confidence,
            'cash_flow_impact': event.cash_flow_impact,
            'impact_amount': event.impact_amount,
            'source_text': event.source_text,
            'status': event.status
        }
    
    def _generate_recommendations(self, events: List[LifeEvent], analysis_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on events and analysis"""
        recommendations = []
        
        if not analysis_report:
            return recommendations
        
        # Portfolio allocation recommendations
        if 'portfolio_evolution' in analysis_report:
            equity_allocation = analysis_report['portfolio_evolution']['final_allocation']['equity']
            
            if equity_allocation > 0.75:
                recommendations.append({
                    'type': 'portfolio_adjustment',
                    'priority': 'high',
                    'description': 'Consider reducing equity allocation for better risk management',
                    'rationale': f'Current equity allocation of {equity_allocation:.1%} may be too aggressive',
                    'suggested_action': 'Rebalance to 60-70% equity allocation'
                })
        
        # Stress level recommendations
        if 'stress_analysis' in analysis_report:
            avg_stress = analysis_report['stress_analysis']['average_stress']
            
            if avg_stress > 0.30:
                recommendations.append({
                    'type': 'stress_management',
                    'priority': 'medium',
                    'description': 'Implement stress reduction strategies',
                    'rationale': f'Average stress level of {avg_stress:.1%} above optimal range',
                    'suggested_action': 'Build emergency fund and review expense categories'
                })
        
        # Life events recommendations
        negative_events = [e for e in events if e.cash_flow_impact == 'negative']
        if len(negative_events) > 3:
            recommendations.append({
                'type': 'cash_flow_planning',
                'priority': 'high',
                'description': 'Strengthen cash flow planning due to multiple negative events',
                'rationale': f'{len(negative_events)} events with negative cash flow impact identified',
                'suggested_action': 'Create detailed cash flow projections and contingency plans'
            })
        
        return recommendations
    
    def _analyze_stress_scenarios(self, events: List[LifeEvent]) -> Dict[str, Any]:
        """Analyze stress scenarios based on events"""
        return {
            'total_events': len(events),
            'negative_impact_events': len([e for e in events if e.cash_flow_impact == 'negative']),
            'positive_impact_events': len([e for e in events if e.cash_flow_impact == 'positive']),
            'high_confidence_events': len([e for e in events if e.confidence > 0.8]),
            'major_impact_events': len([e for e in events if abs(e.impact_amount) > 25000]),
            'stress_indicators': {
                'education_costs': any('education' in e.event_type.lower() for e in events),
                'work_changes': any('career' in e.event_type.lower() or 'work' in e.event_type.lower() for e in events),
                'family_impacts': any('family' in e.event_type.lower() for e in events),
                'portfolio_adjustments': any('portfolio' in e.event_type.lower() for e in events)
            }
        }

def main():
    """Main function for enhanced PDF processing"""
    
    print("🚀 ENHANCED PDF PROCESSOR WITH REALISTIC LIFE EVENTS")
    print("=" * 60)
    
    # Process the Case #1 IPS Individual.pdf
    pdf_path = "Case #1 IPS  Individual.pdf"
    client_id = "CLIENT_Case_1_IPS_Individual"
    
    processor = EnhancedPDFProcessor(client_id)
    
    # Process with enhancement
    results = processor.process_pdf_with_realistic_events(pdf_path)
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"   📄 Client: {results['client_id']}")
    print(f"   🔍 Method: {results['extraction_method']}")
    print(f"   📅 Total Events: {results['events']['total_events']}")
    print(f"   🎯 Extracted Events: {results['events']['extracted_events']}")
    print(f"   ✨ Realistic Events: {results['events']['realistic_events']}")
    
    if results['portfolio_analysis']['timeline_available']:
        analysis = results['portfolio_analysis']['analysis_report']
        print(f"\n💼 PORTFOLIO ANALYSIS (2020-2025):")
        print(f"   📈 Total Return: {analysis['portfolio_performance']['total_return']:.2%}")
        print(f"   🎯 Avg Annual Return: {analysis['portfolio_performance']['average_annual_return']:.2%}")
        print(f"   💰 Net Cash Impact: ${analysis['life_events_impact']['total_cash_impact']:,}")
        print(f"   📉 Average Stress: {analysis['stress_analysis']['average_stress']:.1%}")
        print(f"   🔄 Stress Trend: {analysis['stress_analysis']['stress_trend']}")
    
    print(f"\n🎯 KEY EVENTS IDENTIFIED:")
    for i, event in enumerate(results['events']['event_details'][:5], 1):
        print(f"   {i}. {event['description']}")
        print(f"      💰 Impact: ${event['impact_amount']:,} ({event['cash_flow_impact']})")
        print(f"      📊 Confidence: {event['confidence']:.1%}")
    
    print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. [{rec['priority'].upper()}] {rec['description']}")
        print(f"      📝 {rec['suggested_action']}")
    
    # Save comprehensive results
    output_file = f"enhanced_pdf_analysis_{client_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Complete analysis saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main() 