# Integrated Life Planning System - Client Input Processing

## Overview

This system provides a complete end-to-end pipeline for processing client inputs (emails, PDFs) and generating optimal life decisions. The system uses AI to extract life events from natural language, models stress scenarios, tracks balances across different life paths, and provides real-time recommendations.

## Key Features

### 🔍 **AI-Powered Event Extraction**
- Processes client emails and PDFs using KOR libraries
- Extracts life events using AI (event type, date, amount, impact)
- Timestamps events based on natural language context
- Calculates confidence scores for extracted events

### 📊 **Scenario Modeling & Stress Analysis**
- Generates all possible life scenarios based on configuration
- Models stress scenarios and rules out impossible paths
- Tracks financial balances across different life paths
- Uses individual threads for parallel scenario calculations

### 💰 **Balance Tracking System**
- Maintains balance tracking for all accounts and events
- Tracks inflows and outflows across scenarios
- Recalculates balances when new events are added
- Provides real-time balance updates

### 🎯 **Optimal Decision Engine**
- Generates optimal decisions for each feasible scenario
- Provides confidence scores and stress levels
- Recommends specific actions based on analysis
- Tracks decision history over time

### ⚠️ **Real-Time Monitoring**
- Monitors upcoming events and generates alerts
- Provides recommendations for handling events
- Tracks decision history and configuration drift
- Offers proactive guidance

## System Architecture

```
Client Input → AI Extraction → Event Timestamping → Scenario Modeling → 
Stress Analysis → Balance Tracking → Optimal Decisions → Real-Time Monitoring
```

## Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Install Optional Dependencies**
```bash
# For PDF processing
pip install PyPDF2 pdfplumber

# For AI processing
pip install openai transformers torch spacy

# For email processing
pip install email-validator
```

## Quick Start

### Basic Usage

```python
from integrated_life_planner import IntegratedLifePlanner

# Initialize planner
planner = IntegratedLifePlanner("CLIENT_001")

# Process client email
email_content = """
Hi,
We've decided to send our child to Johns Hopkins University starting next year.
The tuition will be $110,000 per year. My wife is planning to work part-time
starting in 3 months to help with childcare costs.
Thanks,
Client
"""

# Get optimal decisions
result = planner.process_client_update(email_content, 'email')
print(f"Extracted {len(result['extracted_events'])} events")
print(f"Generated {len(result['optimal_decisions'])} optimal decisions")
```

### Advanced Usage

```python
from client_input_processor import ClientInputProcessor

# Initialize processor
processor = ClientInputProcessor("CLIENT_001")

# Process PDF document
events = processor.process_pdf("client_update.pdf")

# Process email
events = processor.process_email(email_content)

# Export configuration space
output_path = processor.export_configuration_space()
```

## Configuration

The system uses the existing IPS configuration (`config/ips_config.json`) and extends it with:

- **Event Types**: education, work, family, health, housing, financial, retirement, charity
- **Stress Thresholds**: Configurable stress levels for scenario ruling
- **Balance Tracking**: Account types and balance calculation methods
- **AI Models**: Event extraction and classification models

## API Reference

### IntegratedLifePlanner

Main class for end-to-end life planning.

#### Methods

- `process_client_update(input_text, input_type='email', input_date=None)`: Process client input and return optimal decisions
- `export_comprehensive_report(output_path=None)`: Export comprehensive life planning report
- `get_decision_history()`: Get history of all decisions and alerts

### ClientInputProcessor

Handles client input processing and event extraction.

#### Methods

- `process_email(email_content, email_date=None)`: Process email and extract events
- `process_pdf(pdf_path)`: Process PDF and extract events
- `export_configuration_space(output_path=None)`: Export configuration space

### AIEventExtractor

AI-powered event extraction from natural language.

#### Features

- Event type classification
- Date extraction (absolute and relative)
- Amount extraction
- Cash flow impact determination
- Confidence scoring

## Demo Scripts

### Run Complete Demo
```bash
python demo_end_to_end_pipeline.py
```

### Run Individual Components
```bash
# Client input processing demo
python src/client_input_processor.py

# Integrated life planner demo
python src/integrated_life_planner.py
```

## Output Files

The system generates several output files:

- `ips_output/{client_id}_config_space.json`: Configuration space with all scenarios
- `ips_output/{client_id}_comprehensive_report.json`: Complete life planning report
- Balance tracking data for each scenario
- Decision history and recommendations

## Example Output

```json
{
  "client_id": "DEMO_CLIENT_001",
  "extracted_events": [
    {
      "event_type": "education",
      "description": "Johns Hopkins University tuition",
      "planned_date": "2025-09-01T00:00:00",
      "confidence": 0.9,
      "cash_flow_impact": "negative",
      "impact_amount": 110000
    }
  ],
  "optimal_decisions": {
    "SCENARIO_0001": {
      "current_config": {
        "ED_PATH": "JohnsHopkins",
        "HEL_WORK": "Part-time",
        "RISK_BAND": 2
      },
      "optimal_adjustments": {
        "RISK_BAND": 1
      },
      "confidence_score": 0.85,
      "stress_level": 0.35,
      "recommended_actions": [
        "Build emergency fund",
        "Review education funding strategy"
      ]
    }
  },
  "stress_analysis": {
    "total_scenarios": 192,
    "stressful_scenarios": 45,
    "ruled_out_scenarios": 12,
    "average_stress_level": 0.28
  }
}
```

## Key Components

### 1. Event Extraction Pipeline

```python
# Extract events from natural language
events = ai_extractor.extract_events(text, reference_date)

# Each event contains:
# - event_type: education, work, family, etc.
# - planned_date: extracted timestamp
# - impact_amount: monetary impact
# - cash_flow_impact: positive/negative/neutral
# - confidence: AI confidence score
```

### 2. Scenario Management

```python
# Generate all possible scenarios
scenarios = scenario_manager.generate_scenarios()

# Flag stressful scenarios
scenario_manager.flag_stress_scenario(scenario_id, stress_level)

# Rule out impossible scenarios
scenario_manager.rule_out_scenario(scenario_id, reason)
```

### 3. Balance Tracking

```python
# Update balances for all scenarios
balance_tracker.update_scenario_balance(scenario_id, cash_flows, new_event)

# Get current balances
balances = balance_tracker.get_all_balances()
```

### 4. Optimal Decision Generation

```python
# Calculate optimal decisions for each scenario
optimal_decisions = planner._calculate_optimal_decisions()

# Each decision contains:
# - current_config: current configuration
# - optimal_adjustments: recommended changes
# - confidence_score: decision confidence
# - stress_level: scenario stress level
# - recommended_actions: specific actions to take
```

## Stress Analysis

The system analyzes stress levels across multiple dimensions:

- **Overall Stress**: Combined financial and lifestyle stress
- **Cash Flow Stress**: Income vs. expense ratios
- **Income Volatility**: Stability of income sources
- **Expense Pressure**: High-cost events and commitments

Scenarios with stress levels > 0.4 are flagged, and those > 0.6 are ruled out.

## Real-Time Monitoring

The system provides real-time monitoring with:

- **Event Alerts**: Notifications for upcoming events
- **Stress Monitoring**: Continuous stress level tracking
- **Balance Updates**: Real-time balance recalculations
- **Decision Tracking**: History of all decisions and changes

## Integration with Existing System

This system integrates seamlessly with the existing IPS model:

- Uses existing configuration (`ips_config.json`)
- Leverages existing cash flow calculations
- Extends scenario generation capabilities
- Enhances stress analysis with real-time data

## Error Handling

The system includes comprehensive error handling:

- **Missing Dependencies**: Graceful degradation when AI libraries unavailable
- **Invalid Inputs**: Validation and error messages for malformed inputs
- **Processing Errors**: Retry logic and fallback mechanisms
- **Data Integrity**: Validation of extracted events and calculations

## Performance Considerations

- **Parallel Processing**: Uses threading for scenario calculations
- **Caching**: Caches optimal decisions to avoid recalculation
- **Incremental Updates**: Only recalculates affected scenarios
- **Memory Management**: Efficient data structures for large scenario spaces

## Future Enhancements

- **Advanced AI Models**: Integration with GPT-4 or similar for better extraction
- **Natural Language Processing**: Enhanced date and amount extraction
- **Machine Learning**: Learning from client feedback to improve recommendations
- **Real-time APIs**: Integration with financial data providers
- **Mobile Interface**: Client-facing mobile app for updates

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **PDF Processing Errors**
   ```bash
   pip install PyPDF2 pdfplumber
   ```

3. **AI Model Issues**
   ```bash
   pip install openai transformers torch
   ```

4. **Configuration Errors**
   - Ensure `config/ips_config.json` exists
   - Check file permissions
   - Validate JSON format

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the demo scripts
3. Examine the example outputs
4. Check the configuration files

The system is designed to be robust and provide clear error messages for common issues. 