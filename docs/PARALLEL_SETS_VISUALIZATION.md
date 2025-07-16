# Parallel Sets Visualization for IPS Analysis

## Overview

The Parallel Sets Visualization is a powerful addition to the IPS (Investment Policy Statement) analysis system that provides interactive visualizations of relationships between life events, portfolio allocations, market conditions, comfort levels, and performance outcomes. This feature is inspired by the D3.js parallel sets visualization technique and adapted specifically for investment planning scenarios.

## Features

### 🎯 Core Capabilities

- **Multi-dimensional Analysis**: Shows relationships between 5 key dimensions:
  - Life Events (career changes, family events, etc.)
  - Portfolio Allocations (High Equity, Balanced, Conservative)
  - Market Conditions (High/Moderate/Low Stress)
  - Comfort Levels (High/Moderate/Low Comfort)
  - Performance Outcomes (High/Positive/Negative Performance)

- **Interactive Visualizations**: Two visualization types:
  - **Plotly Sankey Diagrams**: Static but detailed flow diagrams
  - **D3.js Interactive Visualizations**: Fully interactive with filtering and hover effects

- **Real-time Data Integration**: Seamlessly integrates with existing IPS system components:
  - Dynamic Portfolio Engine
  - Realistic Life Events Generator
  - Comfort Analysis Engine

### 📊 Data Flow

```
Life Events → Portfolio Allocation → Market Conditions → Comfort → Performance
```

## Architecture

### Components

1. **ParallelSetsVisualizer** (`src/parallel_sets_visualizer.py`)
   - Main visualization class
   - Handles data transformation and visualization creation
   - Supports both Plotly and D3.js outputs

2. **Integration Points**
   - `DynamicPortfolioEngine`: Provides portfolio evolution data
   - `RealisticLifeEventsGenerator`: Provides life event data
   - `InteractiveDashboard`: Can incorporate parallel sets views

### Data Structure

```python
{
    'life_events': [
        {
            'category': 'Career Change',
            'impact': 'High',
            'direction': 'Positive',
            'date': '2021-01-15'
        }
    ],
    'portfolio_allocations': ['High Equity', 'Balanced', 'Conservative'],
    'market_conditions': ['High Stress', 'Moderate Stress', 'Low Stress'],
    'comfort_levels': ['High Comfort', 'Moderate Comfort', 'Low Comfort'],
    'performance_outcomes': ['High Performance', 'Positive Performance', 'Negative Performance']
}
```

## Usage

### Basic Usage

```python
from src.parallel_sets_visualizer import ParallelSetsVisualizer
from src.dynamic_portfolio_engine import DynamicPortfolioEngine

# Create portfolio engine with data
portfolio_engine = DynamicPortfolioEngine(client_config)
portfolio_engine.simulate_portfolio_evolution(life_events)

# Create visualizer
visualizer = ParallelSetsVisualizer(portfolio_engine)

# Generate visualizations
plotly_fig = visualizer.create_parallel_sets_visualization()
html_file = visualizer.save_parallel_sets_html("output.html")
```

### Demo Usage

```bash
# Run the demo
python demos/parallel_sets_demo.py

# Run tests
python tests/test_parallel_sets.py
```

## Visualization Types

### 1. Plotly Sankey Diagram

**Features:**
- Static but detailed flow visualization
- Color-coded by event type and impact
- Hover tooltips with detailed information
- Exportable to various formats

**Best for:**
- Presentations and reports
- Static analysis
- Detailed flow analysis

### 2. D3.js Interactive Visualization

**Features:**
- Fully interactive parallel sets
- Real-time filtering capabilities
- Smooth animations and transitions
- Customizable styling

**Best for:**
- Interactive dashboards
- Exploratory data analysis
- Client presentations

## Integration with IPS System

### Life Events Integration

The parallel sets visualization integrates with the existing life events system:

```python
# Life events are automatically categorized
life_event_mapping = {
    'family_planning': -0.3,      # Conservative impact
    'career_advancement': 0.4,    # Aggressive impact
    'portfolio_rebalancing': 0.2, # Moderate impact
    'education_planning': -0.2,   # Conservative impact
    'education_decision': 0.3,    # Positive impact
    'work_arrangement': -0.4      # Conservative impact
}
```

### Portfolio Engine Integration

The visualization automatically processes portfolio snapshots:

```python
# Portfolio allocations are categorized based on equity percentage
if equity_pct > 0.7:
    allocation_category = 'High Equity'
elif equity_pct > 0.5:
    allocation_category = 'Balanced'
else:
    allocation_category = 'Conservative'
```

### Market Conditions Integration

Market stress levels are automatically categorized:

```python
# Market conditions based on stress levels
if market_stress > 0.7:
    market_category = 'High Stress'
elif market_stress > 0.3:
    market_category = 'Moderate Stress'
else:
    market_category = 'Low Stress'
```

## Analysis Capabilities

### Relationship Analysis

The visualization reveals key insights:

1. **Life Event Impact Patterns**
   - Which events lead to conservative vs. aggressive allocations
   - Correlation between event types and performance outcomes

2. **Market Response Patterns**
   - How portfolio allocations change under different market conditions
   - Effectiveness of stress-based adjustments

3. **Comfort-Performance Trade-offs**
   - Relationship between comfort levels and performance outcomes
   - Optimal comfort-performance balance

### Filtering and Exploration

Interactive features allow users to:

- **Filter by Event Type**: Focus on specific life events
- **Filter by Impact Level**: Analyze high/medium/low impact events
- **Filter by Direction**: Compare positive vs. negative events
- **Filter by Allocation Type**: Focus on specific portfolio strategies

## Example Scenarios

### Scenario 1: Career Change Analysis

```
Career Change (High, Positive) → High Equity → Low Stress → High Comfort → High Performance
```

**Insight**: Career advancements typically lead to more aggressive allocations and better performance.

### Scenario 2: Family Planning Impact

```
Family Planning (Medium, Negative) → Conservative → Low Stress → High Comfort → Positive Performance
```

**Insight**: Family events lead to conservative allocations but still maintain positive performance.

### Scenario 3: Market Stress Response

```
Market Crash (High, Negative) → Conservative → High Stress → Low Comfort → Negative Performance
```

**Insight**: Market crashes lead to defensive positioning and temporary performance declines.

## Technical Implementation

### Data Processing Pipeline

1. **Event Extraction**: Life events are extracted from the portfolio engine
2. **Categorization**: Events are categorized by type, impact, and direction
3. **Allocation Mapping**: Portfolio allocations are mapped to categories
4. **Market Mapping**: Market conditions are categorized by stress levels
5. **Comfort Mapping**: Comfort levels are categorized
6. **Performance Calculation**: Performance outcomes are simulated
7. **Visualization Generation**: Data is formatted for visualization

### Visualization Generation

```python
def create_parallel_sets_data(self):
    """Create data structure for parallel sets visualization"""
    sets_data = {
        'life_events': [],
        'portfolio_allocations': [],
        'market_conditions': [],
        'comfort_levels': [],
        'performance_outcomes': []
    }
    
    # Process life events
    for event in self.data.get('life_events_log', []):
        sets_data['life_events'].append({
            'category': event['type'],
            'impact': self._categorize_impact(event['impact_score']),
            'direction': 'Positive' if event['impact_score'] > 0 else 'Negative'
        })
    
    # Process portfolio snapshots
    for snapshot in self.data.get('portfolio_snapshots', []):
        sets_data['portfolio_allocations'].append(
            self._categorize_allocation(snapshot['portfolio']['equity'])
        )
        # ... additional processing
```

## Testing

### Test Coverage

The parallel sets visualization includes comprehensive tests:

- **Data Creation Tests**: Verify data structure and format
- **Visualization Tests**: Ensure visualizations can be created
- **Integration Tests**: Verify integration with existing components
- **Format Tests**: Ensure all data categories are properly formatted

### Running Tests

```bash
# Run all parallel sets tests
python tests/test_parallel_sets.py

# Run specific test
python -m unittest tests.test_parallel_sets.TestParallelSetsVisualizer.test_parallel_sets_data_creation
```

## Future Enhancements

### Planned Features

1. **Advanced Filtering**: More sophisticated filtering options
2. **Time-based Analysis**: Temporal analysis of relationships
3. **Predictive Modeling**: ML-based relationship prediction
4. **Custom Dimensions**: User-defined analysis dimensions
5. **Export Capabilities**: Enhanced export options

### Potential Integrations

1. **Stress Analyzer**: Integration with stress analysis results
2. **Life Choice Optimizer**: Integration with optimization results
3. **Historical Backtesting**: Integration with backtesting results
4. **Client Input Processor**: Real-time client data integration

## Performance Considerations

### Optimization Strategies

1. **Data Caching**: Cache processed data for repeated visualizations
2. **Lazy Loading**: Load visualization components on demand
3. **Data Compression**: Compress large datasets for web delivery
4. **Progressive Rendering**: Render visualizations progressively

### Scalability

The visualization system is designed to handle:

- **Small Datasets**: < 100 events (instant rendering)
- **Medium Datasets**: 100-1000 events (optimized rendering)
- **Large Datasets**: > 1000 events (sampling and aggregation)

## Conclusion

The Parallel Sets Visualization provides a powerful new dimension to IPS analysis, enabling users to understand complex relationships between life events, portfolio decisions, and outcomes. By integrating seamlessly with the existing IPS system, it enhances the overall analytical capabilities while maintaining the system's core functionality.

The visualization serves as both an analytical tool and a communication device, helping financial advisors and clients understand the impact of life events on investment strategies and outcomes. 