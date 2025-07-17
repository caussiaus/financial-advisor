# 🎓 Mesh Training Engine - Financial Commutator Optimization

## Overview

The Mesh Training Engine is a comprehensive system that generates synthetic people with realistic financial profiles, applies financial shocks to test resilience, and learns optimal commutator routes for financial recovery. This training phase creates a database of successful recovery strategies that can be applied to real financial situations.

## 🎯 Key Features

### 1. **Synthetic People Generation**
- Creates realistic financial profiles with varying ages, occupations, and risk tolerances
- Generates diverse income levels, asset allocations, and debt structures
- Models different life stages and financial goals
- Uses age-based probability distributions for realistic scenarios

### 2. **Financial Shock Simulation**
- **Market Crashes**: Portfolio value reductions (-40% to -10%)
- **Job Loss**: Income reductions (-60% to -20%)
- **Medical Emergencies**: Unexpected expenses (10-30% of assets)
- **Divorce**: Net worth reductions (-50% to -20%)
- **Natural Disasters**: Property damage (15-40% of assets)
- **Interest Rate Spikes**: Bond value reductions (-20% to -5%)
- **Inflation Shocks**: Purchasing power reductions (-15% to -5%)

### 3. **Mesh System Integration**
- Runs cash flows through stochastic mesh engines
- Tracks financial state evolution over time
- Models path-dependent financial scenarios
- Integrates with commutator decision engines

### 4. **Commutator Route Learning**
- Tracks optimal recovery paths through financial state space
- Records edge paths showing sequence of financial moves
- Learns successful commutator sequences
- Determines optimal node changes and routing strategies

## 🏗️ Architecture

### Core Components

#### **TrainingScenario**
```python
@dataclass
class TrainingScenario:
    scenario_id: str
    person: SyntheticClientData
    initial_state: Dict[str, float]
    shocks: List[Dict[str, Any]]
    mesh_engine: Any
    commutator_engine: CommutatorDecisionEngine
    success_metrics: Dict[str, float]
    recovery_path: List[str]
    commutator_sequence: List[CommutatorOperation]
```

#### **CommutatorRoute**
```python
@dataclass
class CommutatorRoute:
    route_id: str
    initial_state: Dict[str, float]
    final_state: Dict[str, float]
    shock_type: str
    shock_magnitude: float
    commutator_sequence: List[CommutatorOperation]
    edge_path: List[str]
    success_score: float
    recovery_time: int
    metadata: Dict[str, Any]
```

#### **TrainingResult**
```python
@dataclass
class TrainingResult:
    session_id: str
    num_scenarios: int
    successful_recoveries: int
    failed_recoveries: int
    average_recovery_time: float
    best_commutator_routes: List[CommutatorRoute]
    shock_type_performance: Dict[str, float]
    mesh_optimization_insights: Dict[str, Any]
```

## 🚀 Usage

### Basic Training Session

```python
from src.training.mesh_training_engine import run_training_session

# Run training with 100 scenarios
result = run_training_session(num_scenarios=100)

print(f"Success Rate: {result.successful_recoveries / result.num_scenarios:.1%}")
print(f"Average Recovery Time: {result.average_recovery_time:.1f} days")
```

### Custom Training Engine

```python
from src.training.mesh_training_engine import MeshTrainingEngine

# Initialize training engine
training_engine = MeshTrainingEngine()

# Generate scenarios with specific age distribution
age_distribution = {25: 0.3, 30: 0.4, 35: 0.3}
scenarios = training_engine.generate_training_scenarios(
    num_scenarios=50,
    age_distribution=age_distribution
)

# Run training session
result = training_engine.run_training_session(scenarios)
```

### Command Line Usage

```bash
# Run with default settings (100 scenarios)
python run_training.py

# Run with custom number of scenarios
python run_training.py 200

# Run demo with visualization
python demos/demo_training_engine.py
```

## 📊 Training Process

### 1. **Scenario Generation**
```python
# Generate synthetic person
person = synthetic_engine.generate_synthetic_client(target_age=35)

# Create initial financial state
initial_state = {
    'cash': 50000,
    'investments': 200000,
    'bonds': 100000,
    'real_estate': 300000,
    'income': 120000,
    'expenses': 84000,
    'net_worth': 570000
}

# Generate financial shocks
shocks = [
    {
        'type': 'market_crash',
        'magnitude': -0.25,
        'timing': datetime.now() + timedelta(days=180),
        'category': 'investment'
    }
]
```

### 2. **Shock Application**
```python
# Apply shocks to initial state
shocked_state = {
    'cash': 50000,
    'investments': 150000,  # Reduced by 25%
    'bonds': 100000,
    'real_estate': 300000,
    'income': 120000,
    'expenses': 84000,
    'net_worth': 520000  # Reduced net worth
}
```

### 3. **Mesh Processing**
```python
# Initialize mesh engine with shocked state
mesh_engine = StochasticMeshEngine(shocked_state)

# Generate commutator operations
operations = commutator_engine.generate_commutator_operations()

# Select optimal sequence
optimal_sequence = commutator_engine.select_optimal_commutator_sequence(operations)
```

### 4. **Route Tracking**
```python
# Execute sequence and track path
success, recovery_time, edge_path = execute_commutator_sequence(
    commutator_engine, optimal_sequence, shocks
)

# Record successful route
route = CommutatorRoute(
    route_id="route_001",
    initial_state=shocked_state,
    final_state=recovered_state,
    shock_type="market_crash",
    shock_magnitude=-0.25,
    commutator_sequence=optimal_sequence,
    edge_path=edge_path,
    success_score=0.85,
    recovery_time=180
)
```

## 📈 Results Analysis

### Success Metrics
- **Recovery Rate**: Percentage of scenarios that successfully recover
- **Average Recovery Time**: Mean time to financial recovery
- **Success Score**: Composite score based on net worth, cash reserves, and debt ratio
- **Route Efficiency**: Number of commutator steps required for recovery

### Shock Type Performance
```python
shock_performance = {
    'market_crash': 0.75,      # 75% recovery rate
    'job_loss': 0.45,          # 45% recovery rate
    'medical_emergency': 0.80, # 80% recovery rate
    'divorce': 0.30,           # 30% recovery rate
    'natural_disaster': 0.60,  # 60% recovery rate
    'interest_rate_spike': 0.85, # 85% recovery rate
    'inflation_shock': 0.70    # 70% recovery rate
}
```

### Commutator Route Patterns
```python
most_frequent_operations = [
    ('rebalance', 45),      # Portfolio rebalancing
    ('debt_restructure', 30), # Debt optimization
    ('income_optimization', 25), # Income enhancement
    ('capital_efficiency', 20), # Capital optimization
    ('risk_management', 15)  # Risk reduction
]
```

## 🎯 Applications

### 1. **Real Financial Planning**
- Apply learned commutator routes to real client situations
- Use successful recovery patterns for financial advice
- Optimize portfolio strategies based on training data

### 2. **Risk Management**
- Identify most effective strategies for different shock types
- Develop contingency plans based on training results
- Create resilience frameworks for financial portfolios

### 3. **Product Development**
- Design financial products based on successful recovery patterns
- Develop insurance products targeting specific shock types
- Create investment strategies optimized for recovery scenarios

## 📁 Output Structure

```
data/outputs/training/
├── successful_routes.json      # Successful commutator routes
├── training_history.json       # Training session history
└── training_analysis.png       # Visualization of results
```

### Route Data Format
```json
{
  "route_id": "route_001",
  "shock_type": "market_crash",
  "shock_magnitude": -0.25,
  "success_score": 0.85,
  "recovery_time": 180,
  "edge_path": ["edge_0_rebalance", "edge_1_debt_restructure"],
  "commutator_sequence": [
    {
      "operation_type": "rebalance",
      "capital_required": 25000,
      "risk_change": -0.05,
      "success_probability": 0.9
    }
  ]
}
```

## 🔧 Configuration

### Shock Type Configuration
```python
shock_types = {
    'market_crash': {
        'probability': 0.15,
        'impact_range': (-0.4, -0.1),
        'recovery_time_range': (12, 36),
        'category': 'investment'
    }
}
```

### Age-Based Adjustments
```python
def calculate_age_factor(age: int, shock_type: str) -> float:
    if shock_type == 'job_loss':
        return 1.5 if age < 35 else 0.8 if age > 55 else 1.0
    elif shock_type == 'medical_emergency':
        return 0.5 if age < 30 else 1.0 if age < 50 else 1.5
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install numpy pandas scipy networkx
   ```

2. **Run Basic Training**
   ```bash
   python run_training.py 50
   ```

3. **Run Demo**
   ```bash
   python demos/demo_training_engine.py
   ```

4. **Analyze Results**
   ```python
   from src.training.mesh_training_engine import MeshTrainingEngine
   
   training_engine = MeshTrainingEngine()
   training_engine.load_training_results()
   ```

## 🎓 Key Insights

### 1. **Synthetic Data Quality**
- Realistic financial profiles based on age, occupation, and life stage
- Probabilistic shock generation with age-appropriate adjustments
- Comprehensive financial state modeling including assets, liabilities, and cash flows

### 2. **Mesh System Integration**
- Cash flows processed through stochastic mesh engines
- Path-dependent analysis of financial evolution
- Integration with commutator decision engines for optimization

### 3. **Route Learning**
- Successful commutator routes tracked and stored
- Edge paths show optimal financial move sequences
- Recovery strategies learned from training data

### 4. **Real-World Application**
- Training data can be applied to real financial situations
- Optimal recovery strategies identified for different shock types
- Commutator routes provide actionable financial advice

## 🔮 Future Enhancements

1. **Advanced Shock Modeling**
   - Correlated shocks (e.g., market crash + job loss)
   - Time-varying shock probabilities
   - Sector-specific shock impacts

2. **Enhanced Route Learning**
   - Machine learning for route optimization
   - Dynamic route adaptation
   - Personalized recovery strategies

3. **Real-Time Integration**
   - Live market data integration
   - Real-time shock detection
   - Dynamic strategy adjustment

4. **Multi-Agent Training**
   - Household-level training scenarios
   - Intergenerational wealth transfer
   - Community-level resilience modeling

---

*The Mesh Training Engine provides a foundation for learning optimal financial recovery strategies through comprehensive simulation and analysis of synthetic financial scenarios.* 