# People Data Structure

## 📋 **Standardized Format for Financial Profiles**

This directory contains standardized financial profiles for testing and analysis. Each person has a consistent structure with all necessary financial information.

---

## 📁 **Directory Structure**

```
data/inputs/people/current/
├── person_001/
│   ├── profile.json           # Basic profile information
│   ├── financial_state.json   # Current financial state
│   ├── goals.json            # Financial goals
│   ├── life_events.json      # Life events and timeline
│   └── preferences.json      # Risk preferences and constraints
├── person_002/
│   └── ...
└── README.md                 # This file
```

---

## 📄 **File Templates**

### **profile.json Template**
```json
{
  "id": "person_001",
  "name": "John Doe",
  "age": 35,
  "occupation": "Software Engineer",
  "location": "San Francisco, CA",
  "family_status": "married_with_children",
  "income_level": "high",
  "risk_tolerance": "moderate",
  "investment_horizon": 25,
  "notes": "Tech professional with family"
}
```

### **financial_state.json Template**
```json
{
  "id": "person_001",
  "timestamp": "2025-01-01T00:00:00Z",
  "assets": {
    "cash": 50000,
    "investments": 250000,
    "real_estate": 800000,
    "retirement_accounts": 150000,
    "other_assets": 25000
  },
  "liabilities": {
    "mortgage": 400000,
    "student_loans": 0,
    "credit_cards": 5000,
    "other_debt": 0
  },
  "income": {
    "annual_salary": 150000,
    "bonus": 25000,
    "other_income": 10000
  },
  "expenses": {
    "monthly_living": 4000,
    "monthly_mortgage": 2500,
    "monthly_utilities": 500,
    "monthly_food": 800,
    "monthly_transportation": 400,
    "monthly_entertainment": 300,
    "monthly_savings": 2000
  }
}
```

### **goals.json Template**
```json
{
  "id": "person_001",
  "short_term_goals": [
    {
      "id": "goal_001",
      "name": "Emergency Fund",
      "target_amount": 50000,
      "target_date": "2025-12-31",
      "priority": "high",
      "description": "Build 6-month emergency fund"
    }
  ],
  "medium_term_goals": [
    {
      "id": "goal_002",
      "name": "Children's Education",
      "target_amount": 200000,
      "target_date": "2035-06-01",
      "priority": "high",
      "description": "College fund for two children"
    }
  ],
  "long_term_goals": [
    {
      "id": "goal_003",
      "name": "Retirement",
      "target_amount": 2000000,
      "target_date": "2055-01-01",
      "priority": "medium",
      "description": "Comfortable retirement at 65"
    }
  ]
}
```

### **life_events.json Template**
```json
{
  "id": "person_001",
  "past_events": [
    {
      "id": "event_001",
      "name": "First Child Born",
      "date": "2020-03-15",
      "financial_impact": -25000,
      "description": "Hospital costs and initial expenses",
      "category": "family"
    }
  ],
  "planned_events": [
    {
      "id": "event_002",
      "name": "Career Advancement",
      "date": "2025-06-01",
      "expected_impact": 30000,
      "description": "Expected promotion and salary increase",
      "category": "career",
      "probability": 0.8
    },
    {
      "id": "event_003",
      "name": "Home Renovation",
      "date": "2026-03-01",
      "expected_impact": -75000,
      "description": "Kitchen and bathroom renovation",
      "category": "housing",
      "probability": 0.9
    },
    {
      "id": "event_004",
      "name": "Second Child",
      "date": "2027-09-01",
      "expected_impact": -35000,
      "description": "Medical costs and increased expenses",
      "category": "family",
      "probability": 0.6
    },
    {
      "id": "event_005",
      "name": "Early Retirement",
      "date": "2040-01-01",
      "expected_impact": -500000,
      "description": "Early retirement at 50",
      "category": "retirement",
      "probability": 0.3
    }
  ]
}
```

### **preferences.json Template**
```json
{
  "id": "person_001",
  "risk_tolerance": {
    "level": "moderate",
    "score": 6,
    "description": "Comfortable with moderate risk for higher returns"
  },
  "investment_preferences": {
    "preferred_asset_classes": ["stocks", "bonds", "real_estate"],
    "avoided_asset_classes": ["commodities", "cryptocurrency"],
    "target_asset_allocation": {
      "stocks": 0.6,
      "bonds": 0.3,
      "cash": 0.1
    }
  },
  "liquidity_needs": {
    "emergency_fund_target": 50000,
    "monthly_cash_flow_needs": 8000,
    "liquidity_priority": "high"
  },
  "tax_considerations": {
    "tax_bracket": "24%",
    "prefer_tax_advantaged_accounts": true,
    "tax_loss_harvesting": true
  },
  "constraints": {
    "ethical_investing": false,
    "geographic_preferences": ["US", "developed_markets"],
    "minimum_investment_amounts": {
      "stocks": 1000,
      "bonds": 5000,
      "real_estate": 50000
    }
  }
}
```

---

## 🚀 **Adding New People**

### **Step 1: Create Directory**
```bash
mkdir data/inputs/people/current/person_XXX
```

### **Step 2: Create Files**
Create the five JSON files using the templates above:
- `profile.json`
- `financial_state.json`
- `goals.json`
- `life_events.json`
- `preferences.json`

### **Step 3: Validate Data**
Ensure all JSON files are valid and follow the template structure.

### **Step 4: Test Integration**
Run the mesh system to verify the new person data works correctly.

---

## 📊 **Data Validation Rules**

### **Profile Requirements**
- `id`: Must be unique across all people
- `age`: Must be between 18 and 100
- `risk_tolerance`: Must be one of ["low", "moderate", "high"]
- `income_level`: Must be one of ["low", "medium", "high"]

### **Financial State Requirements**
- All monetary values must be positive numbers
- `timestamp` must be valid ISO 8601 format
- Assets and liabilities must be properly categorized

### **Goals Requirements**
- Each goal must have a unique `id`
- `target_amount` must be positive
- `target_date` must be in the future
- `priority` must be one of ["low", "medium", "high"]

### **Life Events Requirements**
- Each event must have a unique `id`
- `probability` must be between 0 and 1
- `financial_impact` must be a number (positive or negative)
- Dates must be valid ISO 8601 format

### **Preferences Requirements**
- `risk_tolerance.score` must be between 1 and 10
- Asset allocation percentages must sum to 1.0
- All monetary values must be positive

---

## 🔧 **Integration with Mesh System**

The mesh system will automatically:
1. **Load all people** from this directory
2. **Validate data** against the templates
3. **Generate mesh nodes** for each person
4. **Create similarity matrices** between people
5. **Provide recommendations** based on similar profiles

---

## 📈 **Example Usage**

```python
from src.core import StochasticMeshEngine
from src.analysis import MeshVectorDatabase

# Load people data
people_data = load_people_from_directory("data/inputs/people/current")

# Initialize mesh engine
mesh_engine = StochasticMeshEngine(people_data[0]["financial_state"])

# Generate mesh for each person
for person in people_data:
    mesh_data = mesh_engine.generate_mesh(person)
    # Process mesh data...
```

---

*Ready for new people data in standardized format!* 