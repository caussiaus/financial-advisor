# Root Directory Cleanup Plan

## 🎯 **Current State Analysis**

### **Root Directory Issues**
- **Too many demo files** (15+ demo scripts)
- **Random test files** scattered throughout
- **Multiple dashboard files** with similar functionality
- **Generated images and visualizations** in root
- **Evaluation results** taking up space
- **Multiple documentation files** that could be consolidated

### **Data Directory Issues**
- **Trial people data** is scattered and inconsistent
- **Historical backtests** are mixed with current data
- **Uploads directory** has old files
- **Outputs directory** is bloated with old results

---

## 🧹 **Cleanup Strategy**

### **1. Root Directory Organization**

#### **Move to `demos/` directory:**
- All `demo_*.py` files
- All `test_*.py` files (except core tests)
- All `*_demo.py` files

#### **Move to `dashboards/` directory:**
- `enhanced_mesh_dashboard.py`
- `mesh_congruence_app.py`
- `mesh_market_dashboard.py`
- `flexibility_comfort_dashboard.py`
- `omega_web_app.py`
- `web_ui.py`

#### **Move to `visualizations/` directory:**
- `life_events_portfolio_visualization.html`
- `serve_visualization.py`
- All `.png` files
- All `.html` files

#### **Move to `docs/` directory:**
- All `*.md` files (except README.md)
- All documentation summaries

#### **Move to `scripts/` directory:**
- `start_enhanced_dashboard.py`
- `system_control.py`
- `deploy.sh`
- `setup.py`
- `cleanup_project.py`
- `convert_to_dl_friendly.py`

#### **Move to `evaluations/` directory:**
- `evaluation_results_20250717_012831/`
- `comprehensive_mesh_evaluation.py`
- `stress_test_script.py`

### **2. Data Directory Cleanup**

#### **Clean `data/inputs/trial_people/`:**
- Remove all existing trial people
- Create new standardized structure
- Prepare for new people data

#### **Clean `data/outputs/`:**
- Archive old results
- Keep only essential current outputs
- Create clear organization

#### **Clean `data/uploads/`:**
- Remove old uploads
- Keep only essential files

---

## 📁 **New Directory Structure**

```
ips_case1/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── LICENSE                      # License file
├── src/                        # Source code (already organized)
├── data/                       # Clean data structure
│   ├── inputs/
│   │   ├── people/             # New standardized people data
│   │   ├── uploads/            # Current uploads only
│   │   └── historical/         # Historical data
│   └── outputs/
│       ├── current/            # Current outputs
│       ├── archived/           # Archived results
│       └── visualizations/     # Generated visualizations
├── demos/                      # All demo and test scripts
├── dashboards/                 # All dashboard applications
├── visualizations/             # All visualization files
├── docs/                       # All documentation
├── scripts/                    # Utility scripts
├── evaluations/                # Evaluation results
└── tests/                      # Core tests (keep in root)
```

---

## 🚀 **Implementation Steps**

### **Step 1: Create New Directories**
```bash
mkdir -p demos dashboards visualizations docs scripts evaluations
```

### **Step 2: Move Files by Category**
```bash
# Move demo files
mv demo_*.py demos/
mv test_*.py demos/  # except core tests
mv *_demo.py demos/

# Move dashboard files
mv *_dashboard.py dashboards/
mv *_app.py dashboards/
mv web_ui.py dashboards/

# Move visualization files
mv *.html visualizations/
mv *.png visualizations/
mv serve_visualization.py visualizations/

# Move documentation
mv *.md docs/  # except README.md

# Move scripts
mv start_enhanced_dashboard.py scripts/
mv system_control.py scripts/
mv deploy.sh scripts/
mv setup.py scripts/
mv cleanup_project.py scripts/
mv convert_to_dl_friendly.py scripts/

# Move evaluations
mv evaluation_results_* evaluations/
mv comprehensive_mesh_evaluation.py evaluations/
mv stress_test_script.py evaluations/
```

### **Step 3: Clean Data Directory**
```bash
# Archive old trial people
mkdir -p data/inputs/people/archived
mv data/inputs/trial_people/* data/inputs/people/archived/

# Create new people structure
mkdir -p data/inputs/people/current

# Clean outputs
mkdir -p data/outputs/current data/outputs/archived data/outputs/visualizations
```

### **Step 4: Update Import Paths**
- Update all import statements in moved files
- Update any hardcoded paths
- Test that everything still works

---

## 📋 **New People Data Format**

### **Standardized Structure for Each Person:**
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
```

### **Profile Template:**
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

---

## ✅ **Success Criteria**

- **Clean root directory** with only essential files
- **Organized data structure** ready for new people
- **Clear separation** of demos, dashboards, docs
- **Updated documentation** reflecting new structure
- **Working imports** after file moves
- **Ready for new people data** in standardized format

---

## 🎯 **Next Steps After Cleanup**

1. **Create new people data** in standardized format
2. **Test all systems** with new structure
3. **Update documentation** to reflect new organization
4. **Create user guide** for adding new people
5. **Implement data validation** for new people format 