# Core Modules Summary

## System Architecture Overview

The system is built around several core modules that work together to create a comprehensive financial planning and analysis platform:

### 1. **JSON-to-Vector Converter** (`src/json_to_vector_converter.py`)
**Purpose**: Converts client JSON data into high-dimensional vector representations

**Key Components**:
- `JSONToVectorConverter`: Main converter class
- `ClientVectorProfile`: Vector representation of client data
- `LifestyleEvent`: Individual life events with probabilities
- `LifeStage`: Enum for different life stages (early_career, mid_career, etc.)
- `EventCategory`: Enum for event types (education, career, family, etc.)

**Core Functions**:
```python
# Convert JSON client data to vector profile
vector_profile = converter.convert_json_to_vector_profile(json_data)

# Generate lifestyle events based on age and life stage
events = converter.generate_lifestyle_events(age, income, life_stage)

# Convert events to seed events for mesh processing
seed_events = converter.convert_events_to_seed_events(lifestyle_events)
```

**Output**: 128-dimensional vector embeddings representing client financial profiles

### 2. **Synthetic Lifestyle Engine** (`src/synthetic_lifestyle_engine.py`)
**Purpose**: Generates synthetic client data with realistic lifestyle events

**Key Components**:
- `SyntheticLifestyleEngine`: Main engine for generating synthetic clients
- `SyntheticClientData`: Complete client data structure
- Life stage configurations for different age groups

**Core Functions**:
```python
# Generate synthetic client
client = engine.generate_synthetic_client(target_age=35)

# Process with mesh engine
client = engine.process_with_mesh_engine(client, num_scenarios=1000)

# Generate batch of clients
clients = engine.generate_client_batch(num_clients=50)
```

**Output**: Complete synthetic client profiles with events, cash flows, and mesh data

### 3. **Mesh Vector Database** (`src/mesh_vector_database.py`)
**Purpose**: Vector database for similarity matching and uncertainty estimation

**Key Components**:
- `MeshVectorDatabase`: Main vector database class
- `MeshEmbedding`: Embedding representation with metadata
- `SimilarityMatch`: Similarity match between clients

**Core Functions**:
```python
# Add client to database
vector_db.add_client(client_data)

# Find similar clients
similar_clients = vector_db.find_similar_clients(client_id, top_k=5)

# Get recommendations
recommendations = vector_db.get_recommendations(client_id)
```

**Output**: Similarity matches, uncertainty estimates, and recommendations

### 4. **Trial People Manager** (`src/trial_people_manager.py`)
**Purpose**: Manages multiple trial people for training and analysis

**Key Components**:
- `TrialPeopleManager`: Main manager for trial people
- `TrialPerson`: Individual trial person data
- `InterpolatedSurface`: Interpolated surfaces across people
- `TaskSchedule`: Task scheduling for analysis

**Core Functions**:
```python
# Ingest trial person from files
person = manager.ingest_trial_person(folder_name)

# Process with mesh engine
person = manager.process_trial_person_with_mesh(person)

# Interpolate surfaces across group
surfaces = manager.interpolate_surfaces_across_group()

# Identify less dense sections
density_analysis = manager.identify_less_dense_sections()

# Visualize high-dimensional topology
viz_files = manager.visualize_high_dimensional_topology()
```

**Output**: Interpolated surfaces, density analysis, topology visualizations

### 5. **Time Uncertainty Mesh Engine** (`src/time_uncertainty_mesh.py`)
**Purpose**: Monte Carlo mesh engine for financial scenario simulation

**Key Components**:
- `TimeUncertaintyMeshEngine`: Main mesh engine
- `SeedEvent`: Seed events for mesh initialization
- Mesh state management and risk analysis

**Core Functions**:
```python
# Initialize mesh with time uncertainty
mesh_data, risk_analysis = mesh_engine.initialize_mesh_with_time_uncertainty(
    seed_events, num_scenarios=1000, time_horizon_years=5
)

# Generate mesh nodes
mesh_engine.generate_mesh_nodes(time_horizon_years)

# Perform risk analysis
risk_analysis = mesh_engine.perform_risk_analysis()
```

**Output**: Mesh states, risk metrics, scenario analysis

### 6. **Synthetic Data Generator** (`src/synthetic_data_generator.py`)
**Purpose**: Generates realistic synthetic financial data

**Key Components**:
- `SyntheticFinancialDataGenerator`: Main generator
- `PersonProfile`: Person profile with financial data
- Realistic data generation algorithms

**Core Functions**:
```python
# Generate person profile
profile = generator.generate_person_profile()

# Generate financial data
financial_data = generator.generate_financial_data(profile)
```

**Output**: Realistic synthetic financial profiles

## Module Relationships

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   JSON-to-Vector    │    │  Synthetic Data     │    │  Trial People       │
│     Converter       │    │    Generator        │    │     Manager         │
│                     │    │                     │    │                     │
│ • Convert JSON to   │    │ • Generate realistic│    │ • Manage multiple   │
│   vector profiles   │    │   financial data    │    │   trial people      │
│ • Generate events   │    │ • Create person     │    │ • Interpolate       │
│ • Life stage logic  │    │   profiles          │    │   surfaces          │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                         │                           │
           │                         │                           │
           ▼                         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Synthetic Lifestyle │    │  Time Uncertainty  │    │  Mesh Vector        │
│       Engine        │    │    Mesh Engine      │    │    Database         │
│                     │    │                     │    │                     │
│ • Generate complete │    │ • Monte Carlo       │    │ • Vector similarity │
│   client data       │    │   simulations       │    │   matching          │
│ • Process with mesh │    │ • Risk analysis     │    │ • Uncertainty       │
│ • Export data       │    │ • Scenario modeling │    │   estimation        │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Data Flow

### 1. **Input Processing**
```
JSON Client Data → JSON-to-Vector Converter → Vector Profile
```

### 2. **Synthetic Generation**
```
Synthetic Data Generator → Synthetic Lifestyle Engine → Complete Client Data
```

### 3. **Mesh Processing**
```
Client Data → Time Uncertainty Mesh Engine → Mesh States + Risk Analysis
```

### 4. **Vector Database**
```
Mesh Data → Mesh Vector Database → Similarity Matches + Recommendations
```

### 5. **Trial Analysis**
```
Multiple People → Trial People Manager → Interpolated Surfaces + Visualizations
```

## Key Features by Module

### **JSON-to-Vector Converter**
- ✅ Converts JSON to 128-dimensional vectors
- ✅ Generates lifestyle events by age/life stage
- ✅ Models event probabilities and cash flows
- ✅ Creates discretionary spending surfaces

### **Synthetic Lifestyle Engine**
- ✅ Generates realistic synthetic clients
- ✅ Creates lifestyle events with probabilities
- ✅ Integrates with mesh engine for modeling
- ✅ Exports complete client data

### **Mesh Vector Database**
- ✅ Stores vector embeddings with metadata
- ✅ Finds similar clients using cosine similarity
- ✅ Estimates uncertainties based on similarity
- ✅ Provides recommendations from similar clients

### **Trial People Manager**
- ✅ Manages multiple trial people
- ✅ Interpolates surfaces across the group
- ✅ Identifies less dense mesh sections
- ✅ Visualizes high-dimensional topology
- ✅ Schedules analysis tasks

### **Time Uncertainty Mesh Engine**
- ✅ Monte Carlo scenario generation
- ✅ Risk analysis and metrics
- ✅ Time horizon modeling
- ✅ Vectorized processing

### **Synthetic Data Generator**
- ✅ Realistic financial data generation
- ✅ Person profile creation
- ✅ Asset and debt modeling
- ✅ Risk tolerance simulation

## Integration Points

### **Vector Embedding Pipeline**
```
JSON Data → Vector Converter → Vector Profile → Mesh Database → Similarity Matching
```

### **Synthetic Data Pipeline**
```
Synthetic Generator → Lifestyle Engine → Mesh Engine → Vector Database
```

### **Trial Analysis Pipeline**
```
Trial People → Mesh Processing → Surface Interpolation → Topology Visualization
```

## Output Formats

### **Vector Profiles**
- 128-dimensional normalized vectors
- Life stage and event probability data
- Cash flow and discretionary spending vectors

### **Mesh Data**
- Multi-dimensional mesh states
- Risk analysis metrics
- Scenario probability distributions

### **Similarity Matches**
- Cosine similarity scores
- Matching factor identification
- Uncertainty estimates
- Confidence scores

### **Interpolated Surfaces**
- Grid-based interpolated values
- Confidence maps
- Contributing people tracking

### **Visualizations**
- PCA and t-SNE projections
- 3D topology plots
- Network similarity graphs
- Heatmap matrices

## Usage Examples

### **Basic Client Processing**
```python
# Convert JSON to vector
converter = JSONToVectorConverter()
vector_profile = converter.convert_json_to_vector_profile(client_json)

# Generate synthetic client
engine = SyntheticLifestyleEngine()
client = engine.generate_synthetic_client(target_age=35)

# Process with mesh
client = engine.process_with_mesh_engine(client)

# Add to vector database
vector_db = MeshVectorDatabase()
vector_db.add_client(client)

# Find similar clients
similar = vector_db.find_similar_clients(client.client_id)
```

### **Trial People Analysis**
```python
# Create trial manager
manager = TrialPeopleManager()

# Ingest trial people
people_folders = manager.scan_upload_directory()
for folder in people_folders:
    person = manager.ingest_trial_person(folder)
    person = manager.process_trial_person_with_mesh(person)

# Interpolate surfaces
surfaces = manager.interpolate_surfaces_across_group()

# Visualize topology
viz_files = manager.visualize_high_dimensional_topology()
```

This modular architecture allows for flexible deployment and testing of individual components while maintaining the ability to run comprehensive end-to-end analysis. 