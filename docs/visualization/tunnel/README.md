# Tunnel Visualization Module

## Overview

The tunnel visualization module provides high-dimensional data visualization capabilities for financial analysis and neural mesh exploration. It creates interactive 3D tunnel visualizations that show how financial states evolve over time.

## Components

### 1. GuaranteedTunnelVisualizer
- **Purpose**: Simple, reliable 3D scatter plots
- **Features**: Basic scatter plots for financial metrics
- **Use Case**: When you need guaranteed visualization output

### 2. SimpleTunnelVisualizer  
- **Purpose**: Basic tunnel surface generation
- **Features**: Simple tunnel shapes with basic interpolation
- **Use Case**: Quick tunnel visualization with minimal complexity

### 3. HighDimensionalTunnelVisualizer
- **Purpose**: Advanced tunnel with statistical interpolation
- **Features**: Gaussian distribution tunnels, statistical processing
- **Use Case**: Sophisticated analysis with statistical accuracy

### 4. TunnelDashboard
- **Purpose**: Interactive Flask web interface
- **Features**: Real-time filtering, parameter controls, export functionality
- **Use Case**: Interactive exploration and analysis

## Data Input

The system processes financial simulation data from `outputs/data/horatio_mesh_timelapse.json`:

- **132 Financial Snapshots** (July 2025 to June 2036)
- **26,730 Data Points** across multiple financial states
- **4 Key Metrics**: Total wealth, cash, investments, probability

## Usage

### Command Line
```bash
# Run interactive dashboard
python3 scripts/run_tunnel_dashboard.py

# Access at http://localhost:5020
```

### Programmatic Usage
```python
from src.visualization.tunnel import GuaranteedTunnelVisualizer

# Create visualizer
visualizer = GuaranteedTunnelVisualizer()

# Generate visualization
fig = visualizer.create_guaranteed_visualization(
    time_window=(0, 50),
    node_filter=['omega_0_0', 'omega_1_0']
)
```

## API Endpoints

### GET /api/tunnel-data
- **Parameters**: `time_window`, `node_filter`, `resolution`
- **Returns**: JSON with Plotly figure data

### GET /api/data-summary  
- **Returns**: Statistical summary of available data

### GET /api/export
- **Parameters**: Same as tunnel-data
- **Returns**: HTML file download

## Visualization Types

### 1. Total Wealth Tunnel
Shows wealth evolution over time with 3D scatter plots

### 2. Cash Flow Tunnel  
Displays cash management patterns and liquidity

### 3. Investment Portfolio Tunnel
Visualizes investment allocation changes

### 4. Probability/Confidence Tunnel
Shows state confidence levels and risk assessment

## Features

- **Interactive 3D**: Full rotation, zoom, and pan
- **Real-time Filtering**: Time windows and node selection
- **Statistical Processing**: Mean, standard deviation, Gaussian distributions
- **Export Capability**: Save visualizations as HTML files
- **Responsive Design**: Works across different screen sizes

## Technical Details

- **Data Processing**: 26,730 data points across 132 snapshots
- **Visualization Engine**: Plotly for interactive 3D graphics
- **Web Framework**: Flask for API and web interface
- **Statistical Methods**: Gaussian distribution for tunnel shapes
- **Performance**: < 30 seconds for full dataset processing

## File Structure

```
src/visualization/tunnel/
├── __init__.py
├── guaranteed_tunnel_visualizer.py
├── simple_tunnel_visualizer.py
├── high_dimensional_tunnel_visualizer.py
└── tunnel_dashboard.py

docs/visualization/tunnel/
├── README.md
└── TUNNEL_VISUALIZATION_SUMMARY.md

outputs/visualizations/tunnel/
├── basic_tunnel_visualization.html
├── enhanced_tunnel_visualization.html
├── time_windowed_tunnel_visualization.html
└── guaranteed_visualization.html
```

## Dependencies

- `plotly`: 3D visualization engine
- `flask`: Web framework for dashboard
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `pathlib`: File path handling

## Future Enhancements

- **Animation**: Time-lapse tunnel evolution
- **Multi-metric Correlation**: Cross-tunnel analysis  
- **Machine Learning Integration**: Predictive tunnel modeling
- **Advanced Filtering**: More sophisticated data selection
- **Performance Optimization**: GPU acceleration for large datasets 