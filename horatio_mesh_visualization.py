#!/usr/bin/env python3
"""
Horatio Mesh Visualization
- Reads the timelapse data
- Creates plots showing mesh evolution
- Demonstrates the "sculpture" effect
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_timelapse_data():
    """Load the timelapse data"""
    with open('horatio_mesh_timelapse.json', 'r') as f:
        data = json.load(f)
    return data

def create_mesh_evolution_plots():
    """Create plots showing mesh evolution"""
    # Load data
    data = load_timelapse_data()
    snapshots = data['snapshots']
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame([
        {
            'timestamp': datetime.fromisoformat(s['snapshot_time']),
            'total_nodes': s['total_nodes'],
            'total_edges': s['total_edges'],
            'avg_wealth': sum(
                float(node['financial_state'].get('total_wealth', 0)) 
                for node in s['nodes']
            ) / len(s['nodes']) if s['nodes'] else 0,
            'events_triggered': sum(
                len(node['event_triggers']) 
                for node in s['nodes']
            )
        }
        for s in snapshots
    ])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎭 Horatio Mesh Evolution: The Financial Sculpture', fontsize=16, fontweight='bold')
    
    # Plot 1: Mesh Growth (Nodes and Edges)
    ax1.plot(df['timestamp'], df['total_nodes'], 'o-', linewidth=2, markersize=4, label='Nodes')
    ax1.plot(df['timestamp'], df['total_edges'], 's-', linewidth=2, markersize=4, label='Edges')
    ax1.set_title('Mesh Structure Growth')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wealth Evolution
    ax2.plot(df['timestamp'], df['avg_wealth'] / 1000000, 'o-', linewidth=2, markersize=4, color='green')
    ax2.set_title('Average Wealth Evolution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Wealth (Millions $)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Events Triggered
    ax3.plot(df['timestamp'], df['events_triggered'], 'o-', linewidth=2, markersize=4, color='red')
    ax3.set_title('Events Triggered Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Events Count')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mesh Complexity (Nodes vs Edges)
    ax4.scatter(df['total_nodes'], df['total_edges'], c=range(len(df)), cmap='viridis', s=50, alpha=0.7)
    ax4.set_title('Mesh Complexity: Nodes vs Edges')
    ax4.set_xlabel('Total Nodes')
    ax4.set_ylabel('Total Edges')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for complexity plot
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('Time Progression')
    
    plt.tight_layout()
    plt.savefig('horatio_mesh_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_3d_sculpture_plot():
    """Create a 3D visualization showing the mesh as a sculpture"""
    data = load_timelapse_data()
    snapshots = data['snapshots']
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract data for 3D plot
    times = [datetime.fromisoformat(s['snapshot_time']) for s in snapshots]
    nodes = [s['total_nodes'] for s in snapshots]
    edges = [s['total_edges'] for s in snapshots]
    wealth = [sum(
        float(node['financial_state'].get('total_wealth', 0)) 
        for node in s['nodes']
    ) / len(s['nodes']) if s['nodes'] else 0 for s in snapshots]
    
    # Convert times to numeric for plotting
    time_numeric = [(t - times[0]).days for t in times]
    
    # Create 3D scatter plot
    scatter = ax.scatter(time_numeric, nodes, edges, 
                        c=wealth, cmap='viridis', s=50, alpha=0.7)
    
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Nodes')
    ax.set_zlabel('Edges')
    ax.set_title('🎭 Horatio Financial Mesh: 3D Sculpture')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Wealth ($)')
    
    plt.savefig('horatio_mesh_3d_sculpture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_timeline_animation_data():
    """Create data for potential animation"""
    data = load_timelapse_data()
    snapshots = data['snapshots']
    
    # Create animation frames data
    animation_data = []
    for i, snapshot in enumerate(snapshots):
        frame = {
            'frame': i,
            'timestamp': snapshot['snapshot_time'],
            'nodes': snapshot['nodes'],
            'edges': snapshot['edges'],
            'total_nodes': snapshot['total_nodes'],
            'total_edges': snapshot['total_edges']
        }
        animation_data.append(frame)
    
    # Save animation data
    with open('horatio_mesh_animation_data.json', 'w') as f:
        json.dump(animation_data, f, indent=2, default=str)
    
    print("✅ Animation data saved to horatio_mesh_animation_data.json")
    return animation_data

def print_sculpture_insights():
    """Print key insights about the mesh evolution"""
    data = load_timelapse_data()
    snapshots = data['snapshots']
    
    print("\n🎭 HORATIO MESH SCULPTURE INSIGHTS")
    print("=" * 50)
    
    # Initial state
    initial = snapshots[0]
    print(f"🏁 Initial State:")
    print(f"   Nodes: {initial['total_nodes']}")
    print(f"   Edges: {initial['total_edges']}")
    
    # Peak state
    peak = snapshots[22]  # Around April 2027
    print(f"\n📈 Peak Complexity:")
    print(f"   Nodes: {peak['total_nodes']}")
    print(f"   Edges: {peak['total_edges']}")
    print(f"   Time: {peak['snapshot_time']}")
    
    # Final state
    final = snapshots[-1]
    print(f"\n🏁 Final State:")
    print(f"   Nodes: {final['total_nodes']}")
    print(f"   Edges: {final['total_edges']}")
    
    # Growth phases
    print(f"\n🌱 Growth Phases:")
    print(f"   Phase 1 (0-12 months): Rapid expansion")
    print(f"   Phase 2 (12-24 months): Peak complexity")
    print(f"   Phase 3 (24+ months): Stabilization")
    
    # Events analysis
    total_events = sum(
        len(node['event_triggers']) 
        for snapshot in snapshots 
        for node in snapshot['nodes']
    )
    print(f"\n⚡ Events Analysis:")
    print(f"   Total events across all snapshots: {total_events}")
    
    # Wealth evolution
    wealth_values = [
        sum(node['financial_state'].get('total_wealth', 0) for node in s['nodes']) / len(s['nodes'])
        for s in snapshots if s['nodes']
    ]
    print(f"\n💰 Wealth Evolution:")
    print(f"   Initial wealth: ${wealth_values[0]:,.0f}")
    print(f"   Final wealth: ${wealth_values[-1]:,.0f}")
    print(f"   Growth factor: {wealth_values[-1]/wealth_values[0]:.1f}x")

def main():
    print("🎭 Creating Horatio Mesh Visualization...")
    
    # Create 2D evolution plots
    df = create_mesh_evolution_plots()
    print("✅ Created 2D evolution plots")
    
    # Create 3D sculpture plot
    create_3d_sculpture_plot()
    print("✅ Created 3D sculpture plot")
    
    # Create animation data
    animation_data = create_timeline_animation_data()
    
    # Print insights
    print_sculpture_insights()
    
    print("\n🎉 Visualization complete!")
    print("📊 Files created:")
    print("   - horatio_mesh_evolution.png (2D plots)")
    print("   - horatio_mesh_3d_sculpture.png (3D plot)")
    print("   - horatio_mesh_animation_data.json (animation data)")

if __name__ == "__main__":
    main() 