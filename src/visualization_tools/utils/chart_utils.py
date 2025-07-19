#!/usr/bin/env python3
"""
Chart Utilities
Common utilities for chart creation and manipulation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartUtils:
    """Utility class for chart creation and manipulation"""
    
    def __init__(self):
        """Initialize chart utilities"""
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        logger.info("✅ Chart Utils initialized")
    
    def create_simple_bar_chart(self, data: Dict[str, float], title: str = "Bar Chart") -> go.Figure:
        """Create a simple bar chart"""
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(data.keys()),
                    y=list(data.values()),
                    marker_color=self.default_colors[:len(data)]
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Categories",
                yaxis_title="Values",
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple bar chart: {e}")
            return self._create_error_figure(title, str(e))
    
    def create_simple_line_chart(self, x_data: List, y_data: List, title: str = "Line Chart") -> go.Figure:
        """Create a simple line chart"""
        try:
            fig = go.Figure(data=[
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    line=dict(color=self.default_colors[0], width=2),
                    marker=dict(size=6)
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple line chart: {e}")
            return self._create_error_figure(title, str(e))
    
    def create_simple_pie_chart(self, data: Dict[str, float], title: str = "Pie Chart") -> go.Figure:
        """Create a simple pie chart"""
        try:
            fig = go.Figure(data=[go.Pie(
                labels=list(data.keys()),
                values=list(data.values()),
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                marker_colors=self.default_colors[:len(data)]
            )])
            
            fig.update_layout(
                title=title,
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple pie chart: {e}")
            return self._create_error_figure(title, str(e))
    
    def create_scatter_plot(self, x_data: List, y_data: List, title: str = "Scatter Plot") -> go.Figure:
        """Create a simple scatter plot"""
        try:
            fig = go.Figure(data=[
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.default_colors[0],
                        opacity=0.7
                    )
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return self._create_error_figure(title, str(e))
    
    def create_heatmap(self, data: List[List[float]], x_labels: List[str], 
                      y_labels: List[str], title: str = "Heatmap") -> go.Figure:
        """Create a heatmap"""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=x_labels,
                y=y_labels,
                colorscale='Viridis',
                text=data,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return self._create_error_figure(title, str(e))
    
    def create_subplot_grid(self, charts: List[go.Figure], rows: int, cols: int, 
                           title: str = "Subplot Grid") -> go.Figure:
        """Create a grid of subplots"""
        try:
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[chart.layout.title.text for chart in charts],
                specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
            )
            
            for i, chart in enumerate(charts):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                for trace in chart.data:
                    fig.add_trace(trace, row=row, col=col)
            
            fig.update_layout(
                title=title,
                height=400 * rows,
                width=600 * cols
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating subplot grid: {e}")
            return self._create_error_figure(title, str(e))
    
    def format_currency(self, value: float) -> str:
        """Format value as currency"""
        return f"${value:,.2f}"
    
    def format_percentage(self, value: float) -> str:
        """Format value as percentage"""
        return f"{value:.1f}%"
    
    def normalize_data(self, data: List[float]) -> List[float]:
        """Normalize data to 0-1 range"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return [0.5] * len(data)
        
        return [(x - min_val) / (max_val - min_val) for x in data]
    
    def calculate_percentiles(self, data: List[float], percentiles: List[int] = [25, 50, 75]) -> Dict[int, float]:
        """Calculate percentiles of data"""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        results = {}
        
        for p in percentiles:
            index = (p / 100) * (len(sorted_data) - 1)
            if index.is_integer():
                results[p] = sorted_data[int(index)]
            else:
                lower = sorted_data[int(index)]
                upper = sorted_data[int(index) + 1]
                results[p] = lower + (upper - lower) * (index - int(index))
        
        return results
    
    def _create_error_figure(self, title: str, error_message: str) -> go.Figure:
        """Create error figure when chart creation fails"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Error creating {title}: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title=f"Error: {title}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400
        )
        
        return fig

def main():
    """Test the chart utilities"""
    # Initialize chart utils
    utils = ChartUtils()
    
    # Test simple bar chart
    test_data = {
        "Category A": 100,
        "Category B": 150,
        "Category C": 200,
        "Category D": 75
    }
    
    bar_chart = utils.create_simple_bar_chart(test_data, "Test Bar Chart")
    bar_chart.write_html("test_bar_chart.html")
    print("✅ Bar chart created")
    
    # Test simple line chart
    x_data = [1, 2, 3, 4, 5]
    y_data = [10, 15, 13, 17, 20]
    
    line_chart = utils.create_simple_line_chart(x_data, y_data, "Test Line Chart")
    line_chart.write_html("test_line_chart.html")
    print("✅ Line chart created")
    
    # Test simple pie chart
    pie_chart = utils.create_simple_pie_chart(test_data, "Test Pie Chart")
    pie_chart.write_html("test_pie_chart.html")
    print("✅ Pie chart created")
    
    # Test scatter plot
    scatter_chart = utils.create_scatter_plot(x_data, y_data, "Test Scatter Plot")
    scatter_chart.write_html("test_scatter_chart.html")
    print("✅ Scatter plot created")
    
    # Test heatmap
    heatmap_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x_labels = ["A", "B", "C"]
    y_labels = ["X", "Y", "Z"]
    
    heatmap_chart = utils.create_heatmap(heatmap_data, x_labels, y_labels, "Test Heatmap")
    heatmap_chart.write_html("test_heatmap.html")
    print("✅ Heatmap created")
    
    print("✅ All chart utility tests completed")

if __name__ == "__main__":
    main() 