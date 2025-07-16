#!/usr/bin/env python
# Test Parallel Sets Visualization
# Tests the parallel sets visualizer integration with IPS system
# Author: ChatGPT 2025-01-16

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.parallel_sets_visualizer import ParallelSetsVisualizer
from src.dynamic_portfolio_engine import DynamicPortfolioEngine
from src.realistic_life_events_generator import RealisticLifeEventsGenerator

class TestParallelSetsVisualizer(unittest.TestCase):
    """Test the parallel sets visualizer functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.client_config = {
            'age': 45,
            'retirement_age': 65,
            'portfolio_value': 500000,
            'annual_contribution': 20000,
            'risk_tolerance': 0.6,
            'investment_horizon': 20,
            'income_stability': 0.8,
            'liquidity_needs': 0.3,
            'risk_profile': 3
        }
        
        # Create portfolio engine
        self.portfolio_engine = DynamicPortfolioEngine(self.client_config)
        
        # Create life events generator
        self.life_events_gen = RealisticLifeEventsGenerator("test_client", 2020)
        
        # Generate life events
        self.life_events = self.life_events_gen.generate_complete_life_journey()
        
        # Simulate portfolio evolution
        self.portfolio_engine.simulate_portfolio_evolution(self.life_events)
        
        # Create visualizer
        self.visualizer = ParallelSetsVisualizer(self.portfolio_engine)
    
    def test_parallel_sets_data_creation(self):
        """Test that parallel sets data is created correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        # Check that all required keys exist
        required_keys = ['life_events', 'portfolio_allocations', 'market_conditions', 
                        'comfort_levels', 'performance_outcomes']
        
        for key in required_keys:
            self.assertIn(key, sets_data)
            self.assertIsInstance(sets_data[key], list)
        
        # Check that we have data
        self.assertGreater(len(sets_data['life_events']), 0)
        self.assertGreater(len(sets_data['portfolio_allocations']), 0)
    
    def test_life_events_format(self):
        """Test that life events are formatted correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        for event in sets_data['life_events']:
            self.assertIn('category', event)
            self.assertIn('impact', event)
            self.assertIn('direction', event)
            
            # Check that impact is one of the expected values
            self.assertIn(event['impact'], ['High', 'Medium', 'Low'])
            
            # Check that direction is one of the expected values
            self.assertIn(event['direction'], ['Positive', 'Negative'])
    
    def test_portfolio_allocations_format(self):
        """Test that portfolio allocations are formatted correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        expected_allocations = ['High Equity', 'Balanced', 'Conservative']
        
        for allocation in sets_data['portfolio_allocations']:
            self.assertIn(allocation, expected_allocations)
    
    def test_market_conditions_format(self):
        """Test that market conditions are formatted correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        expected_conditions = ['High Stress', 'Moderate Stress', 'Low Stress']
        
        for condition in sets_data['market_conditions']:
            self.assertIn(condition, expected_conditions)
    
    def test_comfort_levels_format(self):
        """Test that comfort levels are formatted correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        expected_comfort = ['High Comfort', 'Moderate Comfort', 'Low Comfort']
        
        for comfort in sets_data['comfort_levels']:
            self.assertIn(comfort, expected_comfort)
    
    def test_performance_outcomes_format(self):
        """Test that performance outcomes are formatted correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        expected_performance = ['High Performance', 'Positive Performance', 'Negative Performance']
        
        for performance in sets_data['performance_outcomes']:
            self.assertIn(performance, expected_performance)
    
    def test_plotly_visualization_creation(self):
        """Test that Plotly visualization can be created"""
        fig = self.visualizer.create_parallel_sets_visualization()
        
        # Check that the figure has the expected structure
        self.assertIsNotNone(fig)
        self.assertIn('data', fig.to_dict())
        self.assertIn('layout', fig.to_dict())
    
    def test_html_generation(self):
        """Test that HTML file can be generated"""
        html_file = self.visualizer.save_parallel_sets_html("test_parallel_sets.html")
        
        # Check that file was created
        self.assertTrue(os.path.exists(html_file))
        
        # Check that file contains expected content
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('IPS Parallel Sets Analysis', content)
            self.assertIn('D3.js', content)
        
        # Clean up
        os.remove(html_file)
    
    def test_sample_data_creation(self):
        """Test that sample data can be created when no portfolio engine is provided"""
        visualizer_no_data = ParallelSetsVisualizer()
        sets_data = visualizer_no_data.create_parallel_sets_data()
        
        # Check that sample data has the expected structure
        required_keys = ['life_events', 'portfolio_allocations', 'market_conditions', 
                        'comfort_levels', 'performance_outcomes']
        
        for key in required_keys:
            self.assertIn(key, sets_data)
            self.assertIsInstance(sets_data[key], list)
            self.assertGreater(len(sets_data[key]), 0)
    
    def test_data_relationships(self):
        """Test that data relationships are consistent"""
        sets_data = self.visualizer.create_parallel_sets_data()
        
        # Check that all arrays have the same length
        lengths = [len(sets_data[key]) for key in ['life_events', 'portfolio_allocations', 
                                                  'market_conditions', 'comfort_levels', 
                                                  'performance_outcomes']]
        
        # All arrays should have the same length (one entry per life event)
        self.assertTrue(all(length == lengths[0] for length in lengths))
    
    def test_node_labels_generation(self):
        """Test that node labels are generated correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        labels = self.visualizer._get_node_labels(sets_data)
        
        # Check that labels are generated
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 0)
        
        # Check that labels contain expected content
        for label in labels:
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)
    
    def test_node_colors_generation(self):
        """Test that node colors are generated correctly"""
        sets_data = self.visualizer.create_parallel_sets_data()
        colors = self.visualizer._get_node_colors(sets_data)
        
        # Check that colors are generated
        self.assertIsInstance(colors, list)
        self.assertGreater(len(colors), 0)
        
        # Check that colors are valid
        for color in colors:
            self.assertIsInstance(color, str)
            self.assertGreater(len(color), 0)

def run_parallel_sets_tests():
    """Run all parallel sets tests"""
    print("🧪 Running Parallel Sets Visualization Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParallelSetsVisualizer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   - Tests run: {result.testsRun}")
    print(f"   - Failures: {len(result.failures)}")
    print(f"   - Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        for failure in result.failures:
            print(f"   - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"   - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_parallel_sets_tests() 