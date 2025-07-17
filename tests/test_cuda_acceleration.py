import pytest
import numpy as np
try:
    import cupy as cp
    import numba
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from src.stochastic_mesh_engine import StochasticMeshEngine
from src.adaptive_mesh_generator import AdaptiveMeshGenerator

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCUDAAcceleration:
    
    def setup_method(self):
        """Set up test environment"""
        self.initial_state = {
            'total_wealth': 1000000,
            'cash': 200000,
            'investments': 800000
        }
        self.mesh_engine = StochasticMeshEngine(self.initial_state)
        
    def test_cuda_detection(self):
        """Test CUDA availability detection"""
        assert self.mesh_engine.use_cuda == CUDA_AVAILABLE
        
    def test_path_generation_cuda(self):
        """Test CUDA-accelerated path generation"""
        num_paths = 1000
        num_steps = 120  # 10 years monthly
        initial_value = 1000000.0
        
        # Generate paths using CUDA
        paths_cuda = self.mesh_engine._generate_paths_cuda(
            num_paths, num_steps, initial_value,
            drift=0.07, volatility=0.15, dt=1.0/12.0
        )
        
        assert isinstance(paths_cuda, np.ndarray)
        assert paths_cuda.shape == (num_paths, num_steps + 1)
        assert np.all(paths_cuda[:, 0] == initial_value)
        
    def test_state_processing_cuda(self):
        """Test CUDA-accelerated state processing"""
        num_states = 1000
        states = [
            {
                'total_wealth': 1000000.0,
                'cash': 200000.0,
                'investments': 800000.0
            }
            for _ in range(num_states)
        ]
        
        operations = [
            {
                'type': 'investment_growth',
                'return': 0.07,
                'horizon': 1
            }
        ]
        
        # Process states using CUDA
        results = self.mesh_engine._process_states_parallel(states, operations)
        
        assert len(results) == num_states
        assert all(isinstance(r, dict) for r in results)
        assert all('total_wealth' in r for r in results)
        
    def test_performance_comparison(self):
        """Compare CUDA vs CPU performance"""
        import time
        
        num_paths = 10000
        num_steps = 120
        initial_value = 1000000.0
        
        # Time CUDA execution
        start_cuda = time.time()
        _ = self.mesh_engine._generate_paths_cuda(
            num_paths, num_steps, initial_value,
            drift=0.07, volatility=0.15, dt=1.0/12.0
        )
        cuda_time = time.time() - start_cuda
        
        # Time CPU execution
        start_cpu = time.time()
        _ = self.mesh_engine._generate_paths_cpu(
            num_paths, num_steps, initial_value,
            drift=0.07, volatility=0.15, dt=1.0/12.0
        )
        cpu_time = time.time() - start_cpu
        
        # CUDA should be significantly faster for large workloads
        assert cuda_time < cpu_time, \
            f"CUDA ({cuda_time:.2f}s) should be faster than CPU ({cpu_time:.2f}s)"
        
    def test_memory_management(self):
        """Test GPU memory management"""
        if not CUDA_AVAILABLE:
            return
            
        import cupy as cp
        
        # Get initial memory usage
        initial_mem = cp.get_default_memory_pool().used_bytes()
        
        # Generate large paths
        _ = self.mesh_engine._generate_paths_cuda(
            10000, 120, 1000000.0,
            drift=0.07, volatility=0.15, dt=1.0/12.0
        )
        
        # Memory should be released after operation
        cp.get_default_memory_pool().free_all_blocks()
        final_mem = cp.get_default_memory_pool().used_bytes()
        
        assert final_mem <= initial_mem * 1.1  # Allow for small overhead
        
    def test_error_handling(self):
        """Test graceful fallback when CUDA fails"""
        if not CUDA_AVAILABLE:
            return
            
        # Simulate CUDA out of memory
        num_paths = 1000000000  # Unreasonably large
        num_steps = 1000000
        
        # Should fall back to CPU
        paths = self.mesh_engine._generate_paths_cuda(
            num_paths, num_steps, 1000000.0,
            drift=0.07, volatility=0.15, dt=1.0/12.0
        )
        
        assert isinstance(paths, np.ndarray)
        assert paths.shape[0] > 0  # Should still generate some paths 