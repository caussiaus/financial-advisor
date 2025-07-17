"""
DL-friendly file storage format for efficient serialization of deep learning data types.
Handles numpy arrays, tensors, and other non-JSON-serializable objects.
"""
import numpy as np
import json
import pickle
import h5py
import os
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import torch
from pathlib import Path
import zlib
import base64


class DLFriendlyStorage:
    """
    Deep Learning friendly storage system that handles:
    - numpy arrays (float32, float64, int32, int64)
    - PyTorch tensors
    - complex nested structures
    - efficient compression for large datasets
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.supported_types = {
            np.ndarray, np.float32, np.float64, np.int32, np.int64,
            torch.Tensor, datetime, complex
        }
    
    def serialize(self, data: Any, format_type: str = 'json') -> Union[str, bytes]:
        """
        Serialize data to DL-friendly format
        
        Args:
            data: Data to serialize
            format_type: 'json', 'pickle', 'h5', or 'compressed'
        
        Returns:
            Serialized data
        """
        if format_type == 'json':
            return self._serialize_to_json(data)
        elif format_type == 'pickle':
            return self._serialize_to_pickle(data)
        elif format_type == 'h5':
            return self._serialize_to_h5(data)
        elif format_type == 'compressed':
            return self._serialize_compressed(data)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def deserialize(self, serialized_data: Union[str, bytes], format_type: str = 'json') -> Any:
        """
        Deserialize data from DL-friendly format
        
        Args:
            serialized_data: Serialized data
            format_type: 'json', 'pickle', 'h5', or 'compressed'
        
        Returns:
            Deserialized data
        """
        if format_type == 'json':
            return self._deserialize_from_json(serialized_data)
        elif format_type == 'pickle':
            return self._deserialize_from_pickle(serialized_data)
        elif format_type == 'h5':
            return self._deserialize_from_h5(serialized_data)
        elif format_type == 'compressed':
            return self._deserialize_compressed(serialized_data)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _serialize_to_json(self, data: Any) -> str:
        """Convert data to JSON-serializable format"""
        return json.dumps(self._convert_to_json_serializable(data), indent=2)
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert any object to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'numpy_array',
                'dtype': str(obj.dtype),
                'shape': obj.shape,
                'data': obj.tolist()
            }
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return {
                '__type__': 'torch_tensor',
                'dtype': str(obj.dtype),
                'shape': list(obj.shape),
                'data': obj.detach().cpu().numpy().tolist()
            }
        elif isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                'iso': obj.isoformat()
            }
        elif isinstance(obj, complex):
            return {
                '__type__': 'complex',
                'real': obj.real,
                'imag': obj.imag
            }
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _deserialize_from_json(self, json_str: str) -> Any:
        """Convert JSON back to original format"""
        data = json.loads(json_str)
        return self._convert_from_json_serializable(data)
    
    def _convert_from_json_serializable(self, obj: Any) -> Any:
        """Convert JSON-serializable format back to original objects"""
        if isinstance(obj, dict):
            if '__type__' in obj:
                if obj['__type__'] == 'numpy_array':
                    return np.array(obj['data'], dtype=np.dtype(obj['dtype']))
                elif obj['__type__'] == 'torch_tensor':
                    return torch.tensor(obj['data'], dtype=getattr(torch, obj['dtype'].split('.')[-1]))
                elif obj['__type__'] == 'datetime':
                    return datetime.fromisoformat(obj['iso'])
                elif obj['__type__'] == 'complex':
                    return complex(obj['real'], obj['imag'])
            else:
                return {k: self._convert_from_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_from_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _serialize_to_pickle(self, data: Any) -> bytes:
        """Serialize using pickle for complex objects"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_from_pickle(self, pickle_data: bytes) -> Any:
        """Deserialize from pickle"""
        return pickle.loads(pickle_data)
    
    def _serialize_to_h5(self, data: Dict[str, Any]) -> bytes:
        """Serialize to HDF5 format for large datasets"""
        import io
        buffer = io.BytesIO()
        
        with h5py.File(buffer, 'w') as f:
            self._write_dict_to_h5(f, data)
        
        return buffer.getvalue()
    
    def _write_dict_to_h5(self, group, data: Dict[str, Any]):
        """Recursively write dictionary to HDF5"""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._write_dict_to_h5(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip', compression_opts=self.compression_level)
            elif isinstance(value, torch.Tensor):
                group.create_dataset(key, data=value.detach().cpu().numpy(), 
                                  compression='gzip', compression_opts=self.compression_level)
            else:
                group.attrs[key] = value
    
    def _deserialize_from_h5(self, h5_data: bytes) -> Dict[str, Any]:
        """Deserialize from HDF5 format"""
        import io
        buffer = io.BytesIO(h5_data)
        
        with h5py.File(buffer, 'r') as f:
            return self._read_dict_from_h5(f)
    
    def _read_dict_from_h5(self, group) -> Dict[str, Any]:
        """Recursively read dictionary from HDF5"""
        result = {}
        
        # Read datasets
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                result[key] = self._read_dict_from_h5(group[key])
            else:
                result[key] = group[key][:]
        
        # Read attributes
        for key, value in group.attrs.items():
            result[key] = value
        
        return result
    
    def _serialize_compressed(self, data: Any) -> str:
        """Serialize with compression for large datasets"""
        # First convert to JSON-serializable
        json_data = self._convert_to_json_serializable(data)
        json_str = json.dumps(json_data)
        
        # Compress
        compressed = zlib.compress(json_str.encode('utf-8'), level=self.compression_level)
        
        # Base64 encode for safe storage
        return base64.b64encode(compressed).decode('utf-8')
    
    def _deserialize_compressed(self, compressed_data: str) -> Any:
        """Deserialize compressed data"""
        # Base64 decode
        compressed = base64.b64decode(compressed_data.encode('utf-8'))
        
        # Decompress
        json_str = zlib.decompress(compressed).decode('utf-8')
        
        # Parse JSON and convert back
        json_data = json.loads(json_str)
        return self._convert_from_json_serializable(json_data)
    
    def save_to_file(self, data: Any, filepath: str, format_type: str = 'auto'):
        """
        Save data to file with appropriate format
        
        Args:
            data: Data to save
            filepath: Output file path
            format_type: 'auto', 'json', 'pickle', 'h5', or 'compressed'
        """
        if format_type == 'auto':
            format_type = self._detect_best_format(data, filepath)
        
        if format_type == 'json':
            serialized = self._serialize_to_json(data)
            with open(filepath, 'w') as f:
                f.write(serialized)
        elif format_type == 'pickle':
            serialized = self._serialize_to_pickle(data)
            with open(filepath, 'wb') as f:
                f.write(serialized)
        elif format_type == 'h5':
            serialized = self._serialize_to_h5(data)
            with open(filepath, 'wb') as f:
                f.write(serialized)
        elif format_type == 'compressed':
            serialized = self._serialize_compressed(data)
            with open(filepath, 'w') as f:
                f.write(serialized)
    
    def load_from_file(self, filepath: str, format_type: str = 'auto') -> Any:
        """
        Load data from file
        
        Args:
            filepath: Input file path
            format_type: 'auto', 'json', 'pickle', 'h5', or 'compressed'
        
        Returns:
            Loaded data
        """
        if format_type == 'auto':
            format_type = self._detect_format_from_extension(filepath)
        
        if format_type == 'json':
            with open(filepath, 'r') as f:
                return self._deserialize_from_json(f.read())
        elif format_type == 'pickle':
            with open(filepath, 'rb') as f:
                return self._deserialize_from_pickle(f.read())
        elif format_type == 'h5':
            with open(filepath, 'rb') as f:
                return self._deserialize_from_h5(f.read())
        elif format_type == 'compressed':
            with open(filepath, 'r') as f:
                return self._deserialize_compressed(f.read())
    
    def _detect_best_format(self, data: Any, filepath: str) -> str:
        """Detect best format based on data type and size"""
        # Check if data contains numpy arrays or tensors
        has_numpy = self._contains_numpy_arrays(data)
        has_tensors = self._contains_tensors(data)
        
        if has_numpy or has_tensors:
            if self._estimate_size(data) > 100 * 1024 * 1024:  # > 100MB
                return 'h5'
            else:
                return 'compressed'
        else:
            return 'json'
    
    def _detect_format_from_extension(self, filepath: str) -> str:
        """Detect format from file extension"""
        ext = Path(filepath).suffix.lower()
        if ext == '.json':
            return 'json'
        elif ext == '.pkl' or ext == '.pickle':
            return 'pickle'
        elif ext == '.h5' or ext == '.hdf5':
            return 'h5'
        elif ext == '.compressed':
            return 'compressed'
        else:
            return 'json'  # Default
    
    def _contains_numpy_arrays(self, obj: Any) -> bool:
        """Check if object contains numpy arrays"""
        if isinstance(obj, np.ndarray):
            return True
        elif isinstance(obj, dict):
            return any(self._contains_numpy_arrays(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_numpy_arrays(item) for item in obj)
        return False
    
    def _contains_tensors(self, obj: Any) -> bool:
        """Check if object contains PyTorch tensors"""
        if isinstance(obj, torch.Tensor):
            return True
        elif isinstance(obj, dict):
            return any(self._contains_tensors(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_tensors(item) for item in obj)
        return False
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes"""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, dict):
            return sum(self._estimate_size(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        else:
            return len(str(obj).encode('utf-8'))


# Convenience functions for common use cases
def save_mesh_data(mesh_data: Dict[str, Any], filepath: str):
    """Save mesh data with DL-friendly format"""
    storage = DLFriendlyStorage()
    storage.save_to_file(mesh_data, filepath, 'auto')

def load_mesh_data(filepath: str) -> Dict[str, Any]:
    """Load mesh data with DL-friendly format"""
    storage = DLFriendlyStorage()
    return storage.load_from_file(filepath, 'auto')

def save_analysis_results(results: Dict[str, Any], filepath: str):
    """Save analysis results with DL-friendly format"""
    storage = DLFriendlyStorage()
    storage.save_to_file(results, filepath, 'auto')

def load_analysis_results(filepath: str) -> Dict[str, Any]:
    """Load analysis results with DL-friendly format"""
    storage = DLFriendlyStorage()
    return storage.load_from_file(filepath, 'auto')

def convert_existing_json_to_dl_friendly(input_file: str, output_file: str):
    """Convert existing JSON files to DL-friendly format"""
    storage = DLFriendlyStorage()
    
    # Load existing JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Save in DL-friendly format
    storage.save_to_file(data, output_file, 'auto')
    
    print(f"✅ Converted {input_file} to DL-friendly format: {output_file}") 