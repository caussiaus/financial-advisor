"""
Corpus Manager for Bank Statement Data

Monitors statement directories, extracts data, and prepares it for mesh training.
Handles both real and synthetic statements with unified processing.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import torch
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorpusManager:
    """Manages bank statement corpus for training data"""
    
    def __init__(self):
        # Setup directories
        self.real_dir = Path("data/statements/real")
        self.synthetic_dir = Path("data/statements/synthetic")
        self.cache_dir = Path("data/cache/statements")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_synthetic_statements(self) -> Dict:
        """Process all synthetic statements"""
        results = {
            "processed_files": [],
            "total_transactions": 0,
            "errors": []
        }
        
        # Find all JSON files (already processed by synthetic generator)
        json_files = list(self.synthetic_dir.glob("json/*.json"))
        
        if not json_files:
            logger.info("No synthetic JSON statements found")
            return results
        
        for json_file in json_files:
            logger.info(f"Processing synthetic statement: {json_file.name}")
            
            try:
                # Load synthetic data
                with open(json_file, 'r') as f:
                    transactions = json.load(f)
                
                results["processed_files"].append(str(json_file))
                results["total_transactions"] += len(transactions)
                
                logger.info(f"Loaded {len(transactions)} transactions from {json_file.name}")
                
            except Exception as e:
                error_msg = f"Failed to process {json_file.name}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    def vectorize_transactions(self, transactions: List[Dict]) -> Dict:
        """Convert transactions to vector format for training"""
        
        if not transactions:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Extract features
        amounts = df['amount'].to_numpy()
        dates = pd.to_datetime(df['date']).to_numpy()
        categories = df.get('category', pd.Series(['unknown'] * len(df))).to_numpy()
        types = df.get('type', pd.Series(['unknown'] * len(df))).to_numpy()
        
        # Convert to tensors
        tensors = {
            "amounts": torch.tensor(amounts, dtype=torch.float32),
            "dates": torch.tensor(pd.to_datetime(dates).astype(np.int64), dtype=torch.long),
            "categories": categories,  # Keep as strings for now
            "types": types
        }
        
        return tensors
    
    def cache_vectorized_data(self, transactions: List[Dict], filename: str):
        """Cache vectorized transaction data"""
        
        if not transactions:
            return
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(transactions)
        
        # Extract features as lists (avoiding NumPy/PyTorch compatibility issues)
        amounts = df['amount'].tolist()
        dates = df['date'].tolist()
        
        # Handle missing columns
        if 'category' in df.columns:
            categories = df['category'].tolist()
        else:
            categories = ['unknown'] * len(df)
            
        if 'type' in df.columns:
            types = df['type'].tolist()
        else:
            types = ['unknown'] * len(df)
        
        # Save as JSON (simpler format)
        vectorized_data = {
            "filename": filename,
            "num_transactions": len(transactions),
            "amounts": amounts,
            "dates": dates,
            "categories": categories,
            "types": types,
            "vectorization_date": datetime.now().isoformat()
        }
        
        # Save vectorized data
        with open(f"{self.cache_dir}/{filename}_vectorized.json", 'w') as f:
            json.dump(vectorized_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "filename": filename,
            "num_transactions": len(transactions),
            "categories": list(set(categories)),
            "types": list(set(types)),
            "vectorization_date": datetime.now().isoformat()
        }
        
        with open(f"{self.cache_dir}/{filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Cached vectorized data for {filename}")
    
    def process_corpus(self) -> Dict:
        """Process entire corpus (synthetic for now)"""
        
        logger.info("Starting corpus processing...")
        
        results = {
            "synthetic_statements": self.process_synthetic_statements(),
            "vectorized_files": [],
            "total_processed": 0
        }
        
        # Process synthetic files
        all_files = results["synthetic_statements"]["processed_files"]
        
        # Vectorize each file
        for file_path in all_files:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        transactions = json.load(f)
                    
                    # Create cache filename
                    filename = Path(file_path).stem
                    self.cache_vectorized_data(transactions, filename)
                    
                    results["vectorized_files"].append(filename)
                    results["total_processed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to vectorize {file_path}: {e}")
        
        # Save corpus summary
        summary = {
            "processing_date": datetime.now().isoformat(),
            "synthetic_files": len(results["synthetic_statements"]["processed_files"]),
            "total_transactions": results["synthetic_statements"]["total_transactions"],
            "vectorized_files": len(results["vectorized_files"]),
            "errors": results["synthetic_statements"]["errors"]
        }
        
        summary_path = self.cache_dir / "corpus_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Corpus processing complete!")
        logger.info(f"Synthetic files: {summary['synthetic_files']}")
        logger.info(f"Total transactions: {summary['total_transactions']}")
        logger.info(f"Vectorized files: {summary['vectorized_files']}")
        
        return results


def main():
    """Process the entire corpus"""
    corpus_manager = CorpusManager()
    
    # Process corpus
    results = corpus_manager.process_corpus()
    
    print("Corpus processing complete!")
    print(f"Synthetic files processed: {len(results['synthetic_statements']['processed_files'])}")
    print(f"Total vectorized files: {len(results['vectorized_files'])}")


if __name__ == "__main__":
    main() 