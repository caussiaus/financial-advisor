"""
Spending Pattern Vector Database
Stores spending patterns as vectors and enables similarity search for pattern matching
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import chromadb
from chromadb.config import Settings
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpendingPatternVectorizer:
    """Converts spending patterns to high-dimensional vectors"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = MinMaxScaler()
        self.pca = PCA(n_components=50)  # Reduce dimensionality while preserving variance
        self.feature_names = []
        self.is_fitted = False
        
    def create_feature_vector(self, pattern: Dict) -> np.ndarray:
        """Convert spending pattern to feature vector"""
        
        # Define features to extract
        features = []
        feature_names = []
        
        # Basic demographics (normalized)
        features.extend([
            pattern.get('age', 0) / 100.0,  # Normalize age
            pattern.get('income', 0) / 200000.0,  # Normalize income
            pattern.get('household_size', 1) / 6.0,  # Normalize household size
        ])
        feature_names.extend(['age_norm', 'income_norm', 'household_size_norm'])
        
        # Education level (one-hot encoded)
        education_levels = ['High School', 'Some College', 'Bachelor', 'Graduate']
        education = pattern.get('education_level', 'High School')
        education_vector = [1.0 if education == level else 0.0 for level in education_levels]
        features.extend(education_vector)
        feature_names.extend([f'education_{level.replace(" ", "_")}' for level in education_levels])
        
        # Location (one-hot encoded)
        locations = ['Urban', 'Suburban', 'Rural']
        location = pattern.get('location', 'Suburban')
        location_vector = [1.0 if location == loc else 0.0 for loc in locations]
        features.extend(location_vector)
        feature_names.extend([f'location_{loc}' for loc in locations])
        
        # Marital status (one-hot encoded)
        marital_statuses = ['Single', 'Married', 'Divorced']
        marital = pattern.get('marital_status', 'Single')
        marital_vector = [1.0 if marital == status else 0.0 for status in marital_statuses]
        features.extend(marital_vector)
        feature_names.extend([f'marital_{status}' for status in marital_statuses])
        
        # Spending amounts (normalized by income)
        income = max(pattern.get('income', 1), 1)  # Avoid division by zero
        spending_categories = [
            'housing_cost', 'transportation_cost', 'food_cost', 'healthcare_cost',
            'insurance_cost', 'utilities_cost', 'entertainment_cost', 'dining_out_cost',
            'travel_cost', 'hobbies_cost', 'luxury_goods_cost'
        ]
        
        for category in spending_categories:
            monthly_amount = pattern.get(category, 0)
            annual_amount = monthly_amount * 12
            income_ratio = annual_amount / income
            features.append(income_ratio)
            feature_names.append(f'{category}_ratio')
        
        # Savings and investments (normalized by income)
        savings_categories = ['emergency_savings', 'retirement_savings', 'investment_amount']
        for category in savings_categories:
            amount = pattern.get(category, 0)
            income_ratio = amount / income
            features.append(income_ratio)
            feature_names.append(f'{category}_ratio')
        
        # Calculated ratios
        features.extend([
            pattern.get('discretionary_ratio', 0),
            pattern.get('total_nondiscretionary', 0) / income,
            pattern.get('total_discretionary', 0) / income
        ])
        feature_names.extend(['discretionary_ratio', 'nondiscretionary_ratio', 'total_discretionary_ratio'])
        
        # Milestone indicators
        features.extend([
            1.0 if pattern.get('owns_home', False) else 0.0,
            1.0 if pattern.get('married', False) else 0.0,
            1.0 if pattern.get('has_children', False) else 0.0
        ])
        feature_names.extend(['owns_home', 'married', 'has_children'])
        
        # Milestone timing (normalized by age)
        current_age = pattern.get('age', 30)
        milestone_ages = ['home_purchase_age', 'marriage_age', 'first_child_age']
        for milestone in milestone_ages:
            age = pattern.get(milestone, 0)
            if age and age > 0:
                features.append(age / current_age)
            else:
                features.append(0.0)
            feature_names.append(f'{milestone}_ratio')
        
        if not self.feature_names:
            self.feature_names = feature_names
            
        return np.array(features, dtype=np.float32)
    
    def fit_transform(self, patterns: List[Dict]) -> np.ndarray:
        """Fit vectorizer and transform patterns to vectors"""
        
        # Create feature matrix
        feature_matrix = np.array([self.create_feature_vector(pattern) for pattern in patterns])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Fit and transform
        scaled_features = self.scaler.fit_transform(feature_matrix)
        normalized_features = self.normalizer.fit_transform(scaled_features)
        reduced_features = self.pca.fit_transform(normalized_features)
        
        self.is_fitted = True
        logger.info(f"Fitted vectorizer with {feature_matrix.shape[0]} patterns, {feature_matrix.shape[1]} features")
        logger.info(f"Reduced to {reduced_features.shape[1]} dimensions with PCA")
        
        return reduced_features
    
    def transform(self, patterns: List[Dict]) -> np.ndarray:
        """Transform patterns to vectors using fitted vectorizer"""
        
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        feature_matrix = np.array([self.create_feature_vector(pattern) for pattern in patterns])
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        scaled_features = self.scaler.transform(feature_matrix)
        normalized_features = self.normalizer.transform(scaled_features)
        reduced_features = self.pca.transform(normalized_features)
        
        return reduced_features
    
    def save(self, path: str):
        """Save fitted vectorizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'normalizer': self.normalizer,
                'pca': self.pca,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Vectorizer saved to {path}")
    
    def load(self, path: str):
        """Load fitted vectorizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.normalizer = data['normalizer']
        self.pca = data['pca']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Vectorizer loaded from {path}")

class SpendingPatternVectorDB:
    """Vector database for storing and querying spending patterns"""
    
    def __init__(self, db_path: str = "data/spending_vectors", spending_db_path: str = "data/spending_patterns.db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.spending_db_path = spending_db_path
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections
        self.spending_collection = self.client.get_or_create_collection(
            name="spending_patterns",
            metadata={"description": "Spending pattern vectors for similarity search"}
        )
        
        self.milestone_collection = self.client.get_or_create_collection(
            name="milestone_patterns", 
            metadata={"description": "Milestone timing pattern vectors"}
        )
        
        self.vectorizer = SpendingPatternVectorizer()
        self.vectorizer_path = self.db_path / "vectorizer.pkl"
        
        # Try to load existing vectorizer
        if self.vectorizer_path.exists():
            try:
                self.vectorizer.load(str(self.vectorizer_path))
                logger.info("Loaded existing vectorizer")
            except Exception as e:
                logger.warning(f"Could not load vectorizer: {e}")
    
    def load_spending_patterns(self) -> List[Dict]:
        """Load spending patterns from SQLite database"""
        
        conn = sqlite3.connect(self.spending_db_path)
        
        query = """
        SELECT * FROM spending_patterns 
        WHERE age IS NOT NULL AND income IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert to list of dictionaries
        patterns = df.to_dict('records')
        logger.info(f"Loaded {len(patterns)} spending patterns")
        
        return patterns
    
    def vectorize_and_store_patterns(self):
        """Vectorize spending patterns and store in vector database"""
        
        # Load patterns from SQLite
        patterns = self.load_spending_patterns()
        
        if not patterns:
            logger.warning("No patterns found to vectorize")
            return
        
        # Vectorize patterns
        if not self.vectorizer.is_fitted:
            vectors = self.vectorizer.fit_transform(patterns)
            self.vectorizer.save(str(self.vectorizer_path))
        else:
            vectors = self.vectorizer.transform(patterns)
        
        # Clear existing collection
        self.spending_collection.delete()
        self.spending_collection = self.client.get_or_create_collection(
            name="spending_patterns",
            metadata={"description": "Spending pattern vectors for similarity search"}
        )
        
        # Prepare data for ChromaDB
        ids = [str(i) for i in range(len(patterns))]
        embeddings = vectors.tolist()
        
        # Create metadata for each pattern
        metadatas = []
        documents = []
        
        for i, pattern in enumerate(patterns):
            metadata = {
                'age': pattern.get('age', 0),
                'income': pattern.get('income', 0),
                'education_level': pattern.get('education_level', ''),
                'location': pattern.get('location', ''),
                'marital_status': pattern.get('marital_status', ''),
                'discretionary_ratio': pattern.get('discretionary_ratio', 0.0),
                'owns_home': pattern.get('owns_home', False),
                'married': pattern.get('married', False),
                'has_children': pattern.get('has_children', False),
                'source': pattern.get('source', ''),
                'pattern_id': i
            }
            
            # Create document text for search
            document = f"Age: {pattern.get('age')}, Income: {pattern.get('income')}, " \
                      f"Education: {pattern.get('education_level')}, Location: {pattern.get('location')}, " \
                      f"Discretionary Ratio: {pattern.get('discretionary_ratio', 0):.2f}"
            
            metadatas.append(metadata)
            documents.append(document)
        
        # Add to collection
        self.spending_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Stored {len(patterns)} vectorized patterns in database")
    
    def find_similar_patterns(self, target_pattern: Dict, n_results: int = 10) -> List[Dict]:
        """Find similar spending patterns"""
        
        # Vectorize target pattern
        target_vector = self.vectorizer.transform([target_pattern])
        
        # Query similar patterns
        results = self.spending_collection.query(
            query_embeddings=target_vector.tolist(),
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        similar_patterns = []
        for i in range(len(results['ids'][0])):
            similar_patterns.append({
                'id': results['ids'][0][i],
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i],
                'similarity_score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'distance': results['distances'][0][i]
            })
        
        return similar_patterns
    
    def find_patterns_by_criteria(self, 
                                  age_range: Optional[Tuple[int, int]] = None,
                                  income_range: Optional[Tuple[int, int]] = None,
                                  education_level: Optional[str] = None,
                                  location: Optional[str] = None,
                                  has_milestone: Optional[str] = None,
                                  n_results: int = 50) -> List[Dict]:
        """Find patterns matching specific criteria"""
        
        # Build where clause
        where_conditions = {}
        
        if age_range:
            where_conditions["age"] = {"$gte": age_range[0], "$lte": age_range[1]}
        
        if income_range:
            where_conditions["income"] = {"$gte": income_range[0], "$lte": income_range[1]}
        
        if education_level:
            where_conditions["education_level"] = education_level
        
        if location:
            where_conditions["location"] = location
        
        if has_milestone:
            where_conditions[has_milestone] = True
        
        # Query with filters
        results = self.spending_collection.get(
            where=where_conditions if where_conditions else None,
            limit=n_results,
            include=['metadatas', 'documents']
        )
        
        patterns = []
        for i in range(len(results['ids'])):
            patterns.append({
                'id': results['ids'][i],
                'metadata': results['metadatas'][i],
                'document': results['documents'][i]
            })
        
        return patterns
    
    def analyze_spending_surface(self, 
                                income_points: List[int],
                                age_points: List[int],
                                milestone: str) -> Dict:
        """Analyze spending surface for milestone achievement across income/age grid"""
        
        surface_data = {}
        
        for income in income_points:
            surface_data[income] = {}
            
            for age in age_points:
                # Find patterns in this income/age cell
                patterns = self.find_patterns_by_criteria(
                    age_range=(age-2, age+2),
                    income_range=(income-5000, income+5000),
                    has_milestone=milestone,
                    n_results=20
                )
                
                if patterns:
                    # Calculate achievement statistics
                    achieved_count = len(patterns)
                    
                    # Get all patterns in this cell (achieved + not achieved)
                    all_patterns = self.find_patterns_by_criteria(
                        age_range=(age-2, age+2),
                        income_range=(income-5000, income+5000),
                        n_results=50
                    )
                    
                    total_count = len(all_patterns)
                    achievement_rate = achieved_count / total_count if total_count > 0 else 0
                    
                    # Calculate average discretionary spending
                    avg_discretionary = np.mean([p['metadata']['discretionary_ratio'] for p in patterns])
                    
                    surface_data[income][age] = {
                        'achievement_rate': achievement_rate,
                        'achieved_count': achieved_count,
                        'total_count': total_count,
                        'avg_discretionary_ratio': avg_discretionary,
                        'sample_size': len(patterns)
                    }
                else:
                    surface_data[income][age] = {
                        'achievement_rate': 0.0,
                        'achieved_count': 0,
                        'total_count': 0,
                        'avg_discretionary_ratio': 0.0,
                        'sample_size': 0
                    }
        
        return surface_data
    
    def predict_milestone_timing(self, client_pattern: Dict, milestone: str) -> Dict:
        """Predict milestone timing for a client based on similar patterns"""
        
        # Find similar patterns
        similar_patterns = self.find_similar_patterns(client_pattern, n_results=20)
        
        # Filter for patterns that achieved the milestone
        milestone_field = f"{milestone.replace('_', '')}"  # e.g., owns_home, married, has_children
        
        achieved_patterns = [
            p for p in similar_patterns 
            if p['metadata'].get(milestone_field, False)
        ]
        
        if not achieved_patterns:
            return {
                'predicted_age': None,
                'confidence': 0.0,
                'sample_size': 0,
                'similar_patterns': similar_patterns[:5]
            }
        
        # Load original patterns to get milestone ages
        original_patterns = self.load_spending_patterns()
        
        milestone_ages = []
        weights = []
        
        for pattern in achieved_patterns:
            pattern_id = pattern['metadata']['pattern_id']
            if pattern_id < len(original_patterns):
                original = original_patterns[pattern_id]
                milestone_age_field = f"{milestone}_age"
                
                if milestone_age_field in original and original[milestone_age_field]:
                    milestone_ages.append(original[milestone_age_field])
                    weights.append(pattern['similarity_score'])
        
        if not milestone_ages:
            return {
                'predicted_age': None,
                'confidence': 0.0,
                'sample_size': 0,
                'similar_patterns': similar_patterns[:5]
            }
        
        # Calculate weighted average
        milestone_ages = np.array(milestone_ages)
        weights = np.array(weights)
        
        predicted_age = np.average(milestone_ages, weights=weights)
        confidence = np.mean(weights)  # Average similarity as confidence
        
        return {
            'predicted_age': float(predicted_age),
            'confidence': float(confidence),
            'sample_size': len(milestone_ages),
            'age_range': (float(np.min(milestone_ages)), float(np.max(milestone_ages))),
            'median_age': float(np.median(milestone_ages)),
            'similar_patterns': similar_patterns[:5]
        }
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database"""
        
        spending_count = self.spending_collection.count()
        milestone_count = self.milestone_collection.count()
        
        return {
            'spending_patterns_count': spending_count,
            'milestone_patterns_count': milestone_count,
            'vectorizer_fitted': self.vectorizer.is_fitted,
            'feature_count': len(self.vectorizer.feature_names) if self.vectorizer.feature_names else 0
        }

# Usage example
def main():
    # Initialize vector database
    vector_db = SpendingPatternVectorDB()
    
    # Vectorize and store patterns
    vector_db.vectorize_and_store_patterns()
    
    # Example client pattern
    client_pattern = {
        'age': 28,
        'income': 75000,
        'education_level': 'Bachelor',
        'location': 'Urban',
        'marital_status': 'Single',
        'household_size': 1,
        'housing_cost': 1800,
        'transportation_cost': 400,
        'food_cost': 500,
        'healthcare_cost': 200,
        'entertainment_cost': 300,
        'dining_out_cost': 250,
        'discretionary_ratio': 0.15,
        'owns_home': False,
        'married': False,
        'has_children': False
    }
    
    # Find similar patterns
    similar = vector_db.find_similar_patterns(client_pattern, n_results=5)
    print(f"Found {len(similar)} similar patterns")
    
    # Predict milestone timing
    home_prediction = vector_db.predict_milestone_timing(client_pattern, 'home_purchase')
    print(f"Home purchase prediction: {home_prediction}")
    
    # Get database stats
    stats = vector_db.get_database_stats()
    print(f"Database stats: {stats}")

if __name__ == "__main__":
    main() 