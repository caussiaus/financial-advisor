"""
Discretionary Spending Classifier
Classifies expenses as discretionary vs non-discretionary using ML and rule-based approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscretionarySpendingClassifier:
    """Classifies spending as discretionary vs non-discretionary"""
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Rule-based classification patterns
        self.nondiscretionary_keywords = {
            'housing': ['rent', 'mortgage', 'property tax', 'home insurance', 'hoa', 'utilities'],
            'transportation': ['car payment', 'auto loan', 'gas', 'insurance', 'registration', 'maintenance'],
            'food_essential': ['grocery', 'supermarket', 'food basics'],
            'healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'dental', 'vision'],
            'insurance': ['life insurance', 'health insurance', 'disability'],
            'debt': ['credit card minimum', 'student loan', 'personal loan'],
            'utilities': ['electric', 'water', 'sewer', 'trash', 'internet', 'phone'],
            'childcare': ['daycare', 'babysitter', 'school fees'],
            'taxes': ['income tax', 'property tax', 'sales tax']
        }
        
        self.discretionary_keywords = {
            'entertainment': ['movie', 'concert', 'streaming', 'games', 'sports'],
            'dining': ['restaurant', 'takeout', 'delivery', 'coffee shop', 'bar'],
            'travel': ['vacation', 'hotel', 'flight', 'travel', 'tourism'],
            'shopping': ['clothes', 'electronics', 'gadgets', 'jewelry', 'books'],
            'hobbies': ['gym', 'fitness', 'hobby', 'art supplies', 'music'],
            'luxury': ['spa', 'massage', 'luxury', 'premium', 'designer'],
            'subscriptions': ['magazine', 'premium services', 'memberships'],
            'gifts': ['gift', 'donation', 'charity (non-essential)']
        }
        
        # Expense categories and their typical discretionary ratios
        self.category_discretionary_ratios = {
            'housing': 0.0,
            'utilities': 0.0, 
            'insurance': 0.0,
            'healthcare': 0.1,  # Some healthcare is discretionary (cosmetic, etc.)
            'transportation': 0.2,  # Some transport is discretionary
            'food': 0.4,  # Mix of essential groceries and dining out
            'education': 0.3,  # Some education is discretionary
            'entertainment': 0.9,
            'dining_out': 0.9,
            'travel': 0.8,
            'shopping': 0.7,
            'personal_care': 0.6,
            'hobbies': 0.9,
            'gifts': 0.8,
            'miscellaneous': 0.5
        }
    
    def generate_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data for the classifier"""
        
        logger.info(f"Generating {n_samples} training samples")
        
        data = []
        labels = []
        
        # Generate samples based on keyword patterns and categories
        for _ in range(n_samples):
            # Random expense characteristics
            amount = np.random.lognormal(np.log(100), 1)  # Log-normal distribution for amounts
            frequency = np.random.choice(['daily', 'weekly', 'monthly', 'yearly'], p=[0.1, 0.2, 0.6, 0.1])
            
            # Choose category and generate description
            if np.random.random() < 0.5:  # 50% discretionary
                category = np.random.choice(list(self.discretionary_keywords.keys()))
                keywords = self.discretionary_keywords[category]
                description = np.random.choice(keywords)
                is_discretionary = 1
            else:  # 50% non-discretionary
                category = np.random.choice(list(self.nondiscretionary_keywords.keys()))
                keywords = self.nondiscretionary_keywords[category]
                description = np.random.choice(keywords)
                is_discretionary = 0
            
            # Add some variation to descriptions
            if np.random.random() < 0.3:
                description += f" {np.random.choice(['payment', 'bill', 'expense', 'cost'])}"
            
            # Income context (affects discretionary classification)
            income_bracket = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            income_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[income_bracket]
            
            # Life stage context
            life_stage = np.random.choice(['young_single', 'young_couple', 'family', 'empty_nest'], 
                                        p=[0.25, 0.25, 0.3, 0.2])
            
            data.append({
                'amount': amount,
                'description': description,
                'category': category,
                'frequency': frequency,
                'income_bracket': income_bracket,
                'life_stage': life_stage,
                'amount_income_ratio': amount / (50000 * income_multiplier)
            })
            
            labels.append(is_discretionary)
        
        df = pd.DataFrame(data)
        return df, np.array(labels)
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from expense data"""
        
        features = []
        
        for _, row in df.iterrows():
            feature_vector = []
            
            # Amount-based features
            feature_vector.append(row['amount'])
            feature_vector.append(np.log(row['amount'] + 1))  # Log amount
            feature_vector.append(row.get('amount_income_ratio', 0))
            
            # Frequency encoding
            freq_encoding = {'daily': 4, 'weekly': 3, 'monthly': 2, 'yearly': 1}
            feature_vector.append(freq_encoding.get(row['frequency'], 2))
            
            # Income bracket encoding
            income_encoding = {'low': 1, 'medium': 2, 'high': 3}
            feature_vector.append(income_encoding.get(row['income_bracket'], 2))
            
            # Life stage encoding
            life_encoding = {'young_single': 1, 'young_couple': 2, 'family': 3, 'empty_nest': 4}
            feature_vector.append(life_encoding.get(row['life_stage'], 1))
            
            # Text-based features (keyword matching)
            description = str(row['description']).lower()
            
            # Count discretionary keywords
            discretionary_count = 0
            for keywords in self.discretionary_keywords.values():
                discretionary_count += sum(1 for keyword in keywords if keyword in description)
            feature_vector.append(discretionary_count)
            
            # Count non-discretionary keywords
            nondiscretionary_count = 0
            for keywords in self.nondiscretionary_keywords.values():
                nondiscretionary_count += sum(1 for keyword in keywords if keyword in description)
            feature_vector.append(nondiscretionary_count)
            
            # Category-based features
            category = row.get('category', 'miscellaneous')
            category_discretionary_ratio = self.category_discretionary_ratios.get(category, 0.5)
            feature_vector.append(category_discretionary_ratio)
            
            # Description length and complexity
            feature_vector.append(len(description))
            feature_vector.append(len(description.split()))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_classifier(self, df: pd.DataFrame, labels: np.ndarray, test_size: float = 0.2):
        """Train the discretionary spending classifier"""
        
        logger.info("Training discretionary spending classifier")
        
        # Extract features
        X = self.extract_features(df)
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble classifier
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_classifier = None
        
        for name, clf in classifiers.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            mean_score = cv_scores.mean()
            
            logger.info(f"{name} CV score: {mean_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_classifier = clf
        
        # Train best classifier on full training set
        best_classifier.fit(X_train_scaled, y_train)
        
        # Test performance
        test_score = best_classifier.score(X_test_scaled, y_test)
        y_pred = best_classifier.predict(X_test_scaled)
        
        logger.info(f"Test accuracy: {test_score:.3f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        self.classifier = best_classifier
        self.is_fitted = True
        
        return best_classifier
    
    def classify_expense(self, expense_data: Dict) -> Dict:
        """Classify a single expense as discretionary or non-discretionary"""
        
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before use")
        
        # Convert to DataFrame format
        df = pd.DataFrame([expense_data])
        
        # Extract features
        X = self.extract_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.classifier.predict(X_scaled)[0]
        probability = self.classifier.predict_proba(X_scaled)[0]
        
        # Rule-based confidence boost
        rule_based_prediction = self.rule_based_classification(expense_data)
        
        # Combine ML and rule-based predictions
        if rule_based_prediction is not None:
            if rule_based_prediction == prediction:
                confidence = min(0.95, max(probability) + 0.1)  # Boost confidence
            else:
                confidence = max(probability)  # Keep original confidence
        else:
            confidence = max(probability)
        
        return {
            'is_discretionary': bool(prediction),
            'confidence': float(confidence),
            'discretionary_probability': float(probability[1]),
            'nondiscretionary_probability': float(probability[0]),
            'rule_based_prediction': rule_based_prediction,
            'classification_method': 'ml_with_rules'
        }
    
    def rule_based_classification(self, expense_data: Dict) -> Optional[bool]:
        """Rule-based classification as fallback/validation"""
        
        description = str(expense_data.get('description', '')).lower()
        amount = expense_data.get('amount', 0)
        category = expense_data.get('category', '').lower()
        
        # Strong non-discretionary indicators
        strong_nondiscretionary = [
            'rent', 'mortgage', 'utilities', 'insurance', 'healthcare',
            'loan payment', 'tax', 'childcare', 'prescription'
        ]
        
        for keyword in strong_nondiscretionary:
            if keyword in description:
                return False
        
        # Strong discretionary indicators
        strong_discretionary = [
            'vacation', 'entertainment', 'luxury', 'hobby', 'gift',
            'restaurant', 'bar', 'movie', 'concert', 'spa'
        ]
        
        for keyword in strong_discretionary:
            if keyword in description:
                return True
        
        # Category-based rules
        if category in ['housing', 'utilities', 'insurance', 'healthcare']:
            return False
        elif category in ['entertainment', 'travel', 'hobbies', 'luxury']:
            return True
        
        # Amount-based rules (context dependent)
        income = expense_data.get('income', 50000)
        if amount > income * 0.1:  # Large expense relative to income
            # Large expenses are often non-discretionary (rent, car payment)
            return False
        
        return None  # Unable to classify with rules
    
    def analyze_spending_pattern(self, expenses: List[Dict]) -> Dict:
        """Analyze overall spending pattern and discretionary ratio"""
        
        total_amount = 0
        discretionary_amount = 0
        classifications = []
        
        for expense in expenses:
            classification = self.classify_expense(expense)
            classifications.append(classification)
            
            amount = expense.get('amount', 0)
            total_amount += amount
            
            if classification['is_discretionary']:
                discretionary_amount += amount
        
        discretionary_ratio = discretionary_amount / total_amount if total_amount > 0 else 0
        
        # Category breakdown
        category_breakdown = {}
        for i, expense in enumerate(expenses):
            category = expense.get('category', 'miscellaneous')
            amount = expense.get('amount', 0)
            is_discretionary = classifications[i]['is_discretionary']
            
            if category not in category_breakdown:
                category_breakdown[category] = {
                    'total': 0,
                    'discretionary': 0,
                    'nondiscretionary': 0
                }
            
            category_breakdown[category]['total'] += amount
            if is_discretionary:
                category_breakdown[category]['discretionary'] += amount
            else:
                category_breakdown[category]['nondiscretionary'] += amount
        
        return {
            'total_spending': total_amount,
            'discretionary_spending': discretionary_amount,
            'nondiscretionary_spending': total_amount - discretionary_amount,
            'discretionary_ratio': discretionary_ratio,
            'expense_count': len(expenses),
            'discretionary_count': sum(1 for c in classifications if c['is_discretionary']),
            'category_breakdown': category_breakdown,
            'classifications': classifications
        }
    
    def save_model(self, path: str):
        """Save trained classifier"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Classifier saved to {path}")
    
    def load_model(self, path: str):
        """Load trained classifier"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Classifier loaded from {path}")

def main():
    """Demo the discretionary spending classifier"""
    
    # Initialize classifier
    classifier = DiscretionarySpendingClassifier()
    
    # Generate training data
    training_df, training_labels = classifier.generate_training_data(5000)
    
    # Train classifier
    classifier.train_classifier(training_df, training_labels)
    
    # Test on sample expenses
    sample_expenses = [
        {'amount': 1200, 'description': 'rent payment', 'category': 'housing', 'frequency': 'monthly'},
        {'amount': 50, 'description': 'grocery shopping', 'category': 'food', 'frequency': 'weekly'},
        {'amount': 80, 'description': 'dinner at restaurant', 'category': 'dining', 'frequency': 'weekly'},
        {'amount': 15, 'description': 'movie tickets', 'category': 'entertainment', 'frequency': 'monthly'},
        {'amount': 200, 'description': 'car insurance', 'category': 'insurance', 'frequency': 'monthly'},
        {'amount': 500, 'description': 'vacation hotel', 'category': 'travel', 'frequency': 'yearly'}
    ]
    
    print("\nSample Expense Classifications:")
    for expense in sample_expenses:
        result = classifier.classify_expense(expense)
        print(f"${expense['amount']:>6} - {expense['description']:<20} -> "
              f"{'Discretionary' if result['is_discretionary'] else 'Non-discretionary':>15} "
              f"(confidence: {result['confidence']:.2f})")
    
    # Analyze overall pattern
    pattern_analysis = classifier.analyze_spending_pattern(sample_expenses)
    print(f"\nSpending Pattern Analysis:")
    print(f"Total Spending: ${pattern_analysis['total_spending']:.2f}")
    print(f"Discretionary: ${pattern_analysis['discretionary_spending']:.2f} "
          f"({pattern_analysis['discretionary_ratio']:.1%})")
    print(f"Non-discretionary: ${pattern_analysis['nondiscretionary_spending']:.2f}")

if __name__ == "__main__":
    main() 