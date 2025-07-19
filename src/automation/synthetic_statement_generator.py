"""
Synthetic Bank Statement Generator

Generates realistic fake bank statements using Faker for training data.
Produces both CSV/JSON transaction data and PDF statements.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

@dataclass
class Transaction:
    """Transaction data structure"""
    date: str
    description: str
    amount: float
    balance: float
    category: str
    transaction_type: str  # 'debit' or 'credit'

@dataclass
class BankProfile:
    """Bank account profile for synthetic generation"""
    account_name: str
    account_number: str
    starting_balance: float
    monthly_income: float
    spending_pattern: str  # 'high', 'medium', 'low'
    transaction_frequency: int  # transactions per month
    categories: List[str]

class SyntheticStatementGenerator:
    """Generates synthetic bank statements for training data"""
    
    def __init__(self):
        self.merchant_categories = {
            "Groceries": [
                "Whole Foods Market", "Trader Joe's", "Safeway", "Kroger", 
                "Walmart Grocery", "Target Grocery", "Costco", "Aldi",
                "Sprouts Farmers Market", "Food Lion", "Publix", "HEB"
            ],
            "Transportation": [
                "Shell", "Exxon", "BP", "Chevron", "Mobil", "7-Eleven Gas",
                "Uber", "Lyft", "Taxi Service", "Parking Garage", "Metro Transit",
                "Enterprise Rent-A-Car", "Hertz", "Budget Car Rental"
            ],
            "Utilities": [
                "Electric Company", "Water Department", "Gas Company", 
                "Internet Provider", "Phone Company", "Cable TV", "Waste Management",
                "Home Security", "Smart Home Services"
            ],
            "Entertainment": [
                "Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+",
                "Movie Theater", "Restaurant", "Bar", "Concert Venue",
                "Sports Arena", "Gym Membership", "Fitness Center"
            ],
            "Healthcare": [
                "Medical Center", "Pharmacy", "Doctor Office", "Hospital",
                "Dental Office", "Vision Center", "Health Insurance",
                "Medical Equipment", "Lab Services"
            ],
            "Housing": [
                "Rent Payment", "Mortgage Payment", "Home Insurance",
                "Property Tax", "Home Maintenance", "Furniture Store",
                "Home Depot", "Lowe's", "IKEA"
            ],
            "Shopping": [
                "Target", "Walmart", "Amazon", "Best Buy", "Macy's",
                "Nordstrom", "Gap", "Old Navy", "Nike", "Adidas",
                "Apple Store", "Microsoft Store"
            ],
            "Income": [
                "Salary Deposit", "Payroll", "Direct Deposit", "Bonus Payment",
                "Commission", "Freelance Payment", "Investment Income",
                "Refund", "Reimbursement"
            ]
        }
        
        self.spending_patterns = {
            "high": {"min_amount": 50, "max_amount": 500, "debit_ratio": 0.8},
            "medium": {"min_amount": 20, "max_amount": 200, "debit_ratio": 0.7},
            "low": {"min_amount": 10, "max_amount": 100, "debit_ratio": 0.6}
        }
        
        # Setup output directories
        self.output_dir = Path("data/statements/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.output_dir / "json"
        self.json_dir.mkdir(exist_ok=True)
        self.pdf_dir = self.output_dir / "pdf"
        self.pdf_dir.mkdir(exist_ok=True)
    
    def generate_bank_profile(self, profile_id: int) -> BankProfile:
        """Generate a synthetic bank account profile"""
        
        # Generate account details
        account_name = fake.name()
        account_number = fake.credit_card_number()
        
        # Generate financial parameters
        starting_balance = random.uniform(1000, 50000)
        monthly_income = random.uniform(2000, 15000)
        
        # Randomize spending pattern
        spending_pattern = random.choice(["high", "medium", "low"])
        transaction_frequency = random.randint(20, 80)  # transactions per month
        
        # Select categories (exclude Income for now, will add separately)
        available_categories = [cat for cat in self.merchant_categories.keys() if cat != "Income"]
        num_categories = random.randint(4, len(available_categories))
        categories = random.sample(available_categories, num_categories)
        
        return BankProfile(
            account_name=account_name,
            account_number=account_number,
            starting_balance=starting_balance,
            monthly_income=monthly_income,
            spending_pattern=spending_pattern,
            transaction_frequency=transaction_frequency,
            categories=categories
        )
    
    def generate_monthly_transactions(self, profile: BankProfile, month: int) -> List[Transaction]:
        """Generate transactions for a specific month"""
        
        transactions = []
        current_balance = profile.starting_balance
        
        # Determine number of transactions for this month
        num_transactions = profile.transaction_frequency
        
        # Generate transaction dates throughout the month
        start_date = datetime.now() - timedelta(days=30 * month)
        dates = [start_date + timedelta(days=random.randint(0, 29)) for _ in range(num_transactions)]
        dates.sort()
        
        # Add income transactions (1-2 per month)
        income_dates = random.sample(dates[:10], min(2, len(dates[:10])))
        for date in income_dates:
            income_amount = profile.monthly_income + random.uniform(0, 1000)
            current_balance += income_amount
            
            transaction = Transaction(
                date=date.strftime("%Y-%m-%d"),
                description=random.choice(self.merchant_categories["Income"]),
                amount=income_amount,
                balance=current_balance,
                category="Income",
                transaction_type="credit"
            )
            transactions.append(transaction)
        
        # Generate spending transactions
        spending_pattern = self.spending_patterns[profile.spending_pattern]
        remaining_dates = [d for d in dates if d not in income_dates]
        
        for date in remaining_dates:
            # Select category
            category = random.choice(profile.categories)
            merchant = random.choice(self.merchant_categories[category])
            
            # Generate amount
            min_amt = spending_pattern["min_amount"]
            max_amt = spending_pattern["max_amount"]
            amount = random.uniform(min_amt, max_amt)
            
            # Determine if debit or credit
            if random.random() < spending_pattern["debit_ratio"]:
                amount = -amount  # Debit
                transaction_type = "debit"
            else:
                transaction_type = "credit"
            
            current_balance += amount
            
            transaction = Transaction(
                date=date.strftime("%Y-%m-%d"),
                description=merchant,
                amount=amount,
                balance=current_balance,
                category=category,
                transaction_type=transaction_type
            )
            transactions.append(transaction)
        
        # Sort by date
        transactions.sort(key=lambda x: x.date)
        return transactions
    
    def save_transactions_csv(self, transactions: List[Transaction], filename: str):
        """Save transactions to CSV file"""
        df = pd.DataFrame([vars(t) for t in transactions])
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(transactions)} transactions to {csv_path}")
        return csv_path
    
    def save_transactions_json(self, transactions: List[Transaction], filename: str):
        """Save transactions to JSON file"""
        json_data = [vars(t) for t in transactions]
        json_path = self.json_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved {len(transactions)} transactions to {json_path}")
        return json_path
    
    def generate_pdf_statement(self, transactions: List[Transaction], profile: BankProfile, filename: str):
        """Generate PDF bank statement"""
        
        pdf_path = self.pdf_dir / f"{filename}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        
        # Header
        header_text = f"""
        <b>BANK STATEMENT</b><br/>
        Account: {profile.account_name}<br/>
        Account Number: {profile.account_number}<br/>
        Statement Period: {transactions[0].date} to {transactions[-1].date}<br/>
        Starting Balance: ${profile.starting_balance:,.2f}<br/>
        Ending Balance: ${transactions[-1].balance:,.2f}
        """
        story.append(Paragraph(header_text, title_style))
        story.append(Spacer(1, 20))
        
        # Transaction table
        table_data = [['Date', 'Description', 'Amount', 'Balance']]
        for t in transactions:
            amount_str = f"${t.amount:,.2f}" if t.amount >= 0 else f"-${abs(t.amount):,.2f}"
            balance_str = f"${t.balance:,.2f}"
            table_data.append([t.date, t.description, amount_str, balance_str])
        
        table = Table(table_data, colWidths=[1*inch, 3*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        logger.info(f"Generated PDF statement: {pdf_path}")
        return pdf_path
    
    def generate_synthetic_corpus(self, num_profiles: int = 100, months_per_profile: int = 3):
        """Generate a complete synthetic corpus"""
        
        logger.info(f"Generating synthetic corpus: {num_profiles} profiles, {months_per_profile} months each")
        
        corpus_metadata = {
            "generation_date": datetime.now().isoformat(),
            "num_profiles": num_profiles,
            "months_per_profile": months_per_profile,
            "profiles": []
        }
        
        for profile_id in range(num_profiles):
            logger.info(f"Generating profile {profile_id + 1}/{num_profiles}")
            
            # Generate profile
            profile = self.generate_bank_profile(profile_id)
            
            profile_metadata = {
                "profile_id": profile_id,
                "account_name": profile.account_name,
                "account_number": profile.account_number,
                "starting_balance": profile.starting_balance,
                "monthly_income": profile.monthly_income,
                "spending_pattern": profile.spending_pattern,
                "transaction_frequency": profile.transaction_frequency,
                "categories": profile.categories,
                "files": []
            }
            
            # Generate transactions for each month
            for month in range(months_per_profile):
                transactions = self.generate_monthly_transactions(profile, month)
                
                if transactions:
                    # Create filename
                    filename = f"profile_{profile_id:04d}_month_{month:02d}"
                    
                    # Save in different formats
                    csv_path = self.save_transactions_csv(transactions, filename)
                    json_path = self.save_transactions_json(transactions, filename)
                    pdf_path = self.generate_pdf_statement(transactions, profile, filename)
                    
                    profile_metadata["files"].append({
                        "month": month,
                        "csv_file": str(csv_path),
                        "json_file": str(json_path),
                        "pdf_file": str(pdf_path),
                        "num_transactions": len(transactions),
                        "start_date": transactions[0].date,
                        "end_date": transactions[-1].date
                    })
            
            corpus_metadata["profiles"].append(profile_metadata)
        
        # Save corpus metadata
        metadata_path = self.output_dir / "corpus_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(corpus_metadata, f, indent=2)
        
        logger.info(f"Synthetic corpus generation complete!")
        logger.info(f"Total profiles: {num_profiles}")
        logger.info(f"Total files generated: {sum(len(p['files']) for p in corpus_metadata['profiles'])}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return corpus_metadata


def main():
    """Generate synthetic bank statement corpus"""
    generator = SyntheticStatementGenerator()
    
    # Generate a small corpus for testing
    corpus_metadata = generator.generate_synthetic_corpus(
        num_profiles=10,  # Start with 10 profiles
        months_per_profile=2  # 2 months each
    )
    
    print("Synthetic corpus generation complete!")
    print(f"Generated {corpus_metadata['num_profiles']} profiles")
    print(f"Total files: {sum(len(p['files']) for p in corpus_metadata['profiles'])}")


if __name__ == "__main__":
    main() 