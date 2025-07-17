#!/usr/bin/env python3
"""
Single-use utility to generate random people profiles
and output them under data/inputs/people/current/person_XXX/.

Usage:
    pip install faker
    python util/generate_people.py --count 5
"""

import os
import json
import random
import argparse
from datetime import datetime, timedelta
from faker import Faker

faker = Faker()

OUTPUT_DIR = "data/inputs/people/current"
ASSET_CLASSES = ["stocks", "bonds", "real_estate", "crypto", "commodities", "private_equity", "cash"]

FAMILY_STATUSES = [
    "single", "married_no_children", "married_with_children",
    "divorced", "divorced_single_parent", "widowed"
]
RISK_LEVELS = ["low", "moderate", "high"]
INCOME_LEVELS = ["low", "medium", "high"]
TAX_BRACKETS = ["10%", "12%", "22%", "24%", "32%", "35%", "37%"]

def random_allocation(classes, n=3):
    chosen = random.sample(classes, n)
    weights = [random.random() for _ in chosen]
    total = sum(weights)
    return {c: round(w/total, 2) for c, w in zip(chosen, weights)}

def gen_profile(idx):
    name = faker.name()
    age = random.randint(18, 80)
    income_level = random.choice(INCOME_LEVELS)
    risk = random.choice(RISK_LEVELS)
    return {
        "id": f"person_{idx:03d}",
        "name": name,
        "age": age,
        "occupation": faker.job(),
        "location": f"{faker.city()}, {faker.state_abbr()}",
        "family_status": random.choice(FAMILY_STATUSES),
        "income_level": income_level,
        "risk_tolerance": risk,
        "investment_horizon": random.randint(1, 40),
        "notes": f"Auto-generated profile for {name}"
    }

def gen_financial_state(pid, profile):
    # income based on level
    if profile["income_level"] == "low":
        salary = random.randint(20000, 50000)
    elif profile["income_level"] == "medium":
        salary = random.randint(50000, 100000)
    else:
        salary = random.randint(100000, 200000)
    bonus = int(salary * random.uniform(0, 0.2))
    now = datetime.utcnow().isoformat() + "Z"
    # assets & liabilities
    cash = random.randint(0, 100000)
    inv = random.randint(0, 500000)
    real_estate = random.choice([0, random.randint(50000, 1000000)])
    retirement = random.randint(0, 300000)
    other = random.randint(0, 50000)
    mortgage = random.randint(0, real_estate) if real_estate else 0
    student = random.choice([0, random.randint(0, 100000)])
    credit = random.randint(0, 20000)
    expense_base = salary // 12
    return {
        "id": pid,
        "timestamp": now,
        "assets": {
            "cash": cash,
            "investments": inv,
            "real_estate": real_estate,
            "retirement_accounts": retirement,
            "other_assets": other
        },
        "liabilities": {
            "mortgage": mortgage,
            "student_loans": student,
            "credit_cards": credit,
            "other_debt": random.randint(0, 10000)
        },
        "income": {
            "annual_salary": salary,
            "bonus": bonus,
            "other_income": random.randint(0, salary // 5)
        },
        "expenses": {
            "monthly_living": expense_base,
            "monthly_mortgage": round(mortgage/360, 2),
            "monthly_utilities": random.randint(200, 600),
            "monthly_food": random.randint(300, 800),
            "monthly_transportation": random.randint(100, 600),
            "monthly_entertainment": random.randint(100, 500),
            "monthly_savings": round(expense_base * random.uniform(0.05, 0.3), 2)
        }
    }

def gen_goals(pid):
    # one per horizon
    short = {
        "id": "goal_001",
        "name": "Short-term goal",
        "target_amount": random.randint(1000, 50000),
        "target_date": (datetime.utcnow() + timedelta(days=180)).date().isoformat(),
        "priority": random.choice(["low", "medium", "high"]),
        "description": "Auto-generated short-term goal"
    }
    medium = {
        "id": "goal_002",
        "name": "Medium-term goal",
        "target_amount": random.randint(50000, 200000),
        "target_date": (datetime.utcnow() + timedelta(days=365*3)).date().isoformat(),
        "priority": random.choice(["low", "medium", "high"]),
        "description": "Auto-generated medium-term goal"
    }
    long = {
        "id": "goal_003",
        "name": "Long-term goal",
        "target_amount": random.randint(200000, 2000000),
        "target_date": (datetime.utcnow() + timedelta(days=365*20)).date().isoformat(),
        "priority": random.choice(["low", "medium", "high"]),
        "description": "Auto-generated long-term goal"
    }
    return {"id": pid, "short_term_goals": [short], "medium_term_goals": [medium], "long_term_goals": [long]}

def gen_life_events(pid):
    past = [{
        "id": "event_001",
        "name": "Auto past event",
        "date": (datetime.utcnow() - timedelta(days=random.randint(30, 365*5))).date().isoformat(),
        "financial_impact": random.randint(-50000, 50000),
        "description": "Random past event",
        "category": random.choice(["career", "family", "education", "housing"])
    }]
    planned = []
    for i in range(2, 5):
        planned.append({
            "id": f"event_{i:03d}",
            "name": f"Planned event {i}",
            "date": (datetime.utcnow() + timedelta(days=random.randint(90, 365*10))).date().isoformat(),
            "expected_impact": random.randint(-100000, 100000),
            "description": "Auto-generated future event",
            "category": random.choice(["career", "family", "investment", "retirement"]),
            "probability": round(random.uniform(0.1, 0.9), 2)
        })
    return {"id": pid, "past_events": past, "planned_events": planned}

def gen_preferences(pid):
    pref = {
        "id": pid,
        "risk_tolerance": {
            "level": random.choice(RISK_LEVELS),
            "score": random.randint(1, 10),
            "description": "Auto-generated risk profile"
        },
        "investment_preferences": {
            "preferred_asset_classes": random.sample(ASSET_CLASSES, 3),
            "avoided_asset_classes": random.sample(ASSET_CLASSES, 2),
            "target_asset_allocation": random_allocation(ASSET_CLASSES, 3)
        },
        "liquidity_needs": {
            "emergency_fund_target": random.randint(5000, 100000),
            "monthly_cash_flow_needs": random.randint(1000, 10000),
            "liquidity_priority": random.choice(["low", "medium", "high"])
        },
        "tax_considerations": {
            "tax_bracket": random.choice(TAX_BRACKETS),
            "prefer_tax_advantaged_accounts": random.choice([True, False]),
            "tax_loss_harvesting": random.choice([True, False])
        },
        "constraints": {
            "ethical_investing": random.choice([True, False]),
            "geographic_preferences": [faker.country() for _ in range(2)],
            "minimum_investment_amounts": {
                random.choice(ASSET_CLASSES): random.randint(100, 50000)
            }
        }
    }
    return pref

def main(count):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(1, count+1):
        pid = f"person_{i:03d}"
        person_dir = os.path.join(OUTPUT_DIR, pid)
        os.makedirs(person_dir, exist_ok=True)

        profile = gen_profile(i)
        fs = gen_financial_state(pid, profile)
        goals = gen_goals(pid)
        events = gen_life_events(pid)
        prefs = gen_preferences(pid)

        # write files
        for name, obj in [
            ("profile.json", profile),
            ("financial_state.json", fs),
            ("goals.json", goals),
            ("life_events.json", events),
            ("preferences.json", prefs)
        ]:
            path = os.path.join(person_dir, name)
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)

    print(f"Generated {count} profiles in {OUTPUT_DIR}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of profiles to generate")
    args = parser.parse_args()
    main(args.count) 